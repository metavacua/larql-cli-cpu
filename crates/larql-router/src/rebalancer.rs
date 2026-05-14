//! GT6 dynamic rebalancing background task.
//!
//! Runs every `check_interval` seconds and checks for per-layer latency
//! imbalance across replicated shards. When a shard is measurably slower
//! than its peers (ratio > `imbalance_threshold`) and a spare available
//! server exists to replace it, the rebalancer sends `UnassignMsg` to the
//! slow server and triggers gap-fill for the freed layer range.
//!
//! The server receives `UnassignMsg`, drains in-flight requests (up to 30s),
//! sends `DroppingMsg(reason="reassigned")`, and re-enters the available pool.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::{debug, info};

use larql_router_protocol::{RouterMessage, RouterPayload, UnassignMsg};

use crate::grid::GridState;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RebalancerConfig {
    /// How often to run the imbalance check.
    pub check_interval: Duration,
    /// Trigger rebalancing when max(avg_ms) / min(avg_ms) exceeds this ratio
    /// across replicas covering the same layer for at least `sustained_window`.
    pub imbalance_threshold: f32,
    /// Sustained imbalance window before action is taken.
    pub sustained_window: Duration,
    /// Servers that haven't sent a heartbeat within this window are evicted
    /// even if the gRPC stream is still alive. Defensive against deadlocked
    /// servers that keep TCP open but stop sending heartbeats. Default 25 s
    /// = 2.5 × the 10 s heartbeat interval.
    pub stale_heartbeat_timeout: Duration,
}

impl Default for RebalancerConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            imbalance_threshold: 2.0,
            sustained_window: Duration::from_secs(60),
            stale_heartbeat_timeout: Duration::from_secs(25),
        }
    }
}

impl RebalancerConfig {
    pub fn from_cli(interval_secs: u64, threshold: f32) -> Self {
        Self {
            check_interval: Duration::from_secs(interval_secs),
            imbalance_threshold: threshold,
            sustained_window: Duration::from_secs(interval_secs * 2),
            stale_heartbeat_timeout: Duration::from_secs(25),
        }
    }
}

// ── Per-layer imbalance tracker ───────────────────────────────────────────────

/// Tracks how long a given layer has been in imbalanced state.
#[derive(Default)]
struct ImbalanceTracker {
    /// (model_id, layer) → first_seen_imbalanced
    first_seen: HashMap<(String, u32), std::time::Instant>,
}

impl ImbalanceTracker {
    /// Record that this layer is currently imbalanced. Returns true if the
    /// imbalance has been sustained long enough to trigger action.
    fn record(&mut self, key: (String, u32), sustained: Duration) -> bool {
        let entry = self
            .first_seen
            .entry(key)
            .or_insert_with(std::time::Instant::now);
        entry.elapsed() >= sustained
    }

    /// Clear a layer's imbalance record (it is now balanced or was acted on).
    fn clear(&mut self, key: &(String, u32)) {
        self.first_seen.remove(key);
    }
}

// ── Rebalancer task ───────────────────────────────────────────────────────────

/// Spawn the rebalancer background task.
/// Returns immediately; the task runs for the process lifetime.
pub fn spawn(state: Arc<RwLock<GridState>>, cfg: RebalancerConfig) {
    tokio::spawn(rebalancer_task(state, cfg));
}

async fn rebalancer_task(state: Arc<RwLock<GridState>>, cfg: RebalancerConfig) {
    let mut interval = tokio::time::interval(cfg.check_interval);
    let mut tracker = ImbalanceTracker::default();

    loop {
        interval.tick().await;
        evict_stale_heartbeats(&state, cfg.stale_heartbeat_timeout).await;
        check_under_replication(&state).await;
        check_over_replication(&state).await;
        check_imbalance(&state, &cfg, &mut tracker).await;
    }
}

/// Phase 4: pull spares from the available pool to bring under-replicated
/// ranges up to `target_replicas`. Triggered periodically — in addition to
/// the event-driven triggers in `grid.rs` (Available, Ready, deregister).
async fn check_under_replication(state: &Arc<RwLock<GridState>>) {
    let target = state.read().await.target_replicas();
    if target <= 1 {
        return; // No replication target configured.
    }
    let assigned = state.write().await.try_replicate_from_available();
    if assigned > 0 {
        tracing::info!(
            assigned,
            "Rebalancer: replicated under-replicated ranges from available pool"
        );
    }
}

/// Phase 4: drop one replica per over-replicated range each tick by sending
/// `UnassignMsg` to the least-loaded server. Defensive — never drops below
/// `target_replicas` (the over-replicated check already ensures the count is
/// strictly greater).
async fn check_over_replication(state: &Arc<RwLock<GridState>>) {
    // Snapshot ranges + chosen victims while holding only a read lock.
    let plan: Vec<(String, String, u32, u32)> = {
        let g = state.read().await;
        if g.target_replicas() == 0 {
            return;
        }
        g.over_replicated_ranges()
            .into_iter()
            .filter_map(|(model_id, start, end, _surplus)| {
                g.least_loaded_in_range(&model_id, start, end)
                    .map(|e| (e.server_id.clone(), model_id, start, end))
            })
            .collect()
    };
    if plan.is_empty() {
        return;
    }
    for (server_id, model_id, start, end) in plan {
        let tx = {
            let g = state.read().await;
            g.serving_sender(&server_id)
        };
        let Some(tx) = tx else {
            tracing::warn!(server_id, "Over-replication: no sender for victim");
            continue;
        };
        let msg = RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id: model_id.clone(),
                layer_start: start,
                layer_end: end,
                reason: "over_replicated".into(),
            })),
        };
        if tx.try_send(Ok(msg)).is_ok() {
            tracing::info!(
                server_id,
                model_id,
                layers = %format!("{start}-{end}"),
                "Rebalancer: dropping over-replicated replica"
            );
        } else {
            tracing::warn!(
                server_id,
                "Over-replication: UnassignMsg send failed (peer disconnected)"
            );
        }
    }
}

/// Deregister servers that have stopped sending heartbeats. After eviction,
/// re-run gap-fill in case the disappearance exposed a fillable gap.
async fn evict_stale_heartbeats(
    state: &Arc<RwLock<GridState>>,
    timeout: std::time::Duration,
) {
    let stale = state.read().await.stale_server_ids(timeout);
    if stale.is_empty() {
        return;
    }
    let mut guard = state.write().await;
    for sid in &stale {
        tracing::warn!(
            server_id = %sid,
            timeout_s = timeout.as_secs(),
            "Rebalancer: evicting stale server (no heartbeat within timeout)"
        );
        guard.deregister(sid);
    }
    let filled = guard.try_fill_all_gaps();
    if filled > 0 {
        tracing::info!(
            filled,
            "Rebalancer: gap re-fill after stale-heartbeat eviction"
        );
    }
}

async fn check_imbalance(
    state: &Arc<RwLock<GridState>>,
    cfg: &RebalancerConfig,
    tracker: &mut ImbalanceTracker,
) {
    // Collect per-layer latency data across all servers.
    // Group by (model_id, layer) → Vec<(server_id, avg_ms)>.
    let snapshot = {
        let guard = state.read().await;
        let mut by_layer: HashMap<(String, u32), Vec<(String, f32)>> = HashMap::new();
        for (sid, entry) in guard.servers() {
            for (&layer, &(avg_ms, _p99)) in &entry.layer_latencies {
                by_layer
                    .entry((entry.model_id.clone(), layer))
                    .or_default()
                    .push((sid.clone(), avg_ms));
            }
        }
        let has_available = guard.has_available_servers();
        (by_layer, has_available)
    };

    let (by_layer, has_available) = snapshot;

    // Only rebalance if there is a spare server ready to take over.
    if !has_available {
        debug!("Rebalancer: no available servers — skipping imbalance check");
        return;
    }

    for ((model_id, layer), mut servers) in by_layer {
        if servers.len() < 2 {
            // Can't detect imbalance without at least 2 replicas.
            tracker.clear(&(model_id, layer));
            continue;
        }

        servers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let min_ms = servers.first().map(|(_, ms)| *ms).unwrap_or(0.0);
        let max_ms = servers.last().map(|(_, ms)| *ms).unwrap_or(0.0);
        let slowest_server_id = servers.last().map(|(id, _)| id.clone());

        if min_ms <= 0.0 {
            continue;
        }

        let ratio = max_ms / min_ms;
        let key = (model_id.clone(), layer);

        if ratio > cfg.imbalance_threshold {
            let sustained = tracker.record(key.clone(), cfg.sustained_window);
            if sustained {
                // Imbalance has persisted long enough — send UnassignMsg.
                if let Some(ref server_id) = slowest_server_id {
                    info!(
                        model_id = %model_id,
                        layer,
                        ratio = %format!("{ratio:.1}×"),
                        server_id = %server_id,
                        "Rebalancer: sustained imbalance detected — sending UnassignMsg"
                    );
                    send_unassign(state, server_id, &model_id, layer).await;
                    tracker.clear(&key);
                }
            } else {
                debug!(
                    model_id = %model_id,
                    layer,
                    ratio = %format!("{ratio:.1}×"),
                    "Rebalancer: imbalance observed (not yet sustained)"
                );
            }
        } else {
            tracker.clear(&key);
        }
    }
}

/// Send `UnassignMsg` to the serving server identified by `server_id`.
/// The sender channel is stored in `GridState::serving_senders`.
async fn send_unassign(
    state: &Arc<RwLock<GridState>>,
    server_id: &str,
    model_id: &str,
    layer: u32,
) {
    let guard = state.read().await;
    if let Some(tx) = guard.serving_sender(server_id) {
        let msg = RouterMessage {
            payload: Some(RouterPayload::Unassign(UnassignMsg {
                model_id: model_id.to_owned(),
                layer_start: layer,
                layer_end: layer,
                reason: "rebalancing".to_owned(),
            })),
        };
        if let Err(e) = tx.try_send(Ok(msg)) {
            tracing::warn!(server_id, "Rebalancer: failed to send UnassignMsg: {e}");
        }
    } else {
        tracing::warn!(
            server_id,
            "Rebalancer: no sender for server — already disconnected?"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imbalance_tracker_records_and_clears() {
        let mut t = ImbalanceTracker::default();
        let key = ("model".to_string(), 5u32);
        // First record: not sustained yet (window = 1 hour).
        assert!(!t.record(key.clone(), Duration::from_secs(3600)));
        // Clear and re-record: still fresh.
        t.clear(&key);
        assert!(!t.record(key.clone(), Duration::from_secs(3600)));
        // With zero window: sustained immediately.
        let key2 = ("model".to_string(), 6u32);
        assert!(t.record(key2, Duration::from_secs(0)));
    }

    #[test]
    fn rebalancer_config_defaults() {
        let cfg = RebalancerConfig::default();
        assert_eq!(cfg.check_interval, Duration::from_secs(30));
        assert_eq!(cfg.imbalance_threshold, 2.0);
        assert_eq!(cfg.stale_heartbeat_timeout, Duration::from_secs(25));
    }

    #[tokio::test]
    async fn over_replication_drops_least_loaded_replica() {
        use crate::grid::ServerEntry;
        use std::collections::HashMap;
        use tokio::sync::mpsc;

        let state = Arc::new(RwLock::new(GridState::default()));
        let (tx_idle, mut rx_idle) =
            mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (tx_busy, _rx_busy) =
            mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);

        {
            let mut g = state.write().await;
            g.set_target_replicas(2);
            // 3 replicas of model-x 0-4 — surplus 1. idle is least-loaded.
            let busy_1 = ServerEntry {
                server_id: "busy-1".into(),
                listen_url: "http://busy-1".into(),
                model_id: "model-x".into(),
                layer_start: 0,
                layer_end: 4,
                vindex_hash: "h".into(),
                cpu_pct: 0.0,
                ram_used: 0,
                requests_in_flight: 9,
                last_seen: std::time::Instant::now(),
                layer_latencies: HashMap::new(),
            };
            let busy_2 = ServerEntry {
                requests_in_flight: 8,
                server_id: "busy-2".into(),
                listen_url: "http://busy-2".into(),
                ..busy_1.clone()
            };
            let idle = ServerEntry {
                requests_in_flight: 1,
                server_id: "idle".into(),
                listen_url: "http://idle".into(),
                ..busy_1.clone()
            };
            // Idle gets the sender we'll observe; busy-1 gets a separate
            // sender so the test isn't blocked if it ever fired (it shouldn't).
            g.register_with_sender(idle, tx_idle);
            g.register_with_sender(busy_1, tx_busy);
            g.register(busy_2);
        }

        check_over_replication(&state).await;

        let received = rx_idle
            .try_recv()
            .expect("least-loaded server must receive UnassignMsg");
        let Ok(RouterMessage {
            payload: Some(RouterPayload::Unassign(u)),
        }) = received
        else {
            panic!("expected Unassign, got: {received:?}");
        };
        assert_eq!(u.model_id, "model-x");
        assert_eq!(u.layer_start, 0);
        assert_eq!(u.layer_end, 4);
        assert_eq!(u.reason, "over_replicated");
    }

    #[test]
    fn from_cli_derives_sustained_window_from_interval() {
        let cfg = RebalancerConfig::from_cli(15, 2.5);
        assert_eq!(cfg.check_interval, Duration::from_secs(15));
        assert_eq!(cfg.imbalance_threshold, 2.5);
        assert_eq!(cfg.sustained_window, Duration::from_secs(30));
        assert_eq!(cfg.stale_heartbeat_timeout, Duration::from_secs(25));
    }

    fn make_server_entry(
        server_id: &str,
        listen_url: &str,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> crate::grid::ServerEntry {
        crate::grid::ServerEntry {
            server_id: server_id.into(),
            listen_url: listen_url.into(),
            model_id: model_id.into(),
            layer_start,
            layer_end,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
        }
    }

    #[tokio::test]
    async fn under_replication_dispatches_assignment_when_target_above_one() {
        use tokio::sync::mpsc;

        let state = Arc::new(RwLock::new(GridState::default()));
        let (spare_tx, mut spare_rx) =
            mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write().await;
            g.set_target_replicas(2);
            g.register(make_server_entry("a", "http://a", "m", 0, 4));
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        check_under_replication(&state).await;
        let msg = spare_rx
            .try_recv()
            .expect("spare should have been used")
            .expect("ok payload");
        assert!(matches!(msg.payload, Some(RouterPayload::Assign(_))));
    }

    #[tokio::test]
    async fn under_replication_noop_when_target_equals_one() {
        let state = Arc::new(RwLock::new(GridState::default()));
        // target_replicas defaults to 1; should not attempt anything even
        // with a spare in the pool.
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        {
            let mut g = state.write().await;
            g.register(make_server_entry("a", "http://a", "m", 0, 4));
            g.register_available("spare".into(), tx, 1, 0, "/".into());
        }
        check_under_replication(&state).await;
        assert!(rx.try_recv().is_err(), "no assignment should fire at target=1");
    }

    #[tokio::test]
    async fn over_replication_with_no_sender_logs_and_skips() {
        use tokio::sync::mpsc;

        let state = Arc::new(RwLock::new(GridState::default()));
        let (busy_tx, _busy_rx) =
            mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write().await;
            g.set_target_replicas(2);
            // Three replicas, but the least-loaded one has no serving_sender
            // (registered via plain `register`). Test exercises the
            // "no sender for victim" branch.
            let mut idle = make_server_entry("idle", "http://idle", "m", 0, 4);
            idle.requests_in_flight = 1;
            g.register(idle); // no sender path
            let mut busy_1 = make_server_entry("busy-1", "http://busy-1", "m", 0, 4);
            busy_1.requests_in_flight = 5;
            g.register_with_sender(busy_1, busy_tx.clone());
            let mut busy_2 = make_server_entry("busy-2", "http://busy-2", "m", 0, 4);
            busy_2.requests_in_flight = 7;
            g.register_with_sender(busy_2, busy_tx);
        }
        check_over_replication(&state).await;
        // The "no sender" branch is taken; nothing observable on busy_tx
        // because the rebalancer picks `idle` as least-loaded and skips it.
        // Assert through state: no server was removed.
        let g = state.read().await;
        assert_eq!(g.status_response().servers.len(), 3);
    }

    #[tokio::test]
    async fn evict_stale_noop_when_all_fresh() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write().await;
            g.register(make_server_entry("fresh", "http://fresh", "m", 0, 4));
        }
        evict_stale_heartbeats(&state, Duration::from_secs(25)).await;
        let g = state.read().await;
        assert_eq!(g.status_response().servers.len(), 1);
    }

    #[tokio::test]
    async fn check_imbalance_no_op_when_no_available_servers() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let cfg = RebalancerConfig::default();
        let mut tracker = ImbalanceTracker::default();
        {
            let mut g = state.write().await;
            let mut a = make_server_entry("a", "http://a", "m", 0, 0);
            a.layer_latencies.insert(0, (5.0, 10.0));
            let mut b = make_server_entry("b", "http://b", "m", 0, 0);
            b.layer_latencies.insert(0, (50.0, 100.0));
            g.register(a);
            g.register(b);
            // No available pool registered — rebalancer must skip.
        }
        check_imbalance(&state, &cfg, &mut tracker).await;
        // Tracker stays empty — early return before recording.
        assert!(tracker.first_seen.is_empty());
    }

    #[tokio::test]
    async fn check_imbalance_records_then_acts_after_sustained_window() {
        use tokio::sync::mpsc;

        let state = Arc::new(RwLock::new(GridState::default()));
        let (slow_tx, mut slow_rx) =
            mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (fast_tx, _fast_rx) =
            mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (spare_tx, _spare_rx) =
            mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write().await;
            // Two replicas with a 10× latency gap on layer 0.
            let mut slow = make_server_entry("slow", "http://slow", "m", 0, 0);
            slow.layer_latencies.insert(0, (50.0, 100.0));
            let mut fast = make_server_entry("fast", "http://fast", "m", 0, 0);
            fast.layer_latencies.insert(0, (5.0, 10.0));
            g.register_with_sender(slow, slow_tx);
            g.register_with_sender(fast, fast_tx);
            // Spare so the rebalancer is willing to act.
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        // Zero-window config: the first observation immediately becomes
        // sustained, and the rebalancer should send UnassignMsg.
        let cfg = RebalancerConfig {
            check_interval: Duration::from_secs(30),
            imbalance_threshold: 2.0,
            sustained_window: Duration::from_secs(0),
            stale_heartbeat_timeout: Duration::from_secs(25),
        };
        let mut tracker = ImbalanceTracker::default();
        check_imbalance(&state, &cfg, &mut tracker).await;
        let msg = slow_rx
            .try_recv()
            .expect("slow replica should have been Unassigned")
            .expect("ok payload");
        let Some(RouterPayload::Unassign(u)) = msg.payload else {
            panic!("expected Unassign, got {msg:?}");
        };
        assert_eq!(u.reason, "rebalancing");
        assert_eq!(u.layer_start, 0);
        assert_eq!(u.layer_end, 0);
    }

    #[tokio::test]
    async fn check_imbalance_clears_when_balanced() {
        let state = Arc::new(RwLock::new(GridState::default()));
        let (spare_tx, _spare_rx) =
            tokio::sync::mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write().await;
            let mut a = make_server_entry("a", "http://a", "m", 0, 0);
            a.layer_latencies.insert(0, (5.0, 10.0));
            let mut b = make_server_entry("b", "http://b", "m", 0, 0);
            b.layer_latencies.insert(0, (5.5, 10.5));
            g.register(a);
            g.register(b);
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        let cfg = RebalancerConfig {
            sustained_window: Duration::from_secs(60),
            ..RebalancerConfig::default()
        };
        let mut tracker = ImbalanceTracker::default();
        // Pre-populate as if an earlier observation flagged the layer; the
        // balanced state on this tick should clear it.
        tracker
            .first_seen
            .insert(("m".into(), 0), std::time::Instant::now());
        check_imbalance(&state, &cfg, &mut tracker).await;
        assert!(
            tracker.first_seen.is_empty(),
            "balanced layer should clear tracker entry"
        );
    }

    #[tokio::test]
    async fn send_unassign_warns_on_missing_sender() {
        let state = Arc::new(RwLock::new(GridState::default()));
        // No server registered with that id — should hit the "no sender"
        // warn path without panicking.
        send_unassign(&state, "ghost", "m", 0).await;
    }

    #[tokio::test]
    async fn evict_stale_removes_overdue_servers() {
        use crate::grid::ServerEntry;
        use std::collections::HashMap;

        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write().await;
            // Stale server: last_seen 60s ago.
            let stale = ServerEntry {
                server_id: "stale".into(),
                listen_url: "http://stale".into(),
                model_id: "m".into(),
                layer_start: 0,
                layer_end: 4,
                vindex_hash: "h".into(),
                cpu_pct: 0.0,
                ram_used: 0,
                requests_in_flight: 0,
                last_seen: std::time::Instant::now()
                    .checked_sub(Duration::from_secs(60))
                    .unwrap(),
                layer_latencies: HashMap::new(),
            };
            g.register(stale);
        }

        evict_stale_heartbeats(&state, Duration::from_secs(25)).await;

        let g = state.read().await;
        assert_eq!(
            g.status_response().servers.len(),
            0,
            "stale server must be evicted"
        );
    }
}
