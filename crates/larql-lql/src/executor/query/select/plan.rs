//! Logical planning for `SELECT … FROM EDGES` by Apache DataFusion's
//! optimizer (#16).
//!
//! The statement is expressed as a DataFusion logical plan over a
//! schema-only table source, and DataFusion's standard rule set decides
//! what reaches the scan: a `fetch` (limit pushdown), the score predicate
//! (filter pushdown), a top-k sort for `ORDER BY … LIMIT`, and the removal
//! of the scan for `LIMIT 0`. LQL executes the optimised `TableScan` with
//! its own lazy scan; ORDER BY, WHERE score and LIMIT are still applied by
//! the caller afterwards, which is idempotent with what was pushed.
//!
//! API checked against the datafusion-{common,expr,optimizer} 55.1.0 sources:
//! `TableSource` is `Any + Sync + Send` and has no `as_any`.

use std::sync::Arc;

use datafusion_common::arrow::datatypes::{DataType, Field as ArrowField, Schema, SchemaRef};
use datafusion_common::Result as DfResult;
use datafusion_expr::logical_plan::{LogicalPlan, LogicalPlanBuilder, TableScan};
use datafusion_expr::{col, lit, Expr, TableProviderFilterPushDown, TableSource};
use datafusion_optimizer::{Optimizer, OptimizerContext};

/// Column holding an edge's score; the only predicate the plan carries.
const SCORE_COLUMN: &str = "c_score";

/// What the optimised plan asks of the edge scan.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ScanDecision {
    /// The optimiser removed the scan (`LIMIT 0`): no rows can be returned.
    Skip,
    /// Score predicate and limit both reached the scan: stop after `n`
    /// admitted rows.
    Fetch(usize),
    /// The scan must yield every row (e.g. ORDER BY needs all of them).
    All,
}

/// The `EDGES` relation as DataFusion sees it: a schema, and an exact
/// filter pushdown, because the LQL scan evaluates the score predicate
/// itself. No data lives here.
struct EdgesSource {
    schema: SchemaRef,
}

impl EdgesSource {
    fn new() -> Self {
        Self {
            schema: Arc::new(Schema::new(vec![
                ArrowField::new("layer", DataType::UInt64, false),
                ArrowField::new("feature", DataType::UInt64, false),
                ArrowField::new("top_token", DataType::Utf8, false),
                ArrowField::new("also", DataType::Utf8, false),
                ArrowField::new("relation", DataType::Utf8, false),
                ArrowField::new(SCORE_COLUMN, DataType::Float32, false),
            ])),
        }
    }
}

impl TableSource for EdgesSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Exact; filters.len()])
    }
}

/// Plan the statement, optimise it with DataFusion's default rules, and
/// read the decision off the resulting `TableScan`.
pub(super) fn scan_decision(
    has_order_by: bool,
    limit: usize,
    has_score_predicate: bool,
) -> DfResult<ScanDecision> {
    let mut builder = LogicalPlanBuilder::scan("edges", Arc::new(EdgesSource::new()), None)?;
    if has_score_predicate {
        // A stand-in with the predicate's shape: the scan evaluates the real
        // comparison (`EdgeFilters::score_matches`), so only its placement
        // matters to the plan.
        builder = builder.filter(col(SCORE_COLUMN).gt(lit(0.0f32)))?;
    }
    if has_order_by {
        // Any sort key blocks limit pushdown the same way; which LQL field it
        // is does not change where the limit may go.
        builder = builder.sort(vec![col("layer").sort(true, false)])?;
    }
    let plan = builder.limit(0, Some(limit))?.build()?;
    let optimized = Optimizer::new().optimize(plan, &OptimizerContext::new(), |_, _| {})?;
    Ok(match find_scan(&optimized) {
        None => ScanDecision::Skip,
        Some(scan) => match scan.fetch {
            Some(n) if scan.filters.len() == usize::from(has_score_predicate) => {
                ScanDecision::Fetch(n)
            }
            _ => ScanDecision::All,
        },
    })
}

fn find_scan(plan: &LogicalPlan) -> Option<&TableScan> {
    if let LogicalPlan::TableScan(scan) = plan {
        return Some(scan);
    }
    plan.inputs().into_iter().find_map(find_scan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_reaches_the_scan_without_order_by() {
        assert_eq!(
            scan_decision(false, 5, false).unwrap(),
            ScanDecision::Fetch(5)
        );
    }

    #[test]
    fn score_predicate_and_limit_both_reach_the_scan() {
        assert_eq!(
            scan_decision(false, 5, true).unwrap(),
            ScanDecision::Fetch(5)
        );
    }

    #[test]
    fn order_by_keeps_the_full_scan() {
        assert_eq!(scan_decision(true, 5, false).unwrap(), ScanDecision::All);
        assert_eq!(scan_decision(true, 5, true).unwrap(), ScanDecision::All);
    }

    #[test]
    fn limit_zero_removes_the_scan() {
        assert_eq!(scan_decision(false, 0, false).unwrap(), ScanDecision::Skip);
    }
}
