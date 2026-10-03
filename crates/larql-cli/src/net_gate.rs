//! The one place a build without the `net` cargo feature says no.
//!
//! Compiled only without `net`, so nothing here is dead code in a net build.
//! Every call site is itself `cfg(not(feature = "net"))`. The `?`-returning
//! forms are deliberate: an unconditional `return Err(..)` placed before live
//! code trips `unreachable_code` under `-D warnings`, while
//! `require_net(..)?` type-checks as reachable in both feature states.

use std::error::Error;
use std::fmt;

const NET_REBUILD_HINT: &str =
    "this larql binary was built without the `net` cargo feature (rebuild with `--features net`)";

/// The refusal: what was asked for, and why it cannot be served.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NetUnavailable {
    what: String,
}

impl fmt::Display for NetUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} needs networking; {NET_REBUILD_HINT}", self.what)
    }
}

impl Error for NetUnavailable {}

/// Build the refusal error for `what` (a flag, verb or transport name).
pub(crate) fn net_required(what: &str) -> Box<dyn Error> {
    Box::new(NetUnavailable {
        what: what.to_string(),
    })
}

/// Always `Err`: use as `require_net("...")?;` at the point networking is needed.
pub(crate) fn require_net(what: &str) -> Result<(), Box<dyn Error>> {
    Err(net_required(what))
}

/// Refuse a whole verb, e.g. `refuse_command("pull")`.
pub(crate) fn refuse_command(verb: &str) -> Result<(), Box<dyn Error>> {
    require_net(&format!("`larql {verb}`"))
}

/// Refuse the first network-only flag that was actually given.
pub(crate) fn refuse_flags(flags: &[(&str, bool)]) -> Result<(), Box<dyn Error>> {
    match flags.iter().find(|(_, given)| *given) {
        Some((flag, _)) => Err(net_required(&format!("`{flag}`"))),
        None => Ok(()),
    }
}

/// Catch-all payload for a whole-network verb when `net` is off: the verb
/// stays in the clap tree (so `--help` lists it) and swallows any arguments.
#[derive(clap::Args)]
pub(crate) struct NetStubArgs {
    #[arg(
        value_name = "ARGS",
        trailing_var_arg = true,
        allow_hyphen_values = true,
        num_args = 0..
    )]
    #[allow(
        dead_code,
        reason = "clap fills it; the stub exists only so the verb stays listed and refuses"
    )]
    rest: Vec<String>,
}

#[cfg(test)]
mod tests;
