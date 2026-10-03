#!/usr/bin/env python3
# /// script
# dependencies = ["pyyaml"]
# ///
"""Permissions and concurrency rules every workflow in .github/workflows must keep.

Permissions: the repository's default GITHUB_TOKEN may be read/write, so a workflow that
declares nothing hands every step a write token. Each workflow declares its own, and the
only scopes granted are the ones its steps use. Reading the repository (checkout) is
`contents: read`; the exceptions are named below, with the reason.

Concurrency: a run must only ever be superseded by a newer run of the same thing.
  * the PR number is in the group, so two PRs never cancel one another;
  * a push to main is never cancelled by the next push to main (the record of each main
    commit is the point of running on main), so cancel-in-progress is never a bare `true`
    for a workflow that can run on main.

Run: uv run scripts/test_workflow_hygiene.py      (or python3 with pyyaml)
"""

import unittest

from test_workflow_triggers import load, triggers

# workflow -> the permissions its workflow-level block must be exactly, and why.
DEFAULT = {"contents": "read"}
WORKFLOW_PERMISSIONS = {
    # Cancels a closed PR's unfinished runs through the REST API.
    "cancel-closed-pr-runs": {"actions": "write"},
    # Lists a PR's commits through the REST API; never checks the code out.
    "commit-messages": {"contents": "read", "pull-requests": "read"},
}
# (workflow, job) -> job-level permissions that widen the workflow-level block.
JOB_PERMISSIONS = {
    # Creates the GitHub release and uploads its assets.
    ("release", "release"): {"contents": "write"},
}


def jobs_with_permissions(workflows):
    for name, w in workflows.items():
        for job, body in w["jobs"].items():
            if "permissions" in body:
                yield name, job, body["permissions"]


class Permissions(unittest.TestCase):
    def test_every_workflow_declares_its_own(self):
        missing = [n for n, w in load().items() if "permissions" not in w]
        self.assertEqual(missing, [], "no workflow-level `permissions:` -> repo default token")

    def test_workflow_level_permissions_are_exactly_what_is_needed(self):
        for name, w in load().items():
            with self.subTest(workflow=name):
                self.assertEqual(w.get("permissions"), WORKFLOW_PERMISSIONS.get(name, DEFAULT))

    def test_job_level_permissions_are_only_the_named_exceptions(self):
        got = {(n, j): p for n, j, p in jobs_with_permissions(load())}
        self.assertEqual(got, JOB_PERMISSIONS)


class Concurrency(unittest.TestCase):
    def test_pr_workflows_group_by_pull_request(self):
        for name, w in load().items():
            if name == "cancel-closed-pr-runs" or not triggers(w, "pull_request", "main", None):
                continue
            with self.subTest(workflow=name):
                conc = w.get("concurrency")
                self.assertIsNotNone(conc, "a PR workflow with no concurrency group")
                group = conc["group"] if isinstance(conc, dict) else conc
                self.assertIn("github.workflow", group)
                self.assertTrue(
                    "pull_request.number" in group or "head_ref" in group,
                    f"{group!r} does not separate PRs",
                )

    def test_pushes_to_main_are_never_cancelled(self):
        for name, w in load().items():
            if not triggers(w, "push", "main", None):
                continue
            with self.subTest(workflow=name):
                conc = w.get("concurrency")
                if conc is None:
                    continue
                cancel = conc.get("cancel-in-progress", False)
                self.assertTrue(
                    cancel is False or (isinstance(cancel, str) and "pull_request" in cancel),
                    f"cancel-in-progress={cancel!r} would cancel a main push's run",
                )


if __name__ == "__main__":
    unittest.main()
