#!/usr/bin/env bash
# Watch a Core AI Hugging Face Job without interrupting it, then run the full
# release verifier when the job completes successfully.
#
# This is a thin wrapper over scripts/watch_hf_job.sh. It exists to pin the
# Core AI success action (the release verifier) and the Core AI job label.
#
# Usage:
#   ./scripts/watch_core_ai_hf_job.sh                 # newest core-ai job
#   ./scripts/watch_core_ai_hf_job.sh <JOB_ID>
#   ./scripts/watch_core_ai_hf_job.sh <JOB_ID> 30     # poll every 30s
#
# Previous versions of this script defaulted to a hardcoded job id and matched
# status strings (PENDING/QUEUED/SUCCEEDED) that the Jobs API never emits. The
# real stages are SCHEDULING, RUNNING, COMPLETED, ERROR, CANCELED, DELETED.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export LABEL="${LABEL:-solana-clawd-core-ai}"
export ON_SUCCESS="${ON_SUCCESS:-python3 scripts/verify_full_goal_release.py --strict}"

exec ./scripts/watch_hf_job.sh "$@"
