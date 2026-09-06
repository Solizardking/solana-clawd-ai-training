#!/usr/bin/env bash
# Watch a Hugging Face Job until it reaches a terminal stage.
#
# Watching never touches the remote job: Ctrl-C stops the polling loop only.
#
# Usage:
#   ./scripts/watch_hf_job.sh                          # newest job in your namespace
#   ./scripts/watch_hf_job.sh <JOB_ID>
#   ./scripts/watch_hf_job.sh <JOB_ID> 30              # poll every 30s
#   LABEL=solana-clawd-nemotron35 ./scripts/watch_hf_job.sh
#   ON_SUCCESS="python3 scripts/verify_full_goal_release.py --strict" \
#     ./scripts/watch_hf_job.sh <JOB_ID>
#
# Environment:
#   LABEL        Resolve the newest job carrying this label when no JOB_ID given.
#   ON_SUCCESS   Command to run when the job reaches COMPLETED.
#   LOG_TAIL     Lines of logs to print on failure (default 200).
#
# Stages come from huggingface_hub JobStage:
#   SCHEDULING, RUNNING            -> keep waiting
#   COMPLETED                      -> success
#   ERROR, CANCELED, DELETED       -> failure

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JOB_ID="${1:-}"
INTERVAL_SECONDS="${2:-60}"
LOG_TAIL="${LOG_TAIL:-200}"

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found. Install with: pip install --upgrade huggingface_hub" >&2
  exit 1
fi

# Resolve the newest job if the caller did not name one.
if [[ -z "$JOB_ID" ]]; then
  PS_ARGS=(ps --all --json)
  if [[ -n "${LABEL:-}" ]]; then
    PS_ARGS+=(--label "$LABEL")
  fi
  JOB_ID="$(hf jobs "${PS_ARGS[@]}" | python3 -c '
import json, sys

try:
    jobs = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit(0)
if isinstance(jobs, dict):
    jobs = jobs.get("jobs", [])
# Newest first: the API returns most-recent-first, but sort defensively.
jobs.sort(key=lambda j: j.get("created_at") or "", reverse=True)
if jobs:
    print(jobs[0].get("id", ""))
')"
  if [[ -z "$JOB_ID" ]]; then
    echo "Could not find a job to watch${LABEL:+ with label $LABEL}." >&2
    echo "List jobs with: hf jobs ps --all" >&2
    exit 1
  fi
  echo "Resolved newest job${LABEL:+ with label $LABEL}: $JOB_ID"
fi

# Extract .status.stage from `hf jobs inspect --json`, which may return either a
# single object or a list.
read_stage() {
  hf jobs inspect "$JOB_ID" --json 2>/dev/null | python3 -c '
import json, sys

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    print("UNKNOWN|failed to parse inspect output")
    sys.exit(0)
if isinstance(data, list):
    data = data[0] if data else {}
if not isinstance(data, dict):
    data = {}
status = data.get("status") or {}
if isinstance(status, str):
    stage, message = status, ""
else:
    stage = status.get("stage") or "UNKNOWN"
    message = status.get("message") or ""
print(str(stage) + "|" + str(message).replace("\n", " "))
'
}

echo "Watching Hugging Face job: $JOB_ID"
echo "Polling every ${INTERVAL_SECONDS}s. Ctrl-C stops watching; the remote job keeps running."
echo

LAST_STAGE=""
START_TS="$(date +%s)"
UNKNOWN_STREAK=0
# Only emit the "clear line" escape on a real terminal, so piped output and CI
# logs stay clean.
CLEAR_LINE=""
if [[ -t 1 ]]; then
  CLEAR_LINE=$'\033[2K'
fi
# A bad job id or a token without access makes `inspect` fail every time. Give
# up rather than polling forever.
MAX_UNKNOWN="${MAX_UNKNOWN:-5}"

while true; do
  RAW="$(read_stage)"
  STAGE="${RAW%%|*}"
  MESSAGE="${RAW#*|}"
  ELAPSED=$(( $(date +%s) - START_TS ))

  if [[ "$STAGE" != "$LAST_STAGE" ]]; then
    printf '%s[%4ds] %s%s\n' "$CLEAR_LINE" "$ELAPSED" "$STAGE" "${MESSAGE:+ — $MESSAGE}"
    LAST_STAGE="$STAGE"
  elif [[ -n "$CLEAR_LINE" ]]; then
    # Interactive terminal: redraw the elapsed counter in place.
    printf '%s[%4ds] %s\r' "$CLEAR_LINE" "$ELAPSED" "$STAGE"
  fi

  if [[ "$STAGE" == "UNKNOWN" ]]; then
    UNKNOWN_STREAK=$(( UNKNOWN_STREAK + 1 ))
    if (( UNKNOWN_STREAK >= MAX_UNKNOWN )); then
      echo
      echo "Could not read a stage for $JOB_ID after $UNKNOWN_STREAK attempts." >&2
      echo "Check the id and your access with: hf jobs inspect $JOB_ID --json" >&2
      exit 1
    fi
  else
    UNKNOWN_STREAK=0
  fi

  case "$STAGE" in
    RUNNING|SCHEDULING|UNKNOWN)
      sleep "$INTERVAL_SECONDS"
      ;;
    COMPLETED)
      echo
      echo "Job COMPLETED after ${ELAPSED}s."
      if [[ -n "${ON_SUCCESS:-}" ]]; then
        echo "Running: $ON_SUCCESS"
        eval "$ON_SUCCESS"
        exit $?
      fi
      exit 0
      ;;
    ERROR|CANCELED|DELETED)
      echo
      echo "Job ended in stage $STAGE${MESSAGE:+ — $MESSAGE}" >&2
      echo "--- last $LOG_TAIL log lines ---" >&2
      hf jobs logs "$JOB_ID" --tail "$LOG_TAIL" >&2 || true
      exit 1
      ;;
    *)
      echo
      echo "Unrecognized stage: $STAGE" >&2
      echo "Inspect with: hf jobs inspect $JOB_ID --json" >&2
      exit 1
      ;;
  esac
done
