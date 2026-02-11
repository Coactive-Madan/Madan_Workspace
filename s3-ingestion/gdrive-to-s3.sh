#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────
GDRIVE_REMOTE="gdrive"
S3_REMOTE="s3"
GDRIVE_PATH=""
S3_BUCKET=""
S3_PATH=""
TRANSFERS=16
LOG_DIR="$HOME/.local/share/gdrive-to-s3/logs"
DRY_RUN=false
SYNC_MODE=false  # false = copy (add new), true = sync (mirror, deletes extras on dest)

# ─── Usage ───────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Copy (or sync) files from Google Drive to S3 using rclone.

Required:
  -s, --source PATH       Google Drive source path (e.g., "My Folder/subfolder")
  -b, --bucket BUCKET     S3 bucket name
  -d, --dest PATH         S3 destination path within the bucket (default: "")

Options:
  -t, --transfers N       Parallel transfers (default: 16)
  -n, --dry-run           Preview only, don't copy anything
  --sync                  Mirror mode (deletes files on S3 not in Drive)
  --gdrive-remote NAME    Rclone remote name for Google Drive (default: gdrive)
  --s3-remote NAME        Rclone remote name for S3 (default: s3)
  -h, --help              Show this help

Examples:
  $(basename "$0") -s "Projects/2024" -b my-bucket -d backups/projects
  $(basename "$0") -s "Photos" -b my-bucket -d photos --dry-run
  $(basename "$0") -s "Documents" -b my-bucket --sync
EOF
  exit 0
}

# ─── Parse args ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source)       GDRIVE_PATH="$2"; shift 2 ;;
    -b|--bucket)       S3_BUCKET="$2"; shift 2 ;;
    -d|--dest)         S3_PATH="$2"; shift 2 ;;
    -t|--transfers)    TRANSFERS="$2"; shift 2 ;;
    -n|--dry-run)      DRY_RUN=true; shift ;;
    --sync)            SYNC_MODE=true; shift ;;
    --gdrive-remote)   GDRIVE_REMOTE="$2"; shift 2 ;;
    --s3-remote)       S3_REMOTE="$2"; shift 2 ;;
    -h|--help)         usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# ─── Validate ────────────────────────────────────────────────────
if [[ -z "$GDRIVE_PATH" || -z "$S3_BUCKET" ]]; then
  echo "Error: --source and --bucket are required."
  echo ""
  usage
fi

# Check rclone is installed
if ! command -v rclone &>/dev/null; then
  echo "Error: rclone is not installed. Run: brew install rclone"
  exit 1
fi

# Check remotes are configured
for remote in "$GDRIVE_REMOTE" "$S3_REMOTE"; do
  if ! rclone listremotes | grep -q "^${remote}:$"; then
    echo "Error: rclone remote '${remote}' not found."
    echo "Run 'rclone config' to set it up."
    exit 1
  fi
done

# ─── Build paths ─────────────────────────────────────────────────
SRC="${GDRIVE_REMOTE}:${GDRIVE_PATH}"
if [[ -n "$S3_PATH" ]]; then
  DST="${S3_REMOTE}:${S3_BUCKET}/${S3_PATH}"
else
  DST="${S3_REMOTE}:${S3_BUCKET}"
fi

# ─── Logging ─────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"

# ─── Build rclone command ───────────────────────────────────────
CMD="rclone"
if $SYNC_MODE; then
  CMD+=" sync"
else
  CMD+=" copy"
fi

CMD+=" \"$SRC\" \"$DST\""
CMD+=" --transfers $TRANSFERS"
CMD+=" --progress"
CMD+=" --log-file=\"$LOG_FILE\""
CMD+=" --log-level INFO"
CMD+=" --stats 10s"
CMD+=" --checksum"        # verify by checksum, not just size/time
CMD+=" --retries 3"
CMD+=" --low-level-retries 10"

if $DRY_RUN; then
  CMD+=" --dry-run"
fi

# ─── Run ─────────────────────────────────────────────────────────
MODE="copy"
$SYNC_MODE && MODE="sync"
$DRY_RUN && MODE+=" (dry-run)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Google Drive → S3  [$MODE]"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Source:     $SRC"
echo "  Dest:       $DST"
echo "  Transfers:  $TRANSFERS"
echo "  Log:        $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

eval "$CMD"
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  echo ""
  echo "Done. Log saved to: $LOG_FILE"
else
  echo ""
  echo "Failed (exit code $EXIT_CODE). Check log: $LOG_FILE"
  exit $EXIT_CODE
fi
