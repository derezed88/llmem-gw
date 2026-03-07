#!/bin/bash
# setup-llmem-gw.sh
#
# Clones the llmem-gw repo into the current directory, creates a venv,
# installs requirements, then copies working config from the reference
# installation so llmem-gw.py is ready to run immediately.
#
# Usage (local, same machine):
#   cd /some/target/dir
#   ./setup-llmem-gw.sh --source-dir /home/markj/projects/kaliLinuxNWScripts/mymcp
#
# Usage (remote machine — SSH back to the dev host to pull secrets/config):
#   ./setup-llmem-gw.sh --source-host 192.168.10.111 \
#                        --source-dir /home/markj/projects/kaliLinuxNWScripts/mymcp \
#                        [--source-user markj] [--branch <branch>] [--name <dirname>]
#
# Options:
#   --branch <branch>       Git branch to check out after clone (default: main)
#   --name <dirname>        Directory name to clone into (default: llmem-gw)
#   --source-host <host>    Hostname/IP of the dev machine with the reference install
#   --source-user <user>    SSH user on the source host (default: current $USER)
#   --source-dir <path>     Absolute path to the reference install on the source host
#                           (required)
#
# After completion:
#   cd llmem-gw                                   (or --name value)
#   source venv/bin/activate
#   python llmemctl.py port-list             # verify/adjust ports BEFORE starting
#   python llmemctl.py port-set <plugin> <port>  # if ports conflict with other instances
#   python llmem-gw.py      # terminal 1 - server
#   python shell.py          # terminal 2 - client

set -e

REPO_SSH="git@github.com:derezed88/llmem-gw.git"
REPO_HTTPS="https://github.com/derezed88/llmem-gw.git"
BRANCH="main"
DIR_NAME="llmem-gw"
SOURCE_HOST=""
SOURCE_USER="${USER}"
SOURCE_DIR=""   # derived below after arg parsing

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --name)
            DIR_NAME="$2"
            shift 2
            ;;
        --source-host)
            SOURCE_HOST="$2"
            shift 2
            ;;
        --source-user)
            SOURCE_USER="$2"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--branch <branch>] [--name <dirname>]"
            echo "       $0 --source-host <host> [--source-user <user>] [--source-dir <path>]"
            exit 1
            ;;
    esac
done

# SOURCE_DIR is required — no hardcoded default so the script is portable.
if [ -z "$SOURCE_DIR" ]; then
    echo "ERROR: --source-dir is required (absolute path to your reference install)."
    echo "  Local:   --source-dir /path/to/your/dev/install"
    echo "  Remote:  --source-host <host> --source-dir /path/to/dev/install"
    echo ""
    echo "  Tip: the reference install is the directory containing your .env,"
    echo "       credentials.json, llm-models.json, and .system_prompt* files."
    exit 1
fi

TARGET_DIR="$(pwd)/$DIR_NAME"

# ── Helper: copy a single file (local or remote) ─────────────────────────────
# copy_file <relative-path-within-source-dir> [<dest-path>]
# dest defaults to $TARGET_DIR/<relative-path>
copy_file() {
    local rel="$1"
    local dst="${2:-$TARGET_DIR/$rel}"

    if [ -n "$SOURCE_HOST" ]; then
        # Remote: scp from source host
        if scp -q "${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_DIR}/${rel}" "$dst" 2>/dev/null; then
            echo "  ✓ $rel  (from ${SOURCE_USER}@${SOURCE_HOST})"
        else
            echo "  ✗ $rel not found on ${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_DIR} (skipping)"
        fi
    else
        # Local: plain cp
        local src="$SOURCE_DIR/$rel"
        if [ -f "$src" ]; then
            cp "$src" "$dst"
            echo "  ✓ $rel"
        else
            echo "  ✗ $rel not found in reference dir (skipping)"
        fi
    fi
}

# ── Helper: list files matching a glob on source (local or remote) ───────────
list_source_files() {
    local pattern="$1"
    if [ -n "$SOURCE_HOST" ]; then
        ssh "${SOURCE_USER}@${SOURCE_HOST}" "ls ${SOURCE_DIR}/${pattern} 2>/dev/null || true"
    else
        ls ${SOURCE_DIR}/${pattern} 2>/dev/null || true
    fi
}

# ── 0a. System package preflight ─────────────────────────────────────────────
# Only needed when falling back to the system Python (not pyenv).
# pyenv-managed Pythons have venv built in, so skip this check if pyenv is present.
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
if [ ! -x "$PYENV_ROOT/bin/pyenv" ]; then
    MISSING_PKGS=()
    # Check for python3-venv, accepting versioned variants (e.g. python3.12-venv)
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
    if ! dpkg -s "python3-venv" &>/dev/null 2>&1 && \
       ! dpkg -s "python${PY_VER}-venv" &>/dev/null 2>&1; then
        MISSING_PKGS+=("python${PY_VER}-venv")
    fi
    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        echo "ERROR: Missing required system packages: ${MISSING_PKGS[*]}"
        echo "  Install them with:  sudo apt-get install -y ${MISSING_PKGS[*]}"
        echo "  Then re-run this script."
        exit 1
    fi
fi

echo "=== llmem-gw setup ==="
echo "Target:  $TARGET_DIR"
echo "Branch:  $BRANCH"
if [ -n "$SOURCE_HOST" ]; then
    echo "Source:  ${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_DIR}"
else
    echo "Source:  $SOURCE_DIR  (local)"
fi
echo ""

# ── 0. Verify SSH connectivity (remote only) ─────────────────────────────────
if [ -n "$SOURCE_HOST" ]; then
    echo "Testing SSH to ${SOURCE_USER}@${SOURCE_HOST}..."
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${SOURCE_USER}@${SOURCE_HOST}" true 2>/dev/null; then
        echo "ERROR: Cannot SSH to ${SOURCE_USER}@${SOURCE_HOST}."
        echo "  Make sure SSH key auth is set up (ssh-copy-id ${SOURCE_USER}@${SOURCE_HOST})"
        exit 1
    fi
    echo "  SSH OK"
    echo ""
fi

# ── 1. Clone ─────────────────────────────────────────────────────────────────
if [ -d "$TARGET_DIR" ]; then
    echo "ERROR: Directory '$TARGET_DIR' already exists. Remove it first."
    exit 1
fi

echo "Cloning repo (branch: $BRANCH)..."
mkdir -p ~/.ssh && chmod 700 ~/.ssh

# Disable all interactive git prompts — clone must succeed non-interactively
# or fail immediately. This prevents hangs in PTY/non-interactive sessions.
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

# Prefer SSH clone; fall back to HTTPS for machines without a GitHub SSH key.
REPO="$REPO_SSH"
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 git@github.com true 2>/dev/null; then
    echo "  No GitHub SSH key found — using HTTPS clone"
    REPO="$REPO_HTTPS"
fi
git clone --branch "$BRANCH" --depth 1 --no-progress "$REPO" "$TARGET_DIR"
cd "$TARGET_DIR"

# ── 2. Python version check ──────────────────────────────────────────────────
# Ensure pyenv shims are in PATH even in non-interactive SSH sessions.
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

# Prefer pyenv-managed python if available, then fall through explicit versions.
PYTHON_BIN=""
for candidate in "$(pyenv which python 2>/dev/null || true)" python3.11 python3.13 python3.12 python3 python; do
    [ -z "$candidate" ] && continue
    if "$candidate" --version &>/dev/null 2>&1; then
        if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.11+ not found."
    echo "  pyenv is installed at $PYENV_ROOT but 3.11.x may not be built yet."
    echo "  Run:  pyenv install 3.11.10 && pyenv global 3.11.10"
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version)"

# ── 3. Create venv ───────────────────────────────────────────────────────────
echo ""
echo "Creating venv..."
"$PYTHON_BIN" -m venv venv
source venv/bin/activate

# ── 4. Install requirements ──────────────────────────────────────────────────
echo ""
echo "Installing requirements..."
pip install --upgrade pip -q
pip install -r requirements.txt

# ── 5. Copy working config from reference installation ───────────────────────
echo ""
if [ -n "$SOURCE_HOST" ]; then
    echo "Pulling config from ${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_DIR}"
else
    echo "Copying config from reference installation at: $SOURCE_DIR"
fi

# Pin the same Python version as the reference install
copy_file ".python-version"

# Credentials and secrets (not in repo)
copy_file ".env"
copy_file "credentials.json"
copy_file "token.json"

# Model registry (may have local hosts/keys not in repo copy)
copy_file "llm-models.json"

# Instance-specific config (gitignored, must be copied from reference)
copy_file "db-config.json"
copy_file "auto-enrich.json"

# Service account JSON referenced in .env (SERVICE_ACCOUNT_FILE=./gen-lang-*.json)
# Extract the filename from .env if present, then copy it too.
SA_FILE=""
if [ -n "$SOURCE_HOST" ]; then
    SA_FILE=$(ssh "${SOURCE_USER}@${SOURCE_HOST}" \
        "grep -oP 'SERVICE_ACCOUNT_FILE=\"\./\K[^\"]+' ${SOURCE_DIR}/.env 2>/dev/null || true")
else
    SA_FILE=$(grep -oP 'SERVICE_ACCOUNT_FILE="\./\K[^"]+' "${SOURCE_DIR}/.env" 2>/dev/null || true)
fi
if [ -n "$SA_FILE" ]; then
    copy_file "$SA_FILE"
fi

# NOTE: plugins-enabled.json is intentionally NOT copied from the reference.
# The repo's version has clean defaults. Port overrides are applied below via
# llmemctl.py port-set so each instance gets its own port assignments.

# Live system prompt sections (repo has defaults, but reference may have customizations)
while IFS= read -r fpath; do
    [ -f "$fpath" ] || [ -n "$SOURCE_HOST" ] || continue
    fname=$(basename "$fpath")
    if [ -n "$SOURCE_HOST" ]; then
        copy_file "$fname" "$TARGET_DIR/$fname"
    else
        cp "$fpath" "$TARGET_DIR/$fname"
        echo "  ✓ $fname"
    fi
done < <(list_source_files ".system_prompt*")

# ── 6. Verify ────────────────────────────────────────────────────────────────
echo ""
echo "Verifying startup imports..."
python -c "
import sys
missing = []
for mod in ['starlette', 'uvicorn', 'openai', 'google.genai', 'dotenv', 'mcp']:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print('MISSING: ' + ', '.join(missing))
    sys.exit(1)
else:
    print('All core imports OK')
"

# ── 7. Show port configuration ───────────────────────────────────────────────
echo ""
echo "Configured listening ports:"
python llmemctl.py port-list

# ── 8. Summary ───────────────────────────────────────────────────────────────
echo "=== Setup complete ==="
echo ""
echo "  !! IMPORTANT: Always activate the venv before running anything !!"
echo ""
echo "  cd $TARGET_DIR"
echo "  source venv/bin/activate"
echo ""
echo "  If running multiple instances on the same machine, change ports first:"
echo "  python llmemctl.py port-set plugin_client_shellpy 8770"
echo "  python llmemctl.py port-set plugin_client_api     8777"
echo ""
echo "  python llmem-gw.py          # terminal 1 - server"
echo "  python shell.py              # terminal 2 - client"
echo ""
echo "  python llmemctl.py     # manage plugins, models, and ports"
echo ""
