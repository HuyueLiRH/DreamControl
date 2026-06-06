#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${NEWTON_WALL_BRUSH_CONFIG:-${ROOT_DIR}/configs/newton_wall_brush_versions.env}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:?--config requires a path}"
      shift 2
      ;;
    --install)
      INSTALL_DEPS=1
      shift
      ;;
    *)
      CONFIG_PATH="$1"
      shift
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config file: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONFIG_PATH"

clone_or_update() {
  local repo="$1"
  local upstream_repo="$2"
  local ref="$3"
  local target_dir="$4"

  mkdir -p "$(dirname "$target_dir")"
  if [[ ! -d "${target_dir}/.git" ]]; then
    git clone "$repo" "$target_dir"
  fi

  if ! git -C "$target_dir" remote get-url upstream >/dev/null 2>&1; then
    git -C "$target_dir" remote add upstream "$upstream_repo"
  fi

  git -C "$target_dir" fetch origin --tags
  git -C "$target_dir" fetch upstream --tags
  git -C "$target_dir" checkout "$ref"
  git -C "$target_dir" rev-parse --short HEAD
}

echo "Pinning IsaacLab Newton checkout:"
ISAACLAB_SHA="$(clone_or_update "$ISAACLAB_NEWTON_FORK_REPO" "$ISAACLAB_NEWTON_UPSTREAM_REPO" "$ISAACLAB_NEWTON_REF" "$ISAACLAB_NEWTON_DIR")"
echo "  ${ISAACLAB_NEWTON_DIR} @ ${ISAACLAB_SHA}"

echo "Pinning Newton checkout:"
NEWTON_SHA="$(clone_or_update "$NEWTON_FORK_REPO" "$NEWTON_UPSTREAM_REPO" "$NEWTON_REF" "$NEWTON_DIR")"
echo "  ${NEWTON_DIR} @ ${NEWTON_SHA}"

cat <<EOF

Environment source checkouts are ready.

DreamControl training root expected by eval scripts:
  ${NEWTON_WALL_BRUSH_TRAINING_ROOT}

Newton/IsaacLab source checkouts:
  ${ISAACLAB_NEWTON_DIR}
  ${NEWTON_DIR}

If this is a fresh machine, copy or clone DreamControl so that Training lives at
NEWTON_WALL_BRUSH_TRAINING_ROOT, then place the checkpoint and reference npz
before running Training/scripts/newton_wall_brush_eval.sh.
EOF

if [[ "$INSTALL_DEPS" == "1" ]]; then
  python -m pip install -e "${NEWTON_DIR}[examples]"
  (
    cd "$ISAACLAB_NEWTON_DIR"
    ./isaaclab.sh --install
  )
fi
