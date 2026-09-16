#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
APP_EXEC="$SCRIPT_DIR/MOtoolbox.app/Contents/MacOS/MOtoolbox"

if [[ ! -x "$APP_EXEC" ]]; then
  echo "Cannot find MOtoolbox.app next to this launcher."
  echo "Expected: $APP_EXEC"
  read -r "?Press Return to close..."
  exit 1
fi

exec "$APP_EXEC"
