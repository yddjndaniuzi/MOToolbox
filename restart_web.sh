#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

launchctl bootout "gui/$(id -u)/com.motoolbox.pressconf" 2>/dev/null || true
sessions="$(screen -ls 2>/dev/null | awk '/motoolbox-web/ {print $1}' || true)"
printf '%s\n' "$sessions" | while read -r session; do
  [ -n "$session" ] || continue
  screen -S "$session" -X quit || true
done
lsof -tiTCP:5058 -sTCP:LISTEN | xargs -r kill || true

screen -dmS motoolbox-web zsh -lc 'cd /Users/mi/Documents/MOtoolbox && exec .venv/bin/python -u -m pressconf.web > /tmp/motoolbox-screen.log 2> /tmp/motoolbox-screen.err'

sleep 1
curl -fsS http://127.0.0.1:5058/healthz >/dev/null
echo "MOtoolbox web is running at http://127.0.0.1:5058"
