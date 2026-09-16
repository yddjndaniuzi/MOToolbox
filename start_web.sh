#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt >/tmp/motoolbox-pressconf-pip.log 2>&1
exec .venv/bin/python -m pressconf.web

