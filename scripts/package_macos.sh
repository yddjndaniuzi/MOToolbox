#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

with_vault=0
package_name="${MOTOOLBOX_PACKAGE_NAME:-MOtoolbox}"
display_name="${MOTOOLBOX_DISPLAY_NAME:-$package_name}"
bundle_identifier="${MOTOOLBOX_BUNDLE_IDENTIFIER:-com.motoolbox.pressconf}"
target_arch="${MOTOOLBOX_TARGET_ARCH:-arm64}"
for arg in "$@"; do
  case "$arg" in
    --with-vault)
      with_vault=1
      ;;
    --package-name=*)
      package_name="${arg#*=}"
      ;;
    --display-name=*)
      display_name="${arg#*=}"
      ;;
    --bundle-id=*)
      bundle_identifier="${arg#*=}"
      ;;
    --target-arch=*)
      target_arch="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: scripts/package_macos.sh [--with-vault] [--package-name=NAME] [--display-name=NAME] [--bundle-id=ID] [--target-arch=arm64]" >&2
      exit 2
      ;;
  esac
done

if [ -z "$package_name" ] || [ -z "$display_name" ] || [ -z "$bundle_identifier" ]; then
  echo "Package name, display name, and bundle id must be non-empty." >&2
  exit 2
fi

if [ "$target_arch" != "arm64" ]; then
  echo "Only Apple Silicon (arm64) builds are maintained." >&2
  exit 2
fi

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt pyinstaller

rm -rf build/package_resources
mkdir -p build/package_resources/pressconf

if [ "$with_vault" -eq 1 ]; then
  vault_source="${MOTOOLBOX_VAULT_SOURCE:-$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/pressconf-database}"
  if [ ! -d "$vault_source" ]; then
    echo "Vault source does not exist: $vault_source" >&2
    exit 1
  fi
  mkdir -p build/package_resources/pressconf/embedded_vault
  rsync -a \
    --exclude='.git/***' \
    --exclude='.obsidian/***' \
    --exclude='node_modules/***' \
    --include='*/' \
    --include='*.md' \
    --exclude='*' \
    "$vault_source"/ \
    build/package_resources/pressconf/embedded_vault/
fi

rm -rf "dist/$package_name" "dist/$package_name.app"
MOTOOLBOX_PACKAGE_NAME="$package_name" \
MOTOOLBOX_DISPLAY_NAME="$display_name" \
MOTOOLBOX_BUNDLE_IDENTIFIER="$bundle_identifier" \
MOTOOLBOX_TARGET_ARCH="$target_arch" \
  .venv/bin/python -m PyInstaller packaging/motoolbox.spec --noconfirm --clean

echo "Built: dist/$package_name.app"
