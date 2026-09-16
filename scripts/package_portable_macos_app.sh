#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

backend_name="${MOTOOLBOX_BACKEND_APP_NAME:-MOtoolboxBackend}"
backend_display_name="${MOTOOLBOX_BACKEND_DISPLAY_NAME:-MOtoolbox Backend}"
backend_bundle_id="${MOTOOLBOX_BACKEND_BUNDLE_ID:-com.motoolbox.backend}"
shell_name="${MOTOOLBOX_PORTABLE_APP_NAME:-MOtoolbox}"
shell_bundle_id="${MOTOOLBOX_PORTABLE_BUNDLE_ID:-com.motoolbox.app}"
target_arch="${MOTOOLBOX_TARGET_ARCH:-arm64}"

for arg in "$@"; do
  case "$arg" in
    --target-arch=*)
      target_arch="${arg#*=}"
      ;;
    --with-vault)
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: scripts/package_portable_macos_app.sh [--with-vault] [--target-arch=arm64]" >&2
      exit 2
      ;;
  esac
done

if [ "$target_arch" != "arm64" ]; then
  echo "Only Apple Silicon (arm64) builds are maintained." >&2
  exit 2
fi

scripts/package_macos.sh \
  --package-name="$backend_name" \
  --display-name="$backend_display_name" \
  --bundle-id="$backend_bundle_id" \
  "$@"

MOTOOLBOX_SHELL_APP_NAME="$shell_name" \
MOTOOLBOX_SHELL_BUNDLE_ID="$shell_bundle_id" \
MOTOOLBOX_SHELL_TARGET_ARCH="$target_arch" \
  scripts/build_macos_shell.sh

shell_app="dist/$shell_name.app"
backend_app="dist/$backend_name.app"
resources="$shell_app/Contents/Resources"

if [ ! -d "$backend_app" ]; then
  echo "Backend app was not built: $backend_app" >&2
  exit 1
fi

rm -rf "$resources/$backend_name.app"
mkdir -p "$resources"
ditto "$backend_app" "$resources/$backend_name.app"

# Embedding the backend mutates the already-signed shell bundle. Re-sign the
# completed portable bundle so Gatekeeper sees a consistent resource seal.
codesign --force --deep --sign - "$shell_app"

echo "Built portable app: $shell_app"
