#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

swift_package="macos/MOtoolboxShell"
configuration="${MOTOOLBOX_SHELL_CONFIGURATION:-release}"
app_name="${MOTOOLBOX_SHELL_APP_NAME:-MOtoolboxShell}"
bundle_id="${MOTOOLBOX_SHELL_BUNDLE_ID:-com.motoolbox.shell}"
target_arch="${MOTOOLBOX_SHELL_TARGET_ARCH:-arm64}"
build_dir="$swift_package/.build/$configuration"
executable="$build_dir/MOtoolboxShell"
app_dir="dist/$app_name.app"

if [ "$target_arch" != "arm64" ]; then
  echo "Only Apple Silicon (arm64) builds are maintained." >&2
  exit 2
fi

swift_build_args=(--package-path "$swift_package" -c "$configuration")
if [ -n "$target_arch" ]; then
  swift_build_args+=(--arch "$target_arch")
fi
swift build "${swift_build_args[@]}"

rm -rf "$app_dir"
mkdir -p "$app_dir/Contents/MacOS" "$app_dir/Contents/Resources"
cp "$executable" "$app_dir/Contents/MacOS/$app_name"

icon_source="assets/icons/MOtoolbox.icns"
if [ -f "$icon_source" ]; then
  cp "$icon_source" "$app_dir/Contents/Resources/AppIcon.icns"
fi

cat > "$app_dir/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>$app_name</string>
  <key>CFBundleExecutable</key>
  <string>$app_name</string>
  <key>CFBundleIdentifier</key>
  <string>$bundle_id</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>$app_name</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

echo "Built: $app_dir"
