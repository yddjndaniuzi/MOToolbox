// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MOtoolboxShell",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "MOtoolboxShell", targets: ["MOtoolboxShell"]),
    ],
    targets: [
        .executableTarget(
            name: "MOtoolboxShell",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("WebKit"),
            ]
        ),
    ]
)
