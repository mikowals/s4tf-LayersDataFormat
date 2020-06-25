// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "LayersDataFormat",
    platforms: [
      .macOS(.v10_13),
    ],
    products: [
        .library(
            name: "LayersDataFormat",
            targets: ["LayersDataFormat"]),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "LayersDataFormat",
            dependencies: []),
        .testTarget(
            name: "LayersDataFormatTests",
            dependencies: ["LayersDataFormat"]),
    ]
)
