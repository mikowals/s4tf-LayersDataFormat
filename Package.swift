// swift-tools-version:5.1
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
        .package(
            url: "https://github.com/tensorflow/swift-apis.git",
            .branch("master")),
    ],
    targets: [
        .target(
            name: "LayersDataFormat",
            dependencies: ["TensorFlow"]),
        .testTarget(
            name: "LayersDataFormatTests",
            dependencies: ["LayersDataFormat"]),
    ]
)
