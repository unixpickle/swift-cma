// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "CMA",
  platforms: [.macOS(.v13)],
  products: [.library(name: "CMA", targets: ["CMA"])],
  dependencies: [
    .package(url: "https://github.com/unixpickle/honeycrisp.git", from: "0.0.29")
  ],
  targets: [
    .target(
      name: "CMA",
      dependencies: [
        .product(name: "Honeycrisp", package: "honeycrisp"),
        .product(name: "HCBacktrace", package: "honeycrisp"),
      ]
    ),
    .testTarget(
      name: "CMATests",
      dependencies: ["CMA"]),
  ]
)
