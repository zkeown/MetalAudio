// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalAudio",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        // Core GPU primitives and buffer management
        .library(
            name: "MetalAudioKit",
            targets: ["MetalAudioKit"]
        ),
        // DSP operations: FFT, convolution, filters
        .library(
            name: "MetalDSP",
            targets: ["MetalDSP"]
        ),
        // Neural network inference for audio
        .library(
            name: "MetalNN",
            targets: ["MetalNN"]
        ),
        // Full bundle
        .library(
            name: "MetalAudio",
            targets: ["MetalAudioKit", "MetalDSP", "MetalNN"]
        ),
        // Benchmark executable
        .executable(
            name: "Benchmark",
            targets: ["Benchmark"]
        ),
        // Real-time audio demo
        .executable(
            name: "RealtimeAudioDemo",
            targets: ["RealtimeAudioDemo"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-atomics.git", from: "1.2.0"),
    ],
    targets: [
        // MARK: - Core
        .target(
            name: "MetalAudioKit",
            dependencies: [
                .product(name: "Atomics", package: "swift-atomics"),
            ],
            path: "Sources/MetalAudioKit",
            resources: [
                .copy("Shaders"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),

        // MARK: - DSP
        .target(
            name: "MetalDSP",
            dependencies: ["MetalAudioKit"],
            path: "Sources/MetalDSP",
            exclude: ["CLAUDE.md"],
            resources: [
                .copy("Shaders"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),

        // MARK: - Neural Networks
        .target(
            name: "MetalNN",
            dependencies: ["MetalAudioKit"],
            path: "Sources/MetalNN",
            exclude: ["CLAUDE.md"],
            resources: [
                .copy("Shaders"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),

        // MARK: - Tests
        .testTarget(
            name: "MetalAudioKitTests",
            dependencies: ["MetalAudioKit", "MetalDSP"],
            path: "Tests/MetalAudioKitTests"
        ),
        .testTarget(
            name: "MetalDSPTests",
            dependencies: ["MetalDSP", "MetalAudioKit", "MetalNN"],
            path: "Tests/MetalDSPTests"
        ),
        .testTarget(
            name: "MetalNNTests",
            dependencies: ["MetalNN", "MetalAudioKit", "MetalDSP"],
            path: "Tests/MetalNNTests",
            resources: [
                .copy("Resources"),
            ]
        ),

        // MARK: - Benchmark
        .executableTarget(
            name: "Benchmark",
            dependencies: ["MetalAudioKit", "MetalDSP", "MetalNN"],
            path: "Sources/Benchmark"
        ),

        // MARK: - Real-time Audio Demo
        .executableTarget(
            name: "RealtimeAudioDemo",
            dependencies: ["MetalAudioKit", "MetalDSP", "MetalNN"],
            path: "Examples/RealtimeDemo"
        ),
    ]
)
