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
    ],
    targets: [
        // MARK: - Core
        .target(
            name: "MetalAudioKit",
            path: "Sources/MetalAudioKit",
            resources: [
                .copy("Shaders"),
            ]
        ),

        // MARK: - DSP
        .target(
            name: "MetalDSP",
            dependencies: ["MetalAudioKit"],
            path: "Sources/MetalDSP",
            resources: [
                .copy("Shaders"),
            ]
        ),

        // MARK: - Neural Networks
        .target(
            name: "MetalNN",
            dependencies: ["MetalAudioKit"],
            path: "Sources/MetalNN",
            resources: [
                .copy("Shaders"),
            ]
        ),

        // MARK: - Tests
        .testTarget(
            name: "MetalAudioKitTests",
            dependencies: ["MetalAudioKit"],
            path: "Tests/MetalAudioKitTests"
        ),
        .testTarget(
            name: "MetalDSPTests",
            dependencies: ["MetalDSP"],
            path: "Tests/MetalDSPTests"
        ),
        .testTarget(
            name: "MetalNNTests",
            dependencies: ["MetalNN"],
            path: "Tests/MetalNNTests"
        ),
    ]
)
