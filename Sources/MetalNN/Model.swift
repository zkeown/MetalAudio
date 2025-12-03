import Metal
import MetalAudioKit

/// A sequential neural network model for audio inference
public final class Sequential: @unchecked Sendable {

    private let device: AudioDevice
    private let context: ComputeContext
    private var layers: [NNLayer] = []
    private var intermediateBuffers: [Tensor] = []

    /// Initialize sequential model
    /// - Parameter device: Audio device
    public init(device: AudioDevice) {
        self.device = device
        self.context = ComputeContext(device: device)
    }

    /// Add a layer to the model
    public func add(_ layer: NNLayer) {
        layers.append(layer)
    }

    /// Prepare model for inference (allocate intermediate buffers)
    public func build() throws {
        intermediateBuffers.removeAll()

        for i in 0..<layers.count {
            let outputShape = layers[i].outputShape
            let buffer = try Tensor(device: device, shape: outputShape)
            intermediateBuffers.append(buffer)
        }
    }

    /// Run inference
    /// - Parameter input: Input tensor
    /// - Returns: Output tensor
    public func forward(_ input: Tensor) throws -> Tensor {
        guard !layers.isEmpty else {
            throw MetalAudioError.invalidConfiguration("No layers in model")
        }

        var currentInput = input

        try context.executeSync { encoder in
            for (i, layer) in layers.enumerated() {
                let output = intermediateBuffers[i]
                try layer.forward(input: currentInput, output: output, encoder: encoder)
                currentInput = output
            }
        }

        return intermediateBuffers.last!
    }

    /// Run inference asynchronously
    public func forwardAsync(_ input: Tensor, completion: @escaping (Result<Tensor, Error>) -> Void) {
        guard !layers.isEmpty else {
            completion(.failure(MetalAudioError.invalidConfiguration("No layers in model")))
            return
        }

        var currentInput = input

        context.executeAsync({ encoder in
            for (i, layer) in self.layers.enumerated() {
                let output = self.intermediateBuffers[i]
                try layer.forward(input: currentInput, output: output, encoder: encoder)
                currentInput = output
            }
        }, completion: { error in
            if let error = error {
                completion(.failure(error))
            } else {
                completion(.success(self.intermediateBuffers.last!))
            }
        })
    }

    /// Number of layers
    public var layerCount: Int {
        layers.count
    }

    /// Get layer at index
    public func layer(at index: Int) -> NNLayer? {
        guard index >= 0 && index < layers.count else { return nil }
        return layers[index]
    }
}

// MARK: - Model Loading

/// Protocol for loading model weights
public protocol ModelWeights {
    func loadInto(model: Sequential) throws
}

/// Load model from a simple binary format
public final class BinaryModelLoader: @unchecked Sendable {

    /// Header format for binary model files
    public struct Header {
        public let magic: UInt32 = 0x4D544C41  // "MTLA"
        public let version: UInt32 = 1
        public let numTensors: UInt32
        public let reserved: UInt32 = 0

        public init(numTensors: Int) {
            self.numTensors = UInt32(numTensors)
        }
    }

    /// Tensor entry in binary file
    public struct TensorEntry {
        public let nameLength: UInt32
        public let numDims: UInt32
        public let dataType: UInt32  // 0 = float32
        public let dataSize: UInt32
    }

    private let device: AudioDevice

    public init(device: AudioDevice) {
        self.device = device
    }

    /// Load tensors from binary file
    /// - Parameter url: File URL
    /// - Returns: Dictionary of tensor name to Tensor
    public func load(from url: URL) throws -> [String: Tensor] {
        let data = try Data(contentsOf: url)
        var offset = 0

        // Read header
        let headerSize = MemoryLayout<Header>.size
        guard data.count >= headerSize else {
            throw MetalAudioError.invalidConfiguration("File too small")
        }

        let magic = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
        guard magic == 0x4D544C41 else {
            throw MetalAudioError.invalidConfiguration("Invalid magic number")
        }

        let numTensors = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self) }
        offset = headerSize

        var tensors: [String: Tensor] = [:]

        for _ in 0..<numTensors {
            // Read tensor entry
            let nameLength = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            let numDims = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: UInt32.self) }
            let dataSize = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 12, as: UInt32.self) }
            offset += 16

            // Read name
            let nameData = data.subdata(in: offset..<(offset + Int(nameLength)))
            let name = String(data: nameData, encoding: .utf8) ?? ""
            offset += Int(nameLength)

            // Read shape
            var shape: [Int] = []
            for _ in 0..<numDims {
                let dim = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
                shape.append(Int(dim))
                offset += 4
            }

            // Create tensor and read data
            let tensor = try Tensor(device: device, shape: shape)
            data.withUnsafeBytes { ptr in
                memcpy(tensor.buffer.contents(), ptr.baseAddress! + offset, Int(dataSize))
            }
            offset += Int(dataSize)

            tensors[name] = tensor
        }

        return tensors
    }

    /// Save tensors to binary file
    public func save(tensors: [String: Tensor], to url: URL) throws {
        var data = Data()

        // Write header
        var header = Header(numTensors: tensors.count)
        withUnsafeBytes(of: &header) { data.append(contentsOf: $0) }

        // Write tensors
        for (name, tensor) in tensors {
            let nameData = name.data(using: .utf8) ?? Data()

            var entry = TensorEntry(
                nameLength: UInt32(nameData.count),
                numDims: UInt32(tensor.shape.count),
                dataType: 0,
                dataSize: UInt32(tensor.byteSize)
            )
            withUnsafeBytes(of: &entry) { data.append(contentsOf: $0) }

            data.append(nameData)

            for dim in tensor.shape {
                var d = UInt32(dim)
                withUnsafeBytes(of: &d) { data.append(contentsOf: $0) }
            }

            data.append(Data(bytes: tensor.buffer.contents(), count: tensor.byteSize))
        }

        try data.write(to: url)
    }
}
