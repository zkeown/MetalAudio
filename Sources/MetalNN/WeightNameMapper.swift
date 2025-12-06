import Foundation

/// Naming convention for neural network weights.
///
/// Different frameworks and models use different naming conventions for their weight tensors.
/// This enum helps identify and convert between conventions.
public enum WeightNamingConvention: String, CaseIterable {
    /// MetalAudio convention used by this framework
    /// - Time encoder: `time_encoder.{level}.conv.weight`
    /// - Freq encoder: `freq_encoder.{level}.conv.weight`
    /// - Cross transformer: `cross_transformer.layers.{i}.self_attn_freq.*`
    case metalaudio

    /// Meta/Facebook Demucs convention
    /// - Time encoder: `tencoder.{level}.conv.weight`
    /// - Freq encoder: `encoder.{level}.conv.weight`
    /// - Cross transformer: `crosstransformer.layers.{i}.self_attn.*`
    case demucs

    /// Unknown or custom convention
    case unknown
}

/// Maps weight names between different naming conventions.
///
/// ## Usage
///
/// ### Auto-detection
/// ```swift
/// let loader = try SafeTensorsLoader(fileURL: weightsURL)
/// let mapper = WeightNameMapper(tensorNames: loader.availableTensors)
///
/// switch mapper.detectedConvention {
/// case .metalaudio:
///     // Already in correct format
///     break
/// case .demucs:
///     // Use mapper to convert names
///     let metalAudioName = mapper.toMetalAudio(name: "tencoder.0.conv.weight")
/// }
/// ```
///
/// ### Explicit Mapping
/// ```swift
/// let mapper = WeightNameMapper.forConvention(.demucs)
/// let convertedName = mapper.toMetalAudio(name: "crosstransformer.layers.0.self_attn.in_proj_weight")
/// // Returns: "cross_transformer.layers.0.self_attn_freq.in_proj_weight"
/// ```
public struct WeightNameMapper {

    // MARK: - Properties

    /// The detected or configured naming convention
    public let convention: WeightNamingConvention

    // MARK: - Initialization

    /// Create a mapper for a specific convention.
    ///
    /// - Parameter convention: The source naming convention to map from
    public init(convention: WeightNamingConvention) {
        self.convention = convention
    }

    /// Create a mapper with auto-detection from tensor names.
    ///
    /// Examines the tensor names to determine the naming convention.
    ///
    /// - Parameter tensorNames: Array of tensor names from a weights file
    public init(tensorNames: [String]) {
        self.convention = Self.detectConvention(from: tensorNames)
    }

    /// Convenience constructor for a specific convention.
    public static func forConvention(_ convention: WeightNamingConvention) -> WeightNameMapper {
        WeightNameMapper(convention: convention)
    }

    // MARK: - Detection

    /// The detected naming convention (alias for `convention`).
    public var detectedConvention: WeightNamingConvention { convention }

    /// Detect the naming convention from a list of tensor names.
    ///
    /// - Parameter names: Array of tensor names
    /// - Returns: Detected convention, or `.unknown` if not recognized
    public static func detectConvention(from names: [String]) -> WeightNamingConvention {
        var demucsScore = 0
        var metalAudioScore = 0

        for name in names {
            // Demucs-specific patterns
            if name.hasPrefix("tencoder.") || name.hasPrefix("tdecoder.") {
                demucsScore += 2
            }
            if name.hasPrefix("crosstransformer.") {
                demucsScore += 2
            }
            if name.contains(".conv_tr.") {
                demucsScore += 1
            }
            if name.contains(".norm1.") || name.contains(".norm2.") {
                demucsScore += 1
            }
            if name.hasPrefix("channel_upsampler.") || name.hasPrefix("channel_downsampler.") {
                demucsScore += 1
            }

            // MetalAudio-specific patterns
            if name.hasPrefix("time_encoder.") || name.hasPrefix("time_decoder.") {
                metalAudioScore += 2
            }
            if name.hasPrefix("freq_encoder.") || name.hasPrefix("freq_decoder.") {
                metalAudioScore += 2
            }
            if name.hasPrefix("cross_transformer.") {
                metalAudioScore += 2
            }
            if name.contains(".conv_transpose.") {
                metalAudioScore += 1
            }
            if name.contains("_to_transformer.") || name.contains("transformer_to_") {
                metalAudioScore += 1
            }
            if name.contains(".self_attn_freq.") || name.contains(".self_attn_time.") {
                metalAudioScore += 1
            }
        }

        if metalAudioScore > demucsScore && metalAudioScore >= 3 {
            return .metalaudio
        } else if demucsScore > metalAudioScore && demucsScore >= 3 {
            return .demucs
        }

        return .unknown
    }

    // MARK: - Mapping

    /// Convert a weight name to MetalAudio convention.
    ///
    /// If the current convention is already `.metalaudio`, returns the name unchanged.
    ///
    /// - Parameter name: Original tensor name
    /// - Returns: Name in MetalAudio convention
    public func toMetalAudio(name: String) -> String {
        switch convention {
        case .metalaudio:
            return name
        case .demucs:
            return Self.demucsToMetalAudio(name)
        case .unknown:
            return name
        }
    }

    /// Convert a weight name from MetalAudio to another convention.
    ///
    /// - Parameters:
    ///   - name: Name in MetalAudio convention
    ///   - targetConvention: Target convention
    /// - Returns: Converted name
    public func fromMetalAudio(name: String, to targetConvention: WeightNamingConvention) -> String {
        switch targetConvention {
        case .metalaudio:
            return name
        case .demucs:
            return Self.metalAudioToDemucs(name)
        case .unknown:
            return name
        }
    }

    // MARK: - Demucs → MetalAudio Mapping

    /// Convert a single weight name from Demucs to MetalAudio convention.
    public static func demucsToMetalAudio(_ name: String) -> String {
        var result = name

        // Time encoder: tencoder.{level}.{layer} → time_encoder.{level}.{layer}
        if result.hasPrefix("tencoder.") {
            result = result.replacingOccurrences(of: "tencoder.", with: "time_encoder.")
            result = result.replacingOccurrences(of: ".norm1.", with: ".norm.")
        }
        // Time decoder: tdecoder.{level}.{layer} → time_decoder.{level}.{layer}
        else if result.hasPrefix("tdecoder.") {
            result = result.replacingOccurrences(of: "tdecoder.", with: "time_decoder.")
            result = result.replacingOccurrences(of: ".conv_tr.", with: ".conv_transpose.")
            result = result.replacingOccurrences(of: ".norm2.", with: ".norm.")
        }
        // Frequency encoder: encoder.{level}.{layer} → freq_encoder.{level}.{layer}
        else if result.hasPrefix("encoder.") && !result.hasPrefix("encoder_") {
            result = result.replacingOccurrences(of: "encoder.", with: "freq_encoder.")
            result = result.replacingOccurrences(of: ".norm1.", with: ".norm.")
        }
        // Frequency decoder: decoder.{level}.{layer} → freq_decoder.{level}.{layer}
        else if result.hasPrefix("decoder.") && !result.hasPrefix("decoder_") {
            result = result.replacingOccurrences(of: "decoder.", with: "freq_decoder.")
            result = result.replacingOccurrences(of: ".conv_tr.", with: ".conv_transpose.")
            result = result.replacingOccurrences(of: ".norm2.", with: ".norm.")
        }
        // Channel upsamplers/downsamplers (projection layers)
        else if result.hasPrefix("channel_upsampler.") {
            result = result.replacingOccurrences(of: "channel_upsampler.", with: "freq_to_transformer.")
        } else if result.hasPrefix("channel_downsampler.") {
            result = result.replacingOccurrences(of: "channel_downsampler.", with: "transformer_to_freq.")
        } else if result.hasPrefix("channel_upsampler_t.") {
            result = result.replacingOccurrences(of: "channel_upsampler_t.", with: "time_to_transformer.")
        } else if result.hasPrefix("channel_downsampler_t.") {
            result = result.replacingOccurrences(of: "channel_downsampler_t.", with: "transformer_to_time.")
        }
        // Cross-transformer
        else if result.hasPrefix("crosstransformer.") {
            result = result.replacingOccurrences(of: "crosstransformer.", with: "cross_transformer.")
            result = remapCrossTransformerName(result)
        }

        return result
    }

    /// Remap cross-transformer sublayer names.
    private static func remapCrossTransformerName(_ name: String) -> String {
        var result = name

        // Pattern: cross_transformer.layers_t.{i}.{sublayer} → cross_transformer.layers.{i}.{sublayer_time}
        // Pattern: cross_transformer.layers.{i}.{sublayer} → cross_transformer.layers.{i}.{sublayer_freq}

        // Handle time path (layers_t)
        if result.range(of: #"cross_transformer\.layers_t\.(\d+)\."#, options: .regularExpression) != nil {
            let prefix = "cross_transformer.layers_t."
            let afterPrefix = result[result.index(result.startIndex, offsetBy: prefix.count)...]
            if let dotIndex = afterPrefix.firstIndex(of: ".") {
                let levelStr = String(afterPrefix[..<dotIndex])
                let rest = String(afterPrefix[afterPrefix.index(after: dotIndex)...])
                let remappedRest = remapTransformerSublayer(rest, path: "time")
                result = "cross_transformer.layers.\(levelStr).\(remappedRest)"
            }
        }
        // Handle frequency path (layers)
        else if result.hasPrefix("cross_transformer.layers.") && !result.contains(".layers_t.") {
            let prefix = "cross_transformer.layers."
            let afterPrefix = result[result.index(result.startIndex, offsetBy: prefix.count)...]
            if let dotIndex = afterPrefix.firstIndex(of: ".") {
                let levelStr = String(afterPrefix[..<dotIndex])
                let rest = String(afterPrefix[afterPrefix.index(after: dotIndex)...])
                let remappedRest = remapTransformerSublayer(rest, path: "freq")
                result = "cross_transformer.layers.\(levelStr).\(remappedRest)"
            }
        }

        return result
    }

    /// Remap transformer sublayer names with time/freq suffix.
    private static func remapTransformerSublayer(_ sublayer: String, path: String) -> String {
        // Self-attention
        if sublayer.hasPrefix("self_attn.") {
            return "self_attn_\(path)." + String(sublayer.dropFirst("self_attn.".count))
        }
        // Cross-attention
        if sublayer.hasPrefix("cross_attn.") {
            return "cross_attn_\(path)." + String(sublayer.dropFirst("cross_attn.".count))
        }
        // Feed-forward
        if sublayer.hasPrefix("linear1.") || sublayer.hasPrefix("linear2.") {
            return "ffn_\(path).\(sublayer)"
        }
        // Layer norms
        if sublayer.hasPrefix("norm1.") {
            return "norm1_\(path)." + String(sublayer.dropFirst("norm1.".count))
        }
        if sublayer.hasPrefix("norm2.") {
            return "norm2_\(path)." + String(sublayer.dropFirst("norm2.".count))
        }
        if sublayer.hasPrefix("norm3.") {
            return "norm3_\(path)." + String(sublayer.dropFirst("norm3.".count))
        }

        // Unknown, add suffix
        return "\(sublayer)_\(path)"
    }

    // MARK: - MetalAudio → Demucs Mapping

    /// Convert a single weight name from MetalAudio to Demucs convention.
    public static func metalAudioToDemucs(_ name: String) -> String {
        var result = name

        // Time encoder: time_encoder.{level}.{layer} → tencoder.{level}.{layer}
        if result.hasPrefix("time_encoder.") {
            result = result.replacingOccurrences(of: "time_encoder.", with: "tencoder.")
            result = result.replacingOccurrences(of: ".norm.", with: ".norm1.")
        }
        // Time decoder: time_decoder.{level}.{layer} → tdecoder.{level}.{layer}
        else if result.hasPrefix("time_decoder.") {
            result = result.replacingOccurrences(of: "time_decoder.", with: "tdecoder.")
            result = result.replacingOccurrences(of: ".conv_transpose.", with: ".conv_tr.")
            result = result.replacingOccurrences(of: ".norm.", with: ".norm2.")
        }
        // Frequency encoder: freq_encoder.{level}.{layer} → encoder.{level}.{layer}
        else if result.hasPrefix("freq_encoder.") {
            result = result.replacingOccurrences(of: "freq_encoder.", with: "encoder.")
            result = result.replacingOccurrences(of: ".norm.", with: ".norm1.")
        }
        // Frequency decoder: freq_decoder.{level}.{layer} → decoder.{level}.{layer}
        else if result.hasPrefix("freq_decoder.") {
            result = result.replacingOccurrences(of: "freq_decoder.", with: "decoder.")
            result = result.replacingOccurrences(of: ".conv_transpose.", with: ".conv_tr.")
            result = result.replacingOccurrences(of: ".norm.", with: ".norm2.")
        }
        // Projection layers
        else if result.hasPrefix("freq_to_transformer.") {
            result = result.replacingOccurrences(of: "freq_to_transformer.", with: "channel_upsampler.")
        } else if result.hasPrefix("transformer_to_freq.") {
            result = result.replacingOccurrences(of: "transformer_to_freq.", with: "channel_downsampler.")
        } else if result.hasPrefix("time_to_transformer.") {
            result = result.replacingOccurrences(of: "time_to_transformer.", with: "channel_upsampler_t.")
        } else if result.hasPrefix("transformer_to_time.") {
            result = result.replacingOccurrences(of: "transformer_to_time.", with: "channel_downsampler_t.")
        }
        // Cross-transformer
        else if result.hasPrefix("cross_transformer.") {
            result = result.replacingOccurrences(of: "cross_transformer.", with: "crosstransformer.")
            // Note: Full reverse mapping for cross-transformer sublayers would be complex
            // and is not typically needed (we convert TO MetalAudio, not FROM)
        }

        return result
    }

    // MARK: - Batch Operations

    /// Convert all names in a dictionary to MetalAudio convention.
    ///
    /// - Parameter weights: Dictionary with original weight names as keys
    /// - Returns: Dictionary with converted names
    public func mapAllToMetalAudio<T>(_ weights: [String: T]) -> [String: T] {
        var result: [String: T] = [:]
        for (name, value) in weights {
            let mappedName = toMetalAudio(name: name)
            result[mappedName] = value
        }
        return result
    }

    /// Generate a complete mapping table.
    ///
    /// - Parameter names: Original tensor names
    /// - Returns: Dictionary mapping original names to MetalAudio names
    public func generateMapping(for names: [String]) -> [String: String] {
        var mapping: [String: String] = [:]
        for name in names {
            mapping[name] = toMetalAudio(name: name)
        }
        return mapping
    }

    /// Check if a name is already in MetalAudio convention.
    public static func isMetalAudioNaming(_ name: String) -> Bool {
        name.hasPrefix("time_encoder.") ||
        name.hasPrefix("time_decoder.") ||
        name.hasPrefix("freq_encoder.") ||
        name.hasPrefix("freq_decoder.") ||
        name.hasPrefix("cross_transformer.") ||
        name.hasPrefix("time_to_transformer.") ||
        name.hasPrefix("freq_to_transformer.")
    }

    /// Check if a name is in Demucs convention.
    public static func isDemucsNaming(_ name: String) -> Bool {
        name.hasPrefix("tencoder.") ||
        name.hasPrefix("tdecoder.") ||
        name.hasPrefix("crosstransformer.") ||
        name.hasPrefix("channel_upsampler.")
    }
}

// MARK: - SafeTensorsLoader Extension

extension SafeTensorsLoader {

    /// Create a weight name mapper based on the tensors in this file.
    ///
    /// - Returns: WeightNameMapper configured for auto-detected convention
    public func createWeightMapper() -> WeightNameMapper {
        WeightNameMapper(tensorNames: availableTensors)
    }

    /// Load a tensor by MetalAudio name, automatically mapping from the file's convention.
    ///
    /// - Parameters:
    ///   - metalAudioName: Name in MetalAudio convention
    ///   - mapper: WeightNameMapper configured for this file's convention
    /// - Returns: Float32 array of tensor data
    public func loadTensorMapped(metalAudioName: String, mapper: WeightNameMapper) throws -> [Float] {
        // If convention is MetalAudio, use name directly
        // Otherwise, we need to reverse-map to find the original name
        if mapper.convention == .metalaudio {
            return try loadTensor(name: metalAudioName)
        }

        // For other conventions, we need to find the original name
        // by reverse-mapping from MetalAudio
        let originalName = mapper.fromMetalAudio(name: metalAudioName, to: mapper.convention)

        // Try the reverse-mapped name first
        if tensorInfos[originalName] != nil {
            return try loadTensor(name: originalName)
        }

        // If that didn't work, try to find a matching tensor by iterating
        for tensorName in availableTensors {
            if mapper.toMetalAudio(name: tensorName) == metalAudioName {
                return try loadTensor(name: tensorName)
            }
        }

        throw LoaderError.tensorNotFound(name: metalAudioName)
    }
}
