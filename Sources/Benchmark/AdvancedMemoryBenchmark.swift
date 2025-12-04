import Foundation
import Metal
import MetalAudioKit

/// Run advanced memory techniques benchmark
public func runAdvancedMemoryBenchmark(device: AudioDevice) {
    print("\n" + String(repeating: "=", count: 50))
    print("ADVANCED MEMORY TECHNIQUES IMPACT")
    print(String(repeating: "=", count: 50))

    // 1. MappedTensor vs Regular Tensor
    print("\n--- 1. MappedTensor vs Tensor (1MB data) ---")

    let beforeMapped = MemorySnapshot.capture(device: device.device)

    let mappedTensor = try! MappedTensor(shape: [256, 1024])  // 1MB
    mappedTensor.withUnsafeMutableBufferPointer { ptr in
        for i in 0..<ptr.count { ptr[i] = Float(i) }
    }

    let afterMapped = MemorySnapshot.capture(device: device.device)
    let mappedProcessDelta = Int64(afterMapped.processFootprint) - Int64(beforeMapped.processFootprint)
    let mappedGPUDelta = Int64(afterMapped.gpuAllocated) - Int64(beforeMapped.gpuAllocated)

    print("  MappedTensor:")
    print("    Process: +\(String(format: "%.2f", Double(mappedProcessDelta) / 1_000_000))MB")
    print("    GPU: +\(String(format: "%.2f", Double(mappedGPUDelta) / 1_000_000))MB (no GPU allocation!)")

    let beforeRegular = MemorySnapshot.capture(device: device.device)
    let regularTensor = try! Tensor(device: device, shape: [256, 1024])
    var data = [Float](repeating: 0, count: 256 * 1024)
    for i in 0..<data.count { data[i] = Float(i) }
    try! regularTensor.copy(from: data)
    let afterRegular = MemorySnapshot.capture(device: device.device)
    let regularProcessDelta = Int64(afterRegular.processFootprint) - Int64(beforeRegular.processFootprint)
    let regularGPUDelta = Int64(afterRegular.gpuAllocated) - Int64(beforeRegular.gpuAllocated)

    print("  Regular Tensor:")
    print("    Process: +\(String(format: "%.2f", Double(regularProcessDelta) / 1_000_000))MB")
    print("    GPU: +\(String(format: "%.2f", Double(regularGPUDelta) / 1_000_000))MB")

    // Test paging hint
    mappedTensor.advise(.dontneed)
    print("  MappedTensor after .dontneed: \(String(format: "%.0f", mappedTensor.residencyRatio * 100))% resident")

    // 2. AudioAllocatorZone
    print("\n--- 2. AudioAllocatorZone Reuse Efficiency ---")
    let zone = AudioAllocatorZone()

    // Warm up - first allocation goes to system
    let warmup = zone.allocate(byteSize: 4096)!
    zone.deallocate(warmup, byteSize: 4096)

    // Now measure reuse
    for _ in 0..<99 {
        let ptr = zone.allocate(byteSize: 4096)!
        ptr.storeBytes(of: 1.0, as: Float.self)
        zone.deallocate(ptr, byteSize: 4096)
    }

    print("  After 100 alloc/free cycles (4KB each):")
    print("    Total allocations: \(zone.totalAllocations)")
    print("    Freelist hits: \(zone.freelistHits)")
    print("    Reuse ratio: \(String(format: "%.0f", zone.reuseRatio * 100))%")
    print("    Malloc calls avoided: \(zone.freelistHits)")

    // 3. SpeculativeBuffer
    print("\n--- 3. SpeculativeBuffer Cold Eviction ---")
    var specBuffers: [SpeculativeBuffer] = []
    let beforeSpec = MemorySnapshot.capture(device: device.device)

    // Allocate 10 x 100KB buffers
    for _ in 0..<10 {
        let buf = try! SpeculativeBuffer(device: device.device, byteSize: 102400)
        specBuffers.append(buf)
    }

    let afterAlloc = MemorySnapshot.capture(device: device.device)
    let allocGPUDelta = Int64(afterAlloc.gpuAllocated) - Int64(beforeSpec.gpuAllocated)

    print("  Allocated 10 x 100KB buffers:")
    print("    GPU memory used: +\(String(format: "%.2f", Double(allocGPUDelta) / 1_000_000))MB")

    // Evict all (simulate going cold)
    var totalFreed = 0
    for buf in specBuffers {
        totalFreed += buf.speculativeDeallocate()
    }

    let afterEvict = MemorySnapshot.capture(device: device.device)
    let evictGPUDelta = Int64(afterEvict.gpuAllocated) - Int64(beforeSpec.gpuAllocated)

    print("  After speculative eviction:")
    print("    Freed: \(String(format: "%.2f", Double(totalFreed) / 1_000_000))MB")
    print("    GPU memory now: +\(String(format: "%.2f", Double(evictGPUDelta) / 1_000_000))MB")

    // Access one - transparent realloc
    try! specBuffers[0].withContents { _ in }
    let afterRealloc = MemorySnapshot.capture(device: device.device)
    let reallocDelta = Int64(afterRealloc.gpuAllocated) - Int64(afterEvict.gpuAllocated)

    print("  After accessing 1 buffer (transparent realloc):")
    print("    Reallocated: +\(String(format: "%.2f", Double(reallocDelta) / 1_000_000))MB")

    // Summary
    print("\n--- SUMMARY ---")
    print("  MappedTensor: No GPU memory, OS can page out")
    print("  AllocatorZone: \(zone.freelistHits)/\(zone.totalAllocations) reused (\(String(format: "%.0f", zone.reuseRatio * 100))% less malloc)")
    print("  SpeculativeBuffer: \(totalFreed / 1024)KB freed on eviction, reallocates on access")
}
