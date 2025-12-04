# Contributing to MetalAudio ⚡

First off, thanks for wanting to contribute! Whether you're fixing a bug, adding a feature, or improving docs — you're about to make MetalAudio even more metal. 🤘

> *"Many hands make light work. Many GPUs make lighter work."*

*We're Not Gonna Take It — bad code, that is. But yours? Bring it on.*

## Getting Started

### Prerequisites

- **macOS 13+** (for Metal development)
- **Xcode 15+** with Swift 5.9+
- A Mac with Apple Silicon or discrete GPU (recommended for testing)

### Building the Project

```bash
# Clone the repo
git clone https://github.com/yourorg/MetalAudio.git
cd MetalAudio

# Build all targets
swift build

# Run tests
swift test

# Run benchmarks (optional, but satisfying)
swift run Benchmark
```

## How to Contribute

### Reporting Bugs 🐛

Found something that doesn't rock? Please open an issue with:

1. **Environment**: macOS/iOS version, device model, Xcode version
2. **Description**: What happened vs. what you expected
3. **Reproduction steps**: Minimal code to trigger the issue
4. **Logs/output**: Console output, crash logs, or screenshots

*Pro tip: Issues with sample code get fixed faster than vague descriptions. We can't debug vibes.*

### Suggesting Features 💡

Have an idea that would amp up MetalAudio? Open an issue with:

1. **Use case**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: What else did you think about?

### Submitting Pull Requests

Ready to shred some code? Here's the process:

1. **Fork** the repository
2. **Create a branch** from `main` (`git checkout -b feature/my-awesome-feature`)
3. **Make your changes** (see guidelines below)
4. **Test thoroughly** (`swift test`)
5. **Commit** with clear messages
6. **Push** to your fork
7. **Open a PR** against `main`

## Code Guidelines

### Style

- Follow existing code style (we're not picky, but consistency is key)
- Use Swift naming conventions
- Keep functions focused and reasonably sized
- Add doc comments for public APIs

### Real-Time Audio Safety ⚡

This is critical. If your code runs in an audio callback path:

- **No allocations** after initialization
- **No locks** that can block (use `os_unfair_lock` or lock-free structures)
- **No syscalls** that can block (file I/O, network, etc.)
- **Pre-allocate** all buffers during setup

```swift
// GOOD: Pre-allocated, lock-free
func process(input: UnsafePointer<Float>, output: UnsafeMutablePointer<Float>, count: Int) {
    // Direct memory operations only
}

// BAD: Allocates, can block
func process(input: [Float]) -> [Float] {  // Array allocation!
    return input.map { $0 * 2 }            // More allocations!
}
```

*Remember: In audio land, a dropped sample is a sin. Glitches are the enemy.* 🎸

### Testing

- Add tests for new functionality
- Ensure existing tests pass (`swift test`)
- For performance-sensitive code, add benchmark coverage
- Test on both Intel and Apple Silicon if possible

### Commit Messages

Keep them clear and descriptive:

```
feat: Add partitioned convolution for long impulse responses

- Implements overlap-save algorithm
- Supports impulse responses up to 10 seconds
- Includes benchmark showing 4x speedup over direct convolution
```

Prefixes we use:
- `feat:` New features
- `fix:` Bug fixes
- `perf:` Performance improvements
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring (no behavior change)
- `chore:` Build/tooling changes

## Architecture Notes

Before diving deep, familiarize yourself with the module structure:

| Module | Purpose | Key Concerns |
|--------|---------|--------------|
| **MetalAudioKit** | Core GPU infrastructure | Thread safety, device management |
| **MetalDSP** | Signal processing | Numerical accuracy, hybrid CPU/GPU |
| **MetalNN** | Neural inference | Zero-allocation paths, BNNS integration |

See the [README](README.md) for detailed architecture docs.

## Review Process

1. All PRs require at least one approving review
2. CI must pass (build + tests)
3. Performance-sensitive changes need benchmark data
4. Breaking API changes need discussion first

## Automated Issue Handling (Claude)

This repository uses AI-assisted automation for certain issue types. Here's how it works:

### Tier System

| Tier | Issue Type | Automation Level |
|------|------------|------------------|
| **Tier 1** | Documentation fixes, typos | Auto-PR, auto-merge after review |
| **Tier 2** | Simple code fixes (< 3 files) | Auto-PR, human merge required |
| **Tier 3** | Features, architecture | Research only, human implementation |

### What Gets Automated?

**Tier 1** (fully automated):

- Typo fixes in documentation
- README/docs updates
- Comment corrections
- Changelog entries

**Tier 2** (auto-PR, human approval):

- Simple bug fixes with clear scope
- Single-file changes
- Test additions
- Minor refactors

**Tier 3** (research only):

- New features
- Performance optimization
- Anything touching thread safety
- Multi-module changes
- Architecture decisions

### Hard Blocks (Never Automated)

These files/patterns are **never** touched by automation:

- `Package.swift`, `Package.resolved`
- `.github/` workflows
- `**/*.metal` shaders
- `SECURITY.md`
- Code containing `os_unfair_lock`, `Sendable`, `@unchecked`

### How to Use Issue Templates

When opening an issue, select the appropriate template:

- **Documentation Fix** → Tier 1, eligible for full automation
- **Simple Code Fix** → Tier 2, auto-PR with human review
- **Bug Report** → Tier 2-3, depending on complexity
- **Feature Request** → Tier 3, research only

### Opting Out

Add the label `no-automation` to any issue to prevent Claude from processing it.

### Budget & Rate Limits

- Monthly budget: $25 (prevents runaway costs)
- Max 5 PRs per day
- 24-hour cooldown after failed attempts
- All decisions logged in issue comments

## Community

- Be respectful and constructive
- Help others learn (we were all beginners once)
- Assume good intentions

See our [Code of Conduct](CODE_OF_CONDUCT.md) for community guidelines.

---

*Thanks for helping make MetalAudio awesome. Now go forth and make some noise!*

*Welcome to the Jungle — we've got fun and games (and GPU shaders).* 🤘⚡
