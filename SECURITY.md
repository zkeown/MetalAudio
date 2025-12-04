# Security Policy ⚡🔒

We take security seriously — because nobody wants their audio pipeline to be the weak link in the chain.

> *"Keep your buffers safe and your pointers safer."*

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please report it responsibly:

1. **Email**: Send details to the project maintainers (see repository contacts)
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (if you have them)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days, including:
- Confirmation of the vulnerability
- Planned fix timeline
- Any immediate mitigations

*Think of it like a controlled demolition — we want to know about the problem before the building comes down.* 🎸

## Security Considerations for MetalAudio

### GPU Buffer Safety

MetalAudio handles raw GPU memory. When using the framework:

- **Validate input sizes** before GPU operations
- **Check tensor dimensions** to prevent out-of-bounds access
- **Use the built-in NaN/Inf validation** on tensor copies

### Real-Time Thread Safety

Our real-time paths use `os_unfair_lock` for thread safety without priority inversion. However:

- **Never call blocking APIs** from audio callbacks
- **Pre-allocate all buffers** during initialization
- **Avoid shared mutable state** between audio and UI threads when possible

### Metal Shader Security

Metal shaders are compiled from source at runtime. If you're loading custom shaders:

- **Only load shaders from trusted sources**
- **Validate shader paths** to prevent path traversal
- **Use the bundled `.metallib` files** in production when possible

## Disclosure Policy

When we fix a security vulnerability:

1. We release a patch as quickly as possible
2. We update the CHANGELOG with security fix notes
3. We credit the reporter (unless they prefer anonymity)
4. For critical issues, we may issue a security advisory

---

*Stay secure, stay metal.*

*General Patton once said "I'm Paranoid" — oh wait, that was Ozzy. Either way, good security advice.* 🤘🔒
