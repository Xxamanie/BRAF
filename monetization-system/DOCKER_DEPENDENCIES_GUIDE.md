# Docker Dependencies Guide for BRAF Cryptocurrency System

## System Dependencies Overview

This guide explains all system dependencies required for the BRAF cryptocurrency system Docker deployment.

## 🔧 **Core Build Dependencies**

### Essential Build Tools
```dockerfile
# Core compilation tools
gcc                    # GNU C Compiler
g++                    # GNU C++ Compiler  
build-essential        # Meta-package for build tools
make                   # Build automation tool
cmake                  # Cross-platform build system
```

### System Utilities
```dockerfile
# Network and download tools
wget                   # Web file retrieval
curl                   # Data transfer tool
git                    # Version control system

# Archive and compression
unzip                  # Archive extraction
software-properties-common  # Repository management
apt-transport-https    # HTTPS repository support
ca-certificates        # SSL certificate authorities
lsb-release           # Linux Standard Base info
```

## 🐍 **Python Development Dependencies**

```dockerfile
# Python development environment
python3-dev            # Python development headers
python3-pip            # Python package installer
python3-setuptools     # Python package setup tools
python3-wheel          # Python wheel package format
```

**Purpose**: Required for compiling Python packages with C extensions, especially cryptography libraries.

## 🔐 **Cryptography & Security Libraries**

```dockerfile
# Cryptographic libraries
libssl-dev             # OpenSSL development libraries
libffi-dev             # Foreign Function Interface library
libcrypto++-dev        # C++ cryptography library
```

**Purpose**: Essential for NOWPayments API integration, SSL/TLS connections, and cryptocurrency operations.

## 🗄️ **Database Connectivity**

```dockerfile
# PostgreSQL support
libpq-dev              # PostgreSQL development libraries
postgresql-client      # PostgreSQL client tools
```

**Purpose**: Required for connecting to PostgreSQL database for storing cryptocurrency transactions and user data.

## 🖼️ **Image Processing Libraries**

```dockerfile
# Image manipulation (for CAPTCHA solving)
libjpeg-dev            # JPEG image library
libpng-dev             # PNG image library  
libwebp-dev            # WebP image library
```

**Purpose**: Used by browser automation for CAPTCHA solving and image processing in earning platforms.

## 📄 **XML/HTML Processing**

```dockerfile
# Web scraping and parsing
libxml2-dev            # XML parsing library
libxslt1-dev           # XSLT transformation library
```

**Purpose**: Required for web scraping, HTML parsing, and data extraction from earning platforms.

## 📦 **Compression Libraries**

```dockerfile
# Data compression support
zlib1g-dev             # Compression library
libbz2-dev             # Bzip2 compression
liblzma-dev            # LZMA compression
```

**Purpose**: Used for data compression, file handling, and network communication optimization.

## 🌐 **Network Libraries**

```dockerfile
# HTTP and network communication
libcurl4-openssl-dev   # HTTP client library with SSL
```

**Purpose**: Required for API communications with NOWPayments, Cloudflare, and other external services.

## 📊 **Dockerfile Variants**

### 1. Production Dockerfile (`Dockerfile.production`)
- **Full dependencies**: All libraries included
- **Multi-stage build**: Optimized for production
- **Playwright support**: Browser automation capabilities
- **Size**: ~800MB
- **Build time**: 5-10 minutes

### 2. Minimal Dockerfile (`Dockerfile.minimal`)
- **Essential dependencies only**: Core libraries
- **Single stage**: Faster build
- **No browser automation**: API-only operations
- **Size**: ~200MB
- **Build time**: 2-3 minutes

## 🚀 **Optimization Strategies**

### Layer Caching
```dockerfile
# Combine RUN commands to reduce layers
RUN apt-get update && apt-get install -y \
    package1 \
    package2 \
    && rm -rf /var/lib/apt/lists/*
```

### Package Cache Cleanup
```dockerfile
# Always clean package cache
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean
```

### Multi-stage Builds
```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim as builder
# Install build tools and compile

# Stage 2: Runtime
FROM python:3.11-slim
# Copy only compiled artifacts
```

## 🔍 **Dependency Analysis**

### Critical Dependencies (Required)
- `gcc`, `g++`: Compile Python C extensions
- `libssl-dev`, `libffi-dev`: Cryptography libraries
- `libpq-dev`: Database connectivity
- `python3-dev`: Python package compilation

### Optional Dependencies (Feature-specific)
- `libjpeg-dev`, `libpng-dev`: Image processing
- `libxml2-dev`, `libxslt1-dev`: Web scraping
- Browser automation libraries: Playwright support

### Development Dependencies (Build-time only)
- `build-essential`: Compilation tools
- `cmake`: Build system
- `git`: Source code management

## 🐛 **Common Issues & Solutions**

### 1. Cryptography Installation Fails
```dockerfile
# Solution: Install crypto dependencies first
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && pip install cryptography
```

### 2. Database Connection Issues
```dockerfile
# Solution: Install PostgreSQL development libraries
RUN apt-get install -y libpq-dev postgresql-client
```

### 3. Image Processing Errors
```dockerfile
# Solution: Install image libraries
RUN apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev
```

### 4. XML Parsing Failures
```dockerfile
# Solution: Install XML processing libraries
RUN apt-get install -y \
    libxml2-dev \
    libxslt1-dev
```

## 📈 **Performance Considerations**

### Build Time Optimization
1. **Use .dockerignore**: Exclude unnecessary files
2. **Layer ordering**: Put changing layers last
3. **Parallel builds**: Use BuildKit for faster builds
4. **Base image choice**: Use slim variants

### Runtime Optimization
1. **Multi-stage builds**: Separate build and runtime
2. **Minimal runtime**: Only include necessary libraries
3. **User permissions**: Run as non-root user
4. **Resource limits**: Set memory and CPU limits

## 🔒 **Security Best Practices**

### Package Management
```dockerfile
# Always update package lists
RUN apt-get update

# Install specific versions when possible
RUN apt-get install -y package=version

# Clean up after installation
RUN rm -rf /var/lib/apt/lists/*
```

### User Security
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 braf_user
USER braf_user
```

## 📋 **Dependency Checklist**

### ✅ **Essential for Cryptocurrency Operations**
- [x] `libssl-dev` - SSL/TLS for API communications
- [x] `libffi-dev` - Cryptography library support
- [x] `libpq-dev` - Database connectivity
- [x] `python3-dev` - Python package compilation
- [x] `gcc`, `g++` - C/C++ compilation

### ✅ **Required for Web Automation**
- [x] `libxml2-dev` - HTML/XML parsing
- [x] `libxslt1-dev` - Web scraping
- [x] `libjpeg-dev` - Image processing
- [x] `curl`, `wget` - HTTP requests

### ✅ **Optional Enhancements**
- [x] `git` - Source code management
- [x] `unzip` - Archive extraction
- [x] `libcurl4-openssl-dev` - Advanced HTTP features

## 🎯 **Recommended Configuration**

### For Production (Full Features)
```dockerfile
FROM python:3.11-slim
# Install all dependencies from Dockerfile.production
# Includes browser automation, image processing, full crypto support
```

### For API-Only Deployment
```dockerfile
FROM python:3.11-slim  
# Install minimal dependencies from Dockerfile.minimal
# Cryptocurrency API operations only, no browser automation
```

### For Development
```dockerfile
FROM python:3.11
# Use full Python image with development tools
# Add debugging and development utilities
```

---

## 📞 **Support**

### Build Issues
- Check dependency versions
- Verify package availability
- Review build logs for specific errors

### Runtime Issues  
- Ensure all required libraries are installed
- Check file permissions
- Verify environment variables

### Performance Issues
- Use multi-stage builds
- Optimize layer caching
- Consider minimal variants

---

*Docker Dependencies Guide for BRAF Cryptocurrency System*  
*Optimized for NOWPayments Integration*  
*Supporting 150+ Cryptocurrencies*