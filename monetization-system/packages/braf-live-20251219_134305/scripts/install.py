#!/usr/bin/env python3
"""
BRAF Installation Script
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def main():
    print("BRAF Installation Script")
    print("=" * 40)
    
    system = platform.system().lower()
    
    if system == "linux":
        install_linux()
    elif system == "windows":
        install_windows()
    elif system == "darwin":
        install_macos()
    else:
        print(f"Unsupported system: {system}")
        sys.exit(1)

def install_linux():
    print("Installing BRAF on Linux...")
    subprocess.run(["bash", "scripts/deploy_linux.sh"], check=True)

def install_windows():
    print("Installing BRAF on Windows...")
    subprocess.run(["scripts/deploy_windows.bat"], check=True, shell=True)

def install_macos():
    print("Installing BRAF on macOS...")
    # Similar to Linux but with brew
    subprocess.run(["bash", "scripts/deploy_linux.sh"], check=True)

if __name__ == "__main__":
    main()
