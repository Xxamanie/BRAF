#!/usr/bin/env python3
"""
Windows SSL Setup - Alternative to Linux certbot
Sets up SSL certificates for HTTPS on Windows
"""
import subprocess
import sys
import os
from datetime import datetime

def install_ssl_packages():
    """Install SSL packages for Windows"""
    print("🔒 Installing SSL packages for Windows...")
    
    packages = [
        'cryptography>=41.0.0',
        'pyopenssl>=23.0.0',
        'certifi>=2023.0.0'
    ]
    
    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True, text=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {package}")

def create_self_signed_cert():
    """Create self-signed certificate for development"""
    print(f"\n🔐 Creating self-signed certificate...")
    
    cert_script = '''
import ssl
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import ipaddress

# Generate private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Create certificate
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, "Development"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "BRAF Development"),
    x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
])

cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    private_key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    datetime.datetime.utcnow() + datetime.timedelta(days=365)
).add_extension(
    x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.DNSName("127.0.0.1"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]),
    critical=False,
).sign(private_key, hashes.SHA256())

# Save certificate and key
with open("ssl_cert.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

with open("ssl_key.pem", "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ))

print("✅ SSL certificate created: ssl_cert.pem")
print("✅ SSL private key created: ssl_key.pem")
'''
    
    try:
        exec(cert_script)
        return True
    except Exception as e:
        print(f"❌ Certificate creation failed: {e}")
        return False

def setup_https_server():
    """Setup HTTPS server configuration"""
    print(f"\n🌐 Setting up HTTPS server...")
    
    https_server = '''#!/usr/bin/env python3
"""
HTTPS Server for BRAF - Windows SSL
"""
import ssl
import uvicorn
from pathlib import Path

def start_https_server():
    """Start HTTPS server with SSL certificates"""
    
    # Check for SSL files
    cert_file = Path("ssl_cert.pem")
    key_file = Path("ssl_key.pem")
    
    if not cert_file.exists() or not key_file.exists():
        print("❌ SSL certificates not found!")
        print("Run: python setup_windows_ssl.py")
        return
    
    print("🚀 Starting HTTPS server...")
    print("📍 HTTPS URL: https://127.0.0.1:8443")
    print("📍 Dashboard: https://127.0.0.1:8443/dashboard")
    
    # Start HTTPS server
    uvicorn.run(
        "monetization-system.main:app",
        host="127.0.0.1",
        port=8443,
        ssl_keyfile="ssl_key.pem",
        ssl_certfile="ssl_cert.pem",
        reload=False
    )

if __name__ == "__main__":
    start_https_server()
'''
    
    with open("start_https_server.py", "w") as f:
        f.write(https_server)
    
    print("✅ HTTPS server script created: start_https_server.py")

def main():
    """Main SSL setup"""
    print("🔒 WINDOWS SSL SETUP")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Install packages
    install_ssl_packages()
    
    # Create certificate
    cert_created = create_self_signed_cert()
    
    # Setup HTTPS server
    if cert_created:
        setup_https_server()
    
    print(f"\n" + "=" * 50)
    print("✅ SSL SETUP COMPLETE!")
    print("=" * 50)
    
    if cert_created:
        print(f"\n📋 What was created:")
        print(f"   🔐 ssl_cert.pem - SSL certificate")
        print(f"   🔑 ssl_key.pem - Private key")
        print(f"   🌐 start_https_server.py - HTTPS server")
        
        print(f"\n🚀 To start HTTPS server:")
        print(f"   python start_https_server.py")
        
        print(f"\n📍 Access URLs:")
        print(f"   HTTPS: https://127.0.0.1:8443")
        print(f"   Dashboard: https://127.0.0.1:8443/dashboard")
        
        print(f"\n⚠️  Browser Warning:")
        print(f"   Your browser will show 'Not Secure' warning")
        print(f"   Click 'Advanced' → 'Proceed to localhost'")
        print(f"   This is normal for self-signed certificates")
        
    else:
        print(f"\n❌ Certificate creation failed")
        print(f"💡 Your current HTTP server is still working:")
        print(f"   http://127.0.0.1:8003")

if __name__ == "__main__":
    main()