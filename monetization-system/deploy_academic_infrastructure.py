#!/usr/bin/env python3
"""
Academic Infrastructure Deployment Script
Deploys complete academic research framework infrastructure
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from infrastructure.academic_infrastructure_deployer import AcademicInfrastructureDeployer

async def main():
    """Main deployment function"""
    print("🎓 Academic Infrastructure Deployment System")
    print("=" * 60)
    print("📚 Deploying comprehensive academic research framework")
    print("🔬 Infrastructure components:")
    print("   • Research Network Layer")
    print("   • Database Infrastructure (PostgreSQL + Redis)")
    print("   • Data Processing Layer (RabbitMQ + Celery Workers)")
    print("   • Research Applications (Flask + Gunicorn)")
    print("   • Monitoring Stack (Prometheus + Grafana)")
    print("   • Security Layers (VPN + Firewall + Access Controls)")
    print("   • DNS Distribution (Global CDN)")
    print()
    
    try:
        # Initialize deployer
        deployer = AcademicInfrastructureDeployer()
        
        # Deploy infrastructure
        print("🚀 Starting academic infrastructure deployment...")
        deployment_result = await deployer.deploy_academic_infrastructure()
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 ACADEMIC INFRASTRUCTURE DEPLOYMENT RESULTS")
        print("=" * 60)
        
        print(f"🎯 Status: {deployment_result['academic_status']}")
        print(f"⏰ Timestamp: {deployment_result['academic_timestamp']}")
        print()
        
        # Display detailed results
        results = deployment_result['academic_results']
        
        for step_name, step_result in results.items():
            step_display = step_name.replace('_', ' ').title()
            print(f"📋 {step_display}:")
            
            if isinstance(step_result, dict):
                for key, value in step_result.items():
                    if isinstance(value, (str, int, bool)):
                        print(f"   • {key}: {value}")
                    elif isinstance(value, list):
                        print(f"   • {key}: {len(value)} items")
                    else:
                        print(f"   • {key}: {type(value).__name__}")
            else:
                print(f"   • Result: {step_result}")
            print()
        
        # Display access information
        print("🌐 ACADEMIC RESEARCH FRAMEWORK ACCESS")
        print("=" * 60)
        print("📊 Research Interface: http://localhost:8080")
        print("📈 Prometheus Metrics: http://localhost:9090")
        print("📊 Grafana Dashboard: http://localhost:3000")
        print("🐰 RabbitMQ Management: http://localhost:15672")
        print("🔬 Research Applications:")
        print("   • Instance 1: http://localhost:5000")
        print("   • Instance 2: http://localhost:5001")
        print("   • Instance 3: http://localhost:5002")
        print()
        
        print("🔐 SECURITY & ACCESS")
        print("=" * 60)
        print("🔒 VPN Access: Academic WireGuard configured")
        print("🛡️ Firewall: Academic research ports protected")
        print("👥 Access Control: Role-based academic permissions")
        print("📝 Audit Logging: Comprehensive academic logs enabled")
        print()
        
        print("📚 RESEARCH CAPABILITIES")
        print("=" * 60)
        print("🔬 Data Collection: Multi-source research data ingestion")
        print("📊 Data Processing: 10 academic research workers")
        print("📈 Analytics: Real-time research metrics")
        print("🗄️ Storage: PostgreSQL + Redis academic databases")
        print("🌍 Distribution: Global CDN for research access")
        print("📋 Monitoring: Prometheus + Grafana observability")
        print()
        
        print("✅ Academic infrastructure deployment completed successfully!")
        print("🎓 Research framework is ready for academic use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Academic infrastructure deployment failed: {e}")
        print("🔧 Please check Docker installation and try again")
        return False

def check_prerequisites():
    """Check deployment prerequisites"""
    print("🔍 Checking deployment prerequisites...")
    
    # Check Docker
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("✅ Docker is available and running")
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        print("📋 Please install Docker and ensure it's running")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("📋 Python 3.8+ is required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    
    # Check available ports
    import socket
    required_ports = [8080, 9090, 3000, 15672, 5000, 5001, 5002]
    for port in required_ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result == 0:
            print(f"⚠️ Port {port} is already in use")
        else:
            print(f"✅ Port {port} is available")
    
    print("✅ Prerequisites check completed")
    return True

if __name__ == "__main__":
    print("🎓 Academic Research Framework Infrastructure Deployer")
    print("🔬 Comprehensive research infrastructure deployment system")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed")
        sys.exit(1)
    
    print("\n🚀 Starting deployment process...")
    
    try:
        # Run deployment
        success = asyncio.run(main())
        
        if success:
            print("\n🎉 Deployment completed successfully!")
            print("📚 Academic research framework is ready for use")
            sys.exit(0)
        else:
            print("\n❌ Deployment failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Deployment failed with error: {e}")
        sys.exit(1)
