#!/usr/bin/env python3
"""
Docker Deployment Test Suite
Tests the BRAF Docker deployment
"""

import asyncio
import aiohttp
import docker
import time
import subprocess
import sys
from pathlib import Path

class DockerDeploymentTester:
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"❌ Docker client initialization failed: {e}")
            sys.exit(1)
        
        self.services = [
            'braf_postgres',
            'braf_redis', 
            'braf_rabbitmq',
            'braf_main_app',
            'braf_worker_1',
            'braf_worker_2',
            'braf_nginx',
            'braf_prometheus',
            'braf_grafana',
            'braf_flower'
        ]
        
        self.endpoints = {
            'main_app': 'http://localhost:8000/health',
            'dashboard': 'http://localhost:8004',
            'nginx': 'http://localhost/health',
            'prometheus': 'http://localhost:9090/api/v1/query?query=up',
            'grafana': 'http://localhost:3000/api/health',
            'flower': 'http://localhost:5555/api/workers',
            'rabbitmq': 'http://localhost:15672/api/overview'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def check_docker_availability(self):
        """Check if Docker is available and running"""
        print("🔍 Checking Docker availability...")
        
        try:
            # Check Docker daemon
            self.docker_client.ping()
            print("✅ Docker daemon is running")
            
            # Check Docker Compose
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker Compose is available")
                return True
            else:
                print("❌ Docker Compose is not available")
                return False
                
        except Exception as e:
            print(f"❌ Docker check failed: {e}")
            return False
    
    def check_compose_file(self):
        """Check if docker-compose file exists"""
        print("🔍 Checking Docker Compose configuration...")
        
        compose_file = Path('docker-compose.braf.yml')
        if compose_file.exists():
            print("✅ Docker Compose file found")
            return True
        else:
            print("❌ Docker Compose file not found")
            return False
    
    def check_environment_file(self):
        """Check if environment file exists"""
        print("🔍 Checking environment configuration...")
        
        env_file = Path('.env')
        if env_file.exists():
            print("✅ Environment file found")
            return True
        else:
            print("⚠️ Environment file not found - will use defaults")
            return True  # Not critical for testing
    
    def check_container_status(self):
        """Check status of Docker containers"""
        print("🔍 Checking container status...")
        
        running_containers = []
        failed_containers = []
        
        for service in self.services:
            try:
                container = self.docker_client.containers.get(service)
                if container.status == 'running':
                    running_containers.append(service)
                    print(f"   ✅ {service}: Running")
                else:
                    failed_containers.append(service)
                    print(f"   ❌ {service}: {container.status}")
            except docker.errors.NotFound:
                failed_containers.append(service)
                print(f"   ❌ {service}: Not found")
            except Exception as e:
                failed_containers.append(service)
                print(f"   ❌ {service}: Error - {e}")
        
        print(f"📊 Container Status: {len(running_containers)}/{len(self.services)} running")
        return len(failed_containers) == 0
    
    async def test_endpoints(self):
        """Test HTTP endpoints"""
        print("🔍 Testing HTTP endpoints...")
        
        working_endpoints = []
        failed_endpoints = []
        
        for name, url in self.endpoints.items():
            try:
                async with self.session.get(url, timeout=10) as response:
                    if response.status in [200, 401]:  # 401 is OK for auth-protected endpoints
                        working_endpoints.append(name)
                        print(f"   ✅ {name}: HTTP {response.status}")
                    else:
                        failed_endpoints.append(name)
                        print(f"   ❌ {name}: HTTP {response.status}")
            except asyncio.TimeoutError:
                failed_endpoints.append(name)
                print(f"   ❌ {name}: Timeout")
            except Exception as e:
                failed_endpoints.append(name)
                print(f"   ❌ {name}: {str(e)[:50]}...")
        
        print(f"📊 Endpoint Status: {len(working_endpoints)}/{len(self.endpoints)} working")
        return len(failed_endpoints) == 0
    
    def check_docker_network(self):
        """Check Docker network configuration"""
        print("🔍 Checking Docker network...")
        
        try:
            networks = self.docker_client.networks.list()
            braf_networks = [n for n in networks if 'braf' in n.name.lower()]
            
            if braf_networks:
                network = braf_networks[0]
                connected_containers = len(network.containers)
                print(f"   ✅ BRAF network found with {connected_containers} containers")
                return True
            else:
                print("   ❌ BRAF network not found")
                return False
                
        except Exception as e:
            print(f"   ❌ Network check failed: {e}")
            return False
    
    def check_docker_volumes(self):
        """Check Docker volumes"""
        print("🔍 Checking Docker volumes...")
        
        try:
            volumes = self.docker_client.volumes.list()
            braf_volumes = [v for v in volumes if 'braf' in v.name.lower() or 'monetization' in v.name.lower()]
            
            print(f"   ✅ Found {len(braf_volumes)} BRAF-related volumes")
            for volume in braf_volumes[:5]:  # Show first 5
                print(f"      • {volume.name}")
            
            return len(braf_volumes) > 0
            
        except Exception as e:
            print(f"   ❌ Volume check failed: {e}")
            return False
    
    async def run_deployment_test(self):
        """Run comprehensive deployment test"""
        print("🐳 BRAF Docker Deployment Test Suite")
        print("=" * 50)
        
        tests = [
            ("Docker Availability", self.check_docker_availability),
            ("Compose File", self.check_compose_file),
            ("Environment File", self.check_environment_file),
            ("Container Status", self.check_container_status),
            ("Docker Network", self.check_docker_network),
            ("Docker Volumes", self.check_docker_volumes),
            ("HTTP Endpoints", self.test_endpoints)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                
                if result:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
            
            print()
        
        print("=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Docker deployment is fully operational.")
            print("\n🐳 Docker Deployment Status: READY")
            print("🚀 BRAF Framework: OPERATIONAL")
            print("📊 All services: HEALTHY")
            print("🌐 Endpoints: ACCESSIBLE")
            
            print("\n🌐 Access URLs:")
            print("   • Main Dashboard: http://localhost")
            print("   • Enhanced Dashboard: http://localhost/enhanced-dashboard")
            print("   • API Documentation: http://localhost/docs")
            print("   • Grafana: http://localhost:3000")
            print("   • Prometheus: http://localhost:9090")
            print("   • Flower: http://localhost:5555")
            
        elif passed >= total * 0.7:  # 70% pass rate
            print("⚠️ Most tests passed. Deployment is mostly functional.")
            print("🔧 Some components may need attention.")
        else:
            print("❌ Multiple tests failed. Deployment needs attention.")
            print("🔧 Check Docker services and configuration.")
        
        return passed >= total * 0.7

async def main():
    """Main test function"""
    print("🔧 BRAF Docker Deployment Test Suite")
    print("📋 Comprehensive testing of Docker-based deployment")
    print()
    
    async with DockerDeploymentTester() as tester:
        success = await tester.run_deployment_test()
        
        if success:
            print("\n✅ Docker deployment is ready for use!")
            print("🚀 You can now access the BRAF framework")
        else:
            print("\n❌ Docker deployment needs attention")
            print("🔧 Please check the failed tests and fix issues")
            
        return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)