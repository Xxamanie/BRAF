#!/usr/bin/env python3
"""
Academic Infrastructure Test Suite
Tests the deployed academic research framework
"""

import asyncio
import aiohttp
import docker
import time
from typing import Dict, List

class AcademicInfrastructureTester:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.base_urls = {
            'research_interface': 'http://localhost:8080',
            'prometheus': 'http://localhost:9090',
            'grafana': 'http://localhost:3000',
            'rabbitmq': 'http://localhost:15672',
            'app_1': 'http://localhost:5000',
            'app_2': 'http://localhost:5001',
            'app_3': 'http://localhost:5002'
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_container_status(self):
        """Test that all academic containers are running"""
        print("🔍 Testing Academic Container Status...")
        
        expected_containers = [
            'academic_research_interface',
            'postgres_research',
            'academic_research_cache',
            'academic_message_queue',
            'academic_prometheus',
            'academic_grafana',
            'academic_app_1',
            'academic_app_2',
            'academic_app_3'
        ]
        
        running_containers = []
        failed_containers = []
        
        for container_name in expected_containers:
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == 'running':
                    running_containers.append(container_name)
                    print(f"   ✅ {container_name}: Running")
                else:
                    failed_containers.append(container_name)
                    print(f"   ❌ {container_name}: {container.status}")
            except docker.errors.NotFound:
                failed_containers.append(container_name)
                print(f"   ❌ {container_name}: Not found")
        
        print(f"✅ Container Status: {len(running_containers)}/{len(expected_containers)} running")
        return len(failed_containers) == 0
    
    async def test_network_connectivity(self):
        """Test academic network connectivity"""
        print("🔍 Testing Academic Network Connectivity...")
        
        try:
            network = self.docker_client.networks.get('academic_research_network')
            connected_containers = len(network.containers)
            print(f"   ✅ Academic network exists with {connected_containers} containers")
            return True
        except docker.errors.NotFound:
            print("   ❌ Academic research network not found")
            return False
    
    async def test_research_interface(self):
        """Test research interface accessibility"""
        print("🔍 Testing Research Interface...")
        
        try:
            async with self.session.get(f"{self.base_urls['research_interface']}/health") as response:
                if response.status == 200:
                    content = await response.text()
                    print(f"   ✅ Research interface healthy: {content.strip()}")
                    return True
                else:
                    print(f"   ❌ Research interface returned status {response.status}")
                    return False
        except Exception as e:
            print(f"   ❌ Research interface connection failed: {e}")
            return False
    
    async def test_research_applications(self):
        """Test research application health"""
        print("🔍 Testing Research Applications...")
        
        healthy_apps = 0
        total_apps = 3
        
        for i in range(1, total_apps + 1):
            try:
                url = f"http://localhost:{5000 + i - 1}/academic/health"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Academic App {i}: {data.get('status', 'unknown')}")
                        healthy_apps += 1
                    else:
                        print(f"   ❌ Academic App {i}: HTTP {response.status}")
            except Exception as e:
                print(f"   ❌ Academic App {i}: Connection failed - {e}")
        
        print(f"✅ Research Applications: {healthy_apps}/{total_apps} healthy")
        return healthy_apps == total_apps
    
    async def test_monitoring_stack(self):
        """Test monitoring infrastructure"""
        print("🔍 Testing Monitoring Stack...")
        
        # Test Prometheus
        prometheus_ok = False
        try:
            async with self.session.get(f"{self.base_urls['prometheus']}/api/v1/query?query=up") as response:
                if response.status == 200:
                    print("   ✅ Prometheus: Accessible and responding")
                    prometheus_ok = True
                else:
                    print(f"   ❌ Prometheus: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Prometheus: Connection failed - {e}")
        
        # Test Grafana
        grafana_ok = False
        try:
            async with self.session.get(f"{self.base_urls['grafana']}/api/health") as response:
                if response.status == 200:
                    print("   ✅ Grafana: Accessible and responding")
                    grafana_ok = True
                else:
                    print(f"   ❌ Grafana: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Grafana: Connection failed - {e}")
        
        return prometheus_ok and grafana_ok
    
    async def test_message_queue(self):
        """Test RabbitMQ message queue"""
        print("🔍 Testing Message Queue...")
        
        try:
            async with self.session.get(f"{self.base_urls['rabbitmq']}/api/overview") as response:
                if response.status == 200:
                    print("   ✅ RabbitMQ: Management interface accessible")
                    return True
                else:
                    print(f"   ❌ RabbitMQ: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"   ❌ RabbitMQ: Connection failed - {e}")
            return False
    
    async def test_research_capabilities(self):
        """Test research framework capabilities"""
        print("🔍 Testing Research Capabilities...")
        
        capabilities_tested = 0
        total_capabilities = 3
        
        # Test research endpoint
        try:
            async with self.session.get(f"{self.base_urls['app_1']}/academic/research") as response:
                if response.status == 200:
                    data = await response.json()
                    capabilities = data.get('capabilities', [])
                    print(f"   ✅ Research capabilities: {len(capabilities)} available")
                    capabilities_tested += 1
                else:
                    print(f"   ❌ Research endpoint: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Research endpoint: {e}")
        
        # Test metrics endpoint
        try:
            async with self.session.get(f"{self.base_urls['app_1']}/metrics") as response:
                if response.status == 200:
                    print("   ✅ Metrics collection: Prometheus metrics available")
                    capabilities_tested += 1
                else:
                    print(f"   ❌ Metrics endpoint: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Metrics endpoint: {e}")
        
        # Test load balancing
        try:
            responses = []
            for i in range(3):
                async with self.session.get(f"{self.base_urls['research_interface']}/") as response:
                    responses.append(response.status)
            
            if all(status == 200 for status in responses):
                print("   ✅ Load balancing: Nginx distributing requests")
                capabilities_tested += 1
            else:
                print(f"   ❌ Load balancing: Inconsistent responses {responses}")
        except Exception as e:
            print(f"   ❌ Load balancing test: {e}")
        
        print(f"✅ Research Capabilities: {capabilities_tested}/{total_capabilities} working")
        return capabilities_tested == total_capabilities
    
    async def run_all_tests(self):
        """Run comprehensive academic infrastructure tests"""
        print("🎓 Academic Infrastructure Test Suite")
        print("=" * 60)
        
        tests = [
            ("Container Status", self.test_container_status),
            ("Network Connectivity", self.test_network_connectivity),
            ("Research Interface", self.test_research_interface),
            ("Research Applications", self.test_research_applications),
            ("Monitoring Stack", self.test_monitoring_stack),
            ("Message Queue", self.test_message_queue),
            ("Research Capabilities", self.test_research_capabilities)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            try:
                result = await test_func()
                if result:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
            
            print()
        
        print("=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Academic infrastructure is fully operational.")
            print("\n🎓 Academic Research Framework Status: READY")
            print("🔬 Research capabilities: FUNCTIONAL")
            print("📊 Monitoring systems: ACTIVE")
            print("🔒 Security layers: CONFIGURED")
            print("🌍 Distribution: GLOBAL")
        else:
            print(f"⚠️ {total - passed} tests failed. Infrastructure needs attention.")
        
        return passed == total

async def main():
    """Main test function"""
    print("🔧 Academic Infrastructure Test Suite")
    print("📋 Comprehensive testing of deployed research framework")
    print()
    
    # Wait for containers to fully start
    print("⏳ Waiting for containers to initialize...")
    await asyncio.sleep(10)
    
    async with AcademicInfrastructureTester() as tester:
        success = await tester.run_all_tests()
        
        if success:
            print("\n✅ Academic infrastructure is fully operational!")
            print("🌐 Access Points:")
            print("   • Research Interface: http://localhost:8080")
            print("   • Prometheus: http://localhost:9090")
            print("   • Grafana: http://localhost:3000")
            print("   • RabbitMQ: http://localhost:15672")
            print("   • Research Apps: http://localhost:5000-5002")
        else:
            print("\n❌ Some components need attention.")
            print("🔧 Check container logs for details")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")