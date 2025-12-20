#!/usr/bin/env python3
"""
Test BRAF Docker API - Submit tasks to the full Docker deployment
"""

import requests
import json
import time

def test_docker_braf_api():
    """Test the BRAF Docker deployment API."""
    base_url = "http://localhost:8000"
    
    print("🐳 Testing BRAF Docker Deployment...")
    
    # Test health endpoint
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check: {health_data['status']}")
            print(f"   📊 Version: {health_data['version']}")
            print(f"   🐳 Deployment: {health_data['deployment']}")
            print(f"   🔧 Components: {health_data['components']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test task submission
    print("\n2️⃣ Testing task submission...")
    task_data = {
        "id": "docker_test_task_001",
        "profile_id": "docker_profile",
        "actions": [
            {
                "type": "navigate",
                "url": "https://httpbin.org/html",
                "timeout": 30
            },
            {
                "type": "wait",
                "data": "2.0",
                "timeout": 10
            },
            {
                "type": "extract", 
                "selector": "h1",
                "timeout": 10,
                "metadata": {"attribute": "text"}
            },
            {
                "type": "screenshot",
                "data": "docker_test_screenshot.png",
                "timeout": 10
            }
        ],
        "priority": "normal",
        "timeout": 300
    }
    
    try:
        response = requests.post(
            f"{base_url}/tasks",
            json=task_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Task submitted: {result['task_id']}")
            print(f"   📋 Status: {result['status']}")
            print(f"   🐳 Deployment: {result['deployment']}")
            
            # Get task details
            task_id = result['task_id']
            task_response = requests.get(f"{base_url}/tasks/{task_id}")
            if task_response.status_code == 200:
                task_details = task_response.json()
                print(f"   📊 Task details retrieved")
                print(f"   ⏰ Submitted at: {task_details['submitted_at']}")
            
        else:
            print(f"   ❌ Task submission failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Task submission error: {e}")
    
    # Test stats endpoint
    print("\n3️⃣ Testing stats endpoint...")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✅ Stats retrieved")
            print(f"   📊 Tasks submitted: {stats['tasks_submitted']}")
            print(f"   ⏱️ Uptime: {stats['uptime_formatted']}")
            print(f"   🐳 Deployment: {stats['deployment']}")
            print(f"   📦 Containers: {stats['containers']}")
        else:
            print(f"   ❌ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Stats error: {e}")
    
    # Test multiple task submission
    print("\n4️⃣ Testing multiple task submission...")
    for i in range(3):
        task_data = {
            "id": f"docker_batch_task_{i+1:03d}",
            "profile_id": f"batch_profile_{i+1}",
            "actions": [
                {
                    "type": "navigate",
                    "url": f"https://httpbin.org/delay/{i+1}",
                    "timeout": 30
                },
                {
                    "type": "extract",
                    "selector": "body",
                    "timeout": 10
                }
            ]
        }
        
        try:
            response = requests.post(f"{base_url}/tasks", json=task_data)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Batch task {i+1} submitted: {result['task_id']}")
            else:
                print(f"   ❌ Batch task {i+1} failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Batch task {i+1} error: {e}")
    
    print("\n🎉 Docker API testing complete!")
    print(f"\n🌐 Access points:")
    print(f"   • Main Dashboard: {base_url}")
    print(f"   • API Documentation: {base_url}/docs")
    print(f"   • Grafana Monitoring: http://localhost:3000 (admin/admin)")
    print(f"   • Prometheus Metrics: http://localhost:9090")
    print(f"\n🐳 Docker containers running:")
    print(f"   • C2 Server: BRAF command & control")
    print(f"   • Worker Nodes: 2 automation workers")
    print(f"   • PostgreSQL: Database storage")
    print(f"   • Redis: Task queue")
    print(f"   • Prometheus: Metrics collection")
    print(f"   • Grafana: Monitoring dashboards")

if __name__ == "__main__":
    test_docker_braf_api()