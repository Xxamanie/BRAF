#!/usr/bin/env python3
"""
Test BRAF API - Submit a task and check status
"""

import requests
import json
import time

def test_braf_api():
    """Test the BRAF API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing BRAF API...")
    
    # Test health endpoint
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check: {health_data['status']}")
            print(f"   📊 Version: {health_data['version']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test task submission
    print("\n2️⃣ Testing task submission...")
    task_data = {
        "id": "api_test_task_001",
        "profile_id": "test_profile",
        "actions": [
            {
                "type": "navigate",
                "url": "https://httpbin.org/html",
                "timeout": 30
            },
            {
                "type": "extract", 
                "selector": "h1",
                "timeout": 10,
                "metadata": {"attribute": "text"}
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
        else:
            print(f"   ❌ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Stats error: {e}")
    
    # Test tasks list
    print("\n4️⃣ Testing tasks list...")
    try:
        response = requests.get(f"{base_url}/tasks")
        if response.status_code == 200:
            tasks_data = response.json()
            print(f"   ✅ Tasks list retrieved")
            print(f"   📋 Total tasks: {tasks_data['total']}")
        else:
            print(f"   ❌ Tasks list failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Tasks list error: {e}")
    
    print("\n🎉 API testing complete!")
    print(f"\n🌐 Visit the dashboard: {base_url}")
    print(f"📚 API documentation: {base_url}/docs")

if __name__ == "__main__":
    test_braf_api()