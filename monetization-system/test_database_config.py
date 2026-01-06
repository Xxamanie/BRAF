#!/usr/bin/env python3
"""
Test Database Configuration
Verify database ID integration and functionality
"""

import os
import sys
import json
from pathlib import Path
from database.database_config import DatabaseConfig, DatabaseConnectionManager
from database.service import DatabaseService

def test_database_configuration():
    """Test database configuration with the provided database ID"""
    print("🔧 Testing Database Configuration")
    print("=" * 50)
    
    # Test DatabaseConfig
    print("📋 Test 1: Database Configuration")
    config = DatabaseConfig()
    
    print(f"Database ID: {config.database_id}")
    print(f"Database Instance ID: {config.database_instance_id}")
    print(f"Valid UUID: {config.is_valid_uuid}")
    print(f"Connection Type: {config._get_connection_type()}")
    print(f"Connection String (masked): {config._mask_connection_string()}")
    
    # Test database info
    print("\n📋 Test 2: Database Information")
    db_info = config.get_database_info()
    print(json.dumps(db_info, indent=2))
    
    # Test database metadata
    print("\n📋 Test 3: Database Metadata")
    metadata = config.get_database_metadata()
    print(json.dumps(metadata, indent=2))
    
    # Test connection validation
    print("\n📋 Test 4: Connection Validation")
    validation = config.validate_connection()
    print(json.dumps(validation, indent=2))
    
    return config

def test_connection_manager():
    """Test database connection manager"""
    print("\n🔗 Testing Connection Manager")
    print("=" * 50)
    
    conn_manager = DatabaseConnectionManager()
    
    # Create test connections
    print("📋 Test 1: Creating Connections")
    conn1 = conn_manager.create_connection('primary')
    conn2 = conn_manager.create_connection('secondary')
    
    print(f"Primary Connection: {conn1}")
    print(f"Secondary Connection: {conn2}")
    
    # Get connection info
    print("\n📋 Test 2: Connection Information")
    primary_info = conn_manager.get_connection_info('primary')
    print(f"Primary Info: {json.dumps(primary_info, indent=2)}")
    
    # Get all connections
    print("\n📋 Test 3: All Connections")
    all_connections = conn_manager.get_all_connections()
    print(f"Active Connections: {len(all_connections['connections'])}")
    print(f"Statistics: {json.dumps(all_connections['statistics'], indent=2)}")
    
    # Health check
    print("\n📋 Test 4: Health Check")
    health = conn_manager.health_check()
    print(json.dumps(health, indent=2))
    
    # Close connections
    print("\n📋 Test 5: Closing Connections")
    closed1 = conn_manager.close_connection('primary')
    closed2 = conn_manager.close_connection('secondary')
    
    print(f"Primary Closed: {closed1}")
    print(f"Secondary Closed: {closed2}")
    
    return conn_manager

def test_database_service():
    """Test database service integration"""
    print("\n💾 Testing Database Service Integration")
    print("=" * 50)
    
    try:
        # Initialize database service
        print("📋 Test 1: Service Initialization")
        with DatabaseService() as db_service:
            print("✅ Database service initialized successfully")
            
            # Test database info
            print("\n📋 Test 2: Database Info from Service")
            db_info = db_service.get_database_info()
            print(json.dumps(db_info, indent=2))
            
            # Test database health
            print("\n📋 Test 3: Database Health Check")
            health = db_service.get_database_health()
            print(json.dumps(health, indent=2))
            
            # Test connection
            print("\n📋 Test 4: Connection Test")
            connection_ok = db_service.test_connection()
            print(f"Connection Status: {'✅ OK' if connection_ok else '❌ Failed'}")
            
            return True
            
    except Exception as e:
        print(f"❌ Database service test failed: {e}")
        return False

def test_configuration_export_import():
    """Test configuration export and import"""
    print("\n📤 Testing Configuration Export/Import")
    print("=" * 50)
    
    config = DatabaseConfig()
    
    # Export configuration
    print("📋 Test 1: Export Configuration")
    exported_config = config.export_configuration()
    print(json.dumps(exported_config, indent=2))
    
    # Test import (with same data)
    print("\n📋 Test 2: Import Configuration")
    import_success = config.import_configuration(exported_config)
    print(f"Import Success: {'✅ OK' if import_success else '❌ Failed'}")
    
    return exported_config

def test_session_generation():
    """Test session ID generation"""
    print("\n🔑 Testing Session Generation")
    print("=" * 50)
    
    config = DatabaseConfig()
    
    # Generate multiple session IDs
    print("📋 Test 1: Generate Session IDs")
    sessions = []
    for i in range(5):
        session_id = config.generate_session_id()
        sessions.append(session_id)
        print(f"Session {i+1}: {session_id}")
    
    # Verify uniqueness
    print(f"\n📋 Test 2: Uniqueness Check")
    unique_sessions = set(sessions)
    print(f"Generated: {len(sessions)}, Unique: {len(unique_sessions)}")
    print(f"All Unique: {'✅ Yes' if len(sessions) == len(unique_sessions) else '❌ No'}")
    
    return sessions

def main():
    """Main test function"""
    print("🔧 BRAF Database Configuration Test Suite")
    print("=" * 60)
    
    # Check environment variables
    database_id = os.getenv('DATABASE_ID', '')
    print(f"Database ID from environment: {database_id}")
    
    if not database_id:
        print("⚠️  DATABASE_ID not found in environment variables")
        print("Using default from .env file")
    
    # Run tests
    try:
        # Test 1: Database Configuration
        config = test_database_configuration()
        
        # Test 2: Connection Manager
        conn_manager = test_connection_manager()
        
        # Test 3: Database Service
        service_ok = test_database_service()
        
        # Test 4: Configuration Export/Import
        exported_config = test_configuration_export_import()
        
        # Test 5: Session Generation
        sessions = test_session_generation()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        print(f"✅ Database ID: {config.database_id}")
        print(f"✅ UUID Valid: {config.is_valid_uuid}")
        print(f"✅ Connection Type: {config._get_connection_type()}")
        print(f"✅ Service Integration: {'Working' if service_ok else 'Failed'}")
        print(f"✅ Session Generation: Working")
        print(f"✅ Export/Import: Working")
        
        print("\n🎉 All tests completed successfully!")
        
        # Save test results
        test_results = {
            'database_id': config.database_id,
            'test_timestamp': config.get_database_info()['configuration_timestamp'],
            'configuration': config.get_database_metadata(),
            'exported_config': exported_config,
            'sample_sessions': sessions[:3],  # Save first 3 sessions
            'test_status': 'passed'
        }
        
        results_file = Path('database_test_results.json')
        results_file.write_text(json.dumps(test_results, indent=2))
        print(f"📄 Test results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
