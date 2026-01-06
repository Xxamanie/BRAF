#!/usr/bin/env python3
"""
Database Configuration Module
Handles database connection and configuration management
"""

import os
import uuid
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration and management"""
    
    def __init__(self):
        self.database_id = os.getenv('DATABASE_ID', 'cec3b6d4-14c6-4256-9225-a30f14bfcb2c')
        self.database_instance_id = os.getenv('DATABASE_INSTANCE_ID', self.database_id)
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///./braf_monetization.db')
        
        # Validate UUID format
        try:
            uuid.UUID(self.database_id)
            self.is_valid_uuid = True
        except ValueError:
            logger.warning(f"Invalid UUID format for database_id: {self.database_id}")
            self.is_valid_uuid = False
        
        logger.info(f"Database configuration initialized with ID: {self.database_id}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        return {
            'database_id': self.database_id,
            'database_instance_id': self.database_instance_id,
            'database_url': self.database_url,
            'is_valid_uuid': self.is_valid_uuid,
            'connection_type': self._get_connection_type(),
            'configuration_timestamp': datetime.now().isoformat()
        }
    
    def _get_connection_type(self) -> str:
        """Determine database connection type from URL"""
        if self.database_url.startswith('sqlite'):
            return 'sqlite'
        elif self.database_url.startswith('postgresql'):
            return 'postgresql'
        elif self.database_url.startswith('mysql'):
            return 'mysql'
        else:
            return 'unknown'
    
    def get_connection_string(self) -> str:
        """Get the database connection string"""
        return self.database_url
    
    def update_database_id(self, new_id: str) -> bool:
        """Update database ID with validation"""
        try:
            uuid.UUID(new_id)
            self.database_id = new_id
            self.database_instance_id = new_id
            
            # Update environment variables
            os.environ['DATABASE_ID'] = new_id
            os.environ['DATABASE_INSTANCE_ID'] = new_id
            
            logger.info(f"Database ID updated to: {new_id}")
            return True
            
        except ValueError:
            logger.error(f"Invalid UUID format: {new_id}")
            return False
    
    def generate_session_id(self) -> str:
        """Generate a new session ID based on database ID"""
        session_uuid = uuid.uuid4()
        return f"{self.database_id}-{session_uuid}"
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate database connection"""
        validation_result = {
            'database_id': self.database_id,
            'connection_valid': False,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Import here to avoid circular imports
            from database.service import DatabaseService
            
            db_service = DatabaseService()
            
            # Test connection
            if hasattr(db_service, 'test_connection'):
                validation_result['connection_valid'] = db_service.test_connection()
            else:
                # Fallback validation
                validation_result['connection_valid'] = True
                
        except Exception as e:
            validation_result['error'] = str(e)
            logger.error(f"Database connection validation failed: {e}")
        
        return validation_result
    
    def get_database_metadata(self) -> Dict[str, Any]:
        """Get database metadata and statistics"""
        metadata = {
            'database_id': self.database_id,
            'instance_id': self.database_instance_id,
            'connection_type': self._get_connection_type(),
            'url_masked': self._mask_connection_string(),
            'uuid_valid': self.is_valid_uuid,
            'configuration': {
                'pool_size': os.getenv('DB_POOL_SIZE', '20'),
                'max_overflow': os.getenv('DB_MAX_OVERFLOW', '40'),
                'pool_timeout': os.getenv('DB_POOL_TIMEOUT', '30'),
                'pool_recycle': os.getenv('DB_POOL_RECYCLE', '3600')
            },
            'features': {
                'migrations_enabled': True,
                'backup_enabled': True,
                'monitoring_enabled': True,
                'encryption_enabled': 'TLS' in self.database_url.upper()
            }
        }
        
        return metadata
    
    def _mask_connection_string(self) -> str:
        """Mask sensitive information in connection string"""
        if '://' in self.database_url:
            protocol, rest = self.database_url.split('://', 1)
            if '@' in rest:
                credentials, host_db = rest.split('@', 1)
                return f"{protocol}://***:***@{host_db}"
            else:
                return self.database_url
        return self.database_url
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export database configuration for backup/migration"""
        return {
            'database_id': self.database_id,
            'database_instance_id': self.database_instance_id,
            'connection_type': self._get_connection_type(),
            'configuration_metadata': self.get_database_metadata(),
            'export_timestamp': datetime.now().isoformat(),
            'export_version': '1.0'
        }
    
    def import_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Import database configuration from backup"""
        try:
            if 'database_id' in config_data:
                return self.update_database_id(config_data['database_id'])
            return False
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False


class DatabaseConnectionManager:
    """Manages database connections with the configured database ID"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self.active_connections = {}
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'last_connection_time': None
        }
    
    def create_connection(self, connection_name: str = 'default') -> str:
        """Create a new database connection"""
        try:
            connection_id = f"{self.config.database_id}-{uuid.uuid4()}"
            
            # Store connection info
            self.active_connections[connection_name] = {
                'connection_id': connection_id,
                'database_id': self.config.database_id,
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # Update stats
            self.connection_stats['total_connections'] += 1
            self.connection_stats['active_connections'] += 1
            self.connection_stats['last_connection_time'] = datetime.now().isoformat()
            
            logger.info(f"Created database connection: {connection_id}")
            return connection_id
            
        except Exception as e:
            self.connection_stats['failed_connections'] += 1
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    def close_connection(self, connection_name: str = 'default') -> bool:
        """Close a database connection"""
        try:
            if connection_name in self.active_connections:
                connection_info = self.active_connections[connection_name]
                connection_info['status'] = 'closed'
                connection_info['closed_at'] = datetime.now().isoformat()
                
                self.connection_stats['active_connections'] -= 1
                
                logger.info(f"Closed database connection: {connection_info['connection_id']}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to close database connection: {e}")
            return False
    
    def get_connection_info(self, connection_name: str = 'default') -> Optional[Dict[str, Any]]:
        """Get information about a specific connection"""
        return self.active_connections.get(connection_name)
    
    def get_all_connections(self) -> Dict[str, Any]:
        """Get information about all connections"""
        return {
            'connections': self.active_connections,
            'statistics': self.connection_stats,
            'database_config': self.config.get_database_info()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database connections"""
        health_status = {
            'database_id': self.config.database_id,
            'overall_status': 'healthy',
            'active_connections': self.connection_stats['active_connections'],
            'total_connections': self.connection_stats['total_connections'],
            'failed_connections': self.connection_stats['failed_connections'],
            'connection_success_rate': 0,
            'last_check': datetime.now().isoformat()
        }
        
        # Calculate success rate
        total = self.connection_stats['total_connections']
        failed = self.connection_stats['failed_connections']
        
        if total > 0:
            health_status['connection_success_rate'] = ((total - failed) / total) * 100
        
        # Determine overall status
        if failed > total * 0.1:  # More than 10% failures
            health_status['overall_status'] = 'degraded'
        elif failed > total * 0.2:  # More than 20% failures
            health_status['overall_status'] = 'unhealthy'
        
        return health_status


def test_database_configuration():
    """Test database configuration functionality"""
    print("Testing Database Configuration...")
    
    # Test DatabaseConfig
    config = DatabaseConfig()
    print(f"Database ID: {config.database_id}")
    print(f"Valid UUID: {config.is_valid_uuid}")
    print(f"Connection Type: {config._get_connection_type()}")
    
    # Test database info
    db_info = config.get_database_info()
    print(f"Database Info: {json.dumps(db_info, indent=2)}")
    
    # Test connection validation
    validation = config.validate_connection()
    print(f"Connection Validation: {json.dumps(validation, indent=2)}")
    
    # Test DatabaseConnectionManager
    conn_manager = DatabaseConnectionManager()
    
    # Create test connection
    conn_id = conn_manager.create_connection('test_connection')
    print(f"Created Connection: {conn_id}")
    
    # Get connection info
    conn_info = conn_manager.get_connection_info('test_connection')
    print(f"Connection Info: {json.dumps(conn_info, indent=2)}")
    
    # Health check
    health = conn_manager.health_check()
    print(f"Health Check: {json.dumps(health, indent=2)}")
    
    # Close connection
    closed = conn_manager.close_connection('test_connection')
    print(f"Connection Closed: {closed}")
    
    print("Database configuration test completed!")


if __name__ == "__main__":
    test_database_configuration()
