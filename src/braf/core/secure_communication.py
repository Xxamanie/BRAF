"""
Secure Communication Layer for BRAF.

This module provides secure gRPC/WebSocket communication between C2 and workers
with TLS encryption, authorization, and HashiCorp Vault integration.
"""

import asyncio
import json
import logging
import ssl
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path

import grpc
from grpc import aio as aio_grpc
import websockets
import hvac
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

from braf.core.models import WorkerStatus, AutomationTask, TaskResult

logger = logging.getLogger(__name__)


@dataclass
class TLSConfig:
    """TLS configuration for secure communication."""
    
    cert_file: str
    key_file: str
    ca_file: Optional[str] = None
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
    check_hostname: bool = True


@dataclass
class AuthToken:
    """Authorization token for worker authentication."""
    
    token: str
    worker_id: str
    issued_at: datetime
    expires_at: datetime
    permissions: List[str]
    
    def is_valid(self) -> bool:
        """Check if token is still valid."""
        return datetime.now(timezone.utc) < self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if token has specific permission."""
        return permission in self.permissions


class VaultClient:
    """HashiCorp Vault client for production credential management."""
    
    def __init__(
        self,
        vault_url: str,
        vault_token: Optional[str] = None,
        vault_role_id: Optional[str] = None,
        vault_secret_id: Optional[str] = None
    ):
        """
        Initialize Vault client.
        
        Args:
            vault_url: Vault server URL
            vault_token: Direct token (for development)
            vault_role_id: AppRole role ID
            vault_secret_id: AppRole secret ID
        """
        self.vault_url = vault_url
        self.client = hvac.Client(url=vault_url)
        
        # Authenticate with Vault
        if vault_token:
            self.client.token = vault_token
        elif vault_role_id and vault_secret_id:
            self._authenticate_approle(vault_role_id, vault_secret_id)
        else:
            raise ValueError("Must provide either vault_token or AppRole credentials")
    
    def _authenticate_approle(self, role_id: str, secret_id: str):
        """Authenticate using AppRole method."""
        try:
            response = self.client.auth.approle.login(
                role_id=role_id,
                secret_id=secret_id
            )
            self.client.token = response['auth']['client_token']
            logger.info("Successfully authenticated with Vault using AppRole")
        except Exception as e:
            logger.error(f"Vault AppRole authentication failed: {e}")
            raise
    
    def get_secret(self, path: str) -> Dict[str, Any]:
        """Get secret from Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data']
        except Exception as e:
            logger.error(f"Failed to read secret from Vault: {e}")
            raise
    
    def store_secret(self, path: str, secret: Dict[str, Any]):
        """Store secret in Vault."""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=secret
            )
            logger.info(f"Secret stored in Vault at path: {path}")
        except Exception as e:
            logger.error(f"Failed to store secret in Vault: {e}")
            raise
    
    def generate_certificate(self, common_name: str, ttl: str = "24h") -> Dict[str, str]:
        """Generate certificate using Vault PKI."""
        try:
            response = self.client.secrets.pki.generate_certificate(
                name='braf-role',
                common_name=common_name,
                ttl=ttl
            )
            return {
                'certificate': response['data']['certificate'],
                'private_key': response['data']['private_key'],
                'ca_chain': response['data']['ca_chain']
            }
        except Exception as e:
            logger.error(f"Failed to generate certificate: {e}")
            raise


class CertificateManager:
    """Manages TLS certificates for secure communication."""
    
    def __init__(self, vault_client: Optional[VaultClient] = None):
        """Initialize certificate manager."""
        self.vault_client = vault_client
        self.certificates: Dict[str, Dict[str, str]] = {}
    
    def generate_self_signed_cert(
        self,
        common_name: str,
        key_size: int = 2048,
        validity_days: int = 365
    ) -> TLSConfig:
        """Generate self-signed certificate for development."""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "BRAF"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
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
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(common_name),
                x509.DNSName("localhost"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Save to files
        cert_file = f"{common_name}.crt"
        key_file = f"{common_name}.key"
        
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(Encoding.PEM))
        
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.PKCS8,
                NoEncryption()
            ))
        
        logger.info(f"Generated self-signed certificate for {common_name}")
        
        return TLSConfig(
            cert_file=cert_file,
            key_file=key_file,
            verify_mode=ssl.CERT_NONE  # For self-signed certs
        )
    
    def get_vault_certificate(self, common_name: str) -> TLSConfig:
        """Get certificate from Vault PKI."""
        if not self.vault_client:
            raise ValueError("Vault client not configured")
        
        cert_data = self.vault_client.generate_certificate(common_name)
        
        # Save certificate files
        cert_file = f"{common_name}.crt"
        key_file = f"{common_name}.key"
        ca_file = f"{common_name}-ca.crt"
        
        with open(cert_file, "w") as f:
            f.write(cert_data['certificate'])
        
        with open(key_file, "w") as f:
            f.write(cert_data['private_key'])
        
        with open(ca_file, "w") as f:
            f.write('\n'.join(cert_data['ca_chain']))
        
        return TLSConfig(
            cert_file=cert_file,
            key_file=key_file,
            ca_file=ca_file
        )


class TokenManager:
    """Manages authorization tokens for worker authentication."""
    
    def __init__(self, secret_key: str):
        """Initialize token manager."""
        self.secret_key = secret_key
        self.active_tokens: Dict[str, AuthToken] = {}
    
    def generate_token(
        self,
        worker_id: str,
        permissions: List[str],
        ttl_hours: int = 24
    ) -> AuthToken:
        """Generate new authorization token."""
        import jwt
        
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=ttl_hours)
        
        payload = {
            'worker_id': worker_id,
            'permissions': permissions,
            'iat': now.timestamp(),
            'exp': expires_at.timestamp()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        auth_token = AuthToken(
            token=token,
            worker_id=worker_id,
            issued_at=now,
            expires_at=expires_at,
            permissions=permissions
        )
        
        self.active_tokens[token] = auth_token
        logger.info(f"Generated token for worker {worker_id}")
        
        return auth_token
    
    def verify_token(self, token: str) -> Optional[AuthToken]:
        """Verify and return token if valid."""
        try:
            import jwt
            
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            if token in self.active_tokens:
                auth_token = self.active_tokens[token]
                if auth_token.is_valid():
                    return auth_token
                else:
                    # Remove expired token
                    del self.active_tokens[token]
            
            return None
            
        except jwt.InvalidTokenError:
            logger.warning("Invalid token provided")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self.active_tokens:
            del self.active_tokens[token]
            logger.info("Token revoked")
            return True
        return False


class SecureGRPCServer:
    """Secure gRPC server for C2-Worker communication."""
    
    def __init__(
        self,
        tls_config: TLSConfig,
        token_manager: TokenManager,
        host: str = "0.0.0.0",
        port: int = 50051
    ):
        """Initialize secure gRPC server."""
        self.tls_config = tls_config
        self.token_manager = token_manager
        self.host = host
        self.port = port
        self.server: Optional[aio_grpc.Server] = None
        self.message_handlers: Dict[str, Callable] = {}
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add message handler for specific message type."""
        self.message_handlers[message_type] = handler
    
    async def start(self):
        """Start the gRPC server."""
        # Create SSL credentials
        with open(self.tls_config.key_file, 'rb') as f:
            private_key = f.read()
        
        with open(self.tls_config.cert_file, 'rb') as f:
            certificate_chain = f.read()
        
        server_credentials = grpc.ssl_server_credentials(
            [(private_key, certificate_chain)]
        )
        
        # Create server
        self.server = aio_grpc.server()
        
        # Add secure port
        listen_addr = f'{self.host}:{self.port}'
        self.server.add_secure_port(listen_addr, server_credentials)
        
        # Start server
        await self.server.start()
        logger.info(f"Secure gRPC server started on {listen_addr}")
    
    async def stop(self):
        """Stop the gRPC server."""
        if self.server:
            await self.server.stop(grace=5)
            logger.info("Secure gRPC server stopped")
    
    def authenticate_request(self, metadata) -> Optional[AuthToken]:
        """Authenticate gRPC request using metadata."""
        for key, value in metadata:
            if key == 'authorization':
                token = value.replace('Bearer ', '')
                return self.token_manager.verify_token(token)
        return None


class SecureWebSocketServer:
    """Secure WebSocket server for real-time communication."""
    
    def __init__(
        self,
        tls_config: TLSConfig,
        token_manager: TokenManager,
        host: str = "0.0.0.0",
        port: int = 8765
    ):
        """Initialize secure WebSocket server."""
        self.tls_config = tls_config
        self.token_manager = token_manager
        self.host = host
        self.port = port
        self.connected_clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Dict[str, Callable] = {}
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add message handler for specific message type."""
        self.message_handlers[message_type] = handler
    
    async def authenticate_websocket(self, websocket, path):
        """Authenticate WebSocket connection."""
        try:
            # Get token from query parameters or headers
            token = None
            
            # Check query parameters
            if '?' in path:
                query_params = dict(param.split('=') for param in path.split('?')[1].split('&'))
                token = query_params.get('token')
            
            # Check headers if no token in query
            if not token:
                auth_header = websocket.request_headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header[7:]
            
            if not token:
                await websocket.close(code=4001, reason="No authentication token")
                return None
            
            auth_token = self.token_manager.verify_token(token)
            if not auth_token:
                await websocket.close(code=4001, reason="Invalid token")
                return None
            
            return auth_token
            
        except Exception as e:
            logger.error(f"WebSocket authentication error: {e}")
            await websocket.close(code=4000, reason="Authentication error")
            return None
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        auth_token = await self.authenticate_websocket(websocket, path)
        if not auth_token:
            return
        
        worker_id = auth_token.worker_id
        self.connected_clients[worker_id] = websocket
        
        logger.info(f"Worker {worker_id} connected via WebSocket")
        
        try:
            async for message in websocket:
                await self.handle_message(worker_id, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Worker {worker_id} disconnected")
        finally:
            if worker_id in self.connected_clients:
                del self.connected_clients[worker_id]
    
    async def handle_message(self, worker_id: str, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                await handler(worker_id, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def send_to_worker(self, worker_id: str, message: Dict[str, Any]):
        """Send message to specific worker."""
        if worker_id in self.connected_clients:
            websocket = self.connected_clients[worker_id]
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Worker {worker_id} connection closed")
                del self.connected_clients[worker_id]
        else:
            logger.warning(f"Worker {worker_id} not connected")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected workers."""
        if self.connected_clients:
            await asyncio.gather(
                *[self.send_to_worker(worker_id, message) 
                  for worker_id in self.connected_clients.keys()],
                return_exceptions=True
            )
    
    async def start(self):
        """Start the WebSocket server."""
        # Create SSL context
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(self.tls_config.cert_file, self.tls_config.key_file)
        
        # Start server
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ssl=ssl_context
        )
        
        logger.info(f"Secure WebSocket server started on wss://{self.host}:{self.port}")
        return server


class SecureCommunicationManager:
    """Main manager for secure communication between C2 and workers."""
    
    def __init__(
        self,
        vault_config: Optional[Dict[str, str]] = None,
        use_vault: bool = False
    ):
        """Initialize secure communication manager."""
        self.vault_client = None
        self.certificate_manager = None
        self.token_manager = None
        self.grpc_server = None
        self.websocket_server = None
        
        # Initialize Vault if configured
        if use_vault and vault_config:
            self.vault_client = VaultClient(**vault_config)
            self.certificate_manager = CertificateManager(self.vault_client)
        else:
            self.certificate_manager = CertificateManager()
        
        # Initialize token manager
        secret_key = self._get_secret_key()
        self.token_manager = TokenManager(secret_key)
    
    def _get_secret_key(self) -> str:
        """Get secret key for token signing."""
        if self.vault_client:
            try:
                secret = self.vault_client.get_secret('braf/jwt-secret')
                return secret['key']
            except Exception:
                logger.warning("Failed to get JWT secret from Vault, generating new one")
        
        # Generate random secret key
        import secrets
        secret_key = secrets.token_urlsafe(32)
        
        # Store in Vault if available
        if self.vault_client:
            try:
                self.vault_client.store_secret('braf/jwt-secret', {'key': secret_key})
            except Exception as e:
                logger.warning(f"Failed to store JWT secret in Vault: {e}")
        
        return secret_key
    
    async def setup_secure_servers(
        self,
        common_name: str = "braf-c2",
        grpc_port: int = 50051,
        websocket_port: int = 8765
    ):
        """Set up secure gRPC and WebSocket servers."""
        # Get TLS configuration
        if self.vault_client:
            tls_config = self.certificate_manager.get_vault_certificate(common_name)
        else:
            tls_config = self.certificate_manager.generate_self_signed_cert(common_name)
        
        # Create servers
        self.grpc_server = SecureGRPCServer(
            tls_config=tls_config,
            token_manager=self.token_manager,
            port=grpc_port
        )
        
        self.websocket_server = SecureWebSocketServer(
            tls_config=tls_config,
            token_manager=self.token_manager,
            port=websocket_port
        )
        
        # Add message handlers
        self._setup_message_handlers()
        
        logger.info("Secure communication servers configured")
    
    def _setup_message_handlers(self):
        """Set up default message handlers."""
        if self.websocket_server:
            self.websocket_server.add_message_handler('heartbeat', self._handle_heartbeat)
            self.websocket_server.add_message_handler('task_result', self._handle_task_result)
            self.websocket_server.add_message_handler('status_update', self._handle_status_update)
    
    async def _handle_heartbeat(self, worker_id: str, data: Dict[str, Any]):
        """Handle worker heartbeat."""
        logger.debug(f"Heartbeat from worker {worker_id}")
        
        # Send acknowledgment
        response = {
            'type': 'heartbeat_ack',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if self.websocket_server:
            await self.websocket_server.send_to_worker(worker_id, response)
    
    async def _handle_task_result(self, worker_id: str, data: Dict[str, Any]):
        """Handle task result from worker."""
        logger.info(f"Task result received from worker {worker_id}")
        
        # Process task result
        task_result = TaskResult(**data.get('result', {}))
        
        # Store result in database or forward to task executor
        # This would integrate with the task executor and database
        
        # Send acknowledgment
        response = {
            'type': 'task_result_ack',
            'task_id': task_result.task_id
        }
        
        if self.websocket_server:
            await self.websocket_server.send_to_worker(worker_id, response)
    
    async def _handle_status_update(self, worker_id: str, data: Dict[str, Any]):
        """Handle worker status update."""
        logger.debug(f"Status update from worker {worker_id}")
        
        # Process status update
        status = WorkerStatus(**data.get('status', {}))
        
        # Update worker status in database or monitoring system
    
    async def start_servers(self):
        """Start all secure communication servers."""
        if self.grpc_server:
            await self.grpc_server.start()
        
        if self.websocket_server:
            await self.websocket_server.start()
        
        logger.info("All secure communication servers started")
    
    async def stop_servers(self):
        """Stop all secure communication servers."""
        if self.grpc_server:
            await self.grpc_server.stop()
        
        if self.websocket_server:
            # WebSocket server stop is handled by the server object
            pass
        
        logger.info("All secure communication servers stopped")
    
    def generate_worker_token(
        self,
        worker_id: str,
        permissions: Optional[List[str]] = None
    ) -> AuthToken:
        """Generate authentication token for worker."""
        if permissions is None:
            permissions = ['task_execution', 'status_reporting', 'heartbeat']
        
        return self.token_manager.generate_token(worker_id, permissions)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for workers."""
        return {
            'grpc_endpoint': f"grpcs://localhost:50051",
            'websocket_endpoint': f"wss://localhost:8765",
            'certificate_verification': True
        }


# Global secure communication manager instance
_secure_comm_manager: Optional[SecureCommunicationManager] = None


def get_secure_communication_manager() -> Optional[SecureCommunicationManager]:
    """
    Get global secure communication manager instance.
    
    Returns:
        Secure communication manager instance or None if not initialized
    """
    return _secure_comm_manager


def init_secure_communication_manager(
    vault_config: Optional[Dict[str, str]] = None,
    use_vault: bool = False
) -> SecureCommunicationManager:
    """
    Initialize global secure communication manager.
    
    Args:
        vault_config: Vault configuration
        use_vault: Whether to use Vault for secrets
        
    Returns:
        Initialized secure communication manager
    """
    global _secure_comm_manager
    
    _secure_comm_manager = SecureCommunicationManager(
        vault_config=vault_config,
        use_vault=use_vault
    )
    
    return _secure_comm_manager