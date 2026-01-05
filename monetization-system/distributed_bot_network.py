"""
Distributed Bot Network for 24/7 Global Operation
Implements multi-region bot deployment with automatic scaling and load balancing
"""

import os
import json
import time
import random
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import aiohttp
import redis.asyncio as redis
# from geopy.geocoders import Nominatim  # Optional for geo features
# import boto3  # AWS for global deployment - Optional
# from azure.identity import DefaultAzureCredential  # Optional
# from azure.mgmt.compute import ComputeManagementClient  # Optional

logger = logging.getLogger(__name__)

@dataclass
class BotNode:
    """Represents a bot node in the distributed network"""
    node_id: str
    region: str
    provider: str  # aws, azure, gcp, digitalocean
    instance_type: str
    ip_address: str
    status: str  # active, standby, terminated
    last_heartbeat: datetime
    active_sessions: int
    max_sessions: int
    earnings_today: float
    uptime: float

class DistributedBotNetwork:
    """Manages global bot network deployment and scaling"""

    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.regions = [
            'us-east-1', 'us-west-2', 'eu-west-1', 'eu-central-1',
            'ap-southeast-1', 'ap-northeast-1', 'sa-east-1', 'ca-central-1'
        ]
        self.providers = ['aws', 'azure', 'gcp', 'digitalocean']
        self.nodes: Dict[str, BotNode] = {}
        self.scaling_threshold = 80  # Scale up when utilization > 80%
        self.min_nodes_per_region = 3
        self.max_nodes_per_region = 50

        # Cloud provider configurations
        self.provider_configs = {
            'aws': {
                'instance_type': 't3.medium',
                'ami_id': 'ami-0abcdef1234567890',
                'security_group': 'sg-botnetwork',
                'key_name': 'bot-network-key',
                'cost_per_hour': 0.0416,  # t3.medium hourly cost
                'max_concurrent_deployments': 10
            },
            'azure': {
                'vm_size': 'Standard_B2s',
                'image_publisher': 'Canonical',
                'image_offer': 'Ubuntu2204',
                'image_sku': '22.04-LTS',
                'cost_per_hour': 0.026,  # B2s hourly cost
                'max_concurrent_deployments': 8
            },
            'gcp': {
                'machine_type': 'e2-medium',
                'image_project': 'ubuntu-os-cloud',
                'image_family': 'ubuntu-2204-lts',
                'cost_per_hour': 0.033,  # e2-medium hourly cost
                'max_concurrent_deployments': 12
            },
            'digitalocean': {
                'droplet_size': 's-1vcpu-1gb',
                'image_slug': 'ubuntu-22-04-x64',
                'cost_per_hour': 0.0085,  # Basic droplet hourly cost
                'max_concurrent_deployments': 5
            }
        }

        # Initialize cloud SDK clients
        self._initialize_cloud_clients()

    def _initialize_cloud_clients(self):
        """Initialize cloud provider SDK clients"""
        try:
            # AWS SDK initialization
            import boto3
            self.aws_client = boto3.client('ec2')
            self.aws_pricing = boto3.client('pricing', region_name='us-east-1')
            logger.info("AWS SDK initialized")
        except ImportError:
            logger.warning("AWS SDK not available - install boto3")
            self.aws_client = None
        except Exception as e:
            logger.error(f"AWS SDK initialization failed: {e}")
            self.aws_client = None

        try:
            # Azure SDK initialization
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.compute import ComputeManagementClient
            from azure.mgmt.resource import ResourceManagementClient

            self.azure_credential = DefaultAzureCredential()
            self.azure_compute_client = ComputeManagementClient(self.azure_credential, "subscription-id")
            self.azure_resource_client = ResourceManagementClient(self.azure_credential, "subscription-id")
            logger.info("Azure SDK initialized")
        except ImportError:
            logger.warning("Azure SDK not available - install azure-identity azure-mgmt-compute")
            self.azure_credential = None
        except Exception as e:
            logger.error(f"Azure SDK initialization failed: {e}")
            self.azure_credential = None

        try:
            # GCP SDK initialization
            from google.cloud import compute_v1
            self.gcp_compute_client = compute_v1.InstancesClient()
            logger.info("GCP SDK initialized")
        except ImportError:
            logger.warning("GCP SDK not available - install google-cloud-compute")
            self.gcp_compute_client = None
        except Exception as e:
            logger.error(f"GCP SDK initialization failed: {e}")
            self.gcp_compute_client = None

        # DigitalOcean API client (simplified)
        self.do_api_token = os.getenv('DO_API_TOKEN')
        if self.do_api_token:
            logger.info("DigitalOcean API configured")
        else:
            logger.warning("DigitalOcean API token not configured")

    async def initialize_network(self):
        """Initialize the distributed network"""
        logger.info("Initializing distributed bot network...")

        # Deploy minimum nodes in each region
        for region in self.regions:
            await self.ensure_minimum_nodes(region)

        # Start monitoring loop
        asyncio.create_task(self.monitor_and_scale())

        logger.info("Distributed bot network initialized")

    async def ensure_minimum_nodes(self, region: str):
        """Ensure minimum number of nodes are running in a region"""
        active_nodes = [node for node in self.nodes.values()
                       if node.region == region and node.status == 'active']

        nodes_needed = self.min_nodes_per_region - len(active_nodes)

        for i in range(max(0, nodes_needed)):
            provider = random.choice(self.providers)
            await self.deploy_node(region, provider)

    async def deploy_node(self, region: str, provider: str) -> Optional[BotNode]:
        """Deploy a new bot node in specified region with cost tracking"""
        try:
            # Check deployment limits for provider
            config = self.provider_configs[provider]
            current_deployments = len([n for n in self.nodes.values()
                                     if n.provider == provider and n.status in ['active', 'pending']])

            if current_deployments >= config['max_concurrent_deployments']:
                logger.warning(f"Deployment limit reached for {provider}")
                return None

            # Check budget constraints
            if not await self._check_budget_constraints(provider, region):
                logger.warning(f"Budget constraints prevent deployment on {provider} in {region}")
                return None

            node_id = f"{provider}-{region}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Deploy based on provider
            if provider == 'aws':
                instance = await self._deploy_aws_instance(region, node_id)
            elif provider == 'azure':
                instance = await self._deploy_azure_instance(region, node_id)
            elif provider == 'gcp':
                instance = await self._deploy_gcp_instance(region, node_id)
            else:
                instance = await self._deploy_digitalocean_instance(region, node_id)

            if not instance:
                return None

            # Create node with cost tracking
            node = BotNode(
                node_id=node_id,
                region=region,
                provider=provider,
                instance_type=config.get(f'{provider}_instance_type',
                                       config.get('instance_type', 'standard')),
                ip_address=instance['ip_address'],
                status='active',
                last_heartbeat=datetime.now(timezone.utc),
                active_sessions=0,
                max_sessions=100,
                earnings_today=0.0,
                uptime=0.0
            )

            # Track deployment cost
            await self._track_deployment_cost(node, instance)

            self.nodes[node_id] = node
            await self._register_node(node)

            logger.info(f"Deployed bot node {node_id} in {region} (Cost: ${instance.get('hourly_cost', 0):.4f}/hr)")
            return node

        except Exception as e:
            logger.error(f"Failed to deploy node in {region}: {e}")
            return None

    async def _track_deployment_cost(self, node: BotNode, instance: Dict[str, Any]):
        """Track deployment costs for cost optimization"""
        try:
            cost_data = {
                'node_id': node.node_id,
                'provider': node.provider,
                'region': node.region,
                'instance_type': instance.get('instance_type', node.instance_type),
                'hourly_cost': instance.get('hourly_cost', 0),
                'deployment_time': datetime.now(timezone.utc).isoformat(),
                'spot_instance': instance.get('spot_instance', False)
            }

            # Store cost data in Redis for monitoring
            await self.redis_client.hset('deployment_costs', node.node_id, json.dumps(cost_data))

            # Update total cost tracking
            total_cost_key = f"total_cost_{node.provider}_{node.region}"
            current_total = float(await self.redis_client.get(total_cost_key) or 0)
            await self.redis_client.set(total_cost_key, current_total + cost_data['hourly_cost'])

        except Exception as e:
            logger.error(f"Cost tracking failed for {node.node_id}: {e}")

    async def terminate_node(self, node_id: str):
        """Terminate a bot node with proper cleanup and cost tracking"""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        try:
            # Calculate uptime and costs before termination
            uptime_hours = node.uptime
            cost_data = await self.redis_client.hget('deployment_costs', node_id)
            if cost_data:
                cost_info = json.loads(cost_data)
                total_cost = uptime_hours * cost_info['hourly_cost']
                logger.info(f"Node {node_id} cost: ${total_cost:.4f} for {uptime_hours:.2f} hours")

            # Terminate based on provider
            success = False
            if node.provider == 'aws':
                success = await self._terminate_aws_instance(node.region, node.node_id)
            elif node.provider == 'azure':
                success = await self._terminate_azure_instance(node.region, node.node_id)
            elif node.provider == 'gcp':
                success = await self._terminate_gcp_instance(node.region, node.node_id)
            else:
                success = await self._terminate_digitalocean_instance(node.region, node.node_id)

            if success:
                # Update cost tracking
                total_cost_key = f"total_cost_{node.provider}_{node.region}"
                current_total = float(await self.redis_client.get(total_cost_key) or 0)
                cost_info = json.loads(cost_data) if cost_data else {'hourly_cost': 0}
                await self.redis_client.set(total_cost_key,
                                          current_total - cost_info['hourly_cost'])

                # Clean up tracking data
                await self.redis_client.hdel('deployment_costs', node_id)
                await self.redis_client.hdel('bot_nodes', node_id)

                node.status = 'terminated'
                del self.nodes[node_id]

                logger.info(f"Terminated bot node {node_id}")
            else:
                logger.error(f"Failed to terminate {node_id}")

        except Exception as e:
            logger.error(f"Termination failed for {node_id}: {e}")
            # Mark as failed but don't delete from tracking
            node.status = 'termination_failed'

    async def _terminate_aws_instance(self, region: str, instance_id: str) -> bool:
        """Terminate AWS instance with error handling"""
        try:
            if not self.aws_client:
                logger.error("AWS client not available for termination")
                return False

            ec2 = boto3.client('ec2', region_name=region)

            # Terminate instance
            response = ec2.terminate_instances(InstanceIds=[instance_id])

            # Wait for termination
            waiter = ec2.get_waiter('instance_terminated')
            waiter.wait(
                InstanceIds=[instance_id],
                WaiterConfig={'Delay': 5, 'MaxAttempts': 24}  # 2min timeout
            )

            logger.info(f"AWS instance {instance_id} terminated successfully")
            return True

        except Exception as e:
            logger.error(f"AWS termination failed for {instance_id}: {e}")
            return False

    async def get_cost_analysis(self) -> Dict[str, Any]:
        """Get comprehensive cost analysis across all providers and regions"""
        try:
            cost_analysis = {
                'total_hourly_cost': 0.0,
                'cost_by_provider': {},
                'cost_by_region': {},
                'active_instances': len([n for n in self.nodes.values() if n.status == 'active']),
                'total_instances': len(self.nodes),
                'cost_efficiency_score': 0.0
            }

            # Calculate costs by provider and region
            for node in self.nodes.values():
                if node.status == 'active':
                    cost_data = await self.redis_client.hget('deployment_costs', node.node_id)
                    if cost_data:
                        cost_info = json.loads(cost_data)
                        hourly_cost = cost_info.get('hourly_cost', 0)

                        # Provider costs
                        if node.provider not in cost_analysis['cost_by_provider']:
                            cost_analysis['cost_by_provider'][node.provider] = 0.0
                        cost_analysis['cost_by_provider'][node.provider] += hourly_cost

                        # Region costs
                        region_key = f"{node.provider}-{node.region}"
                        if region_key not in cost_analysis['cost_by_region']:
                            cost_analysis['cost_by_region'][region_key] = 0.0
                        cost_analysis['cost_by_region'][region_key] += hourly_cost

                        cost_analysis['total_hourly_cost'] += hourly_cost

            # Calculate cost efficiency (lower is better)
            if cost_analysis['active_instances'] > 0:
                avg_cost_per_instance = cost_analysis['total_hourly_cost'] / cost_analysis['active_instances']
                # Efficiency score: lower cost per instance = higher efficiency
                cost_analysis['cost_efficiency_score'] = max(0, 100 - (avg_cost_per_instance * 100))

            return cost_analysis

        except Exception as e:
            logger.error(f"Cost analysis failed: {e}")
            return {'error': str(e)}

    async def optimize_costs(self) -> Dict[str, Any]:
        """Optimize costs by terminating expensive instances and using cheaper alternatives"""
        try:
            optimizations = {
                'terminated_expensive_instances': [],
                'switched_to_spot_instances': [],
                'regional_optimizations': [],
                'total_savings_hourly': 0.0
            }

            # Find expensive instances to terminate
            cost_analysis = await self.get_cost_analysis()
            avg_cost = cost_analysis['total_hourly_cost'] / max(cost_analysis['active_instances'], 1)

            for node in self.nodes.values():
                if node.status == 'active':
                    cost_data = await self.redis_client.hget('deployment_costs', node.node_id)
                    if cost_data:
                        cost_info = json.loads(cost_data)
                        node_cost = cost_info.get('hourly_cost', 0)

                        # If instance is >50% more expensive than average, consider termination
                        if node_cost > avg_cost * 1.5:
                            await self.terminate_node(node.node_id)
                            optimizations['terminated_expensive_instances'].append({
                                'node_id': node.node_id,
                                'hourly_cost': node_cost,
                                'region': node.region
                            })
                            optimizations['total_savings_hourly'] += node_cost

            # Suggest regional optimizations (pseudo-code for actual implementation)
            optimizations['regional_optimizations'] = [
                "Consider moving US-West-2 instances to US-East-1 for lower latency",
                "Evaluate spot instance usage for cost reduction",
                "Consider reserved instances for long-running deployments"
            ]

            logger.info(f"Cost optimization completed: ${optimizations['total_savings_hourly']:.4f}/hr savings")
            return optimizations

        except Exception as e:
            logger.error(f"Cost optimization failed: {e}")
            return {'error': str(e)}

    async def _check_budget_constraints(self, provider: str, region: str) -> bool:
        """Check if deployment fits within budget constraints"""
        try:
            config = self.provider_configs[provider]

            # Calculate current spending
            current_cost_per_hour = sum(
                config['cost_per_hour'] for node in self.nodes.values()
                if node.provider == provider and node.status == 'active'
            )

            # Check against budget (example: $100/hour max per provider)
            max_budget_per_provider = float(os.getenv('MAX_BUDGET_PER_PROVIDER', '100.0'))
            projected_cost = current_cost_per_hour + config['cost_per_hour']

            if projected_cost > max_budget_per_provider:
                return False

            # Regional cost optimization
            regional_cost = sum(
                self.provider_configs[n.provider]['cost_per_hour']
                for n in self.nodes.values()
                if n.region == region and n.status == 'active'
            )

            # Prefer lower cost regions
            return True

        except Exception as e:
            logger.error(f"Budget check failed: {e}")
            return True  # Allow deployment on error

    async def _deploy_aws_instance(self, region: str, node_id: str) -> Dict[str, Any]:
        """Deploy AWS EC2 instance with proper error handling and cost tracking"""
        if not self.aws_client:
            logger.error("AWS client not initialized")
            return None

        try:
            # Create EC2 client for specific region
            ec2 = boto3.client('ec2', region_name=region)
            config = self.provider_configs['aws']

            # Prepare instance configuration
            instance_config = {
                'ImageId': config['ami_id'],
                'MinCount': 1,
                'MaxCount': 1,
                'InstanceType': config['instance_type'],
                'KeyName': config.get('key_name', 'bot-network-key'),
                'SecurityGroupIds': [config.get('security_group', 'sg-botnetwork')],
                'TagSpecifications': [{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'braf-bot-{node_id}'},
                        {'Key': 'Purpose', 'Value': 'distributed-bot-network'},
                        {'Key': 'ManagedBy', 'Value': 'braf-controller'},
                        {'Key': 'Environment', 'Value': 'production'}
                    ]
                }],
                'UserData': self._get_userdata_script(),
                # Enable detailed monitoring for cost tracking
                'Monitoring': {'Enabled': True},
                # Use spot instances for cost optimization when possible
                'InstanceMarketOptions': {
                    'MarketType': 'spot',
                    'SpotOptions': {
                        'MaxPrice': str(config['cost_per_hour'] * 1.5),  # Max 50% above on-demand
                        'SpotInstanceType': 'one-time'
                    }
                } if os.getenv('USE_SPOT_INSTANCES', 'true').lower() == 'true' else {}
            }

            # Deploy instance
            response = ec2.run_instances(**instance_config)
            instance_id = response['Instances'][0]['InstanceId']

            # Wait for instance to be running with timeout
            waiter = ec2.get_waiter('instance_running')
            waiter.wait(
                InstanceIds=[instance_id],
                WaiterConfig={'Delay': 5, 'MaxAttempts': 60}  # 5min timeout
            )

            # Get instance details
            instances = ec2.describe_instances(InstanceIds=[instance_id])
            instance = instances['Reservations'][0]['Instances'][0]
            ip_address = instance.get('PublicIpAddress')

            if not ip_address:
                # Wait a bit more for IP assignment
                await asyncio.sleep(10)
                instances = ec2.describe_instances(InstanceIds=[instance_id])
                instance = instances['Reservations'][0]['Instances'][0]
                ip_address = instance.get('PublicIpAddress')

            if not ip_address:
                raise Exception("Instance deployed but no public IP assigned")

            return {
                'instance_id': instance_id,
                'ip_address': ip_address,
                'instance_type': config['instance_type'],
                'hourly_cost': config['cost_per_hour'],
                'region': region,
                'spot_instance': bool(instance_config.get('InstanceMarketOptions'))
            }

        except Exception as e:
            logger.error(f"AWS deployment failed: {e}")
            # Attempt cleanup if instance was created
            if 'instance_id' in locals():
                try:
                    ec2.terminate_instances(InstanceIds=[instance_id])
                    logger.info(f"Cleaned up failed instance {instance_id}")
                except:
                    pass
            return None

    async def _deploy_azure_instance(self, region: str, node_id: str) -> Dict[str, Any]:
        """Deploy Azure VM instance"""
        # Simplified Azure deployment
        compute_client = ComputeManagementClient(self.azure_credential, 'subscription-id')

        # Implementation would create VM with similar specs
        return {'instance_id': f'azure-{node_id}', 'ip_address': '192.168.1.100'}

    async def _deploy_gcp_instance(self, region: str, node_id: str) -> Dict[str, Any]:
        """Deploy GCP Compute Engine instance"""
        # Simplified GCP deployment
        return {'instance_id': f'gcp-{node_id}', 'ip_address': '192.168.1.101'}

    async def _deploy_digitalocean_instance(self, region: str, node_id: str) -> Dict[str, Any]:
        """Deploy DigitalOcean droplet"""
        # Simplified DO deployment
        return {'instance_id': f'do-{node_id}', 'ip_address': '192.168.1.102'}

    def _get_userdata_script(self) -> str:
        """Get cloud-init script for bot node setup"""
        return """#!/bin/bash
# Install dependencies
apt update
apt install -y python3 python3-pip docker.io

# Clone and setup BRAF
git clone https://github.com/your-repo/braf.git
cd braf/monetization-system

# Install requirements
pip3 install -r requirements.txt

# Start bot worker
python3 -c "
from automation.browser_automation import ProductionBrowserAutomation
automation = ProductionBrowserAutomation()
# Register with central coordinator
"
"""

    async def _register_node(self, node: BotNode):
        """Register node with central coordinator"""
        node_data = {
            'node_id': node.node_id,
            'region': node.region,
            'ip_address': node.ip_address,
            'max_sessions': node.max_sessions,
            'status': node.status
        }

        await self.redis_client.hset('bot_nodes', node.node_id, json.dumps(node_data))

    async def monitor_and_scale(self):
        """Continuous monitoring and auto-scaling"""
        while True:
            try:
                await self._check_node_health()
                await self._scale_based_on_demand()
                await self._redistribute_load()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)

    async def _check_node_health(self):
        """Check health of all nodes"""
        for node_id, node in list(self.nodes.items()):
            try:
                # Ping node
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{node.ip_address}:8000/health", timeout=10) as response:
                        if response.status == 200:
                            node.last_heartbeat = datetime.now(timezone.utc)
                            node.status = 'active'
                        else:
                            node.status = 'unhealthy'
            except:
                node.status = 'offline'

            # Update Redis
            await self._register_node(node)

    async def _scale_based_on_demand(self):
        """Scale nodes based on current demand"""
        for region in self.regions:
            region_nodes = [n for n in self.nodes.values()
                          if n.region == region and n.status == 'active']

            if not region_nodes:
                continue

            total_capacity = sum(n.max_sessions for n in region_nodes)
            total_active = sum(n.active_sessions for n in region_nodes)
            utilization = (total_active / total_capacity) * 100 if total_capacity > 0 else 0

            if utilization > self.scaling_threshold:
                # Scale up
                await self.deploy_node(region, random.choice(self.providers))
                logger.info(f"Scaled up in {region}: utilization {utilization:.1f}%")

            elif utilization < 30 and len(region_nodes) > self.min_nodes_per_region:
                # Scale down
                node_to_terminate = min(region_nodes, key=lambda n: n.uptime)
                await self.terminate_node(node_to_terminate.node_id)
                logger.info(f"Scaled down in {region}: utilization {utilization:.1f}%")

    async def _redistribute_load(self):
        """Redistribute tasks across healthy nodes"""
        # Get pending tasks from queue
        pending_tasks = await self.redis_client.llen('automation_queue')

        if pending_tasks > 0:
            healthy_nodes = [n for n in self.nodes.values() if n.status == 'active']

            # Sort by available capacity
            healthy_nodes.sort(key=lambda n: n.active_sessions / n.max_sessions)

            # Redistribute tasks to least loaded nodes
            for node in healthy_nodes[:5]:  # Top 5 least loaded
                if node.active_sessions < node.max_sessions:
                    # Send task to node
                    await self._send_task_to_node(node, pending_tasks)

    async def _send_task_to_node(self, node: BotNode, task_count: int):
        """Send tasks to specific node"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {'task_count': min(task_count, node.max_sessions - node.active_sessions)}
                async with session.post(f"http://{node.ip_address}:8000/tasks", json=payload) as response:
                    if response.status == 200:
                        node.active_sessions += payload['task_count']
        except Exception as e:
            logger.error(f"Failed to send tasks to {node.node_id}: {e}")

    async def terminate_node(self, node_id: str):
        """Terminate a bot node"""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        try:
            if node.provider == 'aws':
                await self._terminate_aws_instance(node.region, node.node_id)
            # Similar for other providers

            node.status = 'terminated'
            await self.redis_client.hdel('bot_nodes', node_id)
            del self.nodes[node_id]

            logger.info(f"Terminated bot node {node_id}")

        except Exception as e:
            logger.error(f"Failed to terminate {node_id}: {e}")

    async def _terminate_aws_instance(self, region: str, instance_id: str):
        """Terminate AWS instance"""
        ec2 = self.aws_session.client('ec2', region_name=region)
        ec2.terminate_instances(InstanceIds=[instance_id])

    async def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        active_nodes = [n for n in self.nodes.values() if n.status == 'active']
        total_capacity = sum(n.max_sessions for n in active_nodes)
        total_active = sum(n.active_sessions for n in active_nodes)

        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_nodes),
            'total_capacity': total_capacity,
            'total_active_sessions': total_active,
            'utilization_percent': (total_active / total_capacity * 100) if total_capacity > 0 else 0,
            'regions_covered': len(set(n.region for n in active_nodes)),
            'providers_used': len(set(n.provider for n in active_nodes)),
            'total_earnings_today': sum(n.earnings_today for n in active_nodes),
            'average_uptime': sum(n.uptime for n in active_nodes) / len(active_nodes) if active_nodes else 0
        }

# Global instance
distributed_network = DistributedBotNetwork()