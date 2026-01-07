# BRAF Live Deployment Setup Guide
## Migrating from Simulation to Production with Oracle Cloud & Kuitter.Space

## Executive Summary

This comprehensive guide provides step-by-step instructions for migrating BRAF from simulation to live production deployment using **Oracle Cloud Infrastructure (OCI)** and the **Kuitter.Space domain**. The deployment achieves the **Extended Live Test Framework** with statistical significance validation.

---

## 1. Prerequisites & Requirements

### Oracle Cloud Infrastructure Setup
```bash
# Required OCI Resources
OCI Requirements:
- Budget: $10,000/month (initial deployment)
- Regions: Primary (us-ashburn-1), DR (eu-frankfurt-1)
- Tenancy: Dedicated compartment for BRAF operations
- Networking: Virtual Cloud Network (VCN) with security lists
- Compute: 200 VM instances (AMD-based for cost optimization)
- Storage: 10TB Block Storage + Object Storage
- Database: Autonomous Database (ADB) for Sentinel
```

### Domain & SSL Configuration
```bash
# Kuitter.Space Domain Setup
Domain: kuitter.space
SSL: Let's Encrypt Wildcard Certificate
CDN: Oracle Cloud CDN
DNS: Oracle DNS Zones

Subdomains Required:
- api.kuitter.space          # BRAF API endpoints
- sentinel.kuitter.space     # Sentinel dashboard
- test.kuitter.space         # Live test environment
- monitor.kuitter.space      # Monitoring dashboards
- admin.kuitter.space        # Administrative interface
```

### Local Development Environment
```bash
# Required Software
- Oracle CLI (oci) configured
- Terraform 1.5+
- Ansible 2.15+
- Docker 24+
- kubectl 1.28+
- Helm 3.12+

# Authentication
- OCI API keys configured
- Oracle Container Registry access
- Domain registrar API access
```

---

## 2. Oracle Cloud Infrastructure Provisioning

### Step 1: Tenancy & Compartment Setup
```bash
# Create dedicated compartment for BRAF
oci iam compartment create \
    --name "braf-live-deployment" \
    --description "BRAF Live Testing & Sentinel Training Environment" \
    --parent-compartment-id "ocid1.tenancy.oc1..your-tenancy-ocid"

# Set compartment OCID as environment variable
export COMPARTMENT_OCID="ocid1.compartment.oc1..created-compartment-ocid"
```

### Step 2: Virtual Cloud Network (VCN) Configuration
```bash
# Create VCN with security-first design
oci network vcn create \
    --compartment-id $COMPARTMENT_OCID \
    --cidr-blocks '["10.0.0.0/16"]' \
    --display-name "braf-vcn" \
    --dns-label "braf"

# Create security lists for different components
oci network security-list create \
    --compartment-id $COMPARTMENT_OCID \
    --vcn-id $VCN_OCID \
    --display-name "sentinel-security-list" \
    --egress-security-rules '[{"destination": "0.0.0.0/0", "protocol": "6", "isStateless": false}]' \
    --ingress-security-rules '[{"source": "10.0.0.0/16", "protocol": "6", "isStateless": false}]'
```

### Step 3: Compute Infrastructure Deployment
```bash
# Deploy BRAF attack infrastructure (200 VMs)
for i in {1..200}; do
    oci compute instance launch \
        --compartment-id $COMPARTMENT_OCID \
        --shape "VM.Standard.A1.Flex" \
        --shape-config '{"ocpus": 2, "memory_in_gbs": 4}' \
        --image-id "ocid1.image.oc1.iad.aaaaaaaaxxxx" \
        --subnet-id $SUBNET_OCID \
        --display-name "braf-attack-node-$i" \
        --metadata '{"ssh_authorized_keys": "'"$(cat ~/.ssh/id_rsa.pub)"'"}' \
        --wait-for-state RUNNING
done

# Deploy Sentinel infrastructure (8 GPU-enabled VMs)
for i in {1..8}; do
    oci compute instance launch \
        --compartment-id $COMPARTMENT_OCID \
        --shape "VM.GPU.A10.1" \
        --image-id "ocid1.image.oc1.iad.aaaaaaaaxxxx" \
        --subnet-id $SUBNET_OCID \
        --display-name "sentinel-node-$i" \
        --wait-for-state RUNNING
done
```

### Step 4: Storage & Database Setup
```bash
# Create Autonomous Database for Sentinel
oci database autonomous-database create \
    --compartment-id $COMPARTMENT_OCID \
    --db-name "sentineldb" \
    --display-name "Sentinel Training Database" \
    --admin-password "ComplexPassword123!" \
    --data-storage-size-in-tbs 10 \
    --cpu-core-count 16 \
    --db-workload OLTP \
    --is-auto-scaling-enabled true

# Create Object Storage bucket for logs and backups
oci os bucket create \
    --compartment-id $COMPARTMENT_OCID \
    --name "braf-logs-and-backups" \
    --storage-tier "Standard"
```

### Step 5: Load Balancer Configuration
```bash
# Create load balancer for Sentinel API
oci lb load-balancer create \
    --compartment-id $COMPARTMENT_OCID \
    --display-name "sentinel-lb" \
    --shape-name "100Mbps" \
    --subnets "[$PUBLIC_SUBNET_OCID]" \
    --backend-sets '{"sentinel-backend": {"policy": "ROUND_ROBIN", "health_checker": {"protocol": "HTTP", "url_path": "/health"}}}'
```

---

## 3. Domain Configuration (Kuitter.Space)

### Step 1: DNS Zone Setup
```bash
# Create DNS zone for kuitter.space
oci dns zone create \
    --compartment-id $COMPARTMENT_OCID \
    --name "kuitter.space" \
    --zone-type "PRIMARY"

# Configure DNS records
oci dns record create \
    --zone-name-or-id "kuitter.space" \
    --domain "api.kuitter.space" \
    --type "A" \
    --rdata "$LOAD_BALANCER_IP"

oci dns record create \
    --zone-name-or-id "kuitter.space" \
    --domain "sentinel.kuitter.space" \
    --type "A" \
    --rdata "$SENTINEL_LB_IP"

oci dns record create \
    --zone-name-or-id "kuitter.space" \
    --domain "test.kuitter.space" \
    --type "A" \
    --rdata "$TEST_ENVIRONMENT_IP"
```

### Step 2: SSL Certificate Provisioning
```bash
# Request wildcard SSL certificate
certbot certonly \
    --manual \
    --preferred-challenges dns \
    --email admin@kuitter.space \
    --server https://acme-v02.api.letsencrypt.org/directory \
    --agree-tos \
    -d "*.kuitter.space"

# Upload certificate to OCI Certificate Service
oci certificates-management certificate create \
    --compartment-id $COMPARTMENT_OCID \
    --name "kuitter-space-wildcard" \
    --certificate-pem "$(cat /etc/letsencrypt/live/kuitter.space/fullchain.pem)" \
    --private-key-pem "$(cat /etc/letsencrypt/live/kuitter.space/privkey.pem)"
```

### Step 3: CDN Configuration
```bash
# Configure Oracle Cloud CDN
oci waas custom-protection-rule create \
    --compartment-id $COMPARTMENT_OCID \
    --display-name "kuitter-cdn-protection" \
    --template "PCI_DSS_V3_2_1" \
    --freeform-tags '{"Environment": "Production", "Project": "BRAF"}'
```

---

## 4. BRAF Live Deployment

### Step 1: Container Registry Setup
```bash
# Create Oracle Container Registry repository
oci artifacts container repository create \
    --compartment-id $COMPARTMENT_OCID \
    --display-name "braf-registry" \
    --is-public true

# Login to registry
docker login -u $OCI_USERNAME -p $OCI_AUTH_TOKEN $REGION.ocir.io

# Build and push BRAF components
docker build -t braf-balance-holder ./balance_holder/
docker tag braf-balance-holder $REGION.ocir.io/$NAMESPACE/braf/braf-balance-holder:latest
docker push $REGION.ocir.io/$NAMESPACE/braf/braf-balance-holder:latest

# Repeat for all BRAF components
```

### Step 2: Kubernetes Cluster Deployment
```bash
# Create OKE (Oracle Kubernetes Engine) cluster
oci ce cluster create \
    --compartment-id $COMPARTMENT_OCID \
    --name "braf-cluster" \
    --vcn-id $VCN_OCID \
    --kubernetes-version "v1.28.2" \
    --node-shape "VM.Standard.A1.Flex" \
    --node-count 50

# Configure kubectl
oci ce cluster create-kubeconfig \
    --cluster-id $CLUSTER_OCID \
    --file $HOME/.kube/config \
    --region $REGION
```

### Step 3: Application Deployment
```yaml
# Kubernetes deployment manifest for BRAF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: braf-balance-holder
  namespace: braf-production
spec:
  replicas: 10
  selector:
    matchLabels:
      app: braf-balance-holder
  template:
    metadata:
      labels:
        app: braf-balance-holder
    spec:
      containers:
      - name: braf-balance-holder
        image: $REGION.ocir.io/$NAMESPACE/braf/braf-balance-holder:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: ORACLE_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: oracle-db-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: braf-balance-holder-service
  namespace: braf-production
spec:
  selector:
    app: braf-balance-holder
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Step 4: Database Migration
```bash
# Migrate from simulation to production database
pg_dump simulation_database > simulation_backup.sql

# Restore to Oracle Autonomous Database
oci db autonomous-database restore \
    --autonomous-database-id $ADB_OCID \
    --timestamp $BACKUP_TIMESTAMP

# Run database migrations for production schema
kubectl exec -it $(kubectl get pods -l app=braf-migration -o jsonpath='{.items[0].metadata.name}') \
    -- /app/migrate-to-production.sh
```

---

## 5. Sentinel Production Deployment

### Step 1: GPU Cluster Configuration
```yaml
# Sentinel GPU cluster configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentinel-inference
  namespace: sentinel-production
spec:
  replicas: 8
  selector:
    matchLabels:
      app: sentinel-inference
  template:
    metadata:
      labels:
        app: sentinel-inference
    spec:
      nodeSelector:
        accelerator: nvidia-a10
      containers:
      - name: sentinel
        image: $REGION.ocir.io/$NAMESPACE/sentinel/sentinel-inference:latest
        ports:
        - containerPort: 8080
        - containerPort: 9090  # Metrics
        env:
        - name: MODEL_PATH
          value: "/models/sentinel-v1"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: oracle-db-secret
              key: url
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8000m"
          limits:
            nvidia.com/gpu: 1
            memory: "64Gi"
            cpu: "16000m"
```

### Step 2: Model Training Pipeline
```yaml
# Continuous training pipeline
apiVersion: batch/v1
kind: CronJob
metadata:
  name: sentinel-training-pipeline
  namespace: sentinel-production
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: training
            image: $REGION.ocir.io/$NAMESPACE/sentinel/sentinel-training:latest
            command: ["/app/train.py"]
            args: ["--dataset", "braf_attack_logs", "--model-version", "latest"]
            env:
            - name: ORACLE_DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: oracle-db-secret
                  key: url
          restartPolicy: Never
```

### Step 3: Monitoring & Observability
```yaml
# Prometheus monitoring stack
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sentinel-monitor
  namespace: sentinel-production
spec:
  selector:
    matchLabels:
      app: sentinel-inference
  endpoints:
  - port: metrics
    interval: 30s
---
# Grafana dashboard deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-config
          mountPath: /etc/grafana
      volumes:
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-pvc
      - name: grafana-config
        configMap:
          name: grafana-config
```

---

## 6. Live Test Environment Setup

### Step 1: Isolated Test Network
```bash
# Create isolated VCN for live testing
oci network vcn create \
    --compartment-id $COMPARTMENT_OCID \
    --cidr-blocks '["192.168.0.0/16"]' \
    --display-name "braf-test-vcn" \
    --dns-label "braftest"

# Configure security groups for test isolation
oci network security-list create \
    --compartment-id $COMPARTMENT_OCID \
    --vcn-id $TEST_VCN_OCID \
    --display-name "test-isolation-list" \
    --ingress-security-rules '[{"source": "10.0.0.0/16", "protocol": "6", "isStateless": false}]'
```

### Step 2: Test Application Deployment
```yaml
# Deploy test e-commerce application
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-ecommerce-app
  namespace: braf-testing
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: ecommerce
        image: $REGION.ocir.io/$NAMESPACE/test/test-ecommerce:latest
        ports:
        - containerPort: 3000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: test-db-secret
              key: url
        - name: SENTINEL_ENDPOINT
          value: "http://sentinel.braf-testing.svc.cluster.local:8080"
---
# Service for test application
apiVersion: v1
kind: Service
metadata:
  name: test-ecommerce-service
  namespace: braf-testing
spec:
  selector:
    app: test-ecommerce-app
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

### Step 3: Test Orchestration Setup
```python
# Live test orchestration script
import asyncio
import aiohttp
from datetime import datetime, timedelta

class LiveTestOrchestrator:
    def __init__(self):
        self.braf_endpoints = [
            "http://braf-node-1.braf-production.svc.cluster.local:8080",
            "http://braf-node-2.braf-production.svc.cluster.local:8080",
            # ... 198 more endpoints
        ]
        self.sentinel_endpoint = "http://sentinel.braf-testing.svc.cluster.local:8080"
        self.test_app_endpoint = "http://test-ecommerce.braf-testing.svc.cluster.local"

        self.attack_metrics = {
            'launched': 0,
            'detected': 0,
            'blocked': 0,
            'successful': 0
        }

    async def run_live_test(self, duration_hours=24):
        """Run live test for specified duration"""
        print(f"Starting {duration_hours}-hour live BRAF vs Sentinel test...")

        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        # Launch attack waves
        while datetime.now() < end_time:
            await self.launch_attack_wave()
            await asyncio.sleep(1)  # 1 attack per second

        # Generate final report
        await self.generate_test_report()

    async def launch_attack_wave(self):
        """Launch a coordinated attack wave"""
        attack_config = self.generate_attack_config()

        # Execute attack against test application
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.test_app_endpoint}/api/attack",
                    json=attack_config,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()

                    # Record attack result
                    self.attack_metrics['launched'] += 1
                    if result.get('detected'):
                        self.attack_metrics['detected'] += 1
                    if result.get('blocked'):
                        self.attack_metrics['blocked'] += 1
                    if result.get('successful'):
                        self.attack_metrics['successful'] += 1

            except Exception as e:
                logger.error(f"Attack execution failed: {e}")

    def generate_attack_config(self):
        """Generate realistic attack configuration"""
        attack_types = ['credential_stuffing', 'account_takeover', 'payment_fraud',
                       'money_laundering', 'social_engineering', 'api_abuse']

        return {
            'attack_type': random.choice(attack_types),
            'user_agent': random.choice(self.user_agents),
            'ip_address': random.choice(self.attack_ips),
            'fingerprint': self.generate_browser_fingerprint(),
            'amount': random.uniform(10, 10000),
            'session_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat()
        }
```

---

## 7. Monitoring & Alerting Setup

### Step 1: Comprehensive Monitoring Stack
```yaml
# Prometheus + Grafana monitoring
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    rule_files:
      - /etc/prometheus/prometheus.rules

    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager:9093

    scrape_configs:
    - job_name: 'braf-nodes'
      static_configs:
      - targets: ['braf-node-1:9090', 'braf-node-2:9090']
      labels:
        environment: 'production'

    - job_name: 'sentinel-nodes'
      static_configs:
      - targets: ['sentinel-node-1:9090', 'sentinel-node-2:9090']
      labels:
        component: 'ai_model'
---
# Alerting rules
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  prometheus.rules: |
    groups:
    - name: braf_alerts
      rules:
      - alert: BRAFNodeDown
        expr: up{job="braf-nodes"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "BRAF node is down"
          description: "BRAF node {{ $labels.instance }} has been down for 5 minutes"

      - alert: SentinelAccuracyDrop
        expr: sentinel_detection_rate < 0.95
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Sentinel detection rate dropping"
          description: "Sentinel detection rate is {{ $value }}% (below 95% threshold)"
```

### Step 2: Log Aggregation
```yaml
# ELK stack for log aggregation
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elasticsearch
  namespace: logging
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
        env:
        - name: discovery.type
          value: single-node
        - name: xpack.security.enabled
          value: "false"
        ports:
        - containerPort: 9200
        - containerPort: 9300
        volumeMounts:
        - name: elasticsearch-data
          mountPath: /usr/share/elasticsearch/data
      volumes:
      - name: elasticsearch-data
        persistentVolumeClaim:
          claimName: elasticsearch-pvc
```

---

## 8. Backup & Disaster Recovery

### Step 1: Automated Backups
```bash
# Daily backup script
#!/bin/bash
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$BACKUP_DATE"

# Create backup directory
oci os object bulk-upload \
    --bucket-name "braf-backups" \
    --src-dir "$BACKUP_DIR"

# Database backup
oci db autonomous-database backup \
    --autonomous-database-id $ADB_OCID \
    --display-name "daily_backup_$BACKUP_DATE"

# Container image backup
for image in braf-balance-holder braf-fraud-engine sentinel-inference; do
    docker pull $REGION.ocir.io/$NAMESPACE/braf/$image:latest
    docker save $image > $BACKUP_DIR/${image}_$BACKUP_DATE.tar
done

# Configuration backup
kubectl get all -n braf-production -o yaml > $BACKUP_DIR/k8s_config_$BACKUP_DATE.yaml
```

### Step 2: Disaster Recovery Plan
```yaml
# Disaster recovery configuration
disaster_recovery:
  rto: 4  # Recovery Time Objective: 4 hours
  rpo: 1  # Recovery Point Objective: 1 hour

  recovery_procedures:
    - infrastructure_recovery:
        - oci cli commands for VCN recreation
        - terraform scripts for infrastructure provisioning
        - ansible playbooks for configuration

    - application_recovery:
        - kubectl deployments for application restoration
        - database restore procedures
        - configuration management

    - data_recovery:
        - object storage backup restoration
        - database point-in-time recovery
        - log replay procedures

  failover_regions:
    - primary: us-ashburn-1
    - secondary: eu-frankfurt-1
    - tertiary: ap-sydney-1
```

---

## 9. Security & Compliance

### Step 1: Access Control
```bash
# Oracle Identity and Access Management
oci iam group create \
    --compartment-id $COMPARTMENT_OCID \
    --name "braf-administrators" \
    --description "BRAF system administrators"

oci iam policy create \
    --compartment-id $TENANCY_OCID \
    --name "braf-admin-policy" \
    --description "Policy for BRAF administrators" \
    --statements '["Allow group BRAF-Administrators to manage all-resources in compartment BRAF"]'
```

### Step 2: Encryption & Security
```bash
# Enable encryption at rest
oci kms key create \
    --compartment-id $COMPARTMENT_OCID \
    --display-name "braf-encryption-key" \
    --key-shape '{"algorithm": "AES", "length": 256}'

# Configure network security
oci network security-list update \
    --security-list-id $SECURITY_LIST_OCID \
    --ingress-security-rules '[{"source": "0.0.0.0/0", "protocol": "6", "tcpOptions": {"destinationPortRange": {"max": 22, "min": 22}}, "isStateless": false}]' \
    --egress-security-rules '[{"destination": "0.0.0.0/0", "protocol": "6", "isStateless": false}]'
```

---

## 10. Cost Optimization & Scaling

### Step 1: Auto-Scaling Configuration
```yaml
# Horizontal Pod Autoscaler for BRAF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: braf-hpa
  namespace: braf-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: braf-attack-coordinator
  minReplicas: 10
  maxReplicas: 200
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
---
# Cluster Autoscaler for OCI
apiVersion: cluster.x-k8s.io/v1beta1
kind: MachineDeployment
metadata:
  name: braf-worker-nodes
spec:
  replicas: 50
  selector:
    matchLabels:
      cluster.x-k8s.io/cluster-name: braf-cluster
  template:
    spec:
      bootstrap:
        dataSecretName: ""
      clusterName: braf-cluster
      infrastructureRef:
        apiVersion: infrastructure.cluster.x-k8s.io/v1beta1
        kind: OCIMachineTemplate
        name: braf-machine-template
      version: v1.28.2
```

### Step 2: Cost Monitoring
```bash
# Cost monitoring and alerting
oci usage-api usage-summary \
    --compartment-id $COMPARTMENT_OCID \
    --granularity DAILY \
    --query-text "query usage, cost where compartmentId = '$COMPARTMENT_OCID'" \
    --output table

# Automated cost optimization
#!/bin/bash
# Scale down during low-usage periods
CURRENT_HOUR=$(date +%H)
if [ $CURRENT_HOUR -ge 02 ] && [ $CURRENT_HOUR -le 06 ]; then
    kubectl scale deployment braf-attack-nodes --replicas=20
else
    kubectl scale deployment braf-attack-nodes --replicas=100
fi
```

---

## 11. Migration Checklist

### Pre-Migration Validation
- [ ] All simulation tests passing
- [ ] Infrastructure provisioning complete
- [ ] Domain and SSL configured
- [ ] Security policies implemented
- [ ] Backup and recovery tested
- [ ] Monitoring stack operational

### Migration Execution
- [ ] Deploy BRAF components to production
- [ ] Migrate database from simulation to production
- [ ] Configure Sentinel with live data
- [ ] Update DNS and CDN configurations
- [ ] Enable production monitoring
- [ ] Conduct smoke tests

### Post-Migration Validation
- [ ] All endpoints responding
- [ ] Database connections working
- [ ] Security policies enforced
- [ ] Monitoring data flowing
- [ ] Backup systems operational
- [ ] Performance benchmarks met

---

## 12. Launch Command & Validation

### Production Launch
```bash
# Final production deployment
kubectl apply -f k8s/production/

# Verify deployment
kubectl get pods -n braf-production
kubectl get services -n braf-production

# Test endpoints
curl -k https://api.kuitter.space/health
curl -k https://sentinel.kuitter.space/status

# Start live testing
python live_test_orchestrator.py --duration 168  # 7 days
```

### Success Validation
```bash
# Automated validation script
#!/bin/bash
echo "Validating BRAF Live Deployment..."

# Check all services
services=("api.kuitter.space" "sentinel.kuitter.space" "test.kuitter.space")
for service in "${services[@]}"; do
    if curl -k -f https://$service/health > /dev/null 2>&1; then
        echo "✓ $service is healthy"
    else
        echo "✗ $service is unhealthy"
        exit 1
    fi
done

# Check database connectivity
kubectl exec -it $(kubectl get pods -l app=braf-balance-holder -o jsonpath='{.items[0].metadata.name}') \
    -- python -c "import database; database.test_connection()"

# Start statistical significance testing
echo "Starting statistical significance testing..."
python extended_braf_vs_sentinel_test.py --live --duration-unlimited --confidence 0.95
```

---

## Conclusion

This comprehensive setup guide provides everything needed to migrate BRAF from simulation to full production deployment on Oracle Cloud Infrastructure using the Kuitter.Space domain. The deployment includes:

- **Complete infrastructure provisioning** with Oracle Cloud
- **Domain and SSL configuration** for Kuitter.Space
- **Production-grade deployment** of BRAF and Sentinel
- **Live testing framework** for statistical significance
- **Monitoring, security, and backup systems**
- **Cost optimization and scaling capabilities**

**The migration transforms BRAF from a simulation environment to a live, production-ready cyber fraud simulation platform capable of challenging real-world defenses.** 🚀