# **BRAF AWS DEPLOYMENT COMMAND PROMPTS**

## **OVERVIEW**

This guide provides command-line prompts for deploying BRAF (Browser Automation Framework) on AWS infrastructure using Docker, ECS, and supporting services.

---

## **🔧 PREREQUISITES SETUP**

### **1. Configure AWS CLI**
```bash
# Configure AWS credentials
aws configure

# Set default region
aws configure set default.region us-east-1

# Verify configuration
aws sts get-caller-identity
```

### **2. Install Required Tools**
```bash
# Install AWS CLI v2 (if not installed)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker

# Install AWS Copilot CLI
sudo curl -Lo /usr/local/bin/copilot https://github.com/aws/copilot-cli/releases/latest/download/copilot-linux
sudo chmod +x /usr/local/bin/copilot
```

### **3. Clone BRAF Repository**
```bash
# Clone the BRAF repository
git clone https://github.com/Xxamanie/BRAF.git
cd BRAF

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## **🐳 DOCKER CONTAINER BUILD**

### **1. Build BRAF Docker Images**
```bash
# Build main BRAF application image
docker build -t braf-app:latest .

# Build C2 dashboard image
docker build -f docker/Dockerfile.c2 -t braf-c2:latest .

# Build worker node image
docker build -f docker/Dockerfile.worker -t braf-worker:latest .

# Build monitoring stack image
docker build -f docker/Dockerfile.monitoring -t braf-monitoring:latest .

# List built images
docker images | grep braf
```

### **2. Tag Images for ECR**
```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

# Tag images for ECR
docker tag braf-app:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/braf-app:latest
docker tag braf-c2:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/braf-c2:latest
docker tag braf-worker:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/braf-worker:latest
docker tag braf-monitoring:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/braf-monitoring:latest
```

### **3. Create ECR Repositories**
```bash
# Create ECR repositories
aws ecr create-repository --repository-name braf-app --region ${AWS_REGION}
aws ecr create-repository --repository-name braf-c2 --region ${AWS_REGION}
aws ecr create-repository --repository-name braf-worker --region ${AWS_REGION}
aws ecr create-repository --repository-name braf-monitoring --region ${AWS_REGION}
```

### **4. Push Images to ECR**
```bash
# Authenticate Docker with ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Push images
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/braf-app:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/braf-c2:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/braf-worker:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/braf-monitoring:latest
```

---

## **☁️ AWS INFRASTRUCTURE SETUP**

### **1. Create VPC and Networking**
```bash
# Create VPC
VPC_ID=$(aws ec2 create-vpc --cidr-block 10.0.0.0/16 --query 'Vpc.VpcId' --output text)

# Create subnets
SUBNET_1=$(aws ec2 create-subnet --vpc-id ${VPC_ID} --cidr-block 10.0.1.0/24 --availability-zone us-east-1a --query 'Subnet.SubnetId' --output text)
SUBNET_2=$(aws ec2 create-subnet --vpc-id ${VPC_ID} --cidr-block 10.0.2.0/24 --availability-zone us-east-1b --query 'Subnet.SubnetId' --output text)

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway --query 'InternetGateway.InternetGatewayId' --output text)
aws ec2 attach-internet-gateway --internet-gateway-id ${IGW_ID} --vpc-id ${VPC_ID}

# Create route table
RT_ID=$(aws ec2 create-route-table --vpc-id ${VPC_ID} --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-route --route-table-id ${RT_ID} --destination-cidr-block 0.0.0.0/0 --gateway-id ${IGW_ID}

# Associate subnets with route table
aws ec2 associate-route-table --subnet-id ${SUBNET_1} --route-table-id ${RT_ID}
aws ec2 associate-route-table --subnet-id ${SUBNET_2} --route-table-id ${RT_ID}
```

### **2. Create Security Groups**
```bash
# Create security group for BRAF application
SG_APP=$(aws ec2 create-security-group \
    --group-name braf-app-sg \
    --description "Security group for BRAF application" \
    --vpc-id ${VPC_ID} \
    --query 'GroupId' \
    --output text)

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-id ${SG_APP} \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id ${SG_APP} \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id ${SG_APP} \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Create security group for database
SG_DB=$(aws ec2 create-security-group \
    --group-name braf-db-sg \
    --description "Security group for BRAF database" \
    --vpc-id ${VPC_ID} \
    --query 'GroupId' \
    --output text)

aws ec2 authorize-security-group-ingress \
    --group-id ${SG_DB} \
    --protocol tcp \
    --port 5432 \
    --source-group ${SG_APP}
```

### **3. Create RDS PostgreSQL Database**
```bash
# Create RDS subnet group
aws rds create-db-subnet-group \
    --db-subnet-group-name braf-db-subnet-group \
    --db-subnet-group-description "Subnet group for BRAF database" \
    --subnet-ids ${SUBNET_1} ${SUBNET_2}

# Create PostgreSQL database
aws rds create-db-instance \
    --db-instance-identifier braf-postgres \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 15.0 \
    --master-username braf \
    --master-user-password $(openssl rand -base64 12) \
    --allocated-storage 20 \
    --db-subnet-group-name braf-db-subnet-group \
    --vpc-security-group-ids ${SG_DB} \
    --no-publicly-accessible \
    --backup-retention-period 7

# Get database endpoint
DB_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier braf-postgres \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)
```

### **4. Create ElastiCache Redis**
```bash
# Create Redis subnet group
aws elasticache create-cache-subnet-group \
    --cache-subnet-group-name braf-redis-subnet-group \
    --cache-subnet-group-description "Subnet group for BRAF Redis" \
    --subnet-ids ${SUBNET_1} ${SUBNET_2}

# Create Redis cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id braf-redis \
    --cache-node-type cache.t3.micro \
    --engine redis \
    --num-cache-nodes 1 \
    --cache-subnet-group-name braf-redis-subnet-group \
    --security-group-ids ${SG_APP} \
    --snapshot-retention-limit 7

# Get Redis endpoint
REDIS_ENDPOINT=$(aws elasticache describe-cache-clusters \
    --cache-cluster-id braf-redis \
    --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
    --output text)
```

---

## **🚀 ECS DEPLOYMENT**

### **1. Create ECS Cluster**
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name braf-cluster

# Create capacity provider (for EC2 instances)
aws ecs create-capacity-provider \
    --name braf-capacity-provider \
    --auto-scaling-group-provider \
    '{
        "autoScalingGroupArn": "arn:aws:autoscaling:us-east-1:'${AWS_ACCOUNT_ID}':autoScalingGroup:...:autoScalingGroupName/braf-asg",
        "managedScaling": {
            "status": "ENABLED",
            "targetCapacity": 80,
            "minimumScalingStepSize": 1,
            "maximumScalingStepSize": 10
        },
        "managedTerminationProtection": "ENABLED"
    }'
```

### **2. Create ECS Task Definitions**
```bash
# Create task definition for BRAF application
aws ecs register-task-definition \
    --family braf-app \
    --task-role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole \
    --execution-role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole \
    --network-mode awsvpc \
    --requires-compatibilities FARGATE \
    --cpu 1024 \
    --memory 2048 \
    --container-definitions '[
        {
            "name": "braf-app",
            "image": "'${AWS_ACCOUNT_ID}'.dkr.ecr.'${AWS_REGION}'.amazonaws.com/braf-app:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {"name": "DATABASE_URL", "value": "postgresql://braf:'${DB_PASSWORD}'@'${DB_ENDPOINT}'/braf_dev"},
                {"name": "REDIS_URL", "value": "redis://'${REDIS_ENDPOINT}':6379"},
                {"name": "ENVIRONMENT", "value": "production"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/braf-app",
                    "awslogs-region": "'${AWS_REGION}'",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]'

# Create task definition for C2 dashboard
aws ecs register-task-definition \
    --family braf-c2 \
    --task-role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole \
    --execution-role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole \
    --network-mode awsvpc \
    --requires-compatibilities FARGATE \
    --cpu 512 \
    --memory 1024 \
    --container-definitions '[
        {
            "name": "braf-c2",
            "image": "'${AWS_ACCOUNT_ID}'.dkr.ecr.'${AWS_REGION}'.amazonaws.com/braf-c2:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": 80,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {"name": "DATABASE_URL", "value": "postgresql://braf:'${DB_PASSWORD}'@'${DB_ENDPOINT}'/braf_dev"},
                {"name": "REDIS_URL", "value": "redis://'${REDIS_ENDPOINT}':6379"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/braf-c2",
                    "awslogs-region": "'${AWS_REGION}'",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]'
```

### **3. Create ECS Services**
```bash
# Create BRAF application service
aws ecs create-service \
    --cluster braf-cluster \
    --service-name braf-app-service \
    --task-definition braf-app \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration '{
        "awsvpcConfiguration": {
            "subnets": ["'${SUBNET_1}'", "'${SUBNET_2}'"],
            "securityGroups": ["'${SG_APP}'"],
            "assignPublicIp": "ENABLED"
        }
    }' \
    --load-balancers '[
        {
            "targetGroupArn": "arn:aws:elasticloadbalancing:us-east-1:'${AWS_ACCOUNT_ID}':targetgroup/braf-app/...",
            "containerName": "braf-app",
            "containerPort": 8000
        }
    ]'

# Create C2 dashboard service
aws ecs create-service \
    --cluster braf-cluster \
    --service-name braf-c2-service \
    --task-definition braf-c2 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration '{
        "awsvpcConfiguration": {
            "subnets": ["'${SUBNET_1}'", "'${SUBNET_2}'"],
            "securityGroups": ["'${SG_APP}'"],
            "assignPublicIp": "ENABLED"
        }
    }'
```

---

## **⚖️ LOAD BALANCER SETUP**

### **1. Create Application Load Balancer**
```bash
# Create ALB
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name braf-alb \
    --subnets ${SUBNET_1} ${SUBNET_2} \
    --security-groups ${SG_APP} \
    --scheme internet-facing \
    --type application \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

# Create target groups
TARGET_GROUP_APP=$(aws elbv2 create-target-group \
    --name braf-app-targets \
    --protocol HTTP \
    --port 8000 \
    --vpc-id ${VPC_ID} \
    --target-type ip \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

TARGET_GROUP_C2=$(aws elbv2 create-target-group \
    --name braf-c2-targets \
    --protocol HTTP \
    --port 80 \
    --vpc-id ${VPC_ID} \
    --target-type ip \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

# Create listeners
aws elbv2 create-listener \
    --load-balancer-arn ${ALB_ARN} \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=${TARGET_GROUP_APP}

aws elbv2 create-listener \
    --load-balancer-arn ${ALB_ARN} \
    --protocol HTTP \
    --port 8080 \
    --default-actions Type=forward,TargetGroupArn=${TARGET_GROUP_C2}
```

### **2. Get Load Balancer DNS**
```bash
# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns ${ALB_ARN} \
    --query 'LoadBalancers[0].DNSName' \
    --output text)

echo "BRAF Application URL: http://${ALB_DNS}"
echo "BRAF C2 Dashboard URL: http://${ALB_DNS}:8080"
```

---

## **📊 MONITORING & LOGGING**

### **1. Create CloudWatch Log Groups**
```bash
# Create log groups for ECS services
aws logs create-log-group --log-group-name /ecs/braf-app
aws logs create-log-group --log-group-name /ecs/braf-c2
aws logs create-log-group --log-group-name /ecs/braf-worker
aws logs create-log-group --log-group-name /ecs/braf-monitoring
```

### **2. Set Up CloudWatch Alarms**
```bash
# Create CPU utilization alarm
aws cloudwatch put-metric-alarm \
    --alarm-name braf-high-cpu \
    --alarm-description "BRAF high CPU utilization" \
    --metric-name CPUUtilization \
    --namespace AWS/ECS \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=ClusterName,Value=braf-cluster Name=ServiceName,Value=braf-app-service \
    --evaluation-periods 2

# Create memory utilization alarm
aws cloudwatch put-metric-alarm \
    --alarm-name braf-high-memory \
    --alarm-description "BRAF high memory utilization" \
    --metric-name MemoryUtilization \
    --namespace AWS/ECS \
    --statistic Average \
    --period 300 \
    --threshold 85 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=ClusterName,Value=braf-cluster Name=ServiceName,Value=braf-app-service \
    --evaluation-periods 2
```

### **3. Create CloudWatch Dashboard**
```bash
# Create dashboard for BRAF monitoring
aws cloudwatch put-dashboard \
    --dashboard-name BRAF-Monitoring \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/ECS", "CPUUtilization", "ClusterName", "braf-cluster", "ServiceName", "braf-app-service"]
                    ],
                    "view": "timeSeries",
                    "stacked": false,
                    "region": "'${AWS_REGION}'",
                    "title": "BRAF Application CPU Utilization"
                }
            }
        ]
    }'
```

---

## **🔄 CI/CD PIPELINE**

### **1. Create CodeBuild Project**
```bash
# Create build project
aws codebuild create-project \
    --name braf-build \
    --source '{
        "type": "GITHUB",
        "location": "https://github.com/Xxamanie/BRAF.git",
        "buildspec": "buildspec.yml"
    }' \
    --artifacts '{
        "type": "NO_ARTIFACTS"
    }' \
    --environment '{
        "type": "LINUX_CONTAINER",
        "image": "aws/codebuild/amazonlinux2-x86_64-standard:3.0",
        "computeType": "BUILD_GENERAL1_SMALL"
    }' \
    --service-role arn:aws:iam::${AWS_ACCOUNT_ID}:role/service-role/codebuild-braf-service-role
```

### **2. Create CodePipeline**
```bash
# Create pipeline
aws codepipeline create-pipeline \
    --pipeline '{
        "name": "braf-deployment-pipeline",
        "roleArn": "arn:aws:iam::'${AWS_ACCOUNT_ID}':role/service-role/AWSCodePipelineServiceRole",
        "artifactStore": {
            "type": "S3",
            "location": "braf-artifacts-'${AWS_ACCOUNT_ID}'"
        },
        "stages": [
            {
                "name": "Source",
                "actions": [
                    {
                        "name": "SourceAction",
                        "actionTypeId": {
                            "category": "Source",
                            "owner": "AWS",
                            "provider": "S3",
                            "version": "1"
                        },
                        "configuration": {
                            "S3Bucket": "braf-artifacts-'${AWS_ACCOUNT_ID}'",
                            "S3ObjectKey": "source.zip"
                        }
                    }
                ]
            },
            {
                "name": "Build",
                "actions": [
                    {
                        "name": "BuildAction",
                        "actionTypeId": {
                            "category": "Build",
                            "owner": "AWS",
                            "provider": "CodeBuild",
                            "version": "1"
                        },
                        "configuration": {
                            "ProjectName": "braf-build"
                        }
                    }
                ]
            },
            {
                "name": "Deploy",
                "actions": [
                    {
                        "name": "DeployAction",
                        "actionTypeId": {
                            "category": "Deploy",
                            "owner": "AWS",
                            "provider": "ECS",
                            "version": "1"
                        },
                        "configuration": {
                            "ClusterName": "braf-cluster",
                            "ServiceName": "braf-app-service"
                        }
                    }
                ]
            }
        ]
    }'
```

---

## **💰 COST OPTIMIZATION**

### **1. Set Up Auto Scaling**
```bash
# Create auto scaling policy for BRAF app
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/braf-cluster/braf-app-service \
    --min-capacity 1 \
    --max-capacity 10

# Create scaling policy based on CPU
aws application-autoscaling put-scaling-policy \
    --policy-name braf-cpu-scaling \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/braf-cluster/braf-app-service \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 300
    }'
```

### **2. Set Up Cost Allocation Tags**
```bash
# Tag resources for cost tracking
aws ecs tag-resource \
    --resource-arn arn:aws:ecs:us-east-1:${AWS_ACCOUNT_ID}:cluster/braf-cluster \
    --tags key=Project,value=BRAF key=Environment,value=Production

aws rds add-tags-to-resource \
    --resource-name arn:aws:rds:us-east-1:${AWS_ACCOUNT_ID}:db:braf-postgres \
    --tags Key=Project,Value=BRAF Key=Environment,Value=Production
```

---

## **🔐 SECURITY CONFIGURATION**

### **1. Set Up Secrets Manager**
```bash
# Store database password in Secrets Manager
aws secretsmanager create-secret \
    --name braf/db-password \
    --description "BRAF database password" \
    --secret-string '{"password":"'$(openssl rand -base64 16)'"}'

# Store API keys
aws secretsmanager create-secret \
    --name braf/api-keys \
    --description "BRAF API keys" \
    --secret-string '{
        "maxel_api_key": "your_maxel_key",
        "twitter_bearer_token": "your_twitter_token",
        "crypto_api_keys": {}
    }'
```

### **2. Configure WAF**
```bash
# Create WAF ACL for BRAF
WAF_ACL_ID=$(aws wafv2 create-web-acl \
    --name braf-waf-acl \
    --scope REGIONAL \
    --default-action Allow={} \
    --rules '[
        {
            "Name": "AWSManagedRulesCommonRuleSet",
            "Priority": 0,
            "Statement": {
                "ManagedRuleGroupStatement": {
                    "VendorName": "AWS",
                    "Name": "AWSManagedRulesCommonRuleSet"
                }
            },
            "OverrideAction": {"None": {}},
            "VisibilityConfig": {
                "SampledRequestsEnabled": true,
                "CloudWatchMetricsEnabled": true,
                "MetricName": "braf-waf-common-rules"
            }
        }
    ]' \
    --visibility-config '{
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "braf-waf-acl"
    }' \
    --query 'Summary.Id' \
    --output text)

# Associate WAF with ALB
aws wafv2 associate-web-acl \
    --web-acl-arn arn:aws:wafv2:us-east-1:${AWS_ACCOUNT_ID}:regional/webacl/braf-waf-acl/${WAF_ACL_ID} \
    --resource-arn ${ALB_ARN}
```

---

## **🚀 DEPLOYMENT VERIFICATION**

### **1. Test BRAF Application**
```bash
# Test application health
curl -f http://${ALB_DNS}/health

# Test C2 dashboard
curl -f http://${ALB_DNS}:8080/

# Check ECS services
aws ecs describe-services \
    --cluster braf-cluster \
    --services braf-app-service braf-c2-service
```

### **2. Monitor Deployment**
```bash
# Check CloudWatch logs
aws logs tail /ecs/braf-app --follow

# Check ECS events
aws ecs describe-services \
    --cluster braf-cluster \
    --services braf-app-service \
    --query 'services[0].events[0:5]'

# Monitor ALB targets
aws elbv2 describe-target-health \
    --target-group-arn ${TARGET_GROUP_APP}
```

---

## **🧹 CLEANUP COMMANDS**

### **Delete BRAF Deployment**
```bash
# Delete ECS services
aws ecs delete-service --cluster braf-cluster --service braf-app-service --force
aws ecs delete-service --cluster braf-cluster --service braf-c2-service --force

# Delete ECS cluster
aws ecs delete-cluster --cluster braf-cluster

# Delete load balancer
aws elbv2 delete-load-balancer --load-balancer-arn ${ALB_ARN}

# Delete RDS database
aws rds delete-db-instance --db-instance-identifier braf-postgres --skip-final-snapshot

# Delete Redis cluster
aws elasticache delete-cache-cluster --cache-cluster-id braf-redis

# Delete ECR repositories
aws ecr delete-repository --repository-name braf-app --force
aws ecr delete-repository --repository-name braf-c2 --force
aws ecr delete-repository --repository-name braf-worker --force
aws ecr delete-repository --repository-name braf-monitoring --force

# Delete VPC and networking
aws ec2 delete-vpc --vpc-id ${VPC_ID}
```

---

**This AWS deployment guide provides complete command-line prompts for deploying BRAF on AWS infrastructure. Execute these commands in sequence to set up a production-ready BRAF environment.**