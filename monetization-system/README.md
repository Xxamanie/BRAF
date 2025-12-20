# BRAF Monetization System

Enterprise Browser Automation Revenue Framework with Monetization

## 🚀 Quick Start

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
python run_server.py
```

### Production Deployment
```bash
# Deploy to production server
./deploy.sh

# Manage service
./manage.sh start|stop|restart|status|logs
```

## 📊 Features

### ✅ Core Features
- **Browser Automation Framework**: Complete BRAF integration
- **Enterprise Account Management**: Secure registration and authentication
- **Automation Templates**: Survey, video, and content automation
- **Real-time Dashboard**: Earnings tracking and analytics
- **Multi-provider Withdrawals**: OPay, PalmPay, and cryptocurrency
- **Compliance Monitoring**: Automated compliance checking
- **Security Features**: 2FA, encryption, audit logging

### ✅ Monetization Features
- **Free Beta Mode**: Currently free for all users
- **Subscription Management**: Ready for future monetization
- **Payment Processing**: Mobile money and crypto withdrawals
- **Analytics Dashboard**: Comprehensive earnings tracking
- **Enterprise Features**: Multi-user support and API access

### ✅ Technical Features
- **FastAPI Backend**: High-performance async API
- **SQLite Database**: Lightweight and reliable
- **Docker Support**: Containerized deployment
- **Nginx Integration**: Production-ready web server
- **Systemd Service**: Linux service management
- **Monitoring Ready**: Prometheus and Grafana support

## 🌐 Web Interface

- **Registration**: http://localhost:8003/register
- **Login**: http://localhost:8003/login
- **Dashboard**: http://localhost:8003/dashboard
- **API Docs**: http://localhost:8003/docs

## 📡 API Endpoints

### Authentication
- `POST /api/v1/enterprise/register` - Register new account
- `POST /api/v1/enterprise/login` - Login to account

### Automation
- `GET /api/v1/automation/list/{enterprise_id}` - List automations
- `POST /api/v1/automation/create/{enterprise_id}` - Create automation

### Dashboard
- `GET /api/v1/dashboard/earnings/{enterprise_id}` - Get earnings
- `GET /api/v1/dashboard/withdrawals/{enterprise_id}` - Get withdrawals
- `GET /api/v1/dashboard/overview/{enterprise_id}` - Dashboard overview

### Withdrawals
- `POST /api/v1/withdrawal/create/{enterprise_id}` - Request withdrawal

## 🔧 Configuration

### Environment Variables
```bash
ENVIRONMENT=development|production
DATABASE_URL=sqlite:///./braf.db
SECRET_KEY=your-secret-key
HOST=127.0.0.1
PORT=8003
```

### Database Setup
```bash
# Initialize database
python -c "
from database import engine
from database.models import Base
Base.metadata.create_all(bind=engine)
"

# Seed sample data
python seed_sample_data.py
```

## 🛠️ Management Commands

```bash
# Service management
./manage.sh start|stop|restart|status|logs

# Data management
./manage.sh seed-data        # Add sample data
./manage.sh create-account   # Create new account
./manage.sh backup          # Backup database

# System management
./manage.sh update          # Update system
```

## 🔒 Security

- **Password Hashing**: Secure bcrypt hashing
- **JWT Authentication**: Token-based auth
- **2FA Support**: TOTP two-factor authentication
- **Rate Limiting**: API rate limiting
- **Audit Logging**: Comprehensive activity logs
- **Compliance Monitoring**: Automated compliance checks

## 💰 Monetization

Currently in **Free Beta** mode - all features are free to use.

Future monetization features (ready to enable):
- Subscription tiers (Basic, Pro, Enterprise)
- Usage-based billing
- Premium automation templates
- Priority support
- Advanced analytics

## 📈 Monitoring

- **Health Checks**: `/health` endpoint
- **Metrics**: `/metrics` endpoint (Prometheus)
- **Logging**: Structured JSON logging
- **Alerts**: Security and compliance alerts

## 🐳 Docker Deployment

```bash
# Build and run with Docker
docker-compose up -d

# View logs
docker-compose logs -f
```

## 🤝 Support

- **Documentation**: Full API documentation at `/docs`
- **Health Status**: System health at `/health`
- **Logs**: Service logs via `./manage.sh logs`

## 📄 License

Enterprise License - Contact for commercial use.

---

**BRAF Monetization System v1.0.0**  
Built with FastAPI, SQLite, and modern web technologies.
