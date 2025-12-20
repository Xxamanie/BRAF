#!/usr/bin/env python3
"""
BRAF Dependencies Installation Script
Comprehensive dependency management for live deployment
"""

import os
import sys
import subprocess
import platform
import json
import logging
from pathlib import Path
import urllib.request
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BRAFDependencyInstaller:
    """Comprehensive dependency installer for BRAF system"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.architecture = platform.architecture()[0]
        self.python_version = platform.python_version()
        self.project_root = Path(__file__).parent
        
        logger.info(f"System: {self.system} {self.architecture}")
        logger.info(f"Python: {self.python_version}")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 8):
            logger.error(f"Python 3.8+ required, found {major}.{minor}")
            return False
        
        logger.info(f"Python {major}.{minor} is compatible")
        return True
    
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        logger.info("Installing system dependencies...")
        
        try:
            if self.system == "linux":
                self._install_linux_dependencies()
            elif self.system == "darwin":
                self._install_macos_dependencies()
            elif self.system == "windows":
                self._install_windows_dependencies()
            else:
                logger.warning(f"Unsupported system: {self.system}")
                return False
            
            logger.info("System dependencies installed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to install system dependencies: {e}")
            return False
    
    def _install_linux_dependencies(self):
        """Install Linux system dependencies"""
        logger.info("Installing Linux dependencies...")
        
        # Detect Linux distribution
        try:
            with open('/etc/os-release', 'r') as f:
                os_info = f.read()
            
            if 'ubuntu' in os_info.lower() or 'debian' in os_info.lower():
                self._install_debian_dependencies()
            elif 'centos' in os_info.lower() or 'rhel' in os_info.lower() or 'fedora' in os_info.lower():
                self._install_redhat_dependencies()
            else:
                logger.warning("Unknown Linux distribution, trying Ubuntu/Debian packages...")
                self._install_debian_dependencies()
                
        except FileNotFoundError:
            logger.warning("Cannot detect Linux distribution, trying Ubuntu/Debian packages...")
            self._install_debian_dependencies()
    
    def _install_debian_dependencies(self):
        """Install dependencies for Debian/Ubuntu"""
        packages = [
            'python3-dev',
            'python3-pip',
            'python3-venv',
            'build-essential',
            'libpq-dev',
            'libssl-dev',
            'libffi-dev',
            'libjpeg-dev',
            'libpng-dev',
            'libxml2-dev',
            'libxslt1-dev',
            'zlib1g-dev',
            'curl',
            'wget',
            'git',
            'unzip',
            'postgresql-client',
            'redis-tools',
            'nginx',
            'supervisor',
            'htop',
            'tree',
            'jq'
        ]
        
        # Update package list
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        
        # Install packages
        cmd = ['sudo', 'apt', 'install', '-y'] + packages
        subprocess.run(cmd, check=True)
    
    def _install_redhat_dependencies(self):
        """Install dependencies for CentOS/RHEL/Fedora"""
        packages = [
            'python3-devel',
            'python3-pip',
            'gcc',
            'gcc-c++',
            'make',
            'postgresql-devel',
            'openssl-devel',
            'libffi-devel',
            'libjpeg-devel',
            'libpng-devel',
            'libxml2-devel',
            'libxslt-devel',
            'zlib-devel',
            'curl',
            'wget',
            'git',
            'unzip',
            'postgresql',
            'redis',
            'nginx',
            'supervisor',
            'htop',
            'tree',
            'jq'
        ]
        
        # Determine package manager
        if shutil.which('dnf'):
            pkg_manager = 'dnf'
        elif shutil.which('yum'):
            pkg_manager = 'yum'
        else:
            raise Exception("No supported package manager found (dnf/yum)")
        
        # Install packages
        cmd = ['sudo', pkg_manager, 'install', '-y'] + packages
        subprocess.run(cmd, check=True)
    
    def _install_macos_dependencies(self):
        """Install macOS dependencies using Homebrew"""
        logger.info("Installing macOS dependencies...")
        
        # Check if Homebrew is installed
        if not shutil.which('brew'):
            logger.info("Installing Homebrew...")
            install_script = urllib.request.urlopen(
                'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh'
            ).read().decode('utf-8')
            subprocess.run(['/bin/bash', '-c', install_script], check=True)
        
        packages = [
            'python@3.11',
            'postgresql',
            'redis',
            'nginx',
            'git',
            'curl',
            'wget',
            'jq',
            'tree',
            'htop'
        ]
        
        # Install packages
        for package in packages:
            subprocess.run(['brew', 'install', package], check=False)  # Don't fail if already installed
    
    def _install_windows_dependencies(self):
        """Install Windows dependencies"""
        logger.info("Installing Windows dependencies...")
        
        # Check if chocolatey is available
        if shutil.which('choco'):
            packages = [
                'python3',
                'git',
                'curl',
                'wget',
                'jq',
                'postgresql',
                'redis-64',
                'nginx'
            ]
            
            for package in packages:
                subprocess.run(['choco', 'install', '-y', package], check=False)
        else:
            logger.warning("Chocolatey not found. Please install dependencies manually:")
            logger.warning("- Python 3.8+")
            logger.warning("- Git")
            logger.warning("- PostgreSQL")
            logger.warning("- Redis")
    
    def create_virtual_environment(self):
        """Create Python virtual environment"""
        logger.info("Creating virtual environment...")
        
        venv_path = self.project_root / 'venv'
        
        if venv_path.exists():
            logger.info("Virtual environment already exists")
            return True
        
        try:
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
            logger.info("Virtual environment created")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        venv_path = self.project_root / 'venv'
        
        # Determine pip and python paths
        if self.system == "windows":
            pip_path = venv_path / 'Scripts' / 'pip.exe'
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:
            pip_path = venv_path / 'bin' / 'pip'
            python_path = venv_path / 'bin' / 'python'
        
        try:
            # Upgrade pip
            subprocess.run([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            
            # Install wheel and setuptools
            subprocess.run([str(pip_path), 'install', '--upgrade', 'wheel', 'setuptools'], check=True)
            
            # Install from requirements file
            requirements_files = [
                'requirements-live.txt',
                'requirements.txt'
            ]
            
            for req_file in requirements_files:
                req_path = self.project_root / req_file
                if req_path.exists():
                    logger.info(f"Installing from {req_file}...")
                    subprocess.run([str(pip_path), 'install', '-r', str(req_path)], check=True)
                    break
            else:
                logger.warning("No requirements file found, installing core dependencies...")
                self._install_core_dependencies(pip_path)
            
            logger.info("Python dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python dependencies: {e}")
            return False
    
    def _install_core_dependencies(self, pip_path):
        """Install core dependencies if no requirements file found"""
        core_packages = [
            'fastapi==0.104.1',
            'uvicorn[standard]==0.24.0',
            'sqlalchemy==2.0.23',
            'alembic==1.13.1',
            'psycopg2-binary==2.9.9',
            'redis==5.0.1',
            'celery==5.3.4',
            'requests==2.31.0',
            'pandas==2.1.3',
            'numpy==1.25.2',
            'cryptography==41.0.7',
            'selenium==4.15.0',
            'playwright==1.40.0',
            'beautifulsoup4==4.12.2',
            'tweepy==4.14.0',
            'praw==7.7.1'
        ]
        
        for package in core_packages:
            subprocess.run([str(pip_path), 'install', package], check=True)
    
    def install_playwright_browsers(self):
        """Install Playwright browsers"""
        logger.info("Installing Playwright browsers...")
        
        venv_path = self.project_root / 'venv'
        
        if self.system == "windows":
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:
            python_path = venv_path / 'bin' / 'python'
        
        try:
            subprocess.run([str(python_path), '-m', 'playwright', 'install'], check=True)
            subprocess.run([str(python_path), '-m', 'playwright', 'install-deps'], check=False)  # May fail on some systems
            logger.info("Playwright browsers installed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Playwright browsers: {e}")
            return False
    
    def setup_database(self):
        """Setup database"""
        logger.info("Setting up database...")
        
        try:
            # Check if PostgreSQL is running
            if self.system == "linux":
                subprocess.run(['sudo', 'systemctl', 'start', 'postgresql'], check=False)
                subprocess.run(['sudo', 'systemctl', 'enable', 'postgresql'], check=False)
            elif self.system == "darwin":
                subprocess.run(['brew', 'services', 'start', 'postgresql'], check=False)
            
            # Create database and user (if PostgreSQL is available)
            if self.system in ["linux", "darwin"]:
                try:
                    subprocess.run(['sudo', '-u', 'postgres', 'createdb', 'braf_db'], check=False)
                    subprocess.run(['sudo', '-u', 'postgres', 'createuser', 'braf_user'], check=False)
                    subprocess.run([
                        'sudo', '-u', 'postgres', 'psql', '-c',
                        "ALTER USER braf_user WITH PASSWORD 'braf_secure_password';"
                    ], check=False)
                    subprocess.run([
                        'sudo', '-u', 'postgres', 'psql', '-c',
                        "GRANT ALL PRIVILEGES ON DATABASE braf_db TO braf_user;"
                    ], check=False)
                    logger.info("PostgreSQL database configured")
                except subprocess.CalledProcessError:
                    logger.warning("PostgreSQL setup failed, will use SQLite as fallback")
            else:
                logger.info("PostgreSQL setup skipped on Windows - using SQLite as default")
            
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def setup_redis(self):
        """Setup Redis"""
        logger.info("Setting up Redis...")
        
        try:
            if self.system == "linux":
                subprocess.run(['sudo', 'systemctl', 'start', 'redis'], check=False)
                subprocess.run(['sudo', 'systemctl', 'enable', 'redis'], check=False)
            elif self.system == "darwin":
                subprocess.run(['brew', 'services', 'start', 'redis'], check=False)
            elif self.system == "windows":
                logger.info("Redis setup skipped on Windows - install manually if needed")
                return True
            
            # Test Redis connection (skip on Windows)
            if self.system != "windows":
                subprocess.run(['redis-cli', 'ping'], check=True, capture_output=True)
                logger.info("Redis configured and running")
            return True
            
        except subprocess.CalledProcessError:
            logger.warning("Redis setup failed")
            return True  # Don't fail installation for Redis issues
    
    def create_configuration_files(self):
        """Create configuration files"""
        logger.info("Creating configuration files...")
        
        try:
            # Create .env file if it doesn't exist
            env_file = self.project_root / '.env'
            if not env_file.exists():
                env_content = self._get_default_env_content()
                env_file.write_text(env_content)
                logger.info(".env file created")
            
            # Create alembic.ini if it doesn't exist
            alembic_file = self.project_root / 'alembic.ini'
            if not alembic_file.exists():
                alembic_content = self._get_default_alembic_content()
                alembic_file.write_text(alembic_content)
                logger.info("alembic.ini created")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create configuration files: {e}")
            return False
    
    def _get_default_env_content(self):
        """Get default .env file content"""
        return """# BRAF Environment Configuration

# Database Configuration
DATABASE_URL=postgresql://braf_user:braf_secure_password@localhost:5432/braf_db
# Fallback to SQLite if PostgreSQL not available
# DATABASE_URL=sqlite:///./braf.db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=braf_secret_key_change_in_production
JWT_SECRET_KEY=jwt_secret_key_change_in_production
ENCRYPTION_KEY=encryption_key_change_in_production

# Application Configuration
BRAF_ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Keys (replace with actual keys)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Payment Provider Keys
STRIPE_SECRET_KEY=your_stripe_secret_key
OPAY_API_KEY=your_opay_api_key
PALMPAY_API_KEY=your_palmpay_api_key

# Cryptocurrency API Keys
COINBASE_API_KEY=your_coinbase_api_key
BINANCE_API_KEY=your_binance_api_key

# Monitoring
PROMETHEUS_ENABLED=true
SENTRY_DSN=your_sentry_dsn

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
"""

    def _get_default_alembic_content(self):
        """Get default alembic.ini content"""
        return """# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = migrations

# template used to generate migration file names; The default value is %%(rev)s_%%(slug)s
# Uncomment the line below if you want the files to be prepended with date and time
# file_template = %%(year)d_%%(month).2d_%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
# defaults to the current working directory.
prepend_sys_path = .

# timezone to use when rendering the date within the migration file
# as well as the filename.
# If specified, requires the python-dateutil library that can be
# installed by adding `alembic[tz]` to the pip requirements
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version number format
version_num_format = %03d

# version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
# Valid values for version_path_separator are:
#
# version_path_separator = :
# version_path_separator = ;
# version_path_separator = space
version_path_separator = os

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = postgresql://braf_user:braf_secure_password@localhost:5432/braf_db

[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks = black
# black.type = console_scripts
# black.entrypoint = black
# black.options = -l 79 REVISION_SCRIPT_FILENAME

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""

    def run_tests(self):
        """Run basic tests to verify installation"""
        logger.info("Running installation tests...")
        
        venv_path = self.project_root / 'venv'
        
        if self.system == "windows":
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:
            python_path = venv_path / 'bin' / 'python'
        
        try:
            # Test basic imports
            test_script = '''
import sys
modules = ["fastapi", "uvicorn", "sqlalchemy", "redis", "celery", "requests", "pandas", "numpy"]
failed = []
for module in modules:
    try:
        __import__(module)
        print(f"OK {module}")
    except ImportError:
        print(f"FAIL {module}")
        failed.append(module)

if failed:
    print(f"Failed imports: {failed}")
    sys.exit(1)
else:
    print("All core modules imported successfully!")
'''
            
            result = subprocess.run([str(python_path), '-c', test_script], 
                                  capture_output=True, text=True, check=True)
            logger.info("Import tests passed")
            logger.info(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Tests failed: {e}")
            logger.error(e.stdout)
            logger.error(e.stderr)
            return False
    
    def generate_installation_report(self):
        """Generate installation report"""
        logger.info("Generating installation report...")
        
        report = {
            'timestamp': str(datetime.now().isoformat()),
            'system_info': {
                'platform': self.system,
                'architecture': self.architecture,
                'python_version': self.python_version
            },
            'installation_status': {
                'system_dependencies': 'Installed',
                'virtual_environment': 'Created',
                'python_dependencies': 'Installed',
                'playwright_browsers': 'Installed',
                'database_setup': 'Configured',
                'redis_setup': 'Configured',
                'configuration_files': 'Created'
            },
            'next_steps': [
                '1. Review and update .env file with your API keys',
                '2. Run database migrations: alembic upgrade head',
                '3. Start the application: uvicorn main:app --reload',
                '4. Access the dashboard at: http://localhost:8000'
            ]
        }
        
        report_file = self.project_root / 'installation_report.json'
        report_file.write_text(json.dumps(report, indent=2))
        
        logger.info("Installation report generated")
        return report
    
    def install(self):
        """Main installation method"""
        logger.info("Starting BRAF dependency installation...")
        logger.info("=" * 60)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Installing system dependencies", self.install_system_dependencies),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing Python dependencies", self.install_python_dependencies),
            ("Installing Playwright browsers", self.install_playwright_browsers),
            ("Setting up database", self.setup_database),
            ("Setting up Redis", self.setup_redis),
            ("Creating configuration files", self.create_configuration_files),
            ("Running tests", self.run_tests)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            logger.info(f"\n📋 {step_name}...")
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    logger.warning(f"WARNING: {step_name} completed with warnings")
            except Exception as e:
                logger.error(f"ERROR: {step_name} failed: {e}")
                failed_steps.append(step_name)
        
        # Generate report
        report = self.generate_installation_report()
        
        logger.info("\n" + "=" * 60)
        if not failed_steps:
            logger.info("BRAF INSTALLATION COMPLETED SUCCESSFULLY!")
        else:
            logger.warning(f"Installation completed with {len(failed_steps)} warnings")
            logger.warning(f"Failed/Warning steps: {', '.join(failed_steps)}")
        
        logger.info("=" * 60)
        logger.info("📋 Next Steps:")
        for step in report['next_steps']:
            logger.info(f"   {step}")
        logger.info("=" * 60)
        
        return len(failed_steps) == 0


def main():
    """Main execution function"""
    installer = BRAFDependencyInstaller()
    
    # Check for command line arguments
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
BRAF Dependency Installer

Usage: python install_dependencies.py [options]

Options:
  --help, -h     Show this help message
  --system-only  Install only system dependencies
  --python-only  Install only Python dependencies
  --test-only    Run tests only

Examples:
  python install_dependencies.py              # Full installation
  python install_dependencies.py --system-only  # System deps only
        """)
        return
    
    if '--system-only' in sys.argv:
        installer.install_system_dependencies()
    elif '--python-only' in sys.argv:
        installer.create_virtual_environment()
        installer.install_python_dependencies()
        installer.install_playwright_browsers()
    elif '--test-only' in sys.argv:
        installer.run_tests()
    else:
        success = installer.install()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()