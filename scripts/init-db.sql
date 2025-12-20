-- Initialize BRAF database with required extensions and initial setup

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database user if not exists (for production)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'braf') THEN
        CREATE ROLE braf WITH LOGIN PASSWORD 'password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE braf_dev TO braf;
GRANT ALL ON SCHEMA public TO braf;

-- Create initial tables (will be managed by Alembic migrations)
-- This is just for development setup