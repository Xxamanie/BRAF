from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from config import Config

# Database engine
if Config.DATABASE_URL.startswith('sqlite'):
    # SQLite configuration
    engine = create_engine(
        Config.DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=Config.DEBUG if hasattr(Config, 'DEBUG') else False
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        Config.DATABASE_URL,
        poolclass=StaticPool,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=Config.DEBUG if hasattr(Config, 'DEBUG') else False
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Import models to ensure they're registered
from .models import *
