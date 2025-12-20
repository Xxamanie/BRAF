"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-12-14 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create profiles table
    op.create_table('profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('fingerprint_id', sa.String(length=255), nullable=False),
        sa.Column('proxy_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('session_count', sa.Integer(), nullable=True),
        sa.Column('detection_score', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_profiles_fingerprint_id'), 'profiles', ['fingerprint_id'], unique=False)

    # Create fingerprints table
    op.create_table('fingerprints',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('user_agent', sa.Text(), nullable=False),
        sa.Column('screen_width', sa.Integer(), nullable=False),
        sa.Column('screen_height', sa.Integer(), nullable=False),
        sa.Column('timezone', sa.String(length=100), nullable=False),
        sa.Column('webgl_vendor', sa.String(length=255), nullable=False),
        sa.Column('webgl_renderer', sa.String(length=255), nullable=False),
        sa.Column('canvas_hash', sa.String(length=255), nullable=False),
        sa.Column('audio_context_hash', sa.String(length=255), nullable=False),
        sa.Column('fonts', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('plugins', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('languages', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('platform', sa.String(length=50), nullable=True),
        sa.Column('hardware_concurrency', sa.Integer(), nullable=True),
        sa.Column('device_memory', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_used', sa.DateTime(timezone=True), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create automation_tasks table
    op.create_table('automation_tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('profile_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('target_url', sa.Text(), nullable=False),
        sa.Column('actions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('constraints', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('assigned_worker', sa.String(length=255), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['profile_id'], ['profiles.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_automation_tasks_assigned_worker'), 'automation_tasks', ['assigned_worker'], unique=False)
    op.create_index(op.f('ix_automation_tasks_priority'), 'automation_tasks', ['priority'], unique=False)
    op.create_index(op.f('ix_automation_tasks_status'), 'automation_tasks', ['status'], unique=False)

    # Create compliance_logs table
    op.create_table('compliance_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('action_type', sa.String(length=100), nullable=False),
        sa.Column('target_url', sa.Text(), nullable=True),
        sa.Column('profile_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('worker_id', sa.String(length=255), nullable=False),
        sa.Column('detection_score', sa.Float(), nullable=False),
        sa.Column('ethical_check_passed', sa.Boolean(), nullable=False),
        sa.Column('authorization_token', sa.String(length=255), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['profile_id'], ['profiles.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_compliance_logs_action_type'), 'compliance_logs', ['action_type'], unique=False)
    op.create_index(op.f('ix_compliance_logs_ethical_check_passed'), 'compliance_logs', ['ethical_check_passed'], unique=False)
    op.create_index(op.f('ix_compliance_logs_timestamp'), 'compliance_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_compliance_logs_worker_id'), 'compliance_logs', ['worker_id'], unique=False)

    # Create encrypted_credentials table
    op.create_table('encrypted_credentials',
        sa.Column('profile_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('encrypted_data', sa.Text(), nullable=False),
        sa.Column('salt', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['profile_id'], ['profiles.id'], ),
        sa.PrimaryKeyConstraint('profile_id')
    )

    # Create worker_nodes table
    op.create_table('worker_nodes',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('current_tasks', sa.Integer(), nullable=True),
        sa.Column('max_tasks', sa.Integer(), nullable=True),
        sa.Column('cpu_usage', sa.Float(), nullable=True),
        sa.Column('memory_usage', sa.Float(), nullable=True),
        sa.Column('last_heartbeat', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('capabilities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('configuration', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_worker_nodes_status'), 'worker_nodes', ['status'], unique=False)

    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('metric_type', sa.String(length=100), nullable=False),
        sa.Column('metric_name', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('labels', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('worker_id', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_system_metrics_metric_name'), 'system_metrics', ['metric_name'], unique=False)
    op.create_index(op.f('ix_system_metrics_metric_type'), 'system_metrics', ['metric_type'], unique=False)
    op.create_index(op.f('ix_system_metrics_timestamp'), 'system_metrics', ['timestamp'], unique=False)
    op.create_index(op.f('ix_system_metrics_worker_id'), 'system_metrics', ['worker_id'], unique=False)


def downgrade() -> None:
    op.drop_table('system_metrics')
    op.drop_table('worker_nodes')
    op.drop_table('encrypted_credentials')
    op.drop_table('compliance_logs')
    op.drop_table('automation_tasks')
    op.drop_table('fingerprints')
    op.drop_table('profiles')