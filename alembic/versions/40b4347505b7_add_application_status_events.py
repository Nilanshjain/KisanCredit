"""add application_status_events table for lifecycle timeline

Revision ID: 40b4347505b7
Revises: 60834f3aed34
Create Date: 2026-05-20 17:31:54.924662

Phase 3 lifecycle: every status transition (submitted -> under_review ->
decided -> disbursed -> rejected) is recorded as an append-only event so
the user-facing timeline and the admin override audit trail share one source
of truth. Existing applications (pre-migration) get no backfill — their
timeline starts empty and the UI falls back to the application.submitted_at
timestamp.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '40b4347505b7'
down_revision: Union[str, Sequence[str], None] = '60834f3aed34'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'application_status_events',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('application_id', sa.String(length=50), nullable=False),
        sa.Column('from_status', sa.String(length=50), nullable=True),
        sa.Column('to_status', sa.String(length=50), nullable=False),
        sa.Column('actor_type', sa.String(length=20), nullable=False),
        sa.Column('actor_id', sa.String(length=100), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('occurred_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'idx_status_event_app_time',
        'application_status_events',
        ['application_id', 'occurred_at'],
        unique=False,
    )
    op.create_index(
        'idx_status_event_actor',
        'application_status_events',
        ['actor_type'],
        unique=False,
    )
    op.create_index(
        op.f('ix_application_status_events_application_id'),
        'application_status_events',
        ['application_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_application_status_events_occurred_at'),
        'application_status_events',
        ['occurred_at'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_application_status_events_occurred_at'),
        table_name='application_status_events',
    )
    op.drop_index(
        op.f('ix_application_status_events_application_id'),
        table_name='application_status_events',
    )
    op.drop_index('idx_status_event_app_time', table_name='application_status_events')
    op.drop_index('idx_status_event_actor', table_name='application_status_events')
    op.drop_table('application_status_events')
