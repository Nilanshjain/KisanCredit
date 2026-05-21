"""add role column to users for admin RBAC

Revision ID: 3551308e2b78
Revises: 40b4347505b7
Create Date: 2026-05-21 15:10:24.022769

Phase 4: lender/operator admin view. Gate is `require_admin()` in
src/auth/dependencies.py — checks users.role == 'admin'. Default 'user'
covers every existing applicant; promotion is manual via
scripts/promote_admin.py.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '3551308e2b78'
down_revision: Union[str, Sequence[str], None] = '40b4347505b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # server_default='user' lets the NOT NULL constraint hold for existing rows
    # without a separate backfill step.
    op.add_column(
        'users',
        sa.Column('role', sa.String(length=20), nullable=False, server_default='user'),
    )
    op.create_index(op.f('ix_users_role'), 'users', ['role'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_users_role'), table_name='users')
    op.drop_column('users', 'role')
