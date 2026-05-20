"""add_password_hash_to_users

Revision ID: 60834f3aed34
Revises: 196a9b156de2
Create Date: 2025-11-12 23:31:07.786071

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '60834f3aed34'
down_revision: Union[str, Sequence[str], None] = '196a9b156de2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add password_hash column to users table
    op.add_column('users', sa.Column('password_hash', sa.String(length=255), nullable=True))

    # Add email_verified column
    op.add_column('users', sa.Column('email_verified', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('users', sa.Column('email_verified_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove added columns
    op.drop_column('users', 'email_verified_at')
    op.drop_column('users', 'email_verified')
    op.drop_column('users', 'password_hash')
