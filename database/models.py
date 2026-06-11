"""
models.py - SQLAlchemy ORM models derived from the DBML schema.

Tables
------
- user      : end-user accounts
- invoice   : invoices belonging to a user
- item      : line-items belonging to an invoice

All monetary / decimal fields use ``Numeric(10, 2)`` for
lossless fixed-point arithmetic.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Numeric,
    DateTime,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ──────────────────────────────────────────────
#  User
# ──────────────────────────────────────────────
class User(Base):
    """Application user who owns invoices."""

    __tablename__ = "user"

    user_id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String(255), nullable=False)

    # Relationships
    invoices = relationship(
        "Invoice",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, name={self.name!r})>"


# ──────────────────────────────────────────────
#  Invoice
# ──────────────────────────────────────────────
class Invoice(Base):
    """Invoice header linked to a user, containing totals and metadata."""

    __tablename__ = "invoice"
    __table_args__ = (
        Index("idx_invoice_user_id", "user_id"),
    )

    invoice_id: int = Column(Integer, primary_key=True, autoincrement=True)
    user_id: int = Column(
        Integer,
        ForeignKey("user.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    vendor: str = Column(String(255), nullable=False)
    subtotal = Column(Numeric(10, 2), nullable=False, server_default="0.00")
    discount = Column(Numeric(10, 2), nullable=False, server_default="0.00")
    tax = Column(Numeric(10, 2), nullable=False, server_default="0.00")
    shipping = Column(Numeric(10, 2), nullable=False, server_default="0.00")
    total = Column(Numeric(10, 2), nullable=False, server_default="0.00")
    date: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    user = relationship("User", back_populates="invoices")
    items = relationship(
        "Item",
        back_populates="invoice",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<Invoice(invoice_id={self.invoice_id}, "
            f"vendor={self.vendor!r}, total={self.total})>"
        )


# ──────────────────────────────────────────────
#  Item
# ──────────────────────────────────────────────
class Item(Base):
    """Line-item on an invoice."""

    __tablename__ = "item"
    __table_args__ = (
        Index("idx_item_invoice_id", "invoice_id"),
    )

    item_id: int = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id: int = Column(
        Integer,
        ForeignKey("invoice.invoice_id", ondelete="CASCADE"),
        nullable=False,
    )
    name: str = Column(String(255), nullable=False)
    quantity = Column(Numeric(10, 2), nullable=False, server_default="1.00")
    price = Column(Numeric(10, 2), nullable=False, server_default="0.00")
    savings = Column(Numeric(10, 2), nullable=False, server_default="0.00")

    # Relationships
    invoice = relationship("Invoice", back_populates="items")

    def __repr__(self) -> str:
        return (
            f"<Item(item_id={self.item_id}, "
            f"name={self.name!r}, price={self.price})>"
        )
