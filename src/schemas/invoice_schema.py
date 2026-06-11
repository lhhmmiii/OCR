"""
invoice_schema.py – Pydantic v2 schemas for the LLM-extracted invoice data.

These models sit between the raw LLM JSON response and the SQLAlchemy ORM
layer.  They enforce types, coerce values, and provide a single source of
truth for what a "parsed invoice" looks like.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ItemExtract(BaseModel):
    """A single line-item on the invoice."""

    name: str = Field(..., description="Product or service name")
    quantity: Decimal = Field(default=Decimal("1.00"), ge=0)
    price: Decimal = Field(default=Decimal("0.00"), ge=0)
    savings: Decimal = Field(default=Decimal("0.00"), ge=0)

    @field_validator("quantity", "price", "savings", mode="before")
    @classmethod
    def coerce_decimal(cls, v: object) -> Decimal:
        if v is None or v == "":
            return Decimal("0.00")
        try:
            return Decimal(str(v).replace(",", ""))
        except Exception:
            return Decimal("0.00")


class InvoiceExtract(BaseModel):
    """Structured invoice extracted from OCR text by the LLM."""

    vendor: str = Field(..., description="Vendor / seller name")
    date: Optional[datetime] = Field(
        default=None,
        description="Invoice date (ISO-8601 preferred)",
    )
    subtotal: Decimal = Field(default=Decimal("0.00"), ge=0)
    discount: Decimal = Field(default=Decimal("0.00"), ge=0)
    tax: Decimal = Field(default=Decimal("0.00"), ge=0)
    shipping: Decimal = Field(default=Decimal("0.00"), ge=0)
    total: Decimal = Field(default=Decimal("0.00"), ge=0)
    items: List[ItemExtract] = Field(default_factory=list)

    @field_validator("subtotal", "discount", "tax", "shipping", "total", mode="before")
    @classmethod
    def coerce_decimal(cls, v: object) -> Decimal:
        if v is None or v == "":
            return Decimal("0.00")
        try:
            return Decimal(str(v).replace(",", ""))
        except Exception:
            return Decimal("0.00")

    @field_validator("date", mode="before")
    @classmethod
    def coerce_date(cls, v: object) -> Optional[datetime]:
        if not v:
            return None
        if isinstance(v, datetime):
            return v
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(str(v).strip(), fmt)
            except ValueError:
                continue
        return None
