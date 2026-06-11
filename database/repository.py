"""
repository.py – Database repository for invoice persistence.

Provides a single public function:

    save_invoice(db, extract) -> Invoice

It handles:
- Auto-creating the system user (user_id=1) if absent.
- Inserting the Invoice header row.
- Bulk-inserting all Item rows.
- Committing the transaction atomically.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from database.models import Invoice, Item, User
from src.schemas.invoice_schema import InvoiceExtract

logger = logging.getLogger(__name__)

_SYSTEM_USER_ID = 1
_SYSTEM_USER_NAME = "System"


# ── Public API ─────────────────────────────────────────────────────────────────

def save_invoice(db: Session, extract: InvoiceExtract) -> Invoice:
    """
    Persist an extracted invoice (and its line items) to PostgreSQL.

    Parameters
    ----------
    db:
        An active SQLAlchemy session.
    extract:
        The validated InvoiceExtract object produced by the LLM extractor.

    Returns
    -------
    Invoice
        The newly created ORM Invoice instance (with `invoice_id` populated
        after the commit).
    """
    _ensure_system_user(db)

    invoice_date = extract.date or datetime.now(tz=timezone.utc)

    invoice = Invoice(
        user_id=_SYSTEM_USER_ID,
        vendor=extract.vendor,
        subtotal=extract.subtotal,
        discount=extract.discount,
        tax=extract.tax,
        shipping=extract.shipping,
        total=extract.total,
        date=invoice_date,
    )
    db.add(invoice)
    db.flush()  # populate invoice.invoice_id without committing yet

    for item_extract in extract.items:
        item = Item(
            invoice_id=invoice.invoice_id,
            name=item_extract.name,
            quantity=item_extract.quantity,
            price=item_extract.price,
            savings=item_extract.savings,
        )
        db.add(item)

    db.commit()
    db.refresh(invoice)

    logger.info(
        "Saved invoice_id=%d  vendor=%r  total=%s  items=%d",
        invoice.invoice_id,
        invoice.vendor,
        invoice.total,
        len(extract.items),
    )
    return invoice


# ── Private helpers ────────────────────────────────────────────────────────────

def _ensure_system_user(db: Session) -> None:
    """Create the system user if it doesn't already exist."""
    existing = db.get(User, _SYSTEM_USER_ID)
    if existing is None:
        db.add(User(user_id=_SYSTEM_USER_ID, name=_SYSTEM_USER_NAME))
        db.flush()
        logger.info("Created system user (user_id=%d)", _SYSTEM_USER_ID)
