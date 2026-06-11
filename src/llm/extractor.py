"""
extractor.py – LLM-powered invoice field extractor using Ollama (local inference).

Flow
----
1. Receive plain-text OCR output (one big string).
2. Build a structured prompt asking the model to return strict JSON.
3. Parse the JSON response and validate it with InvoiceExtract (Pydantic).
4. Return the validated InvoiceExtract object.

Environment variables
---------------------
OLLAMA_HOST   - Ollama server base URL, defaults to "http://localhost:11434"
OLLAMA_MODEL  - model name, defaults to "llama3"
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import ollama

from src.schemas.invoice_schema import InvoiceExtract

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "qwen3.5:4b"

_SYSTEM_PROMPT = """\
You are a precise invoice data extraction assistant.
Given the raw OCR text of an invoice, extract the following fields and return
ONLY a valid JSON object — no markdown fences, no explanations.

Required JSON shape:
{
  "vendor":   "<string>",
  "date":     "<YYYY-MM-DD or null>",
  "subtotal": <number or 0>,
  "discount": <number or 0>,
  "tax":      <number or 0>,
  "shipping": <number or 0>,
  "total":    <number or 0>,
  "items": [
    {
      "name":     "<string>",
      "quantity": <number or 1>,
      "price":    <number or 0>,
      "savings":  <number or 0>
    }
  ]
}

Rules:
- All monetary values must be plain numbers (no currency symbols, no commas).
- If a field cannot be found in the text, use null for strings and 0 for numbers.
- items may be an empty list [] if no line-items are present.
- Return ONLY the JSON object.
"""


# ── Main extractor class ───────────────────────────────────────────────────────

class InvoiceExtractor:
    """Extract structured invoice data from OCR text using a local Ollama model."""

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None) -> None:
        self._model = model or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
        _host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=_host)
        logger.info(
            "InvoiceExtractor initialised — model: %s  host: %s",
            self._model, _host,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(
        self,
        ocr_text: str,
        max_retries: int = 3,
        initial_delay: float = 2.0,  # kept for API compatibility; Ollama is local so rarely needed
    ) -> InvoiceExtract:
        """
        Extract invoice fields from raw OCR text.

        Parameters
        ----------
        ocr_text:
            The concatenated plain text produced by the OCR pipeline.
        max_retries:
            Maximum number of retry attempts on transient errors.
        initial_delay:
            Seconds to wait before the first retry (doubles each attempt).

        Returns
        -------
        InvoiceExtract
            A validated Pydantic model containing all extracted fields.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed or validated.
        Exception
            Re-raised after exhausting retries.
        """
        if not ocr_text.strip():
            raise ValueError("OCR text is empty — nothing to extract.")

        prompt = f"Invoice OCR text:\n\n{ocr_text}"
        logger.debug("Sending %d characters to Ollama…", len(prompt))

        import time
        delay = initial_delay
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    options={"temperature": 0},  # deterministic output
                )
                raw = response["message"]["content"].strip()
                logger.debug("Raw Ollama response:\n%s", raw)
                return self._parse_response(raw)

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "Ollama call failed (attempt %d/%d): %s — retrying in %.0fs…",
                        attempt + 1, max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2  # exponential back-off

        raise last_exc  # type: ignore[misc]

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> InvoiceExtract:
        """Strip markdown fences (if any) and parse JSON into InvoiceExtract."""
        # Remove ```json … ``` or ``` … ``` fences that some models emit
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Ollama returned non-JSON content.\n"
                f"Raw response:\n{raw}\n"
                f"JSON error: {exc}"
            ) from exc

        try:
            invoice = InvoiceExtract(**data)
        except Exception as exc:
            raise ValueError(
                f"Extracted JSON does not match the expected schema.\n"
                f"Data: {data}\n"
                f"Error: {exc}"
            ) from exc

        return invoice
