from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

import ollama
from google import genai
from google.genai import types

from src.schemas.invoice_schema import InvoiceExtract
from .base import BaseInvoiceExtractor

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

_DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
_DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

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

# ── Ollama Concrete Extractor ──────────────────────────────────────────────────

class OllamaInvoiceExtractor(BaseInvoiceExtractor):
    """Extract structured invoice data from OCR text using a local Ollama model."""

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None) -> None:
        self._model = model or os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
        _host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=_host)
        logger.info(
            "OllamaInvoiceExtractor initialised — model: %s  host: %s",
            self._model, _host,
        )

    def extract(
        self,
        ocr_text: str,
        max_retries: int = 3,
        initial_delay: float = 2.0,
    ) -> InvoiceExtract:
        if not ocr_text.strip():
            raise ValueError("OCR text is empty — nothing to extract.")

        prompt = f"Invoice OCR text:\n\n{ocr_text}"
        logger.debug("Sending %d characters to Ollama…", len(prompt))

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


# ── Gemini Concrete Extractor ──────────────────────────────────────────────────

class GeminiInvoiceExtractor(BaseInvoiceExtractor):
    """Extract structured invoice data from OCR text using Google Gemini API."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self._model = model or os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
        _api_key = api_key or os.getenv("GEMINI_API_KEY")

        if _api_key:
            self._client = genai.Client(api_key=_api_key)
        else:
            self._client = genai.Client()

        logger.info(
            "GeminiInvoiceExtractor initialised — model: %s",
            self._model,
        )

    def extract(
        self,
        ocr_text: str,
        max_retries: int = 3,
        initial_delay: float = 2.0,
    ) -> InvoiceExtract:
        if not ocr_text.strip():
            raise ValueError("OCR text is empty — nothing to extract.")

        prompt = f"Invoice OCR text:\n\n{ocr_text}"
        logger.debug("Sending %d characters to Gemini…", len(prompt))

        delay = initial_delay
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=InvoiceExtract,
                    ),
                )

                if response.parsed:
                    logger.debug("Successfully parsed structured output directly via Gemini SDK.")
                    return response.parsed

                raw = response.text.strip()
                logger.debug("Raw Gemini response:\n%s", raw)
                return self._parse_response(raw)

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "Gemini call failed (attempt %d/%d): %s — retrying in %.0fs…",
                        attempt + 1, max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2  # exponential back-off

        raise last_exc  # type: ignore[misc]


# ── Backward-compatible Wrapper Class ──────────────────────────────────────────

class InvoiceExtractor(BaseInvoiceExtractor):
    """
    A backward-compatible wrapper that delegates extraction to either Ollama or Gemini.
    By default, it checks the LLM_PROVIDER environment variable to determine the backend.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._provider = provider or os.getenv("LLM_PROVIDER", "gemini")
        self._provider = self._provider.lower()

        if self._provider == "gemini":
            self._delegate: BaseInvoiceExtractor = GeminiInvoiceExtractor(
                model=model,
                api_key=kwargs.get("api_key")
            )
        else:
            self._delegate = OllamaInvoiceExtractor(
                model=model,
                host=kwargs.get("host")
            )

    def extract(
        self,
        ocr_text: str,
        max_retries: int = 3,
        initial_delay: float = 2.0,
    ) -> InvoiceExtract:
        return self._delegate.extract(
            ocr_text=ocr_text,
            max_retries=max_retries,
            initial_delay=initial_delay,
        )
