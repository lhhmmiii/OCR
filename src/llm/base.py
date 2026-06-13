from abc import ABC, abstractmethod
from src.schemas.invoice_schema import InvoiceExtract
import re
import json

class BaseInvoiceExtractor(ABC):
    """Base class for invoice field extractors using LLMs."""

    @abstractmethod
    def extract(
        self,
        ocr_text: str,
        max_retries: int = 3,
        initial_delay: float = 2.0,
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
        pass

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
                f"LLM returned non-JSON content.\n"
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