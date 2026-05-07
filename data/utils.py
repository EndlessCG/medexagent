"""Shared utilities for the data pipeline."""

import logging
import time

from openai import RateLimitError

logger = logging.getLogger(__name__)


def rate_limit_sleep(e: RateLimitError, attempt: int) -> float:
    """Extract retry-after from a RateLimitError and sleep.

    Returns the number of seconds waited.
    """
    retry_after = None
    if hasattr(e, "response") and e.response is not None:
        retry_after = e.response.headers.get("retry-after")
    wait = float(retry_after) if retry_after else min(60, 2 ** attempt * 5)
    logger.warning("Rate limited (attempt %d), waiting %.0fs", attempt + 1, wait)
    time.sleep(wait)
    return wait
