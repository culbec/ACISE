import functools
import os
import subprocess
import time
from typing import Any

import requests

from acise_p2_cidt.utils.logging.log import get_logger

API: str = "https://api.github.com"
API_VERSION: str = "2022-11-28"
PER_PAGE: int = 100  # API maximum
MAX_TRIES: int = 5
TIMEOUT: float = 30.0  # seconds per call

logger = get_logger(__name__)

calls = 0  # API calls made by this process, for the manifest


@functools.cache
def token() -> str:
    """
    GitHub token from `$GITHUB_TOKEN`, else from `gh auth token`. Kept in memory
    only: never logged, never written.

    Returns:
        str: the token.
    """
    if tok := os.environ.get("GITHUB_TOKEN"):
        return tok
    return subprocess.run(
        ["gh", "auth", "token"], check=True, capture_output=True, text=True
    ).stdout.strip()


@functools.cache
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        }
    )
    return s


def _sleep(seconds: float, why: str) -> None:
    # No `log.status` spinner: it would nest a second live display in the
    # collection's progress bar.
    logger.warning("%s: sleeping %.0fs", why, seconds)
    time.sleep(seconds)


def get(path: str, params: dict | None = None) -> Any:
    """
    One API call with rate-limit handling and retries, no caching.

    When the rate limit is used up, it sleeps until the reset time. Server errors
    and secondary limits (403/429) are retried with exponential backoff, honouring
    `Retry-After`; any other error status raises.

    Args:
        path (str): API path under `API`, starting with `/`.
        params (dict | None): query parameters.

    Returns:
        Any: decoded JSON body.
    """
    global calls
    for attempt in range(MAX_TRIES):
        try:
            resp = _session().get(f"{API}{path}", params=params, timeout=TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as e:
            _sleep(2**attempt, f"{type(e).__name__} on {path}")
            continue
        calls += 1

        retryable = resp.status_code >= 500 or resp.status_code in (403, 429)
        if not resp.ok and not retryable:
            resp.raise_for_status()
        if resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = float(resp.headers.get("X-RateLimit-Reset", time.time()))
            _sleep(max(reset - time.time(), 0.0) + 1.0, "rate limit")
        elif not resp.ok:
            wait = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            _sleep(wait, f"HTTP {resp.status_code} on {path}")
        if resp.ok:
            return resp.json()
    raise RuntimeError(f"giving up on {path} after {MAX_TRIES} tries")


def get_all(path: str, params: dict | None = None, key: str | None = None) -> list:
    """
    Every page of a list endpoint, merged.

    Args:
        path (str): API path.
        params (dict | None): query parameters, without paging.
        key (str | None): name of the list in a dict body; None for a list body.

    Returns:
        list: all items.
    """
    items, page = [], 1
    while True:
        body = get(path, {**(params or {}), "per_page": PER_PAGE, "page": page})
        batch = body[key] if key else body
        items += batch
        if len(batch) < PER_PAGE:
            return items
        page += 1
