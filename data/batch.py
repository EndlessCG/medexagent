"""OpenAI Batch API helpers for the data pipeline.

The Batch API submits all requests in a single JSONL file, processes
them asynchronously within a 24-hour window, and provides ~50% cost
savings over synchronous requests.

NOTE: Only compatible with api.openai.com. Custom base URLs (e.g.
local inference servers) do not support the Batch API.
"""

import json
import os
import tempfile
import time

from openai import OpenAI


_TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})


def build_chat_request(
    custom_id: str,
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    response_format: dict | None = None,
) -> dict:
    """Build a single /v1/chat/completions batch request entry."""
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def submit_batch(
    client: OpenAI,
    requests: list[dict],
    description: str = "medagent",
) -> str:
    """Upload requests as a JSONL file and create a batch job.

    Returns:
        The batch id string.
    """
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")
        with open(path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
        )
        print(f"  Batch created: {batch.id}  (file: {uploaded.id})")
        return batch.id
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def poll_batch(
    client: OpenAI,
    batch_id: str,
    poll_interval: int = 60,
    timeout: int = 86400,
):
    """Block until the batch reaches a terminal status and return the batch object.

    Args:
        client: OpenAI client.
        batch_id: The batch id to poll.
        poll_interval: Seconds between status checks.
        timeout: Maximum seconds to wait (default 24h).

    Raises:
        TimeoutError: If the batch does not finish within `timeout` seconds.
    """
    start = time.time()
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in _TERMINAL:
            c = b.request_counts
            print(
                f"  Batch {batch_id}: {b.status} "
                f"({c.completed} completed, {c.failed} failed, {c.total} total)"
            )
            return b
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(
                f"Batch {batch_id} did not finish within {timeout:.0f}s "
                f"(current status: {b.status})"
            )
        c = b.request_counts
        print(
            f"  Batch {batch_id}: {b.status} "
            f"({c.completed}/{c.total} done, {elapsed:.0f}s elapsed) "
            f"— next poll in {poll_interval}s"
        )
        time.sleep(poll_interval)


def download_results(client: OpenAI, batch) -> dict[str, str | None]:
    """Parse the batch output file.

    Returns:
        Dict mapping custom_id -> response content string (None on error).
    """
    results: dict[str, str | None] = {}

    if batch.output_file_id:
        raw = client.files.content(batch.output_file_id)
        for line in raw.text.splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cid = entry["custom_id"]
            try:
                choices = entry["response"]["body"]["choices"]
                results[cid] = choices[0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                results[cid] = None

    if batch.error_file_id:
        try:
            raw = client.files.content(batch.error_file_id)
            for line in raw.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                cid = entry["custom_id"]
                if cid not in results:
                    results[cid] = None
        except Exception:
            pass

    return results


def run_batch_job(
    client: OpenAI,
    requests: list[dict],
    description: str = "medagent",
    poll_interval: int = 60,
    state_file: str | None = None,
) -> dict[str, str | None]:
    """Submit, poll, and download a batch job end-to-end.

    If `state_file` is provided, the batch_id is persisted to disk so the
    job can be resumed if the calling process is interrupted while polling.

    Args:
        client: OpenAI client (must point to api.openai.com).
        requests: List of request dicts from build_chat_request().
        description: Human-readable description stored in batch metadata.
        poll_interval: Seconds between status polls.
        state_file: Path to JSON file for persisting batch_id across restarts.

    Returns:
        Dict mapping custom_id -> response content string (None on failure).

    Raises:
        RuntimeError: If the batch ends in a non-completed terminal state.
    """
    batch_id: str | None = None
    if state_file and os.path.exists(state_file):
        try:
            with open(state_file) as f:
                batch_id = json.load(f).get("batch_id")
            if batch_id:
                print(f"  Resuming batch {batch_id} from {state_file}")
        except Exception:
            batch_id = None

    if not batch_id:
        batch_id = submit_batch(client, requests, description)
        if state_file:
            os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
            with open(state_file, "w") as f:
                json.dump({"batch_id": batch_id}, f)

    batch = poll_batch(client, batch_id, poll_interval)

    if state_file and os.path.exists(state_file):
        try:
            os.unlink(state_file)
        except OSError:
            pass

    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch_id} ended with status '{batch.status}'. "
            "Check the OpenAI dashboard for error details."
        )

    return download_results(client, batch)
