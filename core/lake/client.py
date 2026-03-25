"""Python client for the Trident Lake Query API."""

from __future__ import annotations

import io
import time
from typing import Any, Iterator

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc


class LakeAPIClient:
    """Thin client for the Trident Lake Query API.

    Usage::

        client = LakeAPIClient("http://api.localhost", api_key="my-token-key")
        rows = client.query("SELECT * FROM parsed_emails LIMIT 100")
        table = client.query_arrow("SELECT sender, subject FROM parsed_emails LIMIT 1000")
        for batch in client.query_stream("SELECT * FROM parsed_emails"):
            process(batch)
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, path: str, **kwargs) -> httpx.Response:
        r = httpx.get(
            f"{self._base}{path}",
            headers=self._headers,
            timeout=self._timeout,
            verify=False,
            **kwargs,
        )
        r.raise_for_status()
        return r

    def _post(self, path: str, json: dict, **kwargs) -> httpx.Response:
        r = httpx.post(
            f"{self._base}{path}",
            headers=self._headers,
            json=json,
            timeout=self._timeout,
            verify=False,
            **kwargs,
        )
        r.raise_for_status()
        return r

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return the API health status."""
        return self._get("/health").json()

    # ── Table discovery ───────────────────────────────────────────────────────

    def tables(self) -> list[str]:
        """List tables accessible to the current principal."""
        return self._get("/tables").json()["tables"]

    def schema(self, table: str) -> list[dict]:
        """Return the schema for a table as a list of {column_name, column_type} dicts."""
        return self._get(f"/tables/{table}/schema").json()["schema"]

    def count(self, table: str) -> int:
        """Return the fast row count for a table (read from delta log, no full scan)."""
        return self._get(f"/tables/{table}/count").json()["count"]

    # ── Query (full result) ───────────────────────────────────────────────────

    def query(self, sql: str, limit: int = 10000) -> list[dict]:
        """Execute SQL and return rows as a list of dicts (JSON format)."""
        return self._post("/query", {"sql": sql, "limit": limit}).json()

    def query_arrow(self, sql: str, limit: int = 10000) -> pa.Table:
        """Execute SQL and return the result as a PyArrow Table."""
        r = httpx.post(
            f"{self._base}/query",
            headers={**self._headers, "Accept": "application/vnd.apache.arrow.stream"},
            json={"sql": sql, "limit": limit},
            timeout=self._timeout,
            verify=False,
        )
        r.raise_for_status()
        reader = ipc.open_stream(io.BytesIO(r.content))
        return reader.read_all()

    def query_parquet(self, sql: str, limit: int = 10000) -> bytes:
        """Execute SQL and return the result as raw Parquet bytes."""
        r = httpx.post(
            f"{self._base}/query",
            headers={**self._headers, "Accept": "application/octet-stream"},
            json={"sql": sql, "limit": limit},
            timeout=self._timeout,
            verify=False
        )
        r.raise_for_status()
        return r.content

    # ── Query (streaming) ─────────────────────────────────────────────────────

    def query_stream(self, sql: str) -> Iterator[pa.RecordBatch]:
        """Execute SQL and stream results as PyArrow RecordBatches.

        Memory-efficient for large result sets — yields one batch at a time
        without loading the full result into memory.

        Example::

            for batch in client.query_stream("SELECT * FROM parsed_emails"):
                df = batch.to_pandas()
                process(df)
        """
        max_attempts = 3
        base_delay_seconds = 1.0
        retryable_status_codes = {502, 503, 504}

        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.stream(
                    "POST",
                    f"{self._base}/query/stream",
                    headers={**self._headers, "Accept": "application/vnd.apache.arrow.stream"},
                    json={"sql": sql},
                    timeout=None,  # streaming — no timeout
                    verify=False,
                ) as response:
                    response.raise_for_status()
                    buf = io.BytesIO()
                    for chunk in response.iter_bytes():
                        buf.write(chunk)
                    buf.seek(0)
                    reader = ipc.open_stream(buf)
                    for batch in reader:
                        yield batch
                return
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code not in retryable_status_codes or attempt >= max_attempts:
                    raise
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt >= max_attempts:
                    raise

            sleep_seconds = base_delay_seconds * (2 ** (attempt - 1))
            time.sleep(sleep_seconds)

    def query_stream_arrow(self, sql: str) -> pa.Table:
        """Execute SQL via streaming and collect the full result as a PyArrow Table.

        Useful when you want Arrow format but don't need to process batch-by-batch.
        """
        batches = list(self.query_stream(sql))
        if not batches:
            return pa.table({})
        return pa.Table.from_batches(batches)