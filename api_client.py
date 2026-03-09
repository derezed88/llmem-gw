"""
AgentClient — async HTTP client for llmem-gw API plugin (plugin_client_api).

Protocol (plugin_client_api.py):
  POST /api/v1/submit  {"text":..., "client_id":..., "wait":true/false, "timeout":N}
  GET  /api/v1/stream/{client_id}   SSE: event:tok data:{"text":"..."}, event:done, event:error
  GET  /api/v1/health
  GET  /api/v1/sessions
  Auth: Authorization: Bearer <key>

Usage:
  client = AgentClient("http://localhost:8767", client_id="my-session", api_key="...")
  result = await client.send("hello", timeout=30)
  async for chunk in client.stream("hello", timeout=30):
      print(chunk, end="", flush=True)
"""

import asyncio
import json
import os
import time
from typing import AsyncGenerator, Optional

import httpx


class AgentClient:
    def __init__(
        self,
        base_url: str,
        client_id: str = "",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or f"api-{os.urandom(4).hex()}"
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(self, message: str, timeout: int = 60) -> str:
        """
        Submit a message synchronously (wait=true).
        Returns the full text response as a string.
        Raises httpx.HTTPError or RuntimeError on failure.
        """
        payload = {
            "text": message,
            "client_id": self.client_id,
            "wait": True,
            "timeout": timeout,
        }
        async with httpx.AsyncClient(timeout=timeout + 10) as http:
            resp = await http.post(
                f"{self.base_url}/api/v1/submit",
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status", "")
        if status == "error":
            raise RuntimeError(f"Agent error: {data.get('text', '')}")
        if status == "timeout":
            raise asyncio.TimeoutError(f"Agent timed out: {data.get('text', '')}")
        return data.get("text", "")

    async def stream(self, message: str, timeout: int = 60) -> AsyncGenerator[str, None]:
        """
        Submit a message asynchronously, then consume the SSE stream.
        Yields text chunks as they arrive; raises RuntimeError on error events.
        """
        # 1. Fire the request (wait=false)
        payload = {
            "text": message,
            "client_id": self.client_id,
            "wait": False,
        }
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(
                f"{self.base_url}/api/v1/submit",
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()

        # 2. Connect to SSE stream and yield chunks
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=timeout + 10) as http:
            async with http.stream(
                "GET",
                f"{self.base_url}/api/v1/stream/{self.client_id}",
                headers=self._headers,
            ) as resp:
                resp.raise_for_status()
                event_type = "message"
                async for raw_line in resp.aiter_lines():
                    if time.monotonic() > deadline:
                        raise asyncio.TimeoutError("stream timeout")
                    line = raw_line.strip()
                    if not line:
                        event_type = "message"  # reset after blank separator
                        continue
                    if line.startswith(":"):
                        continue  # keepalive comment
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if event_type == "tok":
                            try:
                                yield json.loads(data_str).get("text", "")
                            except json.JSONDecodeError:
                                yield data_str
                        elif event_type == "done":
                            return
                        elif event_type == "error":
                            try:
                                msg = json.loads(data_str).get("message", data_str)
                            except json.JSONDecodeError:
                                msg = data_str
                            raise RuntimeError(f"Agent error: {msg}")
                        # flush events: ignore (stream stays open until done)

    async def health(self) -> dict:
        """GET /api/v1/health → dict (expects {"status": "ok", ...})"""
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                f"{self.base_url}/api/v1/health",
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def sessions(self) -> list:
        """GET /api/v1/sessions → list of session dicts"""
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                f"{self.base_url}/api/v1/sessions",
                headers=self._headers,
            )
            resp.raise_for_status()
            return resp.json().get("sessions", [])
