"""
Gemini File Extract Plugin for MCP Agent

Provides file_extract tool for extracting/interpreting content from any file
via the Gemini Files API + gemini-2.0-flash multimodal inference.

Supports three mutually exclusive sources (provide exactly one):
  file_id    — Google Drive file ID (downloads via Drive API)
  url        — HTTP/HTTPS URL (fetched by llmem-gw)
  local_path — Absolute path on the local filesystem

Optional prompt parameter allows focused extraction (default: full content extraction).

Requires GEMINI_API_KEY in .env.
Uses extractor-gemini model entry from llm-models.json (gemini-2.0-flash, temp=0.0).
"""

import asyncio
import io
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from config import log

_DEFAULT_PROMPT = (
    "Extract and return the complete content of this file as plain text. "
    "Preserve structure such as headings, tables, and lists. "
    "Do not summarize — return everything."
)

_EXTRACTOR_MODEL = "gemini-2.0-flash"


class _FileExtractArgs(BaseModel):
    file_id: Optional[str] = Field(
        default=None,
        description="Google Drive file ID to download and extract"
    )
    url: Optional[str] = Field(
        default=None,
        description="HTTP/HTTPS URL to fetch and extract"
    )
    local_path: Optional[str] = Field(
        default=None,
        description="Absolute local filesystem path to read and extract"
    )
    prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional extraction prompt. Default: full content extraction. "
            "Examples: 'summarize the key points', 'extract all tables', "
            "'what are the action items?'"
        )
    )

    @model_validator(mode="after")
    def _exactly_one_source(self):
        sources = [s for s in [self.file_id, self.url, self.local_path] if s]
        if len(sources) == 0:
            raise ValueError("Provide exactly one of: file_id, url, local_path")
        if len(sources) > 1:
            raise ValueError("Provide only one of: file_id, url, local_path — not multiple")
        return self


async def _fetch_bytes_from_url(url: str) -> tuple[bytes, str, str]:
    """Fetch bytes from a URL. Returns (bytes, mime_type, filename)."""
    import httpx
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "application/octet-stream").split(";")[0].strip()
        # Guess filename from URL path
        name = url.rstrip("/").split("/")[-1] or "file"
        return resp.content, mime, name


def _fetch_bytes_from_local(path: str) -> tuple[bytes, str, str]:
    """Read bytes from local filesystem. Returns (bytes, mime_type, filename)."""
    import mimetypes
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Local file not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    name = os.path.basename(path)
    with open(path, "rb") as fh:
        return fh.read(), mime, name


async def _extract_with_gemini(data: bytes, mime_type: str, file_name: str, prompt: str) -> str:
    """Upload bytes to Gemini Files API and run inference. Returns extracted text."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set in .env"

    client = genai.Client(api_key=api_key)

    # Upload to Gemini Files API
    def _upload():
        return client.files.upload(
            file=io.BytesIO(data),
            config=types.UploadFileConfig(
                mime_type=mime_type,
                display_name=file_name,
            ),
        )

    log.info("file_extract: uploading %s (%s, %d bytes) to Gemini Files API", file_name, mime_type, len(data))
    gemini_file = await asyncio.to_thread(_upload)

    try:
        def _infer():
            return client.models.generate_content(
                model=_EXTRACTOR_MODEL,
                contents=[
                    types.Part.from_uri(file_uri=gemini_file.uri, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                ),
            )

        response = await asyncio.to_thread(_infer)
        return response.text

    finally:
        def _delete():
            try:
                client.files.delete(name=gemini_file.name)
                log.info("file_extract: deleted Gemini file %s", gemini_file.name)
            except Exception as e:
                log.warning("file_extract: could not delete Gemini file %s: %s", gemini_file.name, e)

        await asyncio.to_thread(_delete)


async def file_extract_executor(
    file_id: Optional[str] = None,
    url: Optional[str] = None,
    local_path: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """Execute file extraction via Gemini multimodal inference."""
    effective_prompt = prompt or _DEFAULT_PROMPT

    try:
        if file_id:
            from drive import (
                _drive_download_bytes, _drive_export_pdf_bytes,
                _GOOGLE_APPS_PDF_EXPORTABLE, _doc_has_images, _get_drive_service,
            )
            meta = await asyncio.to_thread(
                lambda: _get_drive_service().files().get(fileId=file_id, fields="mimeType").execute()
            )
            if meta.get("mimeType") in _GOOGLE_APPS_PDF_EXPORTABLE:
                has_images = await asyncio.to_thread(_doc_has_images, file_id)
                if has_images:
                    data, mime, name = await asyncio.to_thread(_drive_export_pdf_bytes, file_id)
                    log.info("file_extract: Google Doc has images — exported as PDF for image-aware analysis")
                else:
                    data, mime, name = await asyncio.to_thread(_drive_download_bytes, file_id)
                    log.info("file_extract: Google Doc has no images — using text export")
            else:
                data, mime, name = await asyncio.to_thread(_drive_download_bytes, file_id)
            source_label = f"Drive:{file_id}"
        elif url:
            data, mime, name = await _fetch_bytes_from_url(url)
            source_label = f"URL:{url}"
        elif local_path:
            data, mime, name = await asyncio.to_thread(_fetch_bytes_from_local, local_path)
            source_label = f"local:{local_path}"
        else:
            return "Error: no source provided"

        log.info("file_extract: source=%s name=%s mime=%s", source_label, name, mime)
        return await _extract_with_gemini(data, mime, name, effective_prompt)

    except Exception as exc:
        log.exception("file_extract failed")
        return f"file_extract error: {exc}"


class ExtractGeminiPlugin(BasePlugin):
    """Gemini file extraction plugin — supports Drive, URL, and local files."""

    PLUGIN_NAME = "plugin_extract_gemini"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "File content extraction via Gemini Files API (Drive, URL, or local)"
    DEPENDENCIES = ["google-genai", "httpx"]
    ENV_VARS = ["GEMINI_API_KEY"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("ExtractGemini plugin: GEMINI_API_KEY not set in .env")
                return False
            self.enabled = True
            return True
        except ImportError as e:
            print(f"ExtractGemini plugin: google-generativeai not installed: {e}")
            return False
        except Exception as e:
            print(f"ExtractGemini plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=file_extract_executor,
                    name="file_extract",
                    description=(
                        "Extract or interpret the content of any file using Gemini multimodal AI. "
                        "Handles PDFs, images, audio, video, Office docs, plain text, and more. "
                        "Provide exactly ONE source: "
                        "file_id (Google Drive file ID), "
                        "url (HTTP/HTTPS URL to the file), or "
                        "local_path (absolute path on this server). "
                        "Optional prompt focuses the extraction (default: full content as plain text). "
                        "Examples: 'summarize key points', 'extract all tables', 'transcribe audio'."
                    ),
                    args_schema=_FileExtractArgs,
                )
            ]
        }
