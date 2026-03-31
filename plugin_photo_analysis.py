"""
Photo Analysis Plugin for llmem-gw

Analyzes photos/images using vision-capable LLMs. Supports three input sources
(local path, URL, base64 string) and routes by task type:

  general   — Gemini 2.5 Flash (default, cheapest at ~$0.000077/image)
  reasoning — Gemini 2.5 Flash with extended analysis prompt
  ocr       — Gemini 2.5 Flash with OCR-focused prompt

Claude Sonnet vision will be added once the anthropic SDK is installed.

Pricing:
  Gemini 2.5 Flash: $0.30/1M input tokens, $2.50/1M output tokens
  ~258 tokens per image → ~$0.000077/image

Wires into cost_events.py for per-call cost tracking.
"""

import asyncio
import base64
import io
import mimetypes
import os
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, model_validator
from langchain_core.tools import StructuredTool
from plugin_loader import BasePlugin
from config import log

# Default model for photo analysis
_PHOTO_MODEL = "gemini-2.5-flash"

# Approximate tokens per image for cost estimation
_GEMINI_TOKENS_PER_IMAGE = 258

_PROMPTS = {
    "general": (
        "Analyze this image thoroughly. Describe what you see: the main subjects, "
        "setting, notable objects, colors, mood, and any text visible. "
        "Be specific and detailed."
    ),
    "reasoning": (
        "Analyze this image in depth. Describe all visible elements, then interpret "
        "what is happening, infer context and purpose, note anything unusual or notable, "
        "and provide any relevant insights about the content."
    ),
    "ocr": (
        "Extract all text visible in this image exactly as written, preserving formatting "
        "where possible. Include all text — labels, captions, signs, handwriting. "
        "If no text is present, say so briefly."
    ),
}


class _PhotoAnalysisArgs(BaseModel):
    prompt: str = Field(
        description=(
            "What to analyze or ask about the image. "
            "Examples: 'describe what you see', 'extract all text', 'what brand is this?', "
            "'identify the objects in this photo'. "
            "Overrides task_type default prompt if provided."
        )
    )
    file_id: Optional[str] = Field(
        default=None,
        description="Google Drive file ID (downloads via Drive API)"
    )
    local_path: Optional[str] = Field(
        default=None,
        description="Absolute path to an image file on the local filesystem"
    )
    url: Optional[str] = Field(
        default=None,
        description="HTTP/HTTPS URL pointing directly to an image"
    )
    image_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (JPEG, PNG, WebP, GIF supported)"
    )
    mime_type: Optional[str] = Field(
        default=None,
        description=(
            "MIME type for base64 input (e.g. 'image/jpeg', 'image/png'). "
            "Auto-detected for local_path, url, and file_id inputs."
        )
    )
    task_type: str = Field(
        default="general",
        description=(
            "Analysis type: 'general' (describe image), 'reasoning' (deep analysis), "
            "'ocr' (extract text). Determines default prompt if prompt is empty."
        )
    )

    @model_validator(mode="after")
    def _exactly_one_source(self):
        sources = [s for s in [self.file_id, self.local_path, self.url, self.image_b64] if s]
        if len(sources) == 0:
            raise ValueError("Provide exactly one of: file_id, local_path, url, image_b64")
        if len(sources) > 1:
            raise ValueError("Provide only one of: file_id, local_path, url, image_b64 — not multiple")
        return self


async def _fetch_image_bytes(
    local_path: Optional[str],
    url: Optional[str],
    image_b64: Optional[str],
    mime_type: Optional[str],
    file_id: Optional[str] = None,
) -> tuple[bytes, str]:
    """Fetch image bytes and MIME type from the given source. Returns (bytes, mime_type)."""
    if file_id:
        from drive import _drive_download_bytes
        data, detected_mime, _ = await asyncio.to_thread(_drive_download_bytes, file_id)
        return data, detected_mime or mime_type or "image/jpeg"

    if local_path:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Image file not found: {local_path}")
        mime, _ = mimetypes.guess_type(local_path)
        mime = mime or "image/jpeg"
        with open(local_path, "rb") as fh:
            return fh.read(), mime

    if url:
        import httpx
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            detected_mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            return resp.content, detected_mime

    if image_b64:
        # Strip data URI prefix if present: data:image/jpeg;base64,...
        b64_data = image_b64
        detected_mime = mime_type or "image/jpeg"
        if image_b64.startswith("data:"):
            header, _, b64_data = image_b64.partition(",")
            # Extract MIME from header: data:image/png;base64
            if ";" in header:
                detected_mime = header.split(":")[1].split(";")[0]
        return base64.b64decode(b64_data), detected_mime

    raise ValueError("No image source provided")


async def _analyze_with_gemini(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
) -> tuple[str, int, int]:
    """
    Analyze image with Gemini 2.5 Flash using inline image data.
    Returns (response_text, tokens_in, tokens_out).
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set in .env", 0, 0

    client = genai.Client(api_key=api_key)

    # Use inline data for images (avoids Files API overhead for small images)
    b64_data = base64.b64encode(image_bytes).decode("utf-8")

    def _infer():
        return client.models.generate_content(
            model=_PHOTO_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
            ),
        )

    log.info("photo_analysis: analyzing %d bytes (%s) with %s", len(image_bytes), mime_type, _PHOTO_MODEL)
    response = await asyncio.to_thread(_infer)

    # Extract token usage if available
    tokens_in = _GEMINI_TOKENS_PER_IMAGE  # image base cost
    tokens_out = 0
    try:
        usage = response.usage_metadata
        if usage:
            tokens_in = getattr(usage, "prompt_token_count", _GEMINI_TOKENS_PER_IMAGE)
            tokens_out = getattr(usage, "candidates_token_count", 0)
    except Exception:
        pass

    return response.text, tokens_in, tokens_out


async def analyze_photo_executor(
    prompt: str,
    file_id: Optional[str] = None,
    local_path: Optional[str] = None,
    url: Optional[str] = None,
    image_b64: Optional[str] = None,
    mime_type: Optional[str] = None,
    task_type: str = "general",
) -> str:
    """Execute photo analysis via Gemini vision."""
    # Use task_type default prompt if prompt is generic/empty
    effective_prompt = prompt.strip() if prompt.strip() else _PROMPTS.get(task_type, _PROMPTS["general"])

    try:
        image_bytes, detected_mime = await _fetch_image_bytes(local_path, url, image_b64, mime_type, file_id)
    except Exception as e:
        return f"photo_analysis: failed to load image: {e}"

    try:
        result, tokens_in, tokens_out = await _analyze_with_gemini(image_bytes, detected_mime, effective_prompt)
    except Exception as e:
        log.exception("photo_analysis: Gemini call failed")
        return f"photo_analysis error: {e}"

    # Log cost event
    try:
        from cost_events import log_cost_event, estimate_gemini_cost
        cost = estimate_gemini_cost(_PHOTO_MODEL, tokens_in, tokens_out)
        await log_cost_event(
            provider="google",
            service="gemini-vision",
            tool_name="analyze_photo",
            model_key=_PHOTO_MODEL,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            unit="image",
            unit_count=1,
            notes=f"task_type={task_type} mime={detected_mime}",
        )
    except Exception as e:
        log.debug("photo_analysis: cost tracking failed: %s", e)

    return result


class PhotoAnalysisPlugin(BasePlugin):
    """Photo/image analysis plugin via Gemini 2.5 Flash vision."""

    PLUGIN_NAME = "plugin_photo_analysis"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_TYPE = "data_tool"
    DESCRIPTION = "Photo and image analysis via Gemini 2.5 Flash vision (general, reasoning, OCR tasks)"
    DEPENDENCIES = ["google-genai", "httpx"]
    ENV_VARS = ["GEMINI_API_KEY"]

    def __init__(self):
        self.enabled = False

    def init(self, config: dict) -> bool:
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("PhotoAnalysis plugin: GEMINI_API_KEY not set in .env")
                return False
            self.enabled = True
            return True
        except ImportError as e:
            print(f"PhotoAnalysis plugin: google-genai not installed: {e}")
            return False
        except Exception as e:
            print(f"PhotoAnalysis plugin init failed: {e}")
            return False

    def shutdown(self) -> None:
        self.enabled = False

    def get_tools(self) -> Dict[str, Any]:
        return {
            "lc": [
                StructuredTool.from_function(
                    coroutine=analyze_photo_executor,
                    name="analyze_photo",
                    description=(
                        "Analyze a photo or image using Gemini 2.5 Flash vision. "
                        "Provide exactly ONE image source: "
                        "file_id (Google Drive file ID), "
                        "local_path (absolute path on this server), "
                        "url (HTTP/HTTPS URL to image), or "
                        "image_b64 (base64-encoded image data). "
                        "task_type: 'general' (describe image, default), "
                        "'reasoning' (deep analysis), 'ocr' (extract text). "
                        "prompt overrides the default task description if provided."
                    ),
                    args_schema=_PhotoAnalysisArgs,
                )
            ]
        }
