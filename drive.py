import os
import io
import base64
import asyncio
from google.auth.transport.requests import Request as GAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# Absolute imports
from config import log, DRIVE_FOLDER_ID, DRIVE_SCOPES, DRIVE_TOKEN_FILE, DRIVE_CREDS_FILE
from state import current_client_id, sessions

_drive_service = None
_docs_service = None
_verified_subfolders: set[str] = {
    "1JUe7bjxPAuKKxCC9kV2H9eLVMUFkJStl",  # photos subfolder — pre-seeded to survive restarts
}  # cache of folder IDs confirmed as children of DRIVE_FOLDER_ID

def _get_creds():
    """Load and refresh Google OAuth credentials from token file."""
    creds = None
    if os.path.exists(DRIVE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(DRIVE_TOKEN_FILE, DRIVE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GAuthRequest())
        else:
            if not os.path.exists(DRIVE_CREDS_FILE):
                raise FileNotFoundError(
                    "Missing 'credentials.json'. Download it from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(DRIVE_CREDS_FILE, DRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(DRIVE_TOKEN_FILE, "w") as fh:
            fh.write(creds.to_json())
    return creds

def _get_drive_service():
    global _drive_service
    _drive_service = build("drive", "v3", credentials=_get_creds())
    return _drive_service

def _get_docs_service():
    global _docs_service
    _docs_service = build("docs", "v1", credentials=_get_creds())
    return _docs_service

def _doc_has_images(file_id: str) -> bool:
    """Check if a Google Doc has embedded images without downloading it.

    Uses the Docs API to inspect inlineObjects and positionedObjects —
    both are populated only when images (or other embedded objects) exist.
    Returns False for non-Doc types (caller should check MIME before calling).
    """
    svc = _get_docs_service()
    doc = svc.documents().get(documentId=file_id, fields="inlineObjects,positionedObjects").execute()
    return bool(doc.get("inlineObjects") or doc.get("positionedObjects"))

def _drive_list_files(folder_id: str) -> str:
    svc = _get_drive_service()
    # Sanity check: ensure folder_id is clean
    folder_id = folder_id.replace("'", "").replace('"', "")
    
    query = f"'{folder_id}' in parents and trashed = false"
    results = svc.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get("files", [])
    if not items:
        return f"(Folder {folder_id} is empty)"
    lines = [f"Contents of folder {folder_id}:"]
    for item in items:
        tag = "[DIR] " if item["mimeType"] == "application/vnd.google-apps.folder" else "[FILE]"
        lines.append(f"  {tag} {item['name']}  (id: {item['id']})")
    return "\n".join(lines)

def _drive_create_file(name: str, content: str, folder_id: str) -> str:
    svc = _get_drive_service()
    metadata = {"name": name, "parents": [folder_id]}
    media = MediaIoBaseUpload(io.BytesIO(content.encode("utf-8")), mimetype="text/plain", resumable=True)
    file = svc.files().create(body=metadata, media_body=media, fields="id").execute()
    return f"Created '{name}' — id: {file.get('id')}"


def _drive_create_folder(name: str, parent_folder_id: str) -> str:
    """Create a subfolder in Drive. Returns the new folder's ID."""
    svc = _get_drive_service()
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = svc.files().create(body=metadata, fields="id").execute()
    folder_id = folder.get("id")
    _verified_subfolders.add(folder_id)
    return folder_id


def _drive_get_or_create_folder(name: str, parent_folder_id: str) -> str:
    """Return the ID of a named subfolder under parent, creating it if absent."""
    svc = _get_drive_service()
    parent_folder_id = parent_folder_id.replace("'", "").replace('"', "")
    query = (
        f"'{parent_folder_id}' in parents and "
        f"name = '{name}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    results = svc.files().list(q=query, fields="files(id)").execute()
    items = results.get("files", [])
    if items:
        folder_id = items[0]["id"]
        _verified_subfolders.add(folder_id)
        return folder_id
    return _drive_create_folder(name, parent_folder_id)


def _drive_create_image(name: str, image_b64: str, folder_id: str, mime_type: str = "image/jpeg") -> str:
    """Upload a base64-encoded image to Google Drive."""
    svc = _get_drive_service()
    # Strip data URI prefix if present
    if "," in image_b64 and image_b64.index(",") < 80:
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    metadata = {"name": name, "parents": [folder_id]}
    media = MediaIoBaseUpload(io.BytesIO(raw), mimetype=mime_type, resumable=True)
    file = svc.files().create(body=metadata, media_body=media, fields="id").execute()
    return f"Uploaded '{name}' — id: {file.get('id')}"

def _drive_read_file(file_id: str) -> str:
    data, mime, name = _drive_download_bytes(file_id)
    # Binary files (PDFs, images, etc.) can't be decoded as text
    if not mime.startswith("text/") and mime not in (
        "application/json", "application/xml", "application/javascript",
    ):
        return (
            f"Cannot read binary file '{name}' (mime={mime}) as text. "
            f"Use the file_extract tool with file_id='{file_id}' instead — "
            f"it handles PDFs, images, and other binary formats via Gemini."
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return (
            f"Cannot decode '{name}' as UTF-8 text. "
            f"Use the file_extract tool with file_id='{file_id}' instead."
        )


_GOOGLE_APPS_EXPORT = {
    "application/vnd.google-apps.document":     ("text/plain",  ".txt"),
    "application/vnd.google-apps.spreadsheet":  ("text/csv",    ".csv"),
    "application/vnd.google-apps.presentation": ("text/plain",  ".txt"),
    "application/vnd.google-apps.drawing":      ("image/png",   ".png"),
}

# Google Workspace types that can be exported as PDF (preserves images/formatting)
_GOOGLE_APPS_PDF_EXPORTABLE = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.spreadsheet",
}

def _drive_export_pdf_bytes(file_id: str) -> tuple[bytes, str, str]:
    """Export a Google Workspace file as PDF. Returns (pdf_bytes, 'application/pdf', filename).

    Use this instead of _drive_download_bytes when images and layout must be preserved
    (e.g. for Gemini multimodal analysis). Only valid for Docs, Slides, and Sheets.
    """
    svc = _get_drive_service()
    meta = svc.files().get(fileId=file_id, fields="mimeType,name").execute()
    mime = meta.get("mimeType", "")
    name = meta.get("name", file_id)
    if mime not in _GOOGLE_APPS_PDF_EXPORTABLE:
        raise ValueError(f"File '{name}' (mime={mime}) is not a Google Workspace type — use _drive_download_bytes instead")
    if not name.endswith(".pdf"):
        name = name + ".pdf"
    data = svc.files().export(fileId=file_id, mimeType="application/pdf").execute()
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data, "application/pdf", name

def _drive_download_bytes(file_id: str) -> tuple[bytes, str, str]:
    """Download raw bytes from Drive. Returns (bytes, mime_type, file_name).

    Native Google Workspace types (Docs, Sheets, Slides, Drawings) are exported
    to a compatible format automatically via files().export().
    """
    svc = _get_drive_service()
    meta = svc.files().get(fileId=file_id, fields="mimeType,name").execute()
    mime = meta.get("mimeType", "application/octet-stream")
    name = meta.get("name", file_id)

    if mime in _GOOGLE_APPS_EXPORT:
        export_mime, ext = _GOOGLE_APPS_EXPORT[mime]
        if not name.endswith(ext):
            name = name + ext
        data = svc.files().export(fileId=file_id, mimeType=export_mime).execute()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return data, export_mime, name

    request = svc.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue(), mime, name

def _drive_append_file(file_id: str, new_text: str) -> str:
    current = _drive_read_file(file_id)
    updated = current.rstrip("\n") + "\n" + new_text
    svc = _get_drive_service()
    media = MediaIoBaseUpload(io.BytesIO(updated.encode("utf-8")), mimetype="text/plain", resumable=True)
    svc.files().update(fileId=file_id, media_body=media).execute()
    return f"Appended to file id: {file_id}"

def _drive_delete_file(file_id: str) -> str:
    svc = _get_drive_service()
    svc.files().delete(fileId=file_id).execute()
    return f"Deleted file id: {file_id}"

def _drive_move_file(file_id: str, dest_folder_id: str) -> str:
    svc = _get_drive_service()
    # Get current parents
    f = svc.files().get(fileId=file_id, fields="parents,name").execute()
    prev_parents = ",".join(f.get("parents", []))
    svc.files().update(
        fileId=file_id,
        addParents=dest_folder_id,
        removeParents=prev_parents,
        fields="id, parents"
    ).execute()
    return f"Moved '{f.get('name', file_id)}' to folder {dest_folder_id}"

def _is_subfolder_of_root(folder_id: str) -> bool:
    """Check if folder_id is a direct child folder of DRIVE_FOLDER_ID via the Drive API."""
    try:
        svc = _get_drive_service()
        meta = svc.files().get(fileId=folder_id, fields="parents,mimeType").execute()
        is_folder = meta.get("mimeType") == "application/vnd.google-apps.folder"
        parents = meta.get("parents", [])
        return is_folder and DRIVE_FOLDER_ID in parents
    except Exception as e:
        log.warning("Failed to verify subfolder %r: %s", folder_id, e)
        return False

async def run_drive_op(operation: str, file_id: str | None, file_name: str | None, content: str | None, folder_id: str | None) -> str:
    op = (operation or "").strip().lower()

    # --- INPUT SANITIZATION ---
    # The LLM often sends "<your-folder-id>" or "{{folder_id}}" or descriptive names instead of omitting the argument.
    # We must detect this garbage and discard it so we fall back to the .env variable.
    if folder_id:
        bad_patterns = ["<", ">", "{", "}", "your-folder-id", "placeholder", "root",
                        "folder_id", "folder-id", "chat_history", "chat-history", "configured"]
        if any(p in folder_id.lower() for p in bad_patterns):
            log.warning("Ignoring invalid folder_id argument from LLM: %r (falling back to .env)", folder_id)
            folder_id = None
        # Real Google Drive IDs are alphanumeric+dash+underscore, typically 28-44 chars
        elif not all(c.isalnum() or c in "-_" for c in folder_id):
            log.warning("Ignoring folder_id with invalid characters: %r (falling back to .env)", folder_id)
            folder_id = None
        # Accept the configured root folder, or verified subfolders of it
        elif DRIVE_FOLDER_ID and folder_id != DRIVE_FOLDER_ID:
            if folder_id in _verified_subfolders:
                pass  # already confirmed as a child of root
            elif _is_subfolder_of_root(folder_id):
                _verified_subfolders.add(folder_id)
                log.info("Verified folder_id %r as subfolder of root — cached", folder_id)
            else:
                log.warning("Ignoring folder_id %r that is not a subfolder of configured FOLDER_ID (falling back to .env)", folder_id)
                folder_id = None

    # Resolve ID: 1. Valid Arg -> 2. Env Var -> 3. Empty
    fid = (folder_id or DRIVE_FOLDER_ID or "").strip()
    
    # Final check
    if not fid:
        return "Configuration Error: No valid folder_id found. Check FOLDER_ID in .env."

    _cid = current_client_id.get("")
    _model = sessions.get(_cid, {}).get("model", "?") if _cid else "?"
    log.info("Drive op='%s' model=%s client=%s folder_id='%s'", op, _model, _cid, fid)

    try:
        if op == "list":
            return await asyncio.to_thread(_drive_list_files, fid)
        elif op == "create":
            if not file_name: return "Error: file_name required for 'create'."
            return await asyncio.to_thread(_drive_create_file, file_name, content or "", fid)
        elif op == "create_image":
            if not file_name: return "Error: file_name required for 'create_image'."
            if not content: return "Error: content (base64 image data) required for 'create_image'."
            return await asyncio.to_thread(_drive_create_image, file_name, content, fid)
        elif op == "read":
            if not file_id: return "Error: file_id required."
            return await asyncio.to_thread(_drive_read_file, file_id)
        elif op == "append":
            if not file_id: return "Error: file_id required."
            if content is None: return "Error: content required."
            return await asyncio.to_thread(_drive_append_file, file_id, content)
        elif op == "move":
            if not file_id: return "Error: file_id required for 'move'."
            if not folder_id: return "Error: folder_id (destination) required for 'move'."
            return await asyncio.to_thread(_drive_move_file, file_id, fid)
        elif op == "delete":
            if not file_id: return "Error: file_id required."
            return await asyncio.to_thread(_drive_delete_file, file_id)
        else:
            return "Error: unknown operation. Valid: list|create|read|append|delete|move"
    except Exception as exc:
        log.exception("Drive op '%s' failed", op)
        return f"Drive error ({op}): {exc}"