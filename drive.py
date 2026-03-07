import os
import io
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

def _get_drive_service():
    global _drive_service
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

    _drive_service = build("drive", "v3", credentials=creds)
    return _drive_service

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

def _drive_read_file(file_id: str) -> str:
    svc = _get_drive_service()
    request = svc.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue().decode("utf-8")

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
        # If we have a configured folder ID, only accept it — reject anything else the LLM invents
        elif DRIVE_FOLDER_ID and folder_id != DRIVE_FOLDER_ID:
            log.warning("Ignoring folder_id %r that doesn't match configured FOLDER_ID (falling back to .env)", folder_id)
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
        elif op == "read":
            if not file_id: return "Error: file_id required."
            return await asyncio.to_thread(_drive_read_file, file_id)
        elif op == "append":
            if not file_id: return "Error: file_id required."
            if content is None: return "Error: content required."
            return await asyncio.to_thread(_drive_append_file, file_id, content)
        elif op == "delete":
            if not file_id: return "Error: file_id required."
            return await asyncio.to_thread(_drive_delete_file, file_id)
        else:
            return "Error: unknown operation. Valid: list|create|read|append|delete"
    except Exception as exc:
        log.exception("Drive op '%s' failed", op)
        return f"Drive error ({op}): {exc}"