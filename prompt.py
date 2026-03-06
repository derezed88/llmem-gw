import os
import shutil
from typing import List, Dict, Optional, Tuple, Set
from config import log, SYSTEM_PROMPT_FILE, BASE_DIR, LLM_MODELS_FILE

# ---------------------------------------------------------------------------
# SECTIONAL SYSTEM PROMPT DESIGN
# ---------------------------------------------------------------------------
# The system prompt is a tree of section files:
#
# .system_prompt                  ← root: main paragraph + [SECTIONS] list
#   .system_prompt_tools          ← leaf: body text  OR container with [SECTIONS]
#     .system_prompt_tool_url_extract   ← leaf: body text
#     .system_prompt_tool_db_query      ← leaf: body text
#   .system_prompt_behavior       ← leaf: body text
#
# Rules:
# - A section file can have body text AND a [SECTIONS] list.
#   Text before [SECTIONS] is used as the section's own body (preamble).
#   Text after [SECTIONS] lists child sections to recurse into.
# - All sections at all depths are registered in the flat _sections list.
# - Duplicate section names across the tree are rejected (loop detection also
#   catches cross-branch reuse of the same name).
# - Loop detection: if a section name appears in its own ancestor chain,
#   it is skipped and an error placeholder is inserted.
#
# LLM tools:
# - update_system_prompt(section_name, operation, ...) edits specific sections
# - read_system_prompt() returns full assembled prompt
# - read_system_prompt(section) returns specific section
# ---------------------------------------------------------------------------

DEFAULT_MAIN_PROMPT = """\
You are Robot, an AI assistant with persistent memory via a MySQL database.
You are an autonomous agent. When you need information, you MUST use a tool call. Do not explain your steps. Do not provide Markdown code blocks. If you have a tool available for a task, use it immediately.

[SECTIONS]
memory-hierarchy: Staged Memory Hierarchy (PDDS chain)
tool-guardrails: Refined Tool Usage Guardrails
tools: Available Tool Definitions
tool-logging: Tool Usage Logging & Review
time-bypass: Direct Time Bypass
db-guardrails: Refined db_query Guardrails
behavior: Behaviour rules
"""

# In-memory cache of prompt structure.
# Flat list — all sections at all depths, in tree-traversal order.
# Each entry: {short-section-name, description, body, depth, parent}
_main_paragraph: str = ""
_sections: List[Dict] = []
_cached_full_prompt: Optional[str] = None


SYSTEM_PROMPT_BASE = os.path.join(BASE_DIR, "system_prompt")


def _get_section_file_path(section_name: str, folder: Optional[str] = None) -> str:
    """Return the file path for a given section name.

    If folder is provided, use it as the base directory.
    Otherwise fall back to the directory of SYSTEM_PROMPT_FILE.
    """
    base_dir = folder if folder else os.path.dirname(SYSTEM_PROMPT_FILE)
    return os.path.join(base_dir, f".system_prompt_{section_name}")


def _parse_sections_block(content: str) -> Tuple[Optional[List[Tuple[str, str]]], str]:
    """
    Parse content that may contain a [SECTIONS] marker.

    Returns:
      (section_list, body_text)
      - If [SECTIONS] found: section_list = [(name, desc), ...],
        body_text = text before [SECTIONS] (may be non-empty)
      - If no [SECTIONS]: section_list = None, body_text = content
    """
    lines = content.split('\n')
    sections_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == '[SECTIONS]':
            sections_idx = i
            break

    if sections_idx == -1:
        return None, content

    preamble = '\n'.join(lines[:sections_idx]).rstrip()

    section_list = []
    for line in lines[sections_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            parts = line.split(':', 1)
            short_name = parts[0].strip()
            description = parts[1].strip()
            section_list.append((short_name, description))

    return section_list, preamble


def _load_section_recursive(
    section_name: str,
    description: str,
    depth: int,
    parent: Optional[str],
    ancestors: Set[str],
    all_seen: Set[str],
    folder: Optional[str] = None,
) -> List[Dict]:
    """
    Load a section file recursively.

    Returns a flat list of section dicts in tree-traversal order.
    Each dict: {short-section-name, description, body, depth, parent}

    ancestors: set of section names in the current call stack (loop detection)
    all_seen: set of all section names registered so far (duplicate detection)
    folder: optional base folder path; overrides the default SYSTEM_PROMPT_FILE directory
    """
    # Loop detection
    if section_name in ancestors:
        log.error(
            f"Circular reference detected: '{section_name}' is already in ancestor chain "
            f"{sorted(ancestors)}. Skipping."
        )
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': f"[SECTION ERROR: circular reference to '{section_name}']",
            'depth': depth,
            'parent': parent,
        }]

    # Duplicate name detection (same name in two different branches)
    if section_name in all_seen:
        log.error(
            f"Duplicate section name '{section_name}' detected in prompt tree. "
            f"Each section name must be unique. Skipping second occurrence."
        )
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': f"[SECTION ERROR: duplicate section name '{section_name}']",
            'depth': depth,
            'parent': parent,
        }]

    all_seen.add(section_name)

    file_path = _get_section_file_path(section_name, folder)
    if not os.path.exists(file_path):
        log.warning(f"Section file not found: {file_path}")
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': "",
            'depth': depth,
            'parent': parent,
        }]

    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            raw = fh.read()
    except Exception as exc:
        log.warning(f"Could not read section file {file_path}: {exc}")
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': "",
            'depth': depth,
            'parent': parent,
        }]

    # Strip the ## header line if present
    lines = raw.split('\n', 1)
    content = lines[1] if len(lines) > 1 and lines[0].startswith('## ') else raw

    # Check if this is a container (has [SECTIONS]) or a leaf (body text)
    sub_section_list, body = _parse_sections_block(content)

    if sub_section_list is None:
        # Leaf node — plain body text
        return [{
            'short-section-name': section_name,
            'description': description,
            'body': body,
            'depth': depth,
            'parent': parent,
        }]
    else:
        # Container node — recurse into children.
        # body holds any text written before [SECTIONS] in the file.
        result = [{
            'short-section-name': section_name,
            'description': description,
            'body': body,
            'depth': depth,
            'parent': parent,
            'is_container': True,
        }]
        new_ancestors = ancestors | {section_name}
        for child_name, child_desc in sub_section_list:
            child_sections = _load_section_recursive(
                child_name, child_desc, depth + 1, section_name,
                new_ancestors, all_seen, folder=folder
            )
            result.extend(child_sections)
        return result


def _parse_main_prompt(content: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Parse main .system_prompt file into:
    - main_paragraph (before [SECTIONS])
    - section_list [(short-name, description), ...]
    """
    lines = content.split('\n')
    sections_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == '[SECTIONS]':
            sections_idx = i
            break

    if sections_idx == -1:
        return content.strip(), []

    main_paragraph = '\n'.join(lines[:sections_idx]).strip()
    section_list = []
    for line in lines[sections_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            parts = line.split(':', 1)
            short_name = parts[0].strip()
            description = parts[1].strip()
            section_list.append((short_name, description))

    return main_paragraph, section_list


def load_system_prompt() -> str:
    """
    Load the system prompt from disk recursively.
    Builds a flat _sections list from the full section tree.
    Uses the default SYSTEM_PROMPT_FILE path (backward compatible).
    """
    global _main_paragraph, _sections, _cached_full_prompt

    if not os.path.exists(SYSTEM_PROMPT_FILE):
        try:
            with open(SYSTEM_PROMPT_FILE, 'w', encoding='utf-8') as fh:
                fh.write(DEFAULT_MAIN_PROMPT)
        except Exception as exc:
            log.warning(f"Could not write default prompt: {exc}")
        content = DEFAULT_MAIN_PROMPT
    else:
        try:
            with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as fh:
                content = fh.read()
        except Exception as exc:
            log.warning(f"Could not read prompt: {exc}")
            content = DEFAULT_MAIN_PROMPT

    main_paragraph, section_list = _parse_main_prompt(content)
    _main_paragraph = main_paragraph

    all_seen: Set[str] = set()
    _sections = []
    for short_name, description in section_list:
        sections = _load_section_recursive(
            short_name, description, depth=0, parent=None,
            ancestors=set(), all_seen=all_seen
        )
        _sections.extend(sections)

    _cached_full_prompt = _assemble_full_prompt()

    total = len(_sections)
    leaves = sum(1 for s in _sections if not s.get('is_container'))
    log.info(
        f"Loaded system prompt: {len(_main_paragraph)} chars main, "
        f"{total} sections ({leaves} leaf, {total - leaves} container)"
    )
    return _cached_full_prompt


def load_prompt_for_folder(folder: str, active_tools: set[str] | None = None,
                           cold_tools: list[str] | None = None) -> str:
    """
    Stateless: load and assemble the system prompt from a specific folder.
    Does NOT mutate any global state — safe to call for any model at any time.

    The folder must contain a '.system_prompt' root file and optional
    '.system_prompt_<name>' section files.

    active_tools: if provided, sections whose sp_section maps to a tool section file
        are only included if that tool is in active_tools. Always-active toolset
        sections are always included regardless.
    cold_tools: if provided, a hint line listing cold tool names is appended to the
        tools section for model awareness.

    Returns the assembled prompt string.
    """
    root_file = os.path.join(folder, ".system_prompt")
    if not os.path.exists(root_file):
        log.warning(f"load_prompt_for_folder: no .system_prompt in {folder}")
        return DEFAULT_MAIN_PROMPT

    try:
        with open(root_file, 'r', encoding='utf-8') as fh:
            content = fh.read()
    except Exception as exc:
        log.warning(f"load_prompt_for_folder: could not read {root_file}: {exc}")
        return DEFAULT_MAIN_PROMPT

    main_paragraph, section_list = _parse_main_prompt(content)

    # Build a set of sp_section names that are active (from LLM_TOOLSET_META).
    # Sections not in this set that correspond to a tool section file are skipped.
    active_sp_sections: set[str] | None = None
    if active_tools is not None:
        from config import LLM_TOOLSET_META, LLM_TOOLSETS
        active_sp_sections = set()
        for ts_name, meta in LLM_TOOLSET_META.items():
            sp_sec = meta.get("sp_section")
            if not sp_sec:
                continue
            # Include this sp_section if any of its tools are in active_tools
            ts_tools = LLM_TOOLSETS.get(ts_name, [])
            if any(t in active_tools for t in ts_tools):
                active_sp_sections.add(sp_sec)

    all_seen: Set[str] = set()
    sections: List[Dict] = []
    for short_name, description in section_list:
        # If active_sp_sections is set and this section is a tool section not in it, skip
        if active_sp_sections is not None and short_name.startswith("tool_") and short_name not in active_sp_sections:
            continue
        loaded = _load_section_recursive(
            short_name, description, depth=0, parent=None,
            ancestors=set(), all_seen=all_seen, folder=folder
        )
        sections.extend(loaded)

    # Assemble without touching globals
    parts = [main_paragraph]
    for section in sections:
        depth = section.get('depth', 0)
        prefix = "##" + "#" * depth
        header = f"\n\n{prefix} {section['short-section-name']}: {section['description']}"
        body = section['body']
        if body:
            parts.append(header + "\n" + body)
        else:
            parts.append(header)

    # Append cold tool hint if any tools are cold
    if cold_tools:
        parts.append(f"\n\nCold tools available via tool_list: {', '.join(sorted(cold_tools))}")

    return '\n'.join(parts)


def _assemble_full_prompt() -> str:
    """Assemble the full prompt from main paragraph and flat sections list."""
    parts = [_main_paragraph]
    for section in _sections:
        depth = section.get('depth', 0)
        prefix = "##" + "#" * depth  # ## at depth 0, ### at depth 1, etc.
        header = f"\n\n{prefix} {section['short-section-name']}: {section['description']}"
        body = section['body']
        if body:
            parts.append(header + "\n" + body)
        else:
            parts.append(header)
    return '\n'.join(parts)


def get_current_prompt() -> str:
    """Return the currently cached full system prompt."""
    global _cached_full_prompt
    if _cached_full_prompt is None:
        return load_system_prompt()
    return _cached_full_prompt


def get_section(identifier: str) -> Optional[str]:
    """
    Get a specific section by index (e.g. "0") or name (e.g. "tool_url_extract").
    Searches all sections at all depths.
    Returns section content with header, or None if not found.
    """
    if _cached_full_prompt is None:
        load_system_prompt()

    try:
        idx = int(identifier)
        if 0 <= idx < len(_sections):
            section = _sections[idx]
            depth = section.get('depth', 0)
            prefix = "##" + "#" * depth
            header = f"{prefix} {section['short-section-name']}: {section['description']}"
            return header + "\n" + section['body']
        return None
    except ValueError:
        pass

    for section in _sections:
        if section['short-section-name'] == identifier:
            depth = section.get('depth', 0)
            prefix = "##" + "#" * depth
            header = f"{prefix} {section['short-section-name']}: {section['description']}"
            return header + "\n" + section['body']

    return None


def list_sections() -> List[Dict]:
    """Return metadata for all sections at all depths (without bodies)."""
    if _cached_full_prompt is None:
        load_system_prompt()

    return [
        {
            'index': i,
            'short-section-name': s['short-section-name'],
            'description': s['description'],
            'depth': s.get('depth', 0),
            'parent': s.get('parent'),
            'is_container': s.get('is_container', False),
        }
        for i, s in enumerate(_sections)
    ]


def _write_section_file(section_name: str, content: str, folder: Optional[str] = None) -> None:
    """Write content to a section file, including the ## header.

    For container sections (those with a [SECTIONS] block on disk), only the
    preamble (text before [SECTIONS]) is replaced; the [SECTIONS] block and
    everything after it is preserved.
    """
    file_path = _get_section_file_path(section_name, folder)

    description = ""
    is_container = False
    for section in _sections:
        if section['short-section-name'] == section_name:
            description = section['description']
            is_container = section.get('is_container', False)
            break

    header = f"## {section_name}: {description}\n"

    if is_container and os.path.exists(file_path):
        # Read existing file to recover the [SECTIONS] block
        try:
            with open(file_path, 'r', encoding='utf-8') as fh:
                existing = fh.read()
        except Exception:
            existing = ""
        # Strip leading ## header line if present
        stripped = existing.split('\n', 1)[1] if existing.startswith('## ') else existing
        # Find [SECTIONS] marker and keep everything from it onwards
        idx = stripped.find('\n[SECTIONS]')
        if idx == -1:
            # Fallback: no marker found, just write normally
            sections_tail = ""
        else:
            sections_tail = stripped[idx:]  # includes the \n[SECTIONS]\n... lines
        full_content = header + content + sections_tail
    else:
        full_content = header + content

    try:
        with open(file_path, 'w', encoding='utf-8') as fh:
            fh.write(full_content)
    except Exception as exc:
        log.error(f"Could not write section file {file_path}: {exc}")
        raise


def apply_prompt_operation(
    section_name: str,
    operation: str,
    content: str = "",
    target: str = "",
    confirm_overwrite: bool = False,
) -> Tuple[str, str]:
    """
    Perform a surgical edit on a specific system prompt section.
    Works on any section at any depth in the tree.

    Returns (new_section_text, status_message).
    Raises ValueError for invalid operations or missing arguments.
    """
    global _sections, _cached_full_prompt

    section_idx = -1
    for i, section in enumerate(_sections):
        if section['short-section-name'] == section_name:
            section_idx = i
            break

    if section_idx == -1:
        available = ', '.join(s['short-section-name'] for s in _sections)
        raise ValueError(
            f"Section '{section_name}' not found. "
            f"Available sections: {available}"
        )

    # Container sections are editable when they have a preamble body.
    # Only block if it's a container with no body at all.
    if _sections[section_idx].get('is_container') and not _sections[section_idx].get('body'):
        raise ValueError(
            f"Section '{section_name}' is a container (has sub-sections) with no body text. "
            f"Edit its child sections instead."
        )

    current_body = _sections[section_idx]['body']
    op = operation.strip().lower()

    if op == "append":
        if not content:
            raise ValueError("'append' requires non-empty content.")
        separator = "\n" if current_body.endswith("\n") else "\n\n"
        new_body = current_body + separator + content
        msg = f"Appended {len(content)} chars to section '{section_name}'."

    elif op == "prepend":
        if not content:
            raise ValueError("'prepend' requires non-empty content.")
        separator = "\n\n" if not content.endswith("\n") else ""
        new_body = content + separator + current_body
        msg = f"Prepended {len(content)} chars to section '{section_name}'."

    elif op == "replace":
        if not target:
            raise ValueError("'replace' requires a non-empty target string.")
        if target not in current_body:
            raise ValueError(
                f"'replace' target not found in section '{section_name}'. "
                f"Target (first 80 chars): {target[:80]!r}"
            )
        new_body = current_body.replace(target, content, 1)
        msg = f"Replaced target in section '{section_name}'."

    elif op == "delete":
        if not target:
            raise ValueError("'delete' requires a non-empty target string.")
        lines = current_body.splitlines(keepends=True)
        filtered = [ln for ln in lines if target not in ln]
        removed = len(lines) - len(filtered)
        if removed == 0:
            raise ValueError(
                f"'delete' target not found in section '{section_name}'. "
                f"Target: {target[:80]!r}"
            )
        new_body = "".join(filtered)
        msg = f"Deleted {removed} line(s) from section '{section_name}'."

    elif op == "overwrite":
        if not confirm_overwrite:
            raise ValueError(
                "'overwrite' requires confirm_overwrite=true. "
                "This replaces the ENTIRE section."
            )
        if not content:
            raise ValueError("'overwrite' requires non-empty content.")
        new_body = content
        msg = f"Full overwrite of section '{section_name}' ({len(new_body)} chars)."

    else:
        raise ValueError(
            f"Unknown operation {operation!r}. "
            "Valid operations: append, prepend, replace, delete, overwrite."
        )

    _sections[section_idx]['body'] = new_body
    _write_section_file(section_name, new_body)
    _cached_full_prompt = _assemble_full_prompt()

    log.info(f"apply_prompt_operation({section_name}, {op}): {msg}")
    return new_body, msg


# ---------------------------------------------------------------------------
# Sysprompt CRUD Library
# ---------------------------------------------------------------------------
# Functions for managing per-model system prompt folders.
# All accept a model_key and resolve the folder from LLM_REGISTRY.
# "self" is resolved by callers before passing here.
# ---------------------------------------------------------------------------

import json as _json


def _sp_get_folder(model_key: str, llm_registry: dict) -> Optional[str]:
    """
    Return the absolute folder path for a model's system prompts.
    Returns None if model not found or folder is 'none'.
    """
    model_cfg = llm_registry.get(model_key, {})
    folder_rel = model_cfg.get("system_prompt_folder", "")
    if not folder_rel or folder_rel.lower() == "none":
        return None
    return os.path.join(BASE_DIR, folder_rel)


def sp_resolve_model(model_key_or_self: str, current_model: str) -> str:
    """Resolve 'self' to the current model key."""
    if model_key_or_self.lower() == "self":
        return current_model
    return model_key_or_self


def sp_list_directories() -> str:
    """
    List all subdirectories in the system_prompt/ base directory.
    Returns a formatted string showing each directory name and its file count.
    """
    if not os.path.isdir(SYSTEM_PROMPT_BASE):
        return f"System prompt base directory does not exist: {SYSTEM_PROMPT_BASE}"
    entries = sorted(
        e for e in os.listdir(SYSTEM_PROMPT_BASE)
        if os.path.isdir(os.path.join(SYSTEM_PROMPT_BASE, e))
    )
    if not entries:
        return f"No directories found in {SYSTEM_PROMPT_BASE}"
    lines = [f"System prompt directories in {SYSTEM_PROMPT_BASE}:"]
    for name in entries:
        dpath = os.path.join(SYSTEM_PROMPT_BASE, name)
        files = [f for f in os.listdir(dpath) if f.startswith(".system_prompt")]
        lines.append(f"  {name}  ({len(files)} prompt file(s))")
    return "\n".join(lines)


def sp_list_files(model_key: str, llm_registry: dict) -> str:
    """
    List all .system_prompt* files in the model's folder.
    Returns a formatted string.
    """
    folder = _sp_get_folder(model_key, llm_registry)
    if folder is None:
        return f"Model '{model_key}' has no system_prompt_folder configured (or set to 'none')."
    if not os.path.isdir(folder):
        return f"Folder does not exist: {folder}"

    files = sorted(
        f for f in os.listdir(folder)
        if f.startswith(".system_prompt") or f == ".system_prompt"
    )
    if not files:
        return f"No .system_prompt* files in {folder}"

    lines = [f"System prompt files for '{model_key}' in {folder}:"]
    for fname in files:
        fpath = os.path.join(folder, fname)
        size = os.path.getsize(fpath)
        lines.append(f"  {fname}  ({size} bytes)")
    return "\n".join(lines)


def sp_read_prompt(model_key: str, llm_registry: dict) -> str:
    """
    Load and assemble the full system prompt for a model (stateless).
    """
    folder = _sp_get_folder(model_key, llm_registry)
    if folder is None:
        return f"Model '{model_key}' has no system_prompt_folder configured."
    return load_prompt_for_folder(folder)


def sp_read_file(model_key: str, filename: str, llm_registry: dict) -> str:
    """
    Read a specific file from the model's system prompt folder.
    filename can be a bare section name (e.g. 'behavior') or a full filename
    (e.g. '.system_prompt_behavior' or '.system_prompt').
    """
    folder = _sp_get_folder(model_key, llm_registry)
    if folder is None:
        return f"Model '{model_key}' has no system_prompt_folder configured."

    # Resolve filename
    if filename == ".system_prompt":
        fpath = os.path.join(folder, ".system_prompt")
    elif filename.startswith(".system_prompt"):
        fpath = os.path.join(folder, filename)
    else:
        fpath = os.path.join(folder, f".system_prompt_{filename}")

    if not os.path.exists(fpath):
        return f"File not found: {fpath}"

    try:
        with open(fpath, 'r', encoding='utf-8') as fh:
            return fh.read()
    except Exception as exc:
        return f"Error reading {fpath}: {exc}"


def sp_write_file(model_key: str, filename: str, data: str, llm_registry: dict) -> str:
    """
    Overwrite or create a file in the model's system prompt folder.
    filename can be a bare section name or full filename.
    Creates the folder if it doesn't exist.
    """
    folder = _sp_get_folder(model_key, llm_registry)
    if folder is None:
        return f"Model '{model_key}' has no system_prompt_folder configured."

    os.makedirs(folder, exist_ok=True)

    if filename == ".system_prompt":
        fpath = os.path.join(folder, ".system_prompt")
    elif filename.startswith(".system_prompt"):
        fpath = os.path.join(folder, filename)
    else:
        fpath = os.path.join(folder, f".system_prompt_{filename}")

    try:
        with open(fpath, 'w', encoding='utf-8') as fh:
            fh.write(data)
        log.info(f"sp_write_file: wrote {len(data)} bytes to {fpath}")
        return f"Wrote {len(data)} bytes to {fpath}"
    except Exception as exc:
        log.error(f"sp_write_file: error writing {fpath}: {exc}")
        return f"Error writing {fpath}: {exc}"


def sp_delete_file(model_key: str, filename: str, llm_registry: dict) -> str:
    """
    Delete a specific file from the model's system prompt folder.
    """
    folder = _sp_get_folder(model_key, llm_registry)
    if folder is None:
        return f"Model '{model_key}' has no system_prompt_folder configured."

    if filename == ".system_prompt":
        fpath = os.path.join(folder, ".system_prompt")
    elif filename.startswith(".system_prompt"):
        fpath = os.path.join(folder, filename)
    else:
        fpath = os.path.join(folder, f".system_prompt_{filename}")

    if not os.path.exists(fpath):
        return f"File not found: {fpath}"

    try:
        os.remove(fpath)
        return f"Deleted: {fpath}"
    except Exception as exc:
        return f"Error deleting {fpath}: {exc}"


def sp_delete_directory(model_key: str, llm_registry: dict) -> str:
    """
    Delete the entire system prompt folder for a model and set
    system_prompt_folder to 'none' in llm-models.json.
    """
    folder = _sp_get_folder(model_key, llm_registry)
    if folder is None:
        return f"Model '{model_key}' has no folder to delete."

    try:
        shutil.rmtree(folder)
    except Exception as exc:
        return f"Error deleting folder {folder}: {exc}"

    # Update llm-models.json
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = _json.load(f)
        if model_key in data.get('models', {}):
            data['models'][model_key]['system_prompt_folder'] = 'none'
            with open(LLM_MODELS_FILE, 'w') as f:
                _json.dump(data, f, indent=2)
            llm_registry[model_key]['system_prompt_folder'] = 'none'
    except Exception as exc:
        return f"Folder deleted but failed to update llm-models.json: {exc}"

    return f"Deleted folder {folder} and set system_prompt_folder='none' for '{model_key}'."


def sp_copy_directory(src_model: str, new_dir: str, llm_registry: dict) -> str:
    """
    Copy a model's system prompt folder to system_prompt/<new_dir>.
    new_dir must be a simple directory name (no slashes).
    """
    src_folder = _sp_get_folder(src_model, llm_registry)
    if src_folder is None:
        return f"Model '{src_model}' has no system_prompt_folder configured."
    if not os.path.isdir(src_folder):
        return f"Source folder does not exist: {src_folder}"

    # Sanitize new_dir
    new_dir_clean = os.path.basename(new_dir.strip("/\\"))
    if not new_dir_clean:
        return "ERROR: new_dir must be a non-empty directory name."

    dst_folder = os.path.join(SYSTEM_PROMPT_BASE, new_dir_clean)
    if os.path.exists(dst_folder):
        return f"ERROR: Destination already exists: {dst_folder}"

    try:
        shutil.copytree(src_folder, dst_folder)
        return f"Copied '{src_model}' prompts to {dst_folder}"
    except Exception as exc:
        return f"Error copying to {dst_folder}: {exc}"


def sp_set_directory(model_key: str, new_dir: str) -> str:
    """
    Set a model's system_prompt_folder to system_prompt/<new_dir>
    in llm-models.json and in the in-memory LLM_REGISTRY.
    """
    new_dir_clean = os.path.basename(new_dir.strip("/\\"))
    if not new_dir_clean:
        return "ERROR: new_dir must be a non-empty directory name."

    new_folder_rel = f"system_prompt/{new_dir_clean}"
    new_folder_abs = os.path.join(BASE_DIR, new_folder_rel)

    if not os.path.isdir(new_folder_abs) and new_dir_clean.lower() != "none":
        return f"ERROR: Directory does not exist: {new_folder_abs}"

    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = _json.load(f)
        if model_key not in data.get('models', {}):
            return f"ERROR: Model '{model_key}' not found in llm-models.json."
        data['models'][model_key]['system_prompt_folder'] = new_folder_rel
        with open(LLM_MODELS_FILE, 'w') as f:
            _json.dump(data, f, indent=2)
    except Exception as exc:
        return f"Error updating llm-models.json: {exc}"

    # Update in-memory registry
    from config import LLM_REGISTRY
    if model_key in LLM_REGISTRY:
        LLM_REGISTRY[model_key]['system_prompt_folder'] = new_folder_rel

    return f"Set system_prompt_folder='{new_folder_rel}' for model '{model_key}'."
