"""
plan_engine.py — Two-tier plan decomposition and execution engine.

Lifecycle:
  1. Concept steps created by human or reasoning model (step_type='concept')
  2. Decomposer (Sonnet 4.6) breaks concept steps into task steps (step_type='task')
  3. Executor picks up approved task steps and runs tool_call specs
  4. Parent concept steps auto-complete when all child tasks are done
  5. Goal auto-completes when all concept steps are done

Supports goal-less plans (goal_id=0) for ad-hoc work.
"""

import asyncio
import json
import logging

log = logging.getLogger("plan_engine")

# ---------------------------------------------------------------------------
# Table helpers — uses same prefix pattern as memory.py
# ---------------------------------------------------------------------------

def _PLANS():
    from memory import _PLANS as _P
    return _P()

def _GOALS():
    from memory import _GOALS as _G
    return _G()

# ---------------------------------------------------------------------------
# Decomposer system prompt
# ---------------------------------------------------------------------------

_DECOMPOSER_SYSTEM = """\
You are a plan decomposition engine. Break a concept step into discrete, executable task steps.

You will receive:
- The concept step description
- The list of available tools and their argument schemas
- Optional context (goal title, other concept steps for sequencing awareness)

For EACH task step, output:
- description: short human-readable label
- tool_call: JSON object with "tool" and "args" keys, or null if not tool-executable
- target: "model" if tool_call is populated, "human" if requires human action,
  "investigate" if you cannot determine how to execute with current tools
- reason: brief explanation (only for human/investigate targets)

## Rules

1. Each task step = ONE atomic tool call or ONE human action.
2. **ONLY use tools from the AVAILABLE TOOLS list.** Do not invent tool names or add parameters not in the schema. If a tool is not listed, use target="investigate".
3. **ONLY use parameter names shown in the tool schema.** Do not add extra args.
4. If the concept says "human:" or names a person, set target="human".
5. If unsure, set target="investigate" — do NOT guess.
6. Order steps logically. They execute serially.
7. For db_query, write actual SQL in args.query.
8. If already atomic (one tool call), emit exactly one step.

## Sequential dependency handling

Task steps execute one at a time. If step N needs output from step N-1 (e.g., a file_id returned by a google_drive list), write the args with a placeholder description:
- Use the SAME tool but note the dependency in the description (e.g., "Read the report file using file_id from previous step")
- The executor will resolve the dependency at runtime
- Do NOT hardcode IDs you don't have — leave args that depend on prior output EMPTY and note the dependency

## Tool-specific rules

### google_drive
- Operations: list, read, create, append, delete
- `list` → returns file names and IDs. Args: operation="list" (optional: folder_id)
- `read` → requires file_id (NOT file_name). If you don't have the ID yet, first emit a "list" step, then a "read" step noting it needs the file_id from the list result
- `create` → args: operation="create", file_name="...", content="..."
- Do NOT pass file_name to read — it will fail. Only file_id works for read.

### llm_call
- REQUIRED args: model, prompt
- model MUST be a real model key from the system (e.g., "samaritan-execution", "summarizer-anthropic"). NEVER use "default" or "summarizer-gemini" — they must not be used.
- mode: "text" (default, generates text) or "tool" (delegates a tool call; requires "tool" arg)
- Do NOT use mode="chat" — it does not exist.
- **samaritan-execution REQUIRES mode="tool" with an explicit tool= argument.** It rejects mode="text". For cognitive writes (set_goal, set_plan, set_belief, procedure_save, save_memory_typed), always use: {"tool": "llm_call", "args": {"model": "samaritan-execution", "mode": "tool", "tool": "<tool_name>", "prompt": "..."}}
- For synthesis/summarization tasks, use model="summarizer-anthropic" with mode="text".

### search_tavily / search_xai / search_brightdata
- Args: query (string). search_tavily also accepts search_depth="basic"|"advanced".
- search_xai has only the "query" parameter — do NOT add model or other args.
- Do NOT use search_ddgs — it is not available as a tool.

### assert_belief (via llm_call to samaritan-execution)
- Confidence scoring rules — do NOT default to high confidence:
  - **confidence 9-10**: Only for directly verified facts, official documentation, or first-hand experience
  - **confidence 7-8**: Well-sourced from multiple corroborating references
  - **confidence 5-6**: Single search result, plausible but not cross-verified
  - **confidence ≤4**: Speculative, conflicting sources, or uncertain
- When the belief comes from automated research (search results), cap confidence at **7** unless multiple independent sources confirm it.

### procedure_save (via llm_call to samaritan-execution)
- The `steps` arg must be a JSON array of objects: [{"action": "step description"}, ...]
- Do NOT pass steps as a plain numbered string.

### memory_save / memory_recall / procedure_recall
- memory_save: args: topic, content, importance (1-10)
- memory_recall: args: query (semantic search)
- procedure_recall: args: query

### url_extract / url_extract_brightdata
- Args: url (required), query (optional extraction prompt)

## Output format

Respond with ONLY a JSON array. No markdown fences, no explanation outside the JSON.

Example:
[
  {"description": "List Google Drive files to find the target report", "tool_call": {"tool": "google_drive", "args": {"operation": "list"}}, "target": "model"},
  {"description": "Read the report file (file_id from previous list step)", "tool_call": {"tool": "google_drive", "args": {"operation": "read", "file_id": ""}}, "target": "model"},
  {"description": "Search for competitors in the LLM inference space", "tool_call": {"tool": "search_tavily", "args": {"query": "FriendliAI competitors LLM inference 2026", "search_depth": "advanced"}}, "target": "model"},
  {"description": "Summarize findings into a structured competitor list", "tool_call": {"tool": "llm_call", "args": {"model": "summarizer-gemini", "prompt": "Summarize the competitor research into a structured list with key differentiators.", "mode": "text"}}, "target": "model"},
  {"description": "Review competitor list for completeness", "target": "human", "reason": "Admin should verify the competitor list before proceeding to report generation"}
]
"""

# ---------------------------------------------------------------------------
# Build tool catalog for decomposer context
# ---------------------------------------------------------------------------

def _build_tool_catalog() -> str:
    """
    Build a detailed tool catalog string for the decomposer.
    Includes tool names, parameter schemas with types/defaults, and descriptions.
    """
    try:
        from tools import get_all_openai_tools
        all_tools = get_all_openai_tools()
        lines = []
        for td in all_tools:
            fn = td.get("function", {})
            name = fn.get("name", "?")
            desc = fn.get("description", "")[:200]
            params = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            # Build param details
            param_parts = []
            for pname, pschema in params.items():
                ptype = pschema.get("type", "str")
                pdefault = pschema.get("default", "")
                penum = pschema.get("enum")
                req_mark = "*" if pname in required else ""
                detail = f"{pname}{req_mark}: {ptype}"
                if penum:
                    detail += f" [{'/'.join(str(e) for e in penum)}]"
                elif pdefault not in ("", None):
                    detail += f" (default={pdefault})"
                param_parts.append(detail)
            lines.append(f"### {name}")
            lines.append(f"  {desc}")
            if param_parts:
                lines.append(f"  Args: {', '.join(param_parts)}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        log.warning(f"_build_tool_catalog failed: {e}")
        return "(tool catalog unavailable)"


# ---------------------------------------------------------------------------
# Decompose a concept step into task steps
# ---------------------------------------------------------------------------

async def decompose_concept_step(
    concept_step_id: int,
    model_key: str = "plan-decomposer",
    auto_approve: bool = False,
) -> list[dict]:
    """
    Takes a concept step and decomposes it into task steps via LLM.

    Returns list of created task step dicts (id, description, target, tool_call, etc.)
    """
    from database import fetch_dicts, execute_insert
    from config import LLM_REGISTRY
    from agents import _build_lc_llm, _content_to_str
    from langchain_core.messages import SystemMessage, HumanMessage

    # ---- Load the concept step ----
    rows = await fetch_dicts(
        f"SELECT p.*, g.title as goal_title, g.description as goal_description "
        f"FROM {_PLANS()} p "
        f"LEFT JOIN {_GOALS()} g ON g.id = p.goal_id "
        f"WHERE p.id = {concept_step_id} AND p.step_type = 'concept' "
        f"LIMIT 1"
    )
    if not rows:
        raise ValueError(f"Concept step id={concept_step_id} not found or not a concept step")

    concept = rows[0]
    goal_id = concept["goal_id"]
    goal_title = concept.get("goal_title", "") or ""
    goal_desc = concept.get("goal_description", "") or ""

    # ---- Load sibling concept steps for context ----
    siblings = await fetch_dicts(
        f"SELECT step_order, description, status FROM {_PLANS()} "
        f"WHERE goal_id = {goal_id} AND step_type = 'concept' AND id != {concept_step_id} "
        f"ORDER BY step_order"
    ) or []

    # ---- Build the user prompt ----
    tool_catalog = _build_tool_catalog()

    context_parts = []
    if goal_title:
        context_parts.append(f"Goal: {goal_title}")
    if goal_desc:
        context_parts.append(f"Goal description: {goal_desc}")
    if siblings:
        sib_lines = [
            f"  step {s['step_order']}: {s['description']} [{s['status']}]"
            for s in siblings
        ]
        context_parts.append("Other steps in this plan:\n" + "\n".join(sib_lines))

    # ---- Include results from completed sibling task steps ----
    # This gives the decomposer access to data discovered in earlier steps
    # (e.g., competitor names identified in step 1 feed into step 2 decomposition)
    completed_results = await fetch_dicts(
        f"SELECT p2.description, LEFT(p2.result, 500) as result "
        f"FROM {_PLANS()} p2 "
        f"WHERE p2.goal_id = {goal_id} AND p2.step_type = 'task' "
        f"AND p2.status = 'done' AND p2.result IS NOT NULL "
        f"AND p2.result != '' "
        f"ORDER BY p2.step_order LIMIT 20"
    ) or []
    if completed_results:
        res_lines = []
        for cr in completed_results:
            res_lines.append(f"  - {cr['description']}: {cr['result']}")
        context_parts.append(
            "RESULTS FROM COMPLETED EARLIER STEPS (use these to inform your decomposition):\n"
            + "\n".join(res_lines)
        )

    context_block = "\n".join(context_parts) if context_parts else "(no additional context)"

    user_prompt = f"""\
CONCEPT STEP TO DECOMPOSE:
{concept["description"]}

CONTEXT:
{context_block}

AVAILABLE TOOLS:
{tool_catalog}

Decompose the concept step into discrete task steps. Output JSON array only."""

    # ---- Call the decomposer LLM ----
    if model_key not in LLM_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")

    cfg = LLM_REGISTRY[model_key]
    timeout = cfg.get("llm_call_timeout", 90)
    llm = _build_lc_llm(model_key)
    msgs = [SystemMessage(content=_DECOMPOSER_SYSTEM), HumanMessage(content=user_prompt)]

    try:
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)
    except Exception as e:
        log.error(f"decompose_concept_step: LLM call failed: {e}")
        raise

    # ---- Parse the response ----
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        tasks = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error(f"decompose_concept_step: JSON parse failed: {e}\nRaw: {raw[:500]}")
        raise ValueError(f"Decomposer returned invalid JSON: {e}")

    if not isinstance(tasks, list):
        raise ValueError(f"Decomposer returned {type(tasks).__name__}, expected list")

    # ---- Insert task steps into DB ----
    created = []
    approval = "approved" if auto_approve else "proposed"

    for i, task in enumerate(tasks, start=1):
        desc = str(task.get("description", "")).replace("'", "''")
        target = task.get("target", "model")
        if target not in ("model", "human", "investigate", "claude-code", "claude-cognition"):
            target = "investigate"

        tool_call_raw = task.get("tool_call")
        tool_call_sql = "NULL"
        if tool_call_raw and isinstance(tool_call_raw, dict):
            tc_json = json.dumps(tool_call_raw)
            # Escape for MySQL: double single quotes AND double backslashes
            # so \n in JSON doesn't become a literal newline in MySQL storage
            tc_json = tc_json.replace("\\", "\\\\").replace("'", "''")
            tool_call_sql = f"'{tc_json}'"
        elif target == "model":
            # Model target but no tool_call — downgrade to investigate
            target = "investigate"

        reason = str(task.get("reason", "")).replace("'", "''")
        if reason:
            desc_with_reason = f"{desc} — {reason}" if target in ("human", "investigate") else desc
        else:
            desc_with_reason = desc

        executor = "NULL"
        if target == "model":
            executor = "'plan-executor'"

        sql = (
            f"INSERT INTO {_PLANS()} "
            f"(goal_id, step_order, description, status, step_type, parent_id, "
            f"target, executor, tool_call, approval, source, session_id) "
            f"VALUES ({goal_id}, {i}, '{desc_with_reason}', 'pending', 'task', {concept_step_id}, "
            f"'{target}', {executor}, {tool_call_sql}, '{approval}', 'assistant', NULL)"
        )
        try:
            row_id = await execute_insert(sql)
            created.append({
                "id": row_id,
                "step_order": i,
                "description": task.get("description", ""),
                "target": target,
                "tool_call": tool_call_raw,
                "approval": approval,
            })
        except Exception as e:
            log.error(f"decompose_concept_step: insert failed for task {i}: {e}")

    # ---- Mark concept step as in_progress (decomposition done, awaiting execution) ----
    if created:
        from database import execute_sql
        await execute_sql(
            f"UPDATE {_PLANS()} SET status = 'in_progress' WHERE id = {concept_step_id}"
        )

    log.info(
        f"decompose_concept_step: concept={concept_step_id} → {len(created)} task steps "
        f"(approval={approval})"
    )
    return created


# ---------------------------------------------------------------------------
# Approve / reject proposed task steps
# ---------------------------------------------------------------------------

async def approve_plan(concept_step_id: int, approve: bool = True) -> str:
    """
    Approve or reject all proposed task steps under a concept step.
    """
    from database import execute_sql, fetch_dicts

    new_approval = "approved" if approve else "rejected"
    await execute_sql(
        f"UPDATE {_PLANS()} SET approval = '{new_approval}' "
        f"WHERE parent_id = {concept_step_id} AND approval = 'proposed'"
    )
    rows = await fetch_dicts(
        f"SELECT id FROM {_PLANS()} WHERE parent_id = {concept_step_id}"
    )
    count = len(rows) if rows else 0
    return f"{count} task steps {new_approval} for concept step id={concept_step_id}"


# ---------------------------------------------------------------------------
# Resolve sequential dependencies between sibling task steps
# ---------------------------------------------------------------------------

async def _resolve_dependencies(step: dict, tool_name: str, tool_args: dict) -> dict:
    """
    Inspect tool_args for empty/placeholder values that can be resolved from
    the results of earlier sibling steps (same parent_id, lower step_order).

    Currently handles:
    - google_drive read with empty file_id → scan prior list result for matching file
    """
    from database import fetch_dicts

    # google_drive read with empty file_id
    if tool_name == "google_drive" and tool_args.get("operation") == "read" and not tool_args.get("file_id"):
        parent_id = step.get("parent_id")
        if not parent_id:
            return tool_args
        # Find the most recent completed google_drive list sibling
        siblings = await fetch_dicts(
            f"SELECT result, tool_call FROM {_PLANS()} "
            f"WHERE parent_id = {parent_id} AND step_type = 'task' "
            f"AND status = 'done' AND id < {step['id']} "
            f"ORDER BY step_order DESC"
        ) or []
        for sib in siblings:
            sib_tc = sib.get("tool_call", "")
            sib_result = sib.get("result", "") or ""
            # Check if this sibling was a google_drive list
            if isinstance(sib_tc, str):
                try:
                    sib_tc = json.loads(sib_tc)
                except (json.JSONDecodeError, TypeError):
                    sib_tc = {}
            if (isinstance(sib_tc, dict) and
                sib_tc.get("tool") == "google_drive" and
                sib_tc.get("args", {}).get("operation") == "list"):
                # Parse file IDs from the list result — look for "(id: XXXXX" pattern
                import re
                # Look for file name from description to match
                desc_lower = step.get("description", "").lower()
                # Find all file entries: [FILE] name  (id: XXXXX
                file_entries = re.findall(
                    r'\[FILE\]\s+(.+?)\s+\(id:\s+(\S+)', sib_result
                )
                if file_entries:
                    # Try to match by name similarity
                    best_id = None
                    for fname, fid in file_entries:
                        fname_lower = fname.lower().strip()
                        if any(kw in fname_lower for kw in ["friendliai", "comprehensive", "report"]):
                            best_id = fid
                            break
                    if not best_id and file_entries:
                        # Fallback: use first file
                        best_id = file_entries[0][1]
                    if best_id:
                        # Clean trailing parentheses
                        best_id = best_id.rstrip(")")
                        tool_args = dict(tool_args)
                        tool_args["file_id"] = best_id
                        log.info(
                            f"_resolve_dependencies: resolved file_id={best_id} "
                            f"for task step id={step['id']}"
                        )
                break  # Only check the most recent list result

    # google_drive create with empty content → inject content from most recent
    # sibling llm_call result (synthesis step produces the report text)
    if (tool_name == "google_drive" and
        tool_args.get("operation") == "create" and
        not tool_args.get("content")):
        parent_id = step.get("parent_id")
        if parent_id:
            # Find the most recent completed llm_call sibling before this step
            siblings = await fetch_dicts(
                f"SELECT result, tool_call FROM {_PLANS()} "
                f"WHERE parent_id = {parent_id} AND step_type = 'task' "
                f"AND status = 'done' AND id < {step['id']} "
                f"ORDER BY id DESC LIMIT 5"
            ) or []
            for sib in siblings:
                sib_tc = sib.get("tool_call", "")
                sib_result = sib.get("result", "") or ""
                if isinstance(sib_tc, str):
                    try:
                        sib_tc = json.loads(sib_tc)
                    except (json.JSONDecodeError, TypeError):
                        sib_tc = {}
                # Look for the immediately preceding llm_call (synthesis step)
                # Skip error/timeout results — they're not valid content
                _is_error = (sib_result.startswith("ERROR:") or
                             "timed out" in sib_result[:100] or
                             "Invalid tool_call" in sib_result[:100])
                if (isinstance(sib_tc, dict) and
                    sib_tc.get("tool") == "llm_call" and
                    sib_result and len(sib_result) > 50 and
                    not _is_error):
                    tool_args = dict(tool_args)
                    tool_args["content"] = sib_result
                    log.info(
                        f"_resolve_dependencies: injected content ({len(sib_result)} chars) "
                        f"from prior llm_call into google_drive create for task step id={step['id']}"
                    )
                    break

    return tool_args


# ---------------------------------------------------------------------------
# Execute a single task step
# ---------------------------------------------------------------------------

async def execute_task_step(task_step_id: int) -> str:
    """
    Execute a single approved task step using two-tier execution:

    1. Direct tool call — call the Python executor function directly (zero LLM cost).
    2. LLM executor — model_roles["plan_executor"] (samaritan-execution).
       Provider failover handled by backup_models on the model config.

    The `executor` column is written with the actual execution method for audit trail:
      "direct"       — tier 1 succeeded
      "<model_key>"  — tier 2 LLM model that executed it

    Returns the result text.
    """
    from database import fetch_dicts, execute_sql
    from tools import get_tool_executor
    from state import current_client_id

    rows = await fetch_dicts(
        f"SELECT * FROM {_PLANS()} WHERE id = {task_step_id} AND step_type = 'task' LIMIT 1"
    )
    if not rows:
        return f"Task step id={task_step_id} not found"

    step = rows[0]
    if step["approval"] != "approved":
        return f"Task step id={task_step_id} is not approved (approval={step['approval']})"
    if step["target"] != "model":
        return f"Task step id={task_step_id} target={step['target']} — cannot auto-execute"
    if step["status"] in ("done", "skipped"):
        return f"Task step id={task_step_id} already {step['status']}"

    # Mark in_progress
    await execute_sql(
        f"UPDATE {_PLANS()} SET status = 'in_progress' WHERE id = {task_step_id}"
    )

    tool_call_raw = step.get("tool_call")
    if not tool_call_raw:
        await execute_sql(
            f"UPDATE {_PLANS()} SET status = 'pending', "
            f"result = 'ERROR: no tool_call defined' WHERE id = {task_step_id}"
        )
        return f"Task step id={task_step_id} has no tool_call"

    # Parse tool_call
    if isinstance(tool_call_raw, str):
        try:
            tool_call = json.loads(tool_call_raw)
        except json.JSONDecodeError:
            err = f"Invalid tool_call JSON: {tool_call_raw[:200]}"
            await execute_sql(
                f"UPDATE {_PLANS()} SET status = 'pending', "
                f"result = '{err.replace(chr(39), chr(39)*2)}' WHERE id = {task_step_id}"
            )
            return err
    else:
        tool_call = tool_call_raw

    tool_name = tool_call.get("tool", "")
    tool_args = tool_call.get("args", {})

    # ------------------------------------------------------------------
    # Resolve sequential dependencies from sibling step results
    # ------------------------------------------------------------------
    tool_args = await _resolve_dependencies(step, tool_name, tool_args)

    # ------------------------------------------------------------------
    # Sanitize db_query SQL: fix unescaped single quotes in string literals
    # ------------------------------------------------------------------
    if tool_name == "db_query" and "sql" in tool_args:
        tool_args["sql"] = _fix_sql_quotes(tool_args["sql"])

    # ------------------------------------------------------------------
    # TIER 1: Direct tool call (zero LLM cost)
    # ------------------------------------------------------------------
    executor_fn = get_tool_executor(tool_name)
    if not executor_fn:
        err = f"Unknown tool: {tool_name}"
        await _fail_task(task_step_id, err)
        return err

    direct_err = None
    mapped_args = _map_tool_args(executor_fn, tool_args, tool_name)

    try:
        if isinstance(mapped_args, dict):
            result = await executor_fn(**mapped_args)
        else:
            result = await executor_fn()
        result_str = str(result) if result else "(no output)"

        # Check for error results returned as strings (e.g., timeouts)
        if (result_str.startswith("ERROR:") or
            "timed out" in result_str[:100] or
            "Invalid tool_call" in result_str[:100]):
            await _fail_task(task_step_id, result_str)
            log.warning(f"execute_task_step: id={task_step_id} tool={tool_name} direct returned error: {result_str[:100]}")
            return result_str

        # Direct call succeeded
        await _complete_task(task_step_id, result_str, executor="direct")
        parent_id = step.get("parent_id")
        if parent_id:
            await _check_parent_completion(parent_id)
        log.info(f"execute_task_step: id={task_step_id} tool={tool_name} executor=direct → done")
        return result_str

    except Exception as e:
        direct_err = str(e)
        log.warning(
            f"execute_task_step: id={task_step_id} tool={tool_name} direct failed: {direct_err[:200]}"
        )

    # ------------------------------------------------------------------
    # TIER 2 & 3: LLM executor with failover
    # ------------------------------------------------------------------
    llm_result = await _try_llm_executors(
        task_step_id, step, tool_name, tool_args, direct_err
    )
    if llm_result is not None:
        parent_id = step.get("parent_id")
        if parent_id:
            await _check_parent_completion(parent_id)
        return llm_result

    # All tiers exhausted — fail the task
    err = f"All executors failed. Direct: {direct_err}"
    await _fail_task(task_step_id, err)
    return err


def _fix_sql_quotes(sql: str) -> str:
    """Fix unescaped single quotes inside SQL string literals.

    LLM-generated SQL often contains apostrophes (e.g. Symbotic's) that
    break string literals.  Walks character-by-character: any single quote
    flanked by word characters on both sides (letter'letter) is treated as
    an apostrophe and doubled.
    """
    chars = list(sql)
    i = 0
    patched = False
    while i < len(chars):
        if chars[i] == "'" and i > 0 and i < len(chars) - 1:
            prev_is_word = chars[i - 1].isalpha()
            next_is_word = chars[i + 1].isalpha()
            # Already escaped '' — skip both
            if i + 1 < len(chars) and chars[i + 1] == "'":
                i += 2
                continue
            if prev_is_word and next_is_word:
                chars.insert(i, "'")
                patched = True
                i += 2  # skip past the doubled quote
                continue
        i += 1
    if patched:
        log.info("_fix_sql_quotes: patched unescaped apostrophes in SQL")
    return "".join(chars)


def _map_tool_args(executor_fn, tool_args: dict, tool_name: str) -> dict:
    """Map tool_call args to executor function parameter names."""
    if not isinstance(tool_args, dict):
        return tool_args
    import inspect
    try:
        sig = inspect.signature(executor_fn)
        valid_params = set(sig.parameters.keys())
        mapped = {}
        for k, v in tool_args.items():
            if k in valid_params:
                mapped[k] = v
            else:
                _synonyms = {
                    "query": "sql", "sql": "query",
                    "text": "content", "content": "text",
                    "message": "prompt", "prompt": "message",
                }
                alt = _synonyms.get(k, "")
                if alt and alt in valid_params:
                    mapped[alt] = v
                    log.info(f"_map_tool_args: mapped '{k}' → '{alt}' for {tool_name}")
                else:
                    mapped[k] = v
        return mapped
    except (ValueError, TypeError):
        return tool_args


async def _try_llm_executors(
    task_step_id: int,
    step: dict,
    tool_name: str,
    tool_args: dict,
    direct_err: str | None,
) -> str | None:
    """
    Try the plan_executor LLM model.  Provider-level failover is handled by
    backup_models on the executor model config (e.g. samaritan-execution →
    gpt-4o-execution), so only one role is needed here.
    Returns result string on success, None if all fail.
    """
    from config import get_model_role

    try:
        model_key = get_model_role("plan_executor")
    except KeyError:
        log.warning("_try_llm_executors: plan_executor role not configured")
        return None

    log.info(
        f"_try_llm_executors: id={task_step_id} tool={tool_name} "
        f"trying plan_executor={model_key}"
    )

    result = await _llm_execute_tool(model_key, tool_name, tool_args, step, direct_err)
    if result is not None:
        await _complete_task(task_step_id, result, executor=model_key)
        log.info(
            f"execute_task_step: id={task_step_id} tool={tool_name} "
            f"executor={model_key} → done"
        )
        return result

    log.warning(f"_try_llm_executors: plan_executor={model_key} failed")
    return None


async def _llm_execute_tool(
    model_key: str,
    tool_name: str,
    tool_args: dict,
    step: dict,
    direct_err: str | None,
) -> str | None:
    """
    Execute a tool call via an LLM model using llm_call(mode='tool').
    Returns the result string on success, None on failure.
    """
    from agents import llm_call

    # Build a prompt that tells the executor what to do
    args_json = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
    context = step.get("description", "")
    err_note = f"\nNote: direct execution failed with: {direct_err}" if direct_err else ""

    prompt = (
        f"Execute this tool call:\n"
        f"Tool: {tool_name}\n"
        f"Arguments: {args_json}\n"
        f"Context: {context}{err_note}\n\n"
        f"Call the {tool_name} tool with the arguments above and return the result."
    )

    try:
        result = await llm_call(
            model=model_key,
            prompt=prompt,
            mode="tool",
            sys_prompt="target",
            history="none",
            tool=tool_name,
        )
        # llm_call returns "ERROR: ..." on failure
        if result and result.startswith("ERROR:"):
            log.warning(f"_llm_execute_tool: {model_key}/{tool_name} returned: {result[:200]}")
            return None
        return result or None
    except Exception as e:
        log.error(f"_llm_execute_tool: {model_key}/{tool_name} exception: {e}")
        return None


async def _complete_task(task_step_id: int, result_str: str, executor: str = "direct"):
    """Mark a task step as done with result and executor audit trail."""
    from database import execute_sql
    result_escaped = result_str[:4000].replace("'", "''")
    executor_escaped = executor.replace("'", "''")
    await execute_sql(
        f"UPDATE {_PLANS()} SET status = 'done', "
        f"result = '{result_escaped}', "
        f"executor = '{executor_escaped}' "
        f"WHERE id = {task_step_id}"
    )


def _max_task_retries() -> int:
    """Return the max retry limit for task steps before marking them failed.

    Reads goal_health_failure_replan from plugins-enabled.json (default 3).
    This reuses the same threshold as the goal health system so behaviour is consistent.
    """
    import json, os
    try:
        path = os.path.join(os.path.dirname(__file__), "plugins-enabled.json")
        with open(path) as f:
            raw = json.load(f).get("plugin_config", {}).get("proactive_cognition", {})
        return int(raw.get("goal_health_failure_replan", 3))
    except Exception:
        return 3


async def _fail_task(task_step_id: int, error: str):
    """
    Mark a task step as failed and propagate failure up the stack.

    Cascade:
      task → fail_count++; if fail_count >= max_retries: status='failed'
              else: status='pending' (retry on next processor cycle)
      parent concept → status='failed' (blocked) when child hits hard failure
    """
    from database import execute_sql, fetch_dicts
    escaped = error[:2000].replace("'", "''")
    max_retries = _max_task_retries()

    # Increment fail_count and check threshold
    rows = await fetch_dicts(
        f"SELECT fail_count, parent_id FROM {_PLANS()} WHERE id = {task_step_id} LIMIT 1"
    )
    current_fails = rows[0]["fail_count"] if rows else 0
    new_fails = current_fails + 1
    hard_fail = new_fails >= max_retries

    new_status = "failed" if hard_fail else "pending"
    await execute_sql(
        f"UPDATE {_PLANS()} SET status = '{new_status}', fail_count = {new_fails}, "
        f"result = '{escaped}' WHERE id = {task_step_id}"
    )

    if hard_fail:
        log.warning(
            f"_fail_task: task {task_step_id} reached max retries ({max_retries}), "
            f"marking failed: {error[:100]}"
        )
        # Propagate hard failure to parent concept
        if rows and rows[0].get("parent_id"):
            await _check_parent_failure(rows[0]["parent_id"], error, hard_fail=True)
    else:
        log.warning(
            f"_fail_task: task {task_step_id} failed (attempt {new_fails}/{max_retries}), "
            f"re-queued: {error[:100]}"
        )
        if rows and rows[0].get("parent_id"):
            await _check_parent_failure(rows[0]["parent_id"], error, hard_fail=False)


async def _check_parent_failure(parent_id: int, error: str, hard_fail: bool = False):
    """
    When a task step fails, note the error on the parent concept step.

    On soft failure (retryable): appends error to result for audit trail only.
    On hard failure (max retries reached): also sets concept status to 'failed'
    so the goal health system can detect and handle it.

    Does NOT block the goal directly — the goal processor manages goal-level
    recovery to avoid inconsistent state that manual recovery can't fix.
    """
    from database import execute_sql, fetch_dicts

    escaped = error[:2000].replace("'", "''")
    if hard_fail:
        await execute_sql(
            f"UPDATE {_PLANS()} SET status = 'failed', "
            f"result = CONCAT(COALESCE(result, ''), ' | HARD FAIL: {escaped}') "
            f"WHERE id = {parent_id} AND status IN ('pending', 'in_progress')"
        )
        log.warning(f"_check_parent_failure: concept {parent_id} blocked (hard fail): {error[:100]}")
    else:
        await execute_sql(
            f"UPDATE {_PLANS()} SET "
            f"result = CONCAT(COALESCE(result, ''), ' | FAILED: {escaped}') "
            f"WHERE id = {parent_id} AND status IN ('pending', 'in_progress')"
        )
        log.info(f"_check_parent_failure: concept {parent_id} task failure noted: {error[:100]}")


# ---------------------------------------------------------------------------
# Auto-completion logic
# ---------------------------------------------------------------------------

async def _check_parent_completion(parent_id: int):
    """
    If all task steps under a concept step are done/skipped,
    mark the concept step as done. Then check if the goal is complete.
    """
    from database import fetch_dicts, execute_sql

    remaining = await fetch_dicts(
        f"SELECT COUNT(*) as cnt FROM {_PLANS()} "
        f"WHERE parent_id = {parent_id} AND status NOT IN ('done', 'skipped')"
    )
    if remaining and remaining[0]["cnt"] == 0:
        await execute_sql(
            f"UPDATE {_PLANS()} SET status = 'done' WHERE id = {parent_id}"
        )
        log.info(f"_check_parent_completion: concept step id={parent_id} auto-completed")

        # Check if goal is now complete
        goal_rows = await fetch_dicts(
            f"SELECT goal_id FROM {_PLANS()} WHERE id = {parent_id} LIMIT 1"
        )
        if goal_rows and goal_rows[0]["goal_id"]:
            await _check_goal_completion(goal_rows[0]["goal_id"])


async def _check_goal_completion(goal_id: int):
    """
    If all *approved* concept steps for a goal are done/skipped,
    mark the goal as done. Proposed/rejected concepts don't block completion.
    Orphaned pending/proposed steps are auto-skipped on goal completion.
    """
    from database import fetch_dicts, execute_sql

    # Only approved concepts block completion — proposed/rejected ones don't
    remaining = await fetch_dicts(
        f"SELECT COUNT(*) as cnt FROM {_PLANS()} "
        f"WHERE goal_id = {goal_id} AND step_type = 'concept' "
        f"AND approval = 'approved' "
        f"AND status NOT IN ('done', 'skipped')"
    )
    if remaining and remaining[0]["cnt"] == 0:
        # Must have at least one approved concept to complete (guard against empty plans)
        has_concepts = await fetch_dicts(
            f"SELECT COUNT(*) as cnt FROM {_PLANS()} "
            f"WHERE goal_id = {goal_id} AND step_type = 'concept' AND approval = 'approved'"
        )
        if not has_concepts or has_concepts[0]["cnt"] == 0:
            return

        await execute_sql(
            f"UPDATE {_GOALS()} SET status = 'done', "
            f"auto_process_status = CASE "
            f"  WHEN auto_process_status IS NOT NULL THEN 'completed' "
            f"  ELSE auto_process_status END "
            f"WHERE id = {goal_id} AND status IN ('active', 'blocked')"
        )
        # Close orphaned plan steps (pending/proposed tasks under completed goal)
        await execute_sql(
            f"UPDATE {_PLANS()} SET status = 'skipped', "
            f"result = COALESCE(result, 'Skipped: parent goal completed') "
            f"WHERE goal_id = {goal_id} AND status NOT IN ('done', 'skipped')"
        )
        log.info(f"_check_goal_completion: goal id={goal_id} auto-completed")


# ---------------------------------------------------------------------------
# Execute all pending approved task steps for a goal (or all goals)
# ---------------------------------------------------------------------------

async def execute_pending_tasks(
    goal_id: int | None = None,
    max_steps: int = 10,
) -> list[dict]:
    """
    Execute pending approved model-targeted task steps.
    Returns list of {id, tool, result, status} for each executed step.
    """
    from database import fetch_dicts

    where = (
        f"WHERE step_type = 'task' AND status = 'pending' "
        f"AND approval = 'approved' AND target = 'model'"
    )
    if goal_id is not None:
        where += f" AND goal_id = {goal_id}"

    rows = await fetch_dicts(
        f"SELECT id FROM {_PLANS()} {where} "
        f"ORDER BY goal_id, step_order LIMIT {max_steps}"
    )
    if not rows:
        return []

    results = []
    for row in rows:
        step_id = row["id"]
        try:
            result = await execute_task_step(step_id)
            results.append({"id": step_id, "status": "done", "result": result[:500]})
        except Exception as e:
            results.append({"id": step_id, "status": "error", "result": str(e)[:500]})

    return results


# ---------------------------------------------------------------------------
# Decompose all pending concept steps for a goal (or all goals)
# ---------------------------------------------------------------------------

async def decompose_pending_concepts(
    goal_id: int | None = None,
    model_key: str = "plan-decomposer",
    auto_approve: bool = False,
) -> list[dict]:
    """
    Find concept steps that have no child task steps and decompose them.
    Only decomposes the FIRST pending concept per goal — later concepts wait
    until earlier ones are done, so the decomposer can use their results
    (e.g., competitor names discovered in step 1 feed into step 2).
    Returns summary of decomposition results.
    """
    from database import fetch_dicts

    where = (
        f"WHERE p.step_type = 'concept' AND p.status = 'pending' "
        f"AND NOT EXISTS ("
        f"  SELECT 1 FROM {_PLANS()} c WHERE c.parent_id = p.id"
        f")"
    )
    if goal_id is not None:
        where += f" AND p.goal_id = {goal_id}"

    all_rows = await fetch_dicts(
        f"SELECT p.id, p.goal_id, p.step_order, p.description FROM {_PLANS()} p {where} "
        f"ORDER BY p.goal_id, p.step_order"
    )

    # Only take the first pending concept per goal — later steps may depend
    # on results from earlier steps (sequential decomposition)
    seen_goals = set()
    rows = []
    for row in (all_rows or []):
        gid = row["goal_id"]
        if gid in seen_goals:
            continue
        # Check if all earlier concept steps for this goal are done
        earlier_pending = await fetch_dicts(
            f"SELECT id FROM {_PLANS()} "
            f"WHERE goal_id = {gid} AND step_type = 'concept' "
            f"AND step_order < {row['step_order']} "
            f"AND status NOT IN ('done', 'skipped') LIMIT 1"
        )
        if earlier_pending:
            # Earlier concept still in progress — skip this goal for now
            log.info(
                f"decompose_pending_concepts: skipping concept {row['id']} "
                f"(goal {gid}) — earlier concept steps not yet done"
            )
            seen_goals.add(gid)
            continue
        rows.append(row)
        seen_goals.add(gid)
    if not rows:
        return []

    results = []
    for row in rows:
        try:
            tasks = await decompose_concept_step(
                row["id"], model_key=model_key, auto_approve=auto_approve
            )
            results.append({
                "concept_id": row["id"],
                "description": row["description"][:100],
                "task_count": len(tasks),
                "tasks": tasks,
            })
        except Exception as e:
            results.append({
                "concept_id": row["id"],
                "description": row["description"][:100],
                "error": str(e),
            })

    return results


# ---------------------------------------------------------------------------
# Full pipeline: decompose → approve → execute (for automated runs)
# ---------------------------------------------------------------------------

async def run_plan_pipeline(
    goal_id: int | None = None,
    model_key: str = "plan-decomposer",
    auto_approve: bool = False,
    max_exec_steps: int = 10,
) -> dict:
    """
    Full pipeline:
    1. Decompose any un-decomposed concept steps
    2. Execute approved task steps

    Returns summary dict.
    """
    decomp_results = await decompose_pending_concepts(
        goal_id=goal_id, model_key=model_key, auto_approve=auto_approve
    )

    # Auto-approve any proposed task steps that were created by a prior
    # decompose (without auto_approve) so execute_pending_tasks picks them up.
    if auto_approve:
        from database import execute_sql
        where = (
            f"step_type = 'task' AND status = 'pending' AND approval = 'proposed'"
        )
        if goal_id is not None:
            where += f" AND goal_id = {goal_id}"
        await execute_sql(
            f"UPDATE {_PLANS()} SET approval = 'approved' WHERE {where}"
        )

    exec_results = await execute_pending_tasks(
        goal_id=goal_id, max_steps=max_exec_steps
    )

    return {
        "decomposed": decomp_results,
        "executed": exec_results,
    }


# ---------------------------------------------------------------------------
# Create a concept step (convenience wrapper)
# ---------------------------------------------------------------------------

async def create_concept_step(
    description: str,
    goal_id: int = 0,
    step_order: int = 1,
    source: str = "assistant",
    target: str = "model",
    approval: str = "proposed",
    session_id: str = "",
) -> int:
    """
    Create a concept step. Returns the new row ID.

    source: who created it (user, assistant, directive, session)
    target: default target for decomposition hint (model, human, investigate)
    approval: proposed (needs review) or auto (skip review)
    """
    from database import execute_insert

    desc = description.replace("'", "''")
    source = source if source in ("session", "user", "directive", "assistant") else "assistant"
    target = target if target in ("model", "human", "investigate", "claude-code", "claude-cognition") else "model"
    approval = approval if approval in ("proposed", "approved", "rejected", "auto") else "proposed"
    sid = session_id.replace("'", "''") if session_id else "NULL"
    sid_sql = f"'{sid}'" if session_id else "NULL"

    sql = (
        f"INSERT INTO {_PLANS()} "
        f"(goal_id, step_order, description, status, step_type, parent_id, "
        f"target, executor, tool_call, approval, source, session_id) "
        f"VALUES ({goal_id}, {step_order}, '{desc}', 'pending', 'concept', NULL, "
        f"'{target}', NULL, NULL, '{approval}', '{source}', {sid_sql})"
    )
    return await execute_insert(sql)


# ---------------------------------------------------------------------------
# View plan (for display / inspection)
# ---------------------------------------------------------------------------

async def view_plan(goal_id: int | None = None, include_done: bool = False) -> str:
    """
    Render a human-readable view of the plan hierarchy.
    """
    from database import fetch_dicts

    status_filter = "" if include_done else "AND p.status NOT IN ('done','skipped')"
    where = f"WHERE p.step_type = 'concept' {status_filter}"
    if goal_id is not None:
        where += f" AND p.goal_id = {goal_id}"

    concepts = await fetch_dicts(
        f"SELECT p.*, g.title as goal_title FROM {_PLANS()} p "
        f"LEFT JOIN {_GOALS()} g ON g.id = p.goal_id "
        f"{where} ORDER BY p.goal_id, p.step_order"
    )
    if not concepts:
        return "(no active plan steps)"

    lines = []
    current_goal = None
    for c in concepts:
        gid = c.get("goal_id", 0)
        if gid != current_goal:
            current_goal = gid
            gtitle = c.get("goal_title") or f"(ad-hoc, goal_id={gid})"
            lines.append(f"\n## {gtitle}")

        status_icon = {"pending": "○", "in_progress": "▶", "done": "✓", "skipped": "—"}.get(
            c["status"], "?"
        )
        approval_tag = f" [{c.get('approval', '?')}]" if c.get("approval") != "approved" else ""
        target_label = {"model": "🤖", "human": "👤", "investigate": "🔍", "claude-code": "💻"}.get(
            c.get("target", "model"), c.get("target", "?"))
        lines.append(
            f"  {status_icon} [{c['id']}] step {c['step_order']} {target_label}: "
            f"{c['description']}{approval_tag}"
        )

        # Load child task steps
        task_filter = "" if include_done else "AND status NOT IN ('done','skipped')"
        tasks = await fetch_dicts(
            f"SELECT * FROM {_PLANS()} WHERE parent_id = {c['id']} {task_filter} "
            f"ORDER BY step_order"
        )
        if tasks:
            for t in tasks:
                t_icon = {"pending": "·", "in_progress": "▸", "done": "✓", "skipped": "—"}.get(
                    t["status"], "?"
                )
                t_target_label = {"model": "🤖", "human": "👤", "investigate": "🔍", "claude-code": "💻"}.get(
                    t.get("target", "model"), t.get("target", "?"))
                t_approval = f" [{t.get('approval', '?')}]" if t.get("approval") != "approved" else ""
                tc = ""
                if t.get("tool_call"):
                    try:
                        tc_data = json.loads(t["tool_call"]) if isinstance(t["tool_call"], str) else t["tool_call"]
                        tc = f" tool:{tc_data.get('tool', '?')}"
                    except Exception:
                        tc = " tool:?"
                lines.append(
                    f"      {t_icon} [{t['id']}] {t_target_label} {t['description']}{tc}{t_approval}"
                )
                if t.get("result") and t["status"] == "done":
                    result_preview = str(t["result"])[:80].replace("\n", " ")
                    lines.append(f"           → {result_preview}")

    return "\n".join(lines)
