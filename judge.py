"""
judge.py — LLM-as-judge enforcement layer.

This module provides the core judge invocation logic and session configuration
helpers. It is NOT a plugin itself — it is imported by plugin_history_judge.py
and optionally by agents.py for the tool and memory gate hooks.

Per-model judge config in llm-models.json
------------------------------------------
Each model can carry a `judge_config` dict:

    "judge_config": {
        "model":     "qwen35-judge",
        "gates":     ["prompt", "response", "tool", "memory"],
        "mode":      "block",
        "threshold": 0.7
    }

  model     — judge model key in LLM_REGISTRY
  gates     — which enforcement points are active
  mode      — "block" (deny on fail) or "warn" (log + allow)
  threshold — score floor (0.0–1.0); below = fail

Gate points
-----------
  prompt   — inspects the user message before it reaches the LLM
  response — inspects the assistant response before it is stored / delivered
  tool     — inspects a tool call before the executor runs
  memory   — inspects a memory entry before it is persisted

Session overrides
-----------------
session["judge_override"] overrides any key from the model-level judge_config
for the duration of the session. Set/cleared by !judge commands.

Hook registry for tool + memory gates
--------------------------------------
plugin_history_judge.py registers async hook functions here at import time:
    judge._tool_gate_hook    = async (client_id, model_key, session, tool_name, tool_args) -> (bool, str)
    judge._memory_gate_hook  = async (client_id, model_key, session, topic, content) -> (bool, str)

agents.py calls judge.check_tool_gate() and judge.check_memory_gate().
Both are no-ops (return True, "") when no plugin is registered.
"""

import asyncio
import json
import logging
import os
import re

log = logging.getLogger("AISvc")

# ---------------------------------------------------------------------------
# Hook registry — filled by plugin_history_judge when it loads into the chain
# ---------------------------------------------------------------------------
_tool_gate_hook = None    # async (client_id, model_key, session, tool_name, tool_args) -> (bool, str)
_memory_gate_hook = None  # async (client_id, model_key, session, topic, content) -> (bool, str)


async def check_tool_gate(client_id: str, model_key: str, session: dict,
                          tool_name: str, tool_args: dict) -> tuple[bool, str]:
    """Call the tool gate hook if registered. Returns (allowed, denial_reason)."""
    if _tool_gate_hook is None:
        return True, ""
    try:
        return await _tool_gate_hook(client_id, model_key, session, tool_name, tool_args)
    except Exception as e:
        log.warning(f"judge.check_tool_gate: error: {e}")
        return True, ""


async def check_memory_gate(client_id: str, model_key: str, session: dict,
                            topic: str, content: str) -> tuple[bool, str]:
    """Call the memory gate hook if registered. Returns (allowed, denial_reason)."""
    if _memory_gate_hook is None:
        return True, ""
    try:
        return await _memory_gate_hook(client_id, model_key, session, topic, content)
    except Exception as e:
        log.warning(f"judge.check_memory_gate: error: {e}")
        return True, ""


# ---------------------------------------------------------------------------
# Judge prompt templates per gate type.
# The judge must return a single JSON object:
#   {"passed": true|false, "score": 0.0–1.0, "reason": "short explanation"}
# No markdown, no prose outside the JSON.
# ---------------------------------------------------------------------------

_GATE_PROMPTS: dict[str, str] = {
    "prompt": (
        "You are a content and safety judge. Evaluate the USER PROMPT below.\n"
        "Output ONLY a JSON object with these keys:\n"
        "  passed  (bool)   — true if the prompt is acceptable\n"
        "  score   (float)  — 0.0 (completely unacceptable) to 1.0 (fully acceptable)\n"
        "  reason  (string) — one sentence explaining your decision\n\n"
        "USER PROMPT:\n{content}\n\n"
        "JSON:"
    ),
    "tool": (
        "You are a tool-call safety judge. Evaluate the TOOL CALL below.\n"
        "Output ONLY a JSON object with these keys:\n"
        "  passed  (bool)   — true if the tool call is acceptable\n"
        "  score   (float)  — 0.0 (completely unacceptable) to 1.0 (fully acceptable)\n"
        "  reason  (string) — one sentence explaining your decision\n\n"
        "TOOL CALL:\n{content}\n\n"
        "JSON:"
    ),
    "response": (
        "You are a response quality and safety judge. Evaluate the ASSISTANT RESPONSE below.\n"
        "Output ONLY a JSON object with these keys:\n"
        "  passed  (bool)   — true if the response is acceptable\n"
        "  score   (float)  — 0.0 (completely unacceptable) to 1.0 (fully acceptable)\n"
        "  reason  (string) — one sentence explaining your decision\n\n"
        "ASSISTANT RESPONSE:\n{content}\n\n"
        "JSON:"
    ),
    "memory": (
        "You are a memory quality judge. Evaluate whether the MEMORY ENTRY below should be persisted.\n"
        "Output ONLY a JSON object with these keys:\n"
        "  passed  (bool)   — true if the memory entry is worth storing\n"
        "  score   (float)  — 0.0 (not worth storing) to 1.0 (definitely worth storing)\n"
        "  reason  (string) — one sentence explaining your decision\n\n"
        "MEMORY ENTRY:\ntopic: {topic}\ncontent: {content}\n\n"
        "JSON:"
    ),
}

_JSON_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def _parse_judge_response(text: str) -> tuple[bool, float, str]:
    """Parse the judge's JSON response. Returns (passed, score, reason)."""
    m = _JSON_RE.search(text)
    if not m:
        log.warning(f"judge: could not find JSON in response: {text[:200]!r}")
        return True, 1.0, "parse_error:no_json — defaulting to pass"

    try:
        obj = json.loads(m.group(0))
        passed = bool(obj.get("passed", True))
        score = float(obj.get("score", 1.0))
        reason = str(obj.get("reason", ""))
        return passed, score, reason
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"judge: JSON parse error: {e} — text: {text[:200]!r}")
        return True, 1.0, f"parse_error:{e} — defaulting to pass"


def _get_effective_judge_cfg(model_key: str, session: dict) -> dict | None:
    """
    Merge model-level judge_config with any session-level judge_override.
    Returns None if no judge is configured for this model+session.
    """
    from config import LLM_REGISTRY
    model_cfg = LLM_REGISTRY.get(model_key, {})
    base = model_cfg.get("judge_config")
    override = session.get("judge_override")

    if base is None and not override:
        return None
    if base is None:
        base = {}

    if not override:
        return dict(base)

    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def get_judge_model(model_key: str, session: dict) -> str | None:
    """Return the effective judge model key, or None if not configured."""
    cfg = _get_effective_judge_cfg(model_key, session)
    return cfg.get("model") or None if cfg else None


def is_gate_active(gate: str, model_key: str, session: dict) -> bool:
    """Return True if the named gate is active for this model+session."""
    cfg = _get_effective_judge_cfg(model_key, session)
    if cfg is None:
        return False
    return gate in cfg.get("gates", [])


async def judge_eval(
    gate: str,
    content: str,
    model_key: str,
    session: dict,
    topic: str = "",
) -> tuple[bool, float, str]:
    """
    Run a judge evaluation at the named gate point.

    Returns (passed, score, reason).
    On any error or if judge is not configured, returns (True, 1.0, "no_judge").
    """
    cfg = _get_effective_judge_cfg(model_key, session)
    if cfg is None:
        return True, 1.0, "no_judge"

    judge_model = cfg.get("model")
    if not judge_model:
        return True, 1.0, "no_judge_model"

    gates = cfg.get("gates", [])
    if gate not in gates:
        return True, 1.0, "gate_inactive"

    threshold = float(cfg.get("threshold", 0.7))

    template = _GATE_PROMPTS.get(gate)
    if not template:
        log.warning(f"judge: unknown gate type '{gate}'")
        return True, 1.0, f"unknown_gate:{gate}"

    if gate == "memory":
        prompt = template.format(topic=topic, content=content)
    else:
        prompt = template.format(content=content)

    try:
        from config import LLM_REGISTRY, BASE_DIR
        from agents import _build_lc_llm, _content_to_str
        from prompt import load_prompt_for_folder
        from langchain_core.messages import SystemMessage, HumanMessage

        judge_cfg = LLM_REGISTRY.get(judge_model)
        if not judge_cfg:
            log.warning(f"judge: judge model '{judge_model}' not in registry")
            return True, 1.0, f"judge_model_not_found:{judge_model}"

        timeout = judge_cfg.get("llm_call_timeout", 30)
        msgs = []
        sp_folder_rel = judge_cfg.get("system_prompt_folder", "")
        if sp_folder_rel and sp_folder_rel.lower() != "none":
            sp_abs = os.path.join(BASE_DIR, sp_folder_rel)
            sys_text = load_prompt_for_folder(sp_abs)
            if sys_text:
                msgs.append(SystemMessage(content=sys_text))

        msgs.append(HumanMessage(content=prompt))

        llm = _build_lc_llm(judge_model)
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout)
        raw = _content_to_str(response.content)

        passed, score, reason = _parse_judge_response(raw)
        log.info(
            f"judge: gate={gate} model={model_key} judge={judge_model} "
            f"passed={passed} score={score:.2f} reason={reason!r}"
        )
        if score < threshold:
            passed = False

        return passed, score, reason

    except asyncio.TimeoutError:
        log.warning(f"judge: timeout on gate={gate} judge_model={judge_model}")
        return True, 1.0, "judge_timeout"
    except Exception as e:
        log.warning(f"judge: error on gate={gate}: {e}")
        return True, 1.0, f"judge_error:{e}"


async def judge_gate(
    gate: str,
    content: str,
    model_key: str,
    session: dict,
    client_id: str,
    topic: str = "",
) -> tuple[bool, str]:
    """
    High-level gate check with mode enforcement.

    Returns (allowed, denial_reason).
    allowed=True  → proceed normally
    allowed=False → block (only when mode="block")
    """
    cfg = _get_effective_judge_cfg(model_key, session)
    if cfg is None:
        return True, ""

    mode = cfg.get("mode", "block")

    passed, score, reason = await judge_eval(
        gate=gate,
        content=content,
        model_key=model_key,
        session=session,
        topic=topic,
    )

    if passed:
        return True, ""

    msg = f"[judge/{gate}] score={score:.2f} — {reason}"
    log.info(f"judge_gate: FAIL client={client_id} gate={gate} mode={mode} score={score:.2f}")

    if mode == "warn":
        from state import push_tok
        await push_tok(client_id, f"\n{msg}\n")
        return True, ""
    else:
        return False, (
            f"JUDGE BLOCKED [{gate}]: {reason} (score={score:.2f}). "
            f"Do NOT retry the same content. Acknowledge and continue."
        )


# ---------------------------------------------------------------------------
# Session judge_override helpers — used by !judge command and plugin
# ---------------------------------------------------------------------------

def session_judge_status(model_key: str, session: dict) -> str:
    """Return a human-readable summary of active judge config for the session."""
    from config import LLM_REGISTRY
    model_cfg = LLM_REGISTRY.get(model_key, {})
    base = model_cfg.get("judge_config")
    override = session.get("judge_override", {})
    effective = _get_effective_judge_cfg(model_key, session)

    lines = [f"Judge config — model: {model_key}", ""]

    if base:
        lines.append(
            f"  Model default:    model={base.get('model','(none)')}  "
            f"gates={base.get('gates',[])}  "
            f"mode={base.get('mode','block')}  "
            f"threshold={base.get('threshold',0.7)}"
        )
    else:
        lines.append("  Model default:    (none)")

    if override:
        lines.append(f"  Session override: {json.dumps(override)}")
    else:
        lines.append("  Session override: (none)")

    lines.append("")
    if effective:
        lines.append(
            f"  Effective:        model={effective.get('model','(none)')}  "
            f"gates={effective.get('gates',[])}  "
            f"mode={effective.get('mode','block')}  "
            f"threshold={effective.get('threshold',0.7)}"
        )
        plugin_active = _tool_gate_hook is not None
        lines.append(
            f"  Plugin loaded:    {'yes (tool+memory gates active)' if plugin_active else 'no (only prompt+response gates active)'}"
        )
    else:
        lines.append("  Effective:        (disabled — no judge configured)")

    return "\n".join(lines)


def _ensure_session_override(session: dict) -> dict:
    if "judge_override" not in session:
        session["judge_override"] = {}
    return session["judge_override"]


VALID_GATES = {"prompt", "response", "tool", "memory", "all"}


async def cmd_judge(client_id: str, arg: str, session: dict) -> str:
    """
    Handle the !judge command.

    Subcommands
    -----------
    !judge                         — show status
    !judge status                  — show status
    !judge list                    — list all models' judge configs
    !judge on <gate|all>           — enable gate(s) in session override
    !judge off <gate|all>          — disable gate(s) in session override
    !judge model <model-name>      — set judge model in session override
    !judge mode block|warn         — set enforcement mode in session override
    !judge threshold <float>       — set score threshold in session override
    !judge reset                   — clear all session overrides
    !judge test <text>             — run ad-hoc response evaluation
    !judge set <model> <field> <v> — persist judge_config field to llm-models.json
    """
    from config import LLM_REGISTRY, save_llm_model_field
    model_key = session.get("model", "")

    parts = arg.strip().split(None, 2) if arg.strip() else []
    sub = parts[0].lower() if parts else "status"

    # --- status ---
    if sub in ("status", ""):
        return session_judge_status(model_key, session)

    # --- list ---
    if sub == "list":
        lines = ["Judge configs across all models:"]
        for mname, mcfg in sorted(LLM_REGISTRY.items()):
            jcfg = mcfg.get("judge_config")
            if jcfg:
                lines.append(
                    f"  {mname}: judge={jcfg.get('model','?')}  "
                    f"gates={jcfg.get('gates',[])}  "
                    f"mode={jcfg.get('mode','block')}  "
                    f"threshold={jcfg.get('threshold',0.7)}"
                )
            else:
                lines.append(f"  {mname}: (no judge)")
        return "\n".join(lines)

    # --- on <gate|all> ---
    if sub == "on":
        gate_arg = parts[1].lower() if len(parts) > 1 else ""
        if not gate_arg:
            return "Usage: !judge on <prompt|response|tool|memory|all>"
        if gate_arg not in VALID_GATES:
            return f"Unknown gate '{gate_arg}'. Valid: {', '.join(sorted(VALID_GATES))}"
        ov = _ensure_session_override(session)
        # Get current effective gates (from model default + existing override)
        effective = _get_effective_judge_cfg(model_key, session) or {}
        current_gates = list(ov.get("gates", effective.get("gates", [])))
        if gate_arg == "all":
            new_gates = ["prompt", "response", "tool", "memory"]
        else:
            if gate_arg not in current_gates:
                current_gates.append(gate_arg)
            new_gates = current_gates
        ov["gates"] = new_gates
        return f"Judge gates enabled (session): {new_gates}"

    # --- off <gate|all> ---
    if sub == "off":
        gate_arg = parts[1].lower() if len(parts) > 1 else ""
        if not gate_arg:
            return "Usage: !judge off <prompt|response|tool|memory|all>"
        if gate_arg not in VALID_GATES:
            return f"Unknown gate '{gate_arg}'. Valid: {', '.join(sorted(VALID_GATES))}"
        ov = _ensure_session_override(session)
        effective = _get_effective_judge_cfg(model_key, session) or {}
        current_gates = list(ov.get("gates", effective.get("gates", [])))
        if gate_arg == "all":
            new_gates = []
        else:
            new_gates = [g for g in current_gates if g != gate_arg]
        ov["gates"] = new_gates
        return f"Judge gates (session): {new_gates if new_gates else '(all disabled)'}"

    # --- model <model-name> ---
    if sub == "model":
        model_name = parts[1] if len(parts) > 1 else ""
        if not model_name:
            return "Usage: !judge model <model-name>"
        if model_name not in LLM_REGISTRY:
            available = ", ".join(sorted(LLM_REGISTRY.keys()))
            return f"Model '{model_name}' not found.\nAvailable: {available}"
        ov = _ensure_session_override(session)
        ov["model"] = model_name
        return f"Judge model set (session): {model_name}"

    # --- mode block|warn ---
    if sub == "mode":
        mode_val = parts[1].lower() if len(parts) > 1 else ""
        if mode_val not in ("block", "warn"):
            return "Usage: !judge mode <block|warn>"
        ov = _ensure_session_override(session)
        ov["mode"] = mode_val
        return f"Judge mode set (session): {mode_val}"

    # --- threshold <float> ---
    if sub == "threshold":
        thresh_str = parts[1] if len(parts) > 1 else ""
        try:
            thresh = float(thresh_str)
            if not (0.0 <= thresh <= 1.0):
                raise ValueError("out of range")
        except ValueError:
            return "Usage: !judge threshold <0.0–1.0>"
        ov = _ensure_session_override(session)
        ov["threshold"] = thresh
        return f"Judge threshold set (session): {thresh}"

    # --- reset ---
    if sub == "reset":
        session.pop("judge_override", None)
        return "Session judge overrides cleared. Model defaults restored."

    # --- test <text> ---
    if sub == "test":
        test_text = " ".join(parts[1:]) if len(parts) > 1 else ""
        if not test_text:
            return "Usage: !judge test <text to evaluate>"

        cfg = _get_effective_judge_cfg(model_key, session)
        if not cfg or not cfg.get("model"):
            return (
                "No judge model configured for this session/model.\n"
                "Use: !judge model <model-name>  or set judge_config in llm-models.json"
            )

        # Force response gate for ad-hoc test
        test_cfg = dict(cfg)
        test_cfg["gates"] = ["response"]
        orig_override = session.get("judge_override")
        session["judge_override"] = test_cfg
        try:
            passed, score, reason = await judge_eval(
                gate="response",
                content=test_text,
                model_key=model_key,
                session=session,
            )
        finally:
            if orig_override is None:
                session.pop("judge_override", None)
            else:
                session["judge_override"] = orig_override

        verdict = "PASS" if passed else "FAIL"
        return (
            f"Judge test result:\n"
            f"  Verdict:   {verdict}\n"
            f"  Score:     {score:.2f}\n"
            f"  Reason:    {reason}\n"
            f"  Judge:     {cfg.get('model')}\n"
            f"  Threshold: {cfg.get('threshold', 0.7)}"
        )

    # --- set <model> <field> <value> — persist to llm-models.json ---
    if sub == "set":
        # !judge set <model-name> <field> <value>
        # field: model, mode, threshold, gates (comma-separated)
        if len(parts) < 3:
            return "Usage: !judge set <model-name> <field> <value>\n  fields: model, mode, threshold, gates"
        target_model = parts[1]
        rest = parts[2].split(None, 1)
        field = rest[0].lower() if rest else ""
        value_str = rest[1].strip() if len(rest) > 1 else ""

        if target_model not in LLM_REGISTRY:
            return f"Model '{target_model}' not found."
        if field not in ("model", "mode", "threshold", "gates"):
            return f"Unknown field '{field}'. Valid: model, mode, threshold, gates"

        if field == "gates":
            value = [g.strip() for g in value_str.split(",") if g.strip()]
            invalid = [g for g in value if g not in ("prompt", "response", "tool", "memory")]
            if invalid:
                return f"Invalid gates: {invalid}. Valid: prompt, response, tool, memory"
        elif field == "threshold":
            try:
                value = float(value_str)
                if not (0.0 <= value <= 1.0):
                    raise ValueError()
            except ValueError:
                return "threshold must be a float between 0.0 and 1.0"
        elif field == "mode":
            if value_str not in ("block", "warn"):
                return "mode must be 'block' or 'warn'"
            value = value_str
        else:
            if value_str not in LLM_REGISTRY:
                return f"Judge model '{value_str}' not found in registry."
            value = value_str

        existing_jcfg = dict(LLM_REGISTRY.get(target_model, {}).get("judge_config") or {})
        existing_jcfg[field] = value

        ok = save_llm_model_field(target_model, "judge_config", existing_jcfg)
        if ok:
            LLM_REGISTRY[target_model]["judge_config"] = existing_jcfg
            return f"Persisted judge_config[{field}]={value!r} for model '{target_model}'."
        else:
            return "ERROR: failed to save judge_config to llm-models.json."

    return (
        "Usage: !judge [status|list|on|off|model|mode|threshold|reset|test|set]\n"
        "  !judge status                     — show current judge config\n"
        "  !judge list                       — list all models' judge configs\n"
        "  !judge on <gate|all>              — enable gate for this session\n"
        "  !judge off <gate|all>             — disable gate for this session\n"
        "  !judge model <name>               — set judge model for this session\n"
        "  !judge mode block|warn            — set enforcement mode\n"
        "  !judge threshold <0.0-1.0>        — set score threshold\n"
        "  !judge reset                      — clear session overrides\n"
        "  !judge test <text>                — run ad-hoc evaluation\n"
        "  !judge set <model> <field> <val>  — persist config to llm-models.json"
    )
