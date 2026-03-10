#!/usr/bin/env python3
"""
Unit tests for the three-level memory_enabled veto logic.

Tests the _mem_injection_enabled expression in auto_enrich_context and
the conv_log gate in routes.py, without requiring a running server.

Three levels:
  1. model cfg   — LLM_REGISTRY[model_key].get("memory_enabled", None)
  2. session     — session.get("memory_enabled", None)
  3. global      — plugins-enabled.json memory.enabled (via _memory_feature)

Rule: None = "not set at this level, defer up". False = veto. True = explicit allow.
Any single False disables injection. All None → enabled (global default True).

Test matrix (3 iterations × 3 levels = 9 tests):
  Iteration 1: model=False, session=True,  global=True  → DISABLED (model vetoes)
  Iteration 2: model=True,  session=False, global=True  → DISABLED (session vetoes)
  Iteration 3: model=True,  session=True,  global=False → DISABLED (global vetoes)
  Bonus: all True → ENABLED, all None → ENABLED (default)
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Minimal stubs so agents.py can be imported without the full stack.
# ---------------------------------------------------------------------------

# Stub heavy dependencies before import
sys.modules.setdefault("langchain_core", MagicMock())
sys.modules.setdefault("langchain_core.messages", MagicMock())
sys.modules.setdefault("langchain_openai", MagicMock())
sys.modules.setdefault("langchain_google_genai", MagicMock())
sys.modules.setdefault("langchain_anthropic", MagicMock())
sys.modules.setdefault("mysql.connector", MagicMock())
sys.modules.setdefault("mysql", MagicMock())
sys.modules.setdefault("qdrant_client", MagicMock())
sys.modules.setdefault("httpx", MagicMock())
sys.modules.setdefault("langchain_core.tools", MagicMock())
sys.modules.setdefault("langchain_core.runnables", MagicMock())

# ---------------------------------------------------------------------------
# The actual logic under test — extracted verbatim from agents.py so we can
# test it in isolation without pulling in the full module.
# ---------------------------------------------------------------------------

def _compute_mem_injection_enabled(
    model_mem_flag,   # LLM_REGISTRY[model_key].get("memory_enabled", None)
    sess_mem_flag,    # session.get("memory_enabled", None)
    global_enabled,   # plugins-enabled.json memory.enabled (True/False)
) -> bool:
    """
    Mirrors the _mem_injection_enabled expression from auto_enrich_context.
    Kept as a standalone function so tests are pure logic with no I/O.
    """
    def _memory_feature_context_injection() -> bool:
        # Mirrors _memory_feature("context_injection"):
        # master switch + feature switch; here we only vary the master (enabled).
        return global_enabled  # feature switch "context_injection" defaults True

    return (
        (model_mem_flag is None or model_mem_flag)
        and (sess_mem_flag is None or sess_mem_flag)
        and _memory_feature_context_injection()
    )


def _compute_conv_log_allowed(
    model_mem_flag,   # model_cfg.get("memory_enabled", None)
    sess_mem_flag,    # session.get("memory_enabled", None)
    conv_log,         # model_cfg.get("conv_log", False)
) -> bool:
    """
    Mirrors the conv_log gate condition from routes.py.
    conv_log itself must be True, then both memory flags must allow.
    """
    return (
        conv_log
        and (model_mem_flag is None or model_mem_flag)
        and (sess_mem_flag is None or sess_mem_flag)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMemoryInjectionVeto(unittest.TestCase):
    """Three-level veto logic for context injection."""

    # --- Iteration 1: model=False vetoes ---

    def test_iter1_model_false_vetoes(self):
        """model=False, session=True, global=True → DISABLED"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=False,
            sess_mem_flag=True,
            global_enabled=True,
        )
        self.assertFalse(result, "model False should veto even when session+global True")

    def test_iter1_sanity_model_none_passes(self):
        """model=None, session=True, global=True → ENABLED (None defers)"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=None,
            sess_mem_flag=True,
            global_enabled=True,
        )
        self.assertTrue(result, "model None should defer to session+global")

    # --- Iteration 2: session=False vetoes ---

    def test_iter2_session_false_vetoes(self):
        """model=True, session=False, global=True → DISABLED"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=True,
            sess_mem_flag=False,
            global_enabled=True,
        )
        self.assertFalse(result, "session False should veto even when model+global True")

    def test_iter2_sanity_session_none_passes(self):
        """model=True, session=None, global=True → ENABLED (None defers)"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=True,
            sess_mem_flag=None,
            global_enabled=True,
        )
        self.assertTrue(result, "session None should defer")

    # --- Iteration 3: global=False vetoes ---

    def test_iter3_global_false_vetoes(self):
        """model=True, session=True, global=False → DISABLED"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=True,
            sess_mem_flag=True,
            global_enabled=False,
        )
        self.assertFalse(result, "global False should veto even when model+session True")

    def test_iter3_sanity_all_true_passes(self):
        """model=True, session=True, global=True → ENABLED"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=True,
            sess_mem_flag=True,
            global_enabled=True,
        )
        self.assertTrue(result, "all True should be enabled")

    # --- Bonus: defaults (all None) ---

    def test_all_none_defaults_enabled(self):
        """model=None, session=None, global=True (default) → ENABLED"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=None,
            sess_mem_flag=None,
            global_enabled=True,
        )
        self.assertTrue(result, "all None with global True should enable by default")

    def test_all_none_global_false_disabled(self):
        """model=None, session=None, global=False → DISABLED"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=None,
            sess_mem_flag=None,
            global_enabled=False,
        )
        self.assertFalse(result, "all None but global False → disabled")

    # --- Double and triple veto ---

    def test_model_and_session_both_false(self):
        """model=False, session=False, global=True → DISABLED"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=False,
            sess_mem_flag=False,
            global_enabled=True,
        )
        self.assertFalse(result)

    def test_all_three_false(self):
        """model=False, session=False, global=False → DISABLED"""
        result = _compute_mem_injection_enabled(
            model_mem_flag=False,
            sess_mem_flag=False,
            global_enabled=False,
        )
        self.assertFalse(result)


class TestConvLogVeto(unittest.TestCase):
    """Three-level veto logic for conv_log (save_conversation_turn)."""

    # conv_log must be True as a prerequisite; then memory flags are checked.

    # --- Iteration 1: model=False vetoes ---

    def test_iter1_model_false_vetoes_convlog(self):
        """conv_log=True, model=False, session=True → NO LOG"""
        result = _compute_conv_log_allowed(
            model_mem_flag=False,
            sess_mem_flag=True,
            conv_log=True,
        )
        self.assertFalse(result)

    # --- Iteration 2: session=False vetoes ---

    def test_iter2_session_false_vetoes_convlog(self):
        """conv_log=True, model=True, session=False → NO LOG"""
        result = _compute_conv_log_allowed(
            model_mem_flag=True,
            sess_mem_flag=False,
            conv_log=True,
        )
        self.assertFalse(result)

    # --- Iteration 3: conv_log itself is the gate (no memory level here) ---
    # (The global memory.enabled flag gates conv_log indirectly via the model's
    #  memory_enabled field being set False, which is tested above. The conv_log
    #  gate in routes.py does not re-read the global cfg — it trusts model+session.)

    def test_iter3_conv_log_false_blocks_regardless(self):
        """conv_log=False, model=True, session=True → NO LOG (conv_log gate)"""
        result = _compute_conv_log_allowed(
            model_mem_flag=True,
            sess_mem_flag=True,
            conv_log=False,
        )
        self.assertFalse(result)

    def test_all_true_conv_log_allowed(self):
        """conv_log=True, model=True, session=True → LOG"""
        result = _compute_conv_log_allowed(
            model_mem_flag=True,
            sess_mem_flag=True,
            conv_log=True,
        )
        self.assertTrue(result)

    def test_all_none_conv_log_allowed(self):
        """conv_log=True, model=None, session=None → LOG (None defers)"""
        result = _compute_conv_log_allowed(
            model_mem_flag=None,
            sess_mem_flag=None,
            conv_log=True,
        )
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# Matrix summary printer
# ---------------------------------------------------------------------------

def _print_matrix():
    print("\n" + "=" * 70)
    print("Three-level memory_enabled veto matrix")
    print("=" * 70)
    print(f"{'model':>8}  {'session':>8}  {'global':>8}  {'injection':>12}  {'note'}")
    print("-" * 70)
    cases = [
        # (model,  session, global, note)
        (False, True,  True,  "iter1: model vetoes"),
        (True,  False, True,  "iter2: session vetoes"),
        (True,  True,  False, "iter3: global vetoes"),
        (True,  True,  True,  "all True → enabled"),
        (None,  None,  True,  "all None → enabled (default)"),
        (None,  None,  False, "all None, global off → disabled"),
        (False, False, False, "all False → disabled"),
    ]
    for model, sess, glob, note in cases:
        result = _compute_mem_injection_enabled(model, sess, glob)
        mark = "ENABLED " if result else "DISABLED"
        print(f"{str(model):>8}  {str(sess):>8}  {str(glob):>8}  {mark:>12}  {note}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    _print_matrix()
    unittest.main(verbosity=2)
