import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, ".system_prompt")
DRIVE_TOKEN_FILE = os.path.join(BASE_DIR, "token.json")
DRIVE_CREDS_FILE = os.path.join(BASE_DIR, "credentials.json")
LLM_MODELS_FILE = os.path.join(BASE_DIR, "llm-models.json")
LLM_TOOLS_FILE = os.path.join(BASE_DIR, "llm-tools.json")
PLUGINS_ENABLED_FILE = os.path.join(BASE_DIR, "plugins-enabled.json")

# Google Drive
DRIVE_FOLDER_ID = os.getenv("FOLDER_ID")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

# Logging (setup early so load_llm_registry can use it)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AISvc] %(levelname)s %(message)s",
)
log = logging.getLogger("AISvc")


def load_default_model():
    """Load default model from llm-models.json."""
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        return data.get('default_model', '')
    except (FileNotFoundError, Exception) as e:
        log.warning(f"Could not load default_model from llm-models.json: {e}")
        return ''


def save_default_model(model_key: str) -> bool:
    """Persist default_model to llm-models.json. Returns True on success."""
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        data['default_model'] = model_key
        with open(LLM_MODELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_default_model({model_key}): {e}")
        return False


def load_llm_registry():
    """Load enabled models from llm-models.json with API keys from .env."""
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)

        registry = {}
        for name, config in data.get('models', {}).items():
            # Only include enabled models
            if not config.get('enabled', True):
                continue

            # Get API key from environment if specified
            api_key = None
            env_key = config.get('env_key')
            if env_key:
                api_key = os.getenv(env_key)
            elif config.get('host') and 'localhost' not in config.get('host', ''):
                # Local models don't need keys
                api_key = "local-no-key-required"

            _is_openai = config.get('type') == 'OPENAI'
            registry[name] = {
                "model_id": config.get('model_id'),
                "type": config.get('type'),
                "host": config.get('host'),
                "key": api_key,
                "max_context": config.get('max_context', 50),
                "description": config.get('description', ''),

                "llm_call_timeout": config.get('llm_call_timeout', 60),
                "system_prompt_folder": config.get('system_prompt_folder', ''),
                "temperature": config.get('temperature', 1.0),
                "top_p":       config.get('top_p',        1.0 if _is_openai else 0.95),
                "top_k":       config.get('top_k',        None if _is_openai else 40),
                "token_selection_setting": config.get('token_selection_setting', 'default'),
                "llm_tools": config.get('llm_tools', []),
                "llm_tools_gates": config.get('llm_tools_gates', []),
                "memory_scan": config.get('memory_scan', False),
                "max_tokens": config.get('max_tokens'),
                "tool_suppress": config.get('tool_suppress', False),
            }

        return registry
    except FileNotFoundError:
        log.error("llm-models.json not found — no models available. Create llm-models.json to use the server.")
        return {}
    except Exception as e:
        log.error(f"Error loading llm-models.json: {e} — no models available.")
        return {}


def copy_llm_model(source_name: str, new_name: str) -> tuple[bool, str]:
    """
    Copy a model entry in llm-models.json.
    Returns (success, message). The copy is added to LLM_REGISTRY in memory.
    """
    import copy as _copy
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        models = data.get('models', {})
        if source_name not in models:
            return False, f"Source model '{source_name}' not found in llm-models.json."
        if new_name in models:
            return False, f"Model '{new_name}' already exists."
        models[new_name] = _copy.deepcopy(models[source_name])
        with open(LLM_MODELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        # Update LLM_REGISTRY in memory
        src_cfg = LLM_REGISTRY.get(source_name)
        if src_cfg is not None:
            LLM_REGISTRY[new_name] = _copy.deepcopy(src_cfg)
        return True, f"Model '{new_name}' created as a copy of '{source_name}'."
    except Exception as e:
        log.error(f"copy_llm_model({source_name} -> {new_name}): {e}")
        return False, f"ERROR: {e}"


def delete_llm_model(model_name: str) -> tuple[bool, str]:
    """
    Delete a model entry from llm-models.json.
    Returns (success, message). The model is also removed from LLM_REGISTRY in memory.
    """
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        models = data.get('models', {})
        if model_name not in models:
            return False, f"Model '{model_name}' not found in llm-models.json."
        del models[model_name]
        with open(LLM_MODELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        # Remove from in-memory registry
        LLM_REGISTRY.pop(model_name, None)
        return True, f"Model '{model_name}' deleted from llm-models.json and registry."
    except Exception as e:
        log.error(f"delete_llm_model({model_name}): {e}")
        return False, f"ERROR: {e}"


def enable_llm_model(model_name: str) -> tuple[bool, str]:
    """
    Set enabled=true for a model in llm-models.json.
    Returns (success, message). Does NOT update LLM_REGISTRY — requires restart.
    """
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        models = data.get('models', {})
        if model_name not in models:
            return False, f"Model '{model_name}' not found in llm-models.json."
        if models[model_name].get('enabled', True):
            return True, f"Model '{model_name}' is already enabled."
        models[model_name]['enabled'] = True
        with open(LLM_MODELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True, f"Model '{model_name}' enabled in llm-models.json. Restart server for changes to take effect."
    except Exception as e:
        log.error(f"enable_llm_model({model_name}): {e}")
        return False, f"ERROR: {e}"


def disable_llm_model(model_name: str) -> tuple[bool, str]:
    """
    Set enabled=false for a model in llm-models.json.
    Returns (success, message). Does NOT update LLM_REGISTRY — requires restart.
    Refuses to disable the current default model.
    """
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        models = data.get('models', {})
        if model_name not in models:
            return False, f"Model '{model_name}' not found in llm-models.json."
        if data.get('default_model') == model_name:
            return False, f"Cannot disable '{model_name}' — it is the default model. Set a different default first."
        if not models[model_name].get('enabled', True):
            return True, f"Model '{model_name}' is already disabled."
        models[model_name]['enabled'] = False
        with open(LLM_MODELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True, f"Model '{model_name}' disabled in llm-models.json. Restart server for changes to take effect."
    except Exception as e:
        log.error(f"disable_llm_model({model_name}): {e}")
        return False, f"ERROR: {e}"


def save_llm_model_field(model_name: str, field: str, value) -> bool:
    """Persist a single field for a model in llm-models.json. Returns True on success."""
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        if model_name not in data.get('models', {}):
            log.error(f"save_llm_model_field: model '{model_name}' not found in JSON")
            return False
        data['models'][model_name][field] = value
        with open(LLM_MODELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_llm_model_field({model_name}, {field}): {e}")
        return False


def load_llm_tools() -> dict:
    """Load named toolsets from llm-tools.json. Returns {name: [tool_names]}."""
    try:
        with open(LLM_TOOLS_FILE, 'r') as f:
            data = json.load(f)
        return data.get('toolsets', {})
    except FileNotFoundError:
        log.error("llm-tools.json not found — no toolsets available.")
        return {}
    except Exception as e:
        log.error(f"Error loading llm-tools.json: {e}")
        return {}


def save_llm_toolset(name: str, tools: list[str]) -> bool:
    """Persist a single toolset to llm-tools.json. Returns True on success."""
    try:
        with open(LLM_TOOLS_FILE, 'r') as f:
            data = json.load(f)
        data.setdefault('toolsets', {})[name] = tools
        with open(LLM_TOOLS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_llm_toolset({name}): {e}")
        return False


def delete_llm_toolset(name: str) -> tuple[bool, str]:
    """Delete a toolset from llm-tools.json. Returns (success, message)."""
    try:
        with open(LLM_TOOLS_FILE, 'r') as f:
            data = json.load(f)
        toolsets = data.get('toolsets', {})
        if name not in toolsets:
            return False, f"Toolset '{name}' not found."
        del toolsets[name]
        with open(LLM_TOOLS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True, f"Toolset '{name}' deleted."
    except Exception as e:
        log.error(f"delete_llm_toolset({name}): {e}")
        return False, f"ERROR: {e}"


def load_rate_limits() -> dict:
    """Load rate_limits section from plugins-enabled.json."""
    defaults = {
        "llm_call": {"calls": 3,  "window_seconds": 20, "auto_disable": True},
        "search":   {"calls": 5,  "window_seconds": 10, "auto_disable": False},
        "drive":    {"calls": 10, "window_seconds": 60, "auto_disable": False},
        "db":       {"calls": 20, "window_seconds": 60, "auto_disable": False},
        "system":   {"calls": 0,  "window_seconds": 0,  "auto_disable": False},
    }
    try:
        with open(PLUGINS_ENABLED_FILE, 'r') as f:
            config = json.load(f)
        loaded = config.get('rate_limits', {})
        # Merge with defaults so missing keys always have a value
        for tool_type, def_cfg in defaults.items():
            if tool_type not in loaded:
                loaded[tool_type] = def_cfg
            else:
                for k, v in def_cfg.items():
                    loaded[tool_type].setdefault(k, v)
        return loaded
    except Exception as e:
        log.warning(f"Could not load rate_limits from plugins-enabled.json: {e}, using defaults")
        return defaults


def save_rate_limit(tool_type: str, field: str, value) -> bool:
    """Persist a single rate_limit field in plugins-enabled.json. Returns True on success."""
    try:
        with open(PLUGINS_ENABLED_FILE, 'r') as f:
            data = json.load(f)
        if 'rate_limits' not in data:
            data['rate_limits'] = {}
        if tool_type not in data['rate_limits']:
            data['rate_limits'][tool_type] = {}
        data['rate_limits'][tool_type][field] = value
        with open(PLUGINS_ENABLED_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_rate_limit({tool_type}, {field}): {e}")
        return False



def load_limits() -> dict:
    """Load depth/iteration limits from llm-models.json 'limits' section."""
    try:
        with open(LLM_MODELS_FILE, "r") as f:
            data = json.load(f)
        return data.get("limits", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_limit_field(key: str, value: int) -> bool:
    """Persist a single limits field to llm-models.json."""
    try:
        with open(LLM_MODELS_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("limits", {})[key] = value
        with open(LLM_MODELS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_limit_field failed: key={key} value={value} err={e}")
        return False


# Load LLM Registry from JSON (only enabled models)
LLM_REGISTRY = load_llm_registry()

# Load default model from plugins-enabled.json
DEFAULT_MODEL = load_default_model()
MAX_TOOL_ITERATIONS = 10  # fallback only; runtime value lives in LIVE_LIMITS["max_tool_iterations"]

_limits = load_limits()
MAX_AT_LLM_DEPTH = int(_limits.get("max_at_llm_depth", 1))
MAX_AGENT_CALL_DEPTH = int(_limits.get("max_agent_call_depth", 1))

# Live mutable depth limits — agents.py reads from this dict so runtime changes take effect immediately.
# Populated from llm-models.json at startup; mutated in-place by limit_depth_set / limit_max_iteration_set.
LIVE_LIMITS: dict = {
    "max_at_llm_depth":     MAX_AT_LLM_DEPTH,
    "max_agent_call_depth":  MAX_AGENT_CALL_DEPTH,
    "max_tool_iterations":   int(_limits.get("max_tool_iterations", MAX_TOOL_ITERATIONS)),
}

# Named toolsets — loaded from llm-tools.json
LLM_TOOLSETS = load_llm_tools()

# Rate limits by tool type — loaded from plugins-enabled.json
RATE_LIMITS = load_rate_limits()