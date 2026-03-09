"""
Plugin loader and base class for MCP agent system.

Provides:
- BasePlugin abstract class for all plugins
- PluginLoader for dynamic plugin loading
- Plugin validation and dependency checking
"""

import os
import sys
import json
import importlib
import importlib.util
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from starlette.routing import Route
from config import log


class BasePlugin(ABC):
    """
    Abstract base class for all MCP plugins.

    Plugins must implement:
    - PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_TYPE, DESCRIPTION
    - init() and shutdown() lifecycle methods
    - get_tools() for data_tool plugins OR get_routes() for client_interface plugins
    """

    # Metadata (must be set by subclass)
    PLUGIN_NAME: str = "unknown"
    PLUGIN_VERSION: str = "0.0.0"
    PLUGIN_TYPE: str = "unknown"  # "client_interface" or "data_tool"
    DESCRIPTION: str = ""
    DEPENDENCIES: List[str] = []
    ENV_VARS: List[str] = []

    @abstractmethod
    def init(self, config: dict) -> bool:
        """
        Initialize the plugin with given configuration.

        Args:
            config: Plugin-specific configuration from plugins-enabled.json

        Returns:
            True if initialization succeeded, False otherwise
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources when plugin is unloaded."""
        pass

    def get_tools(self) -> Dict[str, Any]:
        """
        Return tool definitions for data_tool plugins.

        Returns:
            Dict with:
            - "lc": List of LangChain StructuredTool objects
              Executors are auto-extracted from the coroutine attribute of each tool.
        """
        return {"lc": []}

    def get_routes(self) -> List[Route]:
        """
        Return Starlette routes for client_interface plugins.

        Returns:
            List of Route objects
        """
        return []

    def get_commands(self) -> Dict[str, Any]:
        """
        Return special ! commands this plugin provides.

        Returns:
            Dict mapping command name to handler function:
            {
                "mycommand": async_handler_function
            }
        Handler signature: async (args: str) -> str
        """
        return {}

    def get_help(self) -> str:
        """
        Return the !help section string for this plugin's commands.

        The string is appended to the help output from cmd_help().
        Return an empty string if the plugin has no user commands.
        """
        return ""


class PluginLoader:
    """
    Loads and manages plugins based on manifest and configuration.
    """

    def __init__(self, manifest_path: str = "plugin-manifest.json",
                 config_path: str = "plugins-enabled.json"):
        self.manifest_path = manifest_path
        self.config_path = config_path
        self.manifest = {}
        self.config = {}
        self.loaded_plugins: Dict[str, BasePlugin] = {}

    def load_manifest(self) -> bool:
        """Load plugin manifest file."""
        try:
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
            log.info(f"Loaded plugin manifest: {len(self.manifest.get('plugins', {}))} plugins defined")
            return True
        except FileNotFoundError:
            log.error(f"Plugin manifest not found: {self.manifest_path}")
            return False
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in manifest: {e}")
            return False

    def load_config(self) -> bool:
        """Load plugins-enabled configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            log.info(f"Loaded plugin config: {len(self.config.get('enabled_plugins', []))} plugins enabled")
            return True
        except FileNotFoundError:
            log.error(f"Plugin config not found: {self.config_path}")
            return False
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in config: {e}")
            return False

    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            log.error(f"Failed to save config: {e}")
            return False

    def validate_plugin(self, plugin_name: str) -> tuple[bool, List[str]]:
        """
        Validate plugin dependencies and environment variables.

        Returns:
            (is_valid, list_of_issues)
        """
        if plugin_name not in self.manifest.get('plugins', {}):
            return False, [f"Plugin '{plugin_name}' not found in manifest"]

        plugin_meta = self.manifest['plugins'][plugin_name]
        issues = []

        # Check if plugin file exists
        plugin_file = plugin_meta.get('file')
        if not os.path.exists(plugin_file):
            issues.append(f"Plugin file not found: {plugin_file}")

        # Check environment variables
        for env_var in plugin_meta.get('env_vars', []):
            if not os.getenv(env_var):
                issues.append(f"Missing environment variable: {env_var}")

        # Check config files
        for config_file in plugin_meta.get('config_files', []):
            if not os.path.exists(config_file):
                issues.append(f"Missing config file: {config_file}")

        # Check dependencies (basic check - just see if importable)
        for dep in plugin_meta.get('dependencies', []):
            # Parse dependency (e.g., "mysql-connector-python>=8.0" -> "mysql.connector")
            dep_name = dep.split('>=')[0].split('==')[0].split('<')[0].strip()
            # Convert package name to import name
            import_name = dep_name.replace('-', '_')
            try:
                importlib.import_module(import_name)
            except ImportError:
                # Try alternate import patterns
                try:
                    if 'mysql-connector' in dep_name:
                        importlib.import_module('mysql.connector')
                    elif 'google-api-python-client' in dep_name:
                        importlib.import_module('googleapiclient')
                    elif 'google-auth' in dep_name:
                        importlib.import_module('google.auth')
                    elif 'tavily-python' in dep_name:
                        importlib.import_module('tavily')
                    elif 'google-genai' in dep_name:
                        importlib.import_module('google.genai')
                    else:
                        raise
                except ImportError:
                    issues.append(f"Missing dependency: {dep}")

        return len(issues) == 0, issues

    def load_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Load a single plugin by name.

        Returns:
            Plugin instance if successful, None otherwise
        """
        if plugin_name not in self.manifest.get('plugins', {}):
            log.error(f"Plugin '{plugin_name}' not in manifest")
            return None

        # Validate first
        is_valid, issues = self.validate_plugin(plugin_name)
        if not is_valid:
            log.error(f"Plugin '{plugin_name}' validation failed: {', '.join(issues)}")
            return None

        plugin_meta = self.manifest['plugins'][plugin_name]
        plugin_file = plugin_meta['file']

        try:
            # Dynamic import
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            if spec is None or spec.loader is None:
                log.error(f"Failed to load spec for {plugin_file}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)

            # Find the plugin class (should be named same as file without .py)
            class_name = plugin_name.replace('plugin_', '').replace('_', ' ').title().replace(' ', '') + 'Plugin'

            # Try common naming patterns
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BasePlugin) and attr != BasePlugin:
                    plugin_class = attr
                    break

            if plugin_class is None:
                log.error(f"No BasePlugin subclass found in {plugin_file}")
                return None

            # Instantiate
            plugin_instance = plugin_class()

            # Get plugin-specific config
            plugin_config = self.config.get('plugin_config', {}).get(plugin_name, {})

            # Initialize
            if not plugin_instance.init(plugin_config):
                log.error(f"Plugin '{plugin_name}' initialization failed")
                return None

            log.info(f"✓ Loaded plugin: {plugin_name} ({plugin_meta.get('description', '')})")
            return plugin_instance

        except Exception as e:
            log.error(f"Failed to load plugin '{plugin_name}': {e}", exc_info=True)
            return None

    def load_all_enabled(self) -> Dict[str, BasePlugin]:
        """
        Load all enabled plugins from configuration.

        Returns:
            Dict mapping plugin names to plugin instances
        """
        if not self.load_manifest() or not self.load_config():
            log.error("Failed to load manifest or config")
            return {}

        enabled_plugins = self.config.get('enabled_plugins', [])

        for plugin_name in enabled_plugins:
            # Respect explicit enabled: false in plugin_config (allows disabling
            # without removing from enabled_plugins list, e.g. llama proxy)
            plugin_cfg = self.config.get('plugin_config', {}).get(plugin_name, {})
            if plugin_cfg.get('enabled') is False:
                log.info(f"  Skipping '{plugin_name}' (enabled: false in plugin_config)")
                continue

            plugin = self.load_plugin(plugin_name)
            if plugin:
                self.loaded_plugins[plugin_name] = plugin

        log.info(f"Loaded {len(self.loaded_plugins)} / {len(enabled_plugins)} enabled plugins")
        return self.loaded_plugins

    def unload_all(self) -> None:
        """Unload all plugins and clean up resources."""
        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                plugin.shutdown()
                log.info(f"Unloaded plugin: {plugin_name}")
            except Exception as e:
                log.error(f"Error unloading plugin '{plugin_name}': {e}")

        self.loaded_plugins.clear()

    def get_default_model(self) -> str:
        """Get default LLM model from llm-models.json."""
        from config import load_default_model
        return load_default_model()

    def set_default_model(self, model_key: str) -> bool:
        """Set default LLM model in llm-models.json."""
        from config import save_default_model
        return save_default_model(model_key)
