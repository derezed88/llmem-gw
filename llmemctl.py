#!/usr/bin/env python3
"""
Plugin Manager for MCP Agent System

Interactive CLI tool for managing plugins:
- List all plugins with status (enabled, configured, missing, unavailable)
- Enable/disable plugins
- Set default LLM model
- Validate plugin dependencies and environment variables
- Show plugin details
"""

import os
import sys
import json
import importlib
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Status colors
    GREEN = '\033[32m'    # Enabled
    YELLOW = '\033[33m'   # Configured
    RED = '\033[31m'      # Missing
    GRAY = '\033[90m'     # Unavailable
    CYAN = '\033[36m'     # Info
    MAGENTA = '\033[35m'  # Highlight


class PluginManager:
    """Manage MCP agent plugins and LLM models."""

    def __init__(self):
        self.manifest_path = "plugin-manifest.json"
        self.config_path = "plugins-enabled.json"
        self.models_path = "llm-models.json"
        self.manifest = {}
        self.config = {}
        self.models = {}

    def load_files(self) -> bool:
        """Load manifest, config, and models files."""
        try:
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        except FileNotFoundError:
            print(f"{Colors.RED}Error: {self.manifest_path} not found{Colors.RESET}")
            return False
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}Error: Invalid JSON in {self.manifest_path}: {e}{Colors.RESET}")
            return False

        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"{Colors.RED}Error: {self.config_path} not found{Colors.RESET}")
            return False
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}Error: Invalid JSON in {self.config_path}: {e}{Colors.RESET}")
            return False

        try:
            with open(self.models_path, 'r') as f:
                self.models = json.load(f)
        except FileNotFoundError:
            print(f"{Colors.RED}Error: {self.models_path} not found{Colors.RESET}")
            return False
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}Error: Invalid JSON in {self.models_path}: {e}{Colors.RESET}")
            return False

        return True

    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"{Colors.RED}Error saving config: {e}{Colors.RESET}")
            return False

    def get_plugin_status(self, plugin_name: str) -> Tuple[str, str, str]:
        """
        Get plugin status.

        Returns:
            (status, color, symbol) tuple
            - status: "enabled", "configured", "missing", "unavailable"
            - color: ANSI color code
            - symbol: Status symbol
        """
        if plugin_name not in self.manifest.get('plugins', {}):
            return ("unknown", Colors.GRAY, "?")

        plugin_meta = self.manifest['plugins'][plugin_name]
        enabled_plugins = self.config.get('enabled_plugins', [])

        # Check if enabled
        if plugin_name in enabled_plugins:
            # Check for explicit runtime kill switch (enabled: false in plugin_config)
            plugin_cfg = self.config.get('plugin_config', {}).get(plugin_name, {})
            if plugin_cfg.get('enabled') is False:
                return ("disabled", Colors.YELLOW, "–")

            # Verify all requirements
            issues = self.validate_plugin(plugin_name)[1]
            if not issues:
                return ("enabled", Colors.GREEN, "✓")
            else:
                return ("enabled_invalid", Colors.RED, "✗")

        # Check if file exists and dependencies are met
        plugin_file = plugin_meta.get('file')
        if not os.path.exists(plugin_file):
            return ("missing", Colors.RED, "✗")

        # Check dependencies and env vars
        issues = self.validate_plugin(plugin_name)[1]
        if issues:
            return ("unavailable", Colors.GRAY, "⊗")

        return ("configured", Colors.YELLOW, "○")

    def validate_plugin(self, plugin_name: str) -> Tuple[bool, List[str]]:
        """
        Validate plugin requirements.

        Returns:
            (is_valid, list_of_issues)
        """
        if plugin_name not in self.manifest.get('plugins', {}):
            return False, [f"Plugin not in manifest"]

        plugin_meta = self.manifest['plugins'][plugin_name]
        issues = []

        # Check file exists
        plugin_file = plugin_meta.get('file')
        if not os.path.exists(plugin_file):
            issues.append(f"File not found: {plugin_file}")

        # Check environment variables
        for env_var in plugin_meta.get('env_vars', []):
            if not os.getenv(env_var):
                issues.append(f"Missing env var: {env_var}")

        # Check config files
        for config_file in plugin_meta.get('config_files', []):
            if not os.path.exists(config_file):
                issues.append(f"Missing config file: {config_file}")

        # Check dependencies
        for dep in plugin_meta.get('dependencies', []):
            dep_name = dep.split('>=')[0].split('==')[0].split('<')[0].strip()
            import_name = dep_name.replace('-', '_')
            try:
                importlib.import_module(import_name)
            except ImportError:
                # Try alternate names
                try:
                    if 'mysql-connector' in dep_name:
                        importlib.import_module('mysql.connector')
                    elif 'google-api-python-client' in dep_name:
                        importlib.import_module('googleapiclient')
                    elif 'google-auth' in dep_name:
                        importlib.import_module('google.auth')
                    elif 'tavily-python' in dep_name:
                        importlib.import_module('tavily')
                    else:
                        raise
                except ImportError:
                    issues.append(f"Missing dependency: {dep}")

        return len(issues) == 0, issues

    def list_plugins(self):
        """List all plugins with status."""
        print(f"\n{Colors.BOLD}MCP Agent Plugin Manager{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")

        # Show default model
        default_model = self.models.get('default_model', 'unknown')
        print(f"{Colors.BOLD}Default LLM:{Colors.RESET} {Colors.MAGENTA}{default_model}{Colors.RESET}")

        # Show quick status summary
        enabled_plugins = self.config.get('enabled_plugins', [])
        ready_count = 0
        issue_count = 0

        for plugin_name in enabled_plugins:
            if plugin_name in self.manifest.get('plugins', {}):
                plugin_cfg = self.config.get('plugin_config', {}).get(plugin_name, {})
                if plugin_cfg.get('enabled') is False:
                    continue  # disabled plugins don't count toward ready or issues
                is_valid, issues = self.validate_plugin(plugin_name)
                if is_valid:
                    ready_count += 1
                else:
                    issue_count += 1

        if issue_count > 0:
            print(f"{Colors.BOLD}Status:{Colors.RESET} {Colors.GREEN}{ready_count} ready{Colors.RESET}, {Colors.RED}{issue_count} need setup{Colors.RESET}\n")
        else:
            print(f"{Colors.BOLD}Status:{Colors.RESET} {Colors.GREEN}All {ready_count} enabled plugins ready!{Colors.RESET}\n")

        # Group plugins by type
        plugins = self.manifest.get('plugins', {})
        client_plugins = {k: v for k, v in plugins.items() if v.get('type') == 'client_interface'}
        data_plugins = {k: v for k, v in plugins.items() if v.get('type') == 'data_tool'}

        print(f"{Colors.BOLD}Client Interface Plugins:{Colors.RESET}")
        self._print_plugin_group(client_plugins)

        print(f"\n{Colors.BOLD}Data Access Tool Plugins:{Colors.RESET}")
        self._print_plugin_group(data_plugins)

        print(f"\n{Colors.BOLD}Legend:{Colors.RESET}")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Enabled     - Plugin active and ready")
        print(f"  {Colors.YELLOW}–{Colors.RESET} Disabled    - In enabled_plugins but turned off (enabled: false in plugin_config)")
        print(f"  {Colors.YELLOW}○{Colors.RESET} Configured  - Plugin available but not enabled")
        print(f"  {Colors.RED}✗{Colors.RESET} Has Issues  - Missing dependencies, env vars, or config files")
        print(f"  {Colors.GRAY}⊗{Colors.RESET} Unavailable - Not enabled and has unresolved issues")

        # Show helpful tips if there are issues
        enabled_with_issues = []
        for plugin_name in self.config.get('enabled_plugins', []):
            if plugin_name in self.manifest.get('plugins', {}):
                issues = self.validate_plugin(plugin_name)[1]
                if issues:
                    enabled_with_issues.append(plugin_name)

        if enabled_with_issues:
            print(f"\n{Colors.CYAN}Quick Setup:{Colors.RESET}")

            # Collect all missing dependencies from enabled plugins
            all_missing_deps = set()
            all_missing_envs = set()
            all_missing_files = set()

            for plugin_name in enabled_with_issues:
                is_valid, issues = self.validate_plugin(plugin_name)
                for issue in issues:
                    if 'Missing dependency' in issue:
                        all_missing_deps.add(issue.split(': ')[1])
                    elif 'Missing env var' in issue:
                        all_missing_envs.add(issue.split(': ')[1])
                    elif 'Missing config file' in issue:
                        all_missing_files.add(issue.split(': ')[1])

            # Show install all command if there are dependencies
            if all_missing_deps:
                print(f"  Install all: {Colors.BOLD}pip install {' '.join(sorted(all_missing_deps))}{Colors.RESET}")

            if all_missing_envs or all_missing_files:
                print(f"  Then configure: .env vars and config files")

            print(f"\n{Colors.CYAN}Detailed help:{Colors.RESET} {Colors.BOLD}python llmemctl.py info <plugin_name>{Colors.RESET}")

        print()

    def _print_plugin_group(self, plugins: Dict):
        """Print a group of plugins."""
        if not plugins:
            print("  (none)")
            return

        for plugin_name, plugin_meta in sorted(plugins.items()):
            status, color, symbol = self.get_plugin_status(plugin_name)
            desc = plugin_meta.get('description', '')
            print(f"  {color}{symbol}{Colors.RESET} {Colors.BOLD}{plugin_name}{Colors.RESET}")
            print(f"     {desc}")

            # Show brief requirements info if plugin has issues
            if status in ("enabled_invalid", "unavailable", "missing"):
                is_valid, issues = self.validate_plugin(plugin_name)
                if issues:
                    # Group issues by type
                    deps = [issue.split(': ')[1] for issue in issues if 'Missing dependency' in issue]
                    envs = [issue.split(': ')[1] for issue in issues if 'Missing env var' in issue]
                    files = [issue.split(': ')[1] for issue in issues if 'Missing config file' in issue]

                    issue_parts = []
                    if deps:
                        issue_parts.append(f"pip: {', '.join(deps)}")
                    if envs:
                        issue_parts.append(f".env: {', '.join(envs)}")
                    if files:
                        issue_parts.append(f"files: {', '.join(files)}")

                    if issue_parts:
                        print(f"     {Colors.GRAY}Missing: {' | '.join(issue_parts)}{Colors.RESET}")

    def show_plugin_info(self, plugin_name: str):
        """Show detailed information about a plugin."""
        if plugin_name not in self.manifest.get('plugins', {}):
            print(f"{Colors.RED}Plugin '{plugin_name}' not found in manifest{Colors.RESET}")
            return

        plugin_meta = self.manifest['plugins'][plugin_name]
        status, color, symbol = self.get_plugin_status(plugin_name)

        print(f"\n{Colors.BOLD}{plugin_name}{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"Status:       {color}{symbol} {status}{Colors.RESET}")
        print(f"Type:         {plugin_meta.get('type')}")
        print(f"Description:  {plugin_meta.get('description')}")
        print(f"File:         {plugin_meta.get('file')}")
        print(f"Priority:     {plugin_meta.get('priority')}")

        # Dependencies
        deps = plugin_meta.get('dependencies', [])
        if deps:
            print(f"\nDependencies:")
            for dep in deps:
                print(f"  - {dep}")

        # Environment variables
        env_vars = plugin_meta.get('env_vars', [])
        if env_vars:
            print(f"\nEnvironment Variables:")
            for env_var in env_vars:
                has_var = bool(os.getenv(env_var))
                symbol = f"{Colors.GREEN}✓{Colors.RESET}" if has_var else f"{Colors.RED}✗{Colors.RESET}"
                print(f"  {symbol} {env_var}")

        # Config files
        config_files = plugin_meta.get('config_files', [])
        if config_files:
            print(f"\nConfiguration Files:")
            for config_file in config_files:
                exists = os.path.exists(config_file)
                symbol = f"{Colors.GREEN}✓{Colors.RESET}" if exists else f"{Colors.RED}✗{Colors.RESET}"
                print(f"  {symbol} {config_file}")

        # Tools provided
        tools = plugin_meta.get('tools', [])
        if tools:
            print(f"\nTools Provided:")
            for tool in tools:
                print(f"  - {tool}")

        # Validation
        is_valid, issues = self.validate_plugin(plugin_name)
        if not is_valid:
            print(f"\n{Colors.RED}Issues:{Colors.RESET}")
            for issue in issues:
                print(f"  - {issue}")

            # Show resolution steps
            print(f"\n{Colors.CYAN}To resolve:{Colors.RESET}")

            # Check for missing dependencies
            missing_deps = [issue.split(': ')[1] for issue in issues if 'Missing dependency' in issue]
            if missing_deps:
                print(f"\n1. Install missing Python packages:")
                print(f"   {Colors.BOLD}source venv/bin/activate{Colors.RESET}")
                print(f"   {Colors.BOLD}pip install {' '.join(missing_deps)}{Colors.RESET}")

            # Check for missing env vars
            missing_env = [issue.split(': ')[1] for issue in issues if 'Missing env var' in issue]
            if missing_env:
                print(f"\n2. Add environment variables to .env file:")
                for env_var in missing_env:
                    print(f"   {Colors.BOLD}{env_var}=<your_value>{Colors.RESET}")

            # Check for missing config files
            missing_files = [issue.split(': ')[1] for issue in issues if 'Missing config file' in issue]
            if missing_files:
                print(f"\n3. Add required configuration files:")
                for config_file in missing_files:
                    print(f"   - {config_file}")
                if 'credentials.json' in missing_files:
                    print(f"     {Colors.GRAY}(Download from Google Cloud Console){Colors.RESET}")

            print(f"\n4. Restart agent after resolving issues:")
            print(f"   {Colors.BOLD}python agent-mcp.py{Colors.RESET}")

        print()

    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name not in self.manifest.get('plugins', {}):
            print(f"{Colors.RED}Plugin '{plugin_name}' not found in manifest{Colors.RESET}")
            return False

        # Validate first
        is_valid, issues = self.validate_plugin(plugin_name)
        if not is_valid:
            print(f"{Colors.RED}Cannot enable plugin - issues found:{Colors.RESET}")
            for issue in issues:
                print(f"  - {issue}")
            print(f"\nTo resolve:")
            print(f"  1. Copy plugin file to this directory if missing")
            print(f"  2. Install missing dependencies: pip install <dependency>")
            print(f"  3. Set environment variables in .env file")
            print(f"  4. Add required config files (e.g., credentials.json)")
            return False

        enabled_plugins = self.config.get('enabled_plugins', [])
        plugin_cfg = self.config.get('plugin_config', {}).get(plugin_name, {})

        if plugin_name in enabled_plugins and plugin_cfg.get('enabled') is not False:
            print(f"{Colors.YELLOW}Plugin '{plugin_name}' is already enabled{Colors.RESET}")
            return True

        # Add to enabled_plugins list if not already present
        if plugin_name not in enabled_plugins:
            enabled_plugins.append(plugin_name)
            self.config['enabled_plugins'] = enabled_plugins

        # Flip the runtime flag to true
        self.config.setdefault('plugin_config', {}).setdefault(plugin_name, {})['enabled'] = True

        if self.save_config():
            print(f"{Colors.GREEN}✓ Enabled plugin: {plugin_name}{Colors.RESET}")
            print(f"{Colors.CYAN}Restart agent-mcp.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        enabled_plugins = self.config.get('enabled_plugins', [])
        plugin_cfg = self.config.get('plugin_config', {}).get(plugin_name, {})

        if plugin_name not in enabled_plugins or plugin_cfg.get('enabled') is False:
            print(f"{Colors.YELLOW}Plugin '{plugin_name}' is not enabled{Colors.RESET}")
            return True

        # Keep in enabled_plugins to preserve config (port, host, etc.)
        # Only flip the runtime flag — loader checks this before starting the plugin
        self.config.setdefault('plugin_config', {}).setdefault(plugin_name, {})['enabled'] = False

        if self.save_config():
            print(f"{Colors.GREEN}✓ Disabled plugin: {plugin_name}{Colors.RESET}")
            print(f"{Colors.CYAN}Restart agent-mcp.py for changes to take effect{Colors.RESET}")
            return True
        return False

    # ------------------------------------------------------------------
    # Port management
    # ------------------------------------------------------------------

    def _get_client_plugins(self) -> List[str]:
        """Return all client_interface plugin names from the manifest."""
        return [
            name for name, meta in self.manifest.get('plugins', {}).items()
            if meta.get('type') == 'client_interface'
        ]

    def port_list(self):
        """List configured ports for all client interface plugins."""
        print(f"\n{Colors.BOLD}Client Interface Port Configuration{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(f"  {'Plugin':<30} {'Port':>6}  {'Config key'}")
        print(f"  {'-'*30} {'-'*6}  {'-'*20}")

        for plugin_name in self._get_client_plugins():
            meta = self.manifest['plugins'][plugin_name]
            port_key = meta.get('port_config_key', '')
            default_port = meta.get('default_port', '?')
            plugin_cfg = self.config.get('plugin_config', {}).get(plugin_name, {})
            current_port = plugin_cfg.get(port_key, default_port) if port_key else default_port

            enabled = plugin_name in self.config.get('enabled_plugins', [])
            color = Colors.GREEN if enabled else Colors.GRAY
            marker = " (enabled)" if enabled else ""
            print(f"  {color}{plugin_name:<30}{Colors.RESET} {current_port:>6}  {port_key}{marker}")

        print()
        print(f"  Change port: {Colors.BOLD}python llmemctl.py port-set <plugin> <port>{Colors.RESET}")
        print()

    def port_set(self, plugin_name: str, port: int) -> bool:
        """Set the listening port for a client interface plugin."""
        meta = self.manifest.get('plugins', {}).get(plugin_name)
        if not meta:
            print(f"{Colors.RED}✗ Plugin '{plugin_name}' not found in manifest{Colors.RESET}")
            return False
        if meta.get('type') != 'client_interface':
            print(f"{Colors.RED}✗ '{plugin_name}' is not a client interface plugin — only those have ports{Colors.RESET}")
            return False

        port_key = meta.get('port_config_key')
        if not port_key:
            print(f"{Colors.RED}✗ Manifest entry for '{plugin_name}' has no port_config_key{Colors.RESET}")
            return False

        if not (1 <= port <= 65535):
            print(f"{Colors.RED}✗ Port must be between 1 and 65535{Colors.RESET}")
            return False

        # Warn about conflicts with other configured ports
        for other_name in self._get_client_plugins():
            if other_name == plugin_name:
                continue
            other_meta = self.manifest['plugins'][other_name]
            other_key = other_meta.get('port_config_key', '')
            other_default = other_meta.get('default_port')
            other_cfg = self.config.get('plugin_config', {}).get(other_name, {})
            other_port = other_cfg.get(other_key, other_default) if other_key else other_default
            if other_port == port:
                print(f"{Colors.YELLOW}Warning: port {port} is also used by '{other_name}'{Colors.RESET}")

        plugin_config = self.config.setdefault('plugin_config', {})
        plugin_config.setdefault(plugin_name, {})[port_key] = port

        if self.save_config():
            print(f"{Colors.GREEN}✓ Set {plugin_name} port to {port} ({port_key}={port}){Colors.RESET}")
            print(f"{Colors.CYAN}Restart agent-mcp.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def save_models(self) -> bool:
        """Save models configuration to file."""
        try:
            with open(self.models_path, 'w') as f:
                json.dump(self.models, f, indent=2)
            return True
        except Exception as e:
            print(f"{Colors.RED}Error saving models: {e}{Colors.RESET}")
            return False

    def list_models(self):
        """List all LLM models with status."""
        print(f"\n{Colors.BOLD}LLM Model Registry{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")

        default_model = self.models.get('default_model', 'unknown')
        models = self.models.get('models', {})

        if not models:
            print("No models configured.")
            return

        enabled_models = {k: v for k, v in models.items() if v.get('enabled', True)}
        disabled_models = {k: v for k, v in models.items() if not v.get('enabled', True)}

        # Show enabled models
        if enabled_models:
            print(f"{Colors.BOLD}Enabled Models:{Colors.RESET}")
            for name, config in sorted(enabled_models.items()):
                default_marker = f" {Colors.MAGENTA}(default){Colors.RESET}" if name == default_model else ""
                api_key_status = self._check_model_api_key(config)
                status_symbol = f"{Colors.GREEN}✓{Colors.RESET}" if api_key_status else f"{Colors.YELLOW}⚠{Colors.RESET}"

                print(f"  {status_symbol} {Colors.BOLD}{name}{Colors.RESET}{default_marker}")
                print(f"     Type: {config.get('type')} | Model: {config.get('model_id')} | Timeout: {config.get('llm_call_timeout', 60)}s")

                if config.get('description'):
                    print(f"     {Colors.GRAY}{config.get('description')}{Colors.RESET}")

                if not api_key_status and config.get('env_key'):
                    print(f"     {Colors.YELLOW}Missing: .env: {config.get('env_key')}{Colors.RESET}")

        # Show disabled models
        if disabled_models:
            print(f"\n{Colors.BOLD}Disabled Models:{Colors.RESET}")
            for name, config in sorted(disabled_models.items()):
                print(f"  {Colors.GRAY}○ {name}{Colors.RESET}")
                print(f"     Type: {config.get('type')} | Model: {config.get('model_id')}")

        print(f"\n{Colors.BOLD}Legend:{Colors.RESET}")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Ready - API key configured")
        print(f"  {Colors.YELLOW}⚠{Colors.RESET} Missing API key")
        print(f"  {Colors.GRAY}○{Colors.RESET} Disabled")
        print()

    def _check_model_api_key(self, model_config: dict) -> bool:
        """Check if model's API key is configured."""
        env_key = model_config.get('env_key')
        if not env_key:
            return True  # No API key required (e.g., local models)
        return bool(os.getenv(env_key))

    def show_model_info(self, model_name: str):
        """Show detailed information about a model."""
        models = self.models.get('models', {})
        if model_name not in models:
            print(f"{Colors.RED}Model '{model_name}' not found{Colors.RESET}")
            return

        model = models[model_name]
        default_model = self.models.get('default_model')

        print(f"\n{Colors.BOLD}{model_name}{Colors.RESET}")
        if model_name == default_model:
            print(f"{Colors.MAGENTA}(Default Model){Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")

        print(f"Enabled:          {'Yes' if model.get('enabled', True) else 'No'}")
        print(f"Type:             {model.get('type')}")
        print(f"Model ID:         {model.get('model_id')}")
        print(f"Host:             {model.get('host') or 'Default'}")
        print(f"Max Context:      {model.get('max_context')} messages")
        print(f"LLM Call Timeout: {model.get('llm_call_timeout', 60)}s")

        if model.get('description'):
            print(f"Description:      {model.get('description')}")

        # Check API key
        env_key = model.get('env_key')
        if env_key:
            has_key = bool(os.getenv(env_key))
            key_status = f"{Colors.GREEN}✓ Configured{Colors.RESET}" if has_key else f"{Colors.RED}✗ Missing{Colors.RESET}"
            print(f"API Key:      {env_key} - {key_status}")
        else:
            print(f"API Key:      Not required (local or no auth)")

        print()

    def add_model(self, name: str, model_id: str, model_type: str, host: str = None,
                  env_key: str = None, max_context: int = 50, description: str = ""):
        """Add a new model to the registry."""
        models = self.models.get('models', {})

        if name in models:
            print(f"{Colors.RED}Model '{name}' already exists. Use 'model update' to modify.{Colors.RESET}")
            return False

        # Validate type
        if model_type.upper() not in ['OPENAI', 'GEMINI']:
            print(f"{Colors.RED}Invalid model type. Must be 'OPENAI' or 'GEMINI'{Colors.RESET}")
            return False

        models[name] = {
            "model_id": model_id,
            "type": model_type.upper(),
            "host": host,
            "env_key": env_key,
            "max_context": max_context,
            "enabled": True,
            "description": description
        }

        self.models['models'] = models
        if self.save_models():
            print(f"{Colors.GREEN}✓ Added model: {name}{Colors.RESET}")
            print(f"{Colors.CYAN}Restart agent-mcp.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def set_default_model(self, model_key: str):
        """Set the default LLM model."""
        models = self.models.get('models', {})

        if model_key not in models:
            print(f"{Colors.RED}Model '{model_key}' not found{Colors.RESET}")
            print(f"\nAvailable models:")
            for key in models.keys():
                print(f"  - {key}")
            return False

        # Check if model is enabled
        if not models[model_key].get('enabled', True):
            print(f"{Colors.RED}Cannot set disabled model as default. Enable it first.{Colors.RESET}")
            return False

        self.models['default_model'] = model_key
        if self.save_models():
            print(f"{Colors.GREEN}✓ Set default model to: {model_key}{Colors.RESET}")
            print(f"{Colors.CYAN}Restart agent-mcp.py for changes to take effect{Colors.RESET}")
            return True
        return False


    def _require_model(self, model_name: str) -> dict | None:
        """Return model entry from models dict, printing an error if not found."""
        models = self.models.get('models', {})
        if model_name not in models:
            print(f"{Colors.RED}Model '{model_name}' not found{Colors.RESET}")
            available = ", ".join(sorted(models.keys()))
            print(f"  Available: {available}")
            return None
        return models[model_name]

    def set_model_description(self, model_name: str, description: str):
        """Set description for a model."""
        models = self.models.get('models', {})

        if model_name not in models:
            print(f"{Colors.RED}Model '{model_name}' not found{Colors.RESET}")
            print(f"\nAvailable models:")
            for key in models.keys():
                print(f"  - {key}")
            return False

        if not description or len(description) > 200:
            print(f"{Colors.RED}Description must be 1-200 characters{Colors.RESET}")
            return False

        old_description = models[model_name].get('description', 'N/A')
        models[model_name]['description'] = description
        self.models['models'] = models

        if self.save_models():
            print(f"{Colors.GREEN}✓ Updated description for {model_name}{Colors.RESET}")
            print(f"  Old: {old_description}")
            print(f"  New: {description}")
            return True
        return False

    def set_model_host(self, model_name: str, host: str):
        """Set host URL for a model."""
        models = self.models.get('models', {})

        if model_name not in models:
            print(f"{Colors.RED}Model '{model_name}' not found{Colors.RESET}")
            print(f"\nAvailable models:")
            for key in models.keys():
                print(f"  - {key}")
            return False

        # Validate host format (must be URL or null)
        if host.lower() == "null" or host.lower() == "none":
            new_host = None
        elif host.startswith("http://") or host.startswith("https://"):
            new_host = host.rstrip('/')  # Remove trailing slash
        else:
            print(f"{Colors.RED}Host must be a valid URL (http:// or https://) or 'null'{Colors.RESET}")
            print(f"{Colors.YELLOW}Examples:{Colors.RESET}")
            print(f"  https://api.openai.com/v1")
            print(f"  http://192.168.1.100:11434")
            print(f"  null (for Gemini models)")
            return False

        old_host = models[model_name].get('host', 'null')
        models[model_name]['host'] = new_host
        self.models['models'] = models

        if self.save_models():
            print(f"{Colors.GREEN}✓ Updated host for {model_name}{Colors.RESET}")
            print(f"  Old: {old_host}")
            print(f"  New: {new_host if new_host else 'null'}")
            print(f"{Colors.CYAN}Restart agent-mcp.py for changes to take effect{Colors.RESET}")
            return True
        return False

    # ------------------------------------------------------------------
    # Tmux plugin configuration
    # ------------------------------------------------------------------

    def tmux_exec_timeout(self, seconds: float):
        """Set TMUX_EXEC_TIMEOUT in plugins-enabled.json plugin_config.plugin_tmux."""
        if seconds <= 0:
            print(f"{Colors.RED}Timeout must be > 0 seconds{Colors.RESET}")
            return False

        plugin_config = self.config.setdefault('plugin_config', {})
        tmux_cfg = plugin_config.setdefault('plugin_tmux', {})
        old = tmux_cfg.get('TMUX_EXEC_TIMEOUT', 10)
        tmux_cfg['TMUX_EXEC_TIMEOUT'] = seconds

        if self.save_config():
            print(f"{Colors.GREEN}✓ TMUX_EXEC_TIMEOUT: {old}s → {seconds}s{Colors.RESET}")
            print(f"{Colors.CYAN}Restart agent-mcp.py for changes to take effect{Colors.RESET}")
            return True
        return False

    # ------------------------------------------------------------------
    # Unified resource commands for agentctl
    # ------------------------------------------------------------------

    def _load_llm_tools(self) -> dict:
        """Load llm-tools.json."""
        try:
            with open("llm-tools.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"{Colors.RED}Error loading llm-tools.json: {e}{Colors.RESET}")
            return {}

    def _save_llm_tools(self, data: dict) -> bool:
        try:
            with open("llm-tools.json", "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"{Colors.RED}Error saving llm-tools.json: {e}{Colors.RESET}")
            return False

    def llm_tools_cmd(self, args: list):
        """Handle: agentctl llm-tools <action> [name] [tools]"""
        action = args[0] if args else "list"
        name = args[1] if len(args) > 1 else ""
        tools_str = args[2] if len(args) > 2 else ""

        data = self._load_llm_tools()
        if not data:
            return

        toolsets = data.get("toolsets", {})

        if action == "list":
            print(f"\n{Colors.BOLD}Toolsets (llm-tools.json):{Colors.RESET}")
            for ts_name in sorted(toolsets.keys()):
                tool_list = toolsets[ts_name]
                print(f"  {ts_name}: {', '.join(tool_list)} ({len(tool_list)} tools)")
            print(f"\n{Colors.BOLD}Model → toolsets:{Colors.RESET}")
            for model_name in sorted(self.models.get("models", {}).keys()):
                ts = self.models["models"][model_name].get("llm_tools", [])
                enabled = self.models["models"][model_name].get("enabled", True)
                status = "" if enabled else f" {Colors.GRAY}(disabled){Colors.RESET}"
                print(f"  {model_name}: {', '.join(ts) if ts else '(none)'}{status}")

        elif action == "read":
            if not name:
                print(f"{Colors.RED}✗ Name required: agentctl llm-tools read <name>{Colors.RESET}")
                return
            ts = toolsets.get(name)
            if ts is None:
                print(f"{Colors.RED}✗ Toolset '{name}' not found{Colors.RESET}")
                return
            print(f"{name}: {', '.join(ts)} ({len(ts)} tools)")

        elif action == "write":
            if not name or not tools_str:
                print(f"{Colors.RED}✗ Usage: agentctl llm-tools write <name> <tool1,tool2,...>{Colors.RESET}")
                return
            tool_list = [t.strip() for t in tools_str.split(",") if t.strip()]
            toolsets[name] = tool_list
            if self._save_llm_tools(data):
                print(f"{Colors.GREEN}✓{Colors.RESET} Toolset '{name}' written: {', '.join(tool_list)}")

        elif action == "delete":
            if not name:
                print(f"{Colors.RED}✗ Name required: agentctl llm-tools delete <name>{Colors.RESET}")
                return
            if name not in toolsets:
                print(f"{Colors.RED}✗ Toolset '{name}' not found{Colors.RESET}")
                return
            del toolsets[name]
            if self._save_llm_tools(data):
                print(f"{Colors.GREEN}✓{Colors.RESET} Toolset '{name}' deleted")

        elif action == "add":
            if not name or not tools_str:
                print(f"{Colors.RED}✗ Usage: agentctl llm-tools add <name> <tool1,tool2,...>{Colors.RESET}")
                return
            existing = toolsets.get(name, [])
            new_tools = [t.strip() for t in tools_str.split(",") if t.strip()]
            merged = list(dict.fromkeys(existing + new_tools))
            toolsets[name] = merged
            if self._save_llm_tools(data):
                added = [t for t in new_tools if t not in existing]
                print(f"{Colors.GREEN}✓{Colors.RESET} Toolset '{name}': added {', '.join(added) if added else '(no new)'}. Now {len(merged)} tools.")

        else:
            print(f"{Colors.RED}✗ Unknown action '{action}'. Valid: list, read, write, delete, add{Colors.RESET}")

    def model_cfg_cmd(self, args: list):
        """Handle: agentctl model-cfg <action> [name] [field] [value]"""
        action = args[0] if args else "list"
        name = args[1] if len(args) > 1 else ""
        field = args[2] if len(args) > 2 else ""
        value = " ".join(args[3:]) if len(args) > 3 else ""

        models = self.models.get("models", {})

        if action == "list":
            print(f"\n{Colors.BOLD}Models:{Colors.RESET}")
            default = self.models.get("default_model", "")
            for key in sorted(models.keys()):
                cfg = models[key]
                enabled = cfg.get("enabled", True)
                ts = cfg.get("llm_tools", [])
                marker = f" {Colors.GREEN}(default){Colors.RESET}" if key == default else ""
                status = f" {Colors.GRAY}(disabled){Colors.RESET}" if not enabled else ""
                print(f"  {key:<16} {cfg.get('model_id',''):<30} tools=[{','.join(ts)}]{marker}{status}")

        elif action == "read":
            if not name or name not in models:
                print(f"{Colors.RED}✗ Model '{name}' not found{Colors.RESET}" if name else f"{Colors.RED}✗ Name required{Colors.RESET}")
                return
            cfg = models[name]
            print(f"\n{Colors.BOLD}Model: {name}{Colors.RESET}")
            for k, v in sorted(cfg.items()):
                if k == "env_key":
                    val_display = f"{v} ({'set' if os.getenv(v or '') else 'unset'})" if v else "(none)"
                    print(f"  {k}: {val_display}")
                else:
                    print(f"  {k}: {v}")

        elif action == "write":
            if not name or not field:
                print(f"{Colors.RED}✗ Usage: agentctl model-cfg write <name> <field> <value>{Colors.RESET}")
                return
            if name not in models:
                print(f"{Colors.RED}✗ Model '{name}' not found{Colors.RESET}")
                return
            # Type coercion
            if field in ("llm_tools", "llm_tools_gates"):
                coerced = [t.strip() for t in value.split(",") if t.strip()]
            elif field in ("enabled", "conv_log", "conv_log_tools", "memory_scan",
                           "memory_scan_suppress", "tool_suppress", "agent_call_stream"):
                if value.lower() in ("null", "none"):
                    coerced = None
                else:
                    coerced = value.lower() in ("true", "1", "yes")
            elif field in ("llm_call_timeout", "max_context"):
                coerced = int(value)
            elif field in ("temperature", "top_p"):
                coerced = float(value)
            elif field == "top_k":
                coerced = None if value.lower() in ("null", "none") else int(value)
            else:
                coerced = value

            old = models[name].get(field, "(unset)")
            models[name][field] = coerced
            with open(self.models_path, "w") as f:
                json.dump(self.models, f, indent=2)
            print(f"{Colors.GREEN}✓{Colors.RESET} {name}.{field}: {old} → {coerced}")

        elif action in ("copy", "delete", "enable", "disable"):
            # Delegate to existing methods
            if action == "copy":
                if not name or not field:
                    print(f"{Colors.RED}✗ Usage: agentctl model-cfg copy <source> <new_name>{Colors.RESET}")
                    return
                # Use field as new_name for copy
                if name not in models:
                    print(f"{Colors.RED}✗ Source model '{name}' not found{Colors.RESET}")
                    return
                if field in models:
                    print(f"{Colors.RED}✗ Model '{field}' already exists{Colors.RESET}")
                    return
                import copy as _copy
                models[field] = _copy.deepcopy(models[name])
                with open(self.models_path, "w") as f:
                    json.dump(self.models, f, indent=2)
                print(f"{Colors.GREEN}✓{Colors.RESET} Copied '{name}' → '{field}'")
            elif action == "delete":
                if not name or name not in models:
                    print(f"{Colors.RED}✗ Model '{name}' not found{Colors.RESET}")
                    return
                del models[name]
                with open(self.models_path, "w") as f:
                    json.dump(self.models, f, indent=2)
                print(f"{Colors.GREEN}✓{Colors.RESET} Deleted '{name}'")
            elif action == "enable":
                if not name:
                    print(f"{Colors.RED}✗ Name required{Colors.RESET}")
                elif name not in models:
                    print(f"{Colors.RED}✗ Model '{name}' not found{Colors.RESET}")
                else:
                    models[name]["enabled"] = True
                    with open(self.models_path, "w") as f:
                        json.dump(self.models, f, indent=2)
                    print(f"{Colors.GREEN}✓{Colors.RESET} Enabled '{name}'")
            elif action == "disable":
                if not name:
                    print(f"{Colors.RED}✗ Name required{Colors.RESET}")
                elif name not in models:
                    print(f"{Colors.RED}✗ Model '{name}' not found{Colors.RESET}")
                else:
                    models[name]["enabled"] = False
                    with open(self.models_path, "w") as f:
                        json.dump(self.models, f, indent=2)
                    print(f"{Colors.GREEN}✓{Colors.RESET} Disabled '{name}'")
        else:
            print(f"{Colors.RED}✗ Unknown action '{action}'. Valid: list, read, write, copy, delete, enable, disable{Colors.RESET}")

    def limits_cfg_cmd(self, args: list):
        """Handle: agentctl limits <action> [key] [value]"""
        action = args[0] if args else "list"
        key = args[1] if len(args) > 1 else ""
        value = args[2] if len(args) > 2 else ""

        limits = self.models.get("limits", {})

        if action == "list":
            print(f"\n{Colors.BOLD}Depth / iteration limits (llm-models.json):{Colors.RESET}")
            for k in sorted(limits.keys()):
                print(f"  {k}: {limits[k]}")
            # Session / system limits from plugins-enabled.json
            timeout = self.config.get("session_idle_timeout_minutes", 60)
            max_users = self.config.get("max_users", 50)
            print(f"\n{Colors.BOLD}Session limits (plugins-enabled.json):{Colors.RESET}")
            timeout_str = f"{timeout} minutes" if timeout > 0 else "disabled"
            print(f"  session_idle_timeout_minutes: {timeout_str}")
            print(f"  max_users: {max_users}")
            # Rate limits from plugins-enabled.json
            rl = self.config.get("rate_limits", {})
            if rl:
                print(f"\n{Colors.BOLD}Rate limits (plugins-enabled.json):{Colors.RESET}")
                for tool_type in sorted(rl.keys()):
                    cfg = rl[tool_type]
                    print(f"  {tool_type}: {cfg.get('calls',0)} calls / {cfg.get('window_seconds',0)}s"
                          f"{'  (auto-disable)' if cfg.get('auto_disable') else ''}")

        elif action == "read":
            if not key:
                print(f"{Colors.RED}✗ Key required: agentctl limits read <key>{Colors.RESET}")
                return
            if key in limits:
                print(f"{key}: {limits[key]}")
            elif key in ("session_idle_timeout_minutes", "max_users"):
                print(f"{key}: {self.config.get(key, '(not set)')}")
            elif key.startswith("rate_"):
                parts = key.split("_")
                if len(parts) >= 3:
                    tool_type = "_".join(parts[1:-1])
                    field = parts[-1] if parts[-1] == "calls" else "window_seconds"
                    rl = self.config.get("rate_limits", {}).get(tool_type, {})
                    print(f"{key}: {rl.get(field, 0)}")
                else:
                    print(f"{Colors.RED}✗ Unknown key '{key}'{Colors.RESET}")
            else:
                print(f"{Colors.RED}✗ Unknown key '{key}'{Colors.RESET}")

        elif action == "write":
            if not key or not value:
                print(f"{Colors.RED}✗ Usage: agentctl limits write <key> <value>{Colors.RESET}")
                return
            try:
                int_val = int(value)
            except ValueError:
                print(f"{Colors.RED}✗ Value must be an integer{Colors.RESET}")
                return

            if key in ("max_at_llm_depth", "max_agent_call_depth", "max_tool_iterations"):
                limits[key] = int_val
                self.models["limits"] = limits
                with open(self.models_path, "w") as f:
                    json.dump(self.models, f, indent=2)
                print(f"{Colors.GREEN}✓{Colors.RESET} {key}: → {int_val} (persisted)")
            elif key in ("session_idle_timeout_minutes", "max_users"):
                self.config[key] = int_val
                self._save_plugins_enabled()
                extra = ""
                if key == "session_idle_timeout_minutes":
                    extra = " (takes effect at next reaper cycle)" if int_val > 0 else " (reaper disabled)"
                print(f"{Colors.GREEN}✓{Colors.RESET} {key}: → {int_val}{extra}")
            elif key.startswith("rate_"):
                parts = key.split("_")
                if len(parts) >= 3 and parts[-1] in ("calls", "window"):
                    tool_type = "_".join(parts[1:-1])
                    field = parts[-1] if parts[-1] == "calls" else "window_seconds"
                    rl = self.config.setdefault("rate_limits", {})
                    rl.setdefault(tool_type, {})[field] = int_val
                    with open(self.config_path, "w") as f:
                        json.dump(self.config, f, indent=2)
                    print(f"{Colors.GREEN}✓{Colors.RESET} {key}: → {int_val} (persisted)")
                else:
                    print(f"{Colors.RED}✗ Unknown rate key '{key}'{Colors.RESET}")
            else:
                print(f"{Colors.RED}✗ Unknown key '{key}'{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ Unknown action '{action}'. Valid: list, read, write{Colors.RESET}")

    def memory_cmd(self, args: list):
        """Handle: agentctl memory status|enable|disable [feature]

        Features: context_injection, reset_summarize, post_response_scan
        Master switch: 'all' (or no feature arg) toggles the 'enabled' key.
        """
        FEATURES = ("context_injection", "reset_summarize", "post_response_scan", "fuzzy_dedup", "vector_search_qdrant", "tool_call_log")
        action = args[0] if args else "status"
        feature = args[1] if len(args) > 1 else "all"

        mem_cfg = self.config.setdefault("plugin_config", {}).setdefault("memory", {
            "enabled": True,
            "context_injection": True,
            "reset_summarize": True,
            "post_response_scan": True,
            "fuzzy_dedup": True,
            "vector_search_qdrant": True,
            "fuzzy_dedup_threshold": 0.78,
            "summarizer_model": "summarizer-anthropic",
            "auto_memory_age": True,
            "memory_age_entrycount": 50,
            "memory_age_count_timer": 60,
            "memory_age_trigger_minutes": 2880,
            "memory_age_minutes_timer": 360,
        })

        def _bool_str(val: bool) -> str:
            return f"{Colors.GREEN}enabled{Colors.RESET}" if val else f"{Colors.RED}disabled{Colors.RESET}"

        # tool_call_log lives in llm-tools.json metadata, not plugins-enabled.json
        def _get_tool_call_log_default() -> bool:
            try:
                with open("llm-tools.json") as _f:
                    return bool(json.load(_f).get("metadata", {}).get("tool_call_log", False))
            except Exception:
                return False

        def _set_tool_call_log_default(val: bool):
            try:
                with open("llm-tools.json") as _f:
                    data = json.load(_f)
                data.setdefault("metadata", {})["tool_call_log"] = val
                with open("llm-tools.json", "w") as _f:
                    json.dump(data, _f, indent=2)
            except Exception as _e:
                print(f"{Colors.RED}✗ Could not write llm-tools.json: {_e}{Colors.RESET}")

        if action == "status":
            master = mem_cfg.get("enabled", True)
            print(f"\n{Colors.BOLD}Memory system{Colors.RESET}  (plugins-enabled.json → plugin_config.memory)")
            print(f"  master switch       : {_bool_str(master)}")
            for feat in FEATURES:
                if feat == "tool_call_log":
                    val = _get_tool_call_log_default()
                    note = f"  {Colors.GRAY}(global default; override per-model via conv_log_tools){Colors.RESET}"
                    print(f"  {feat:<24}: {_bool_str(val)}{note}")
                    continue
                val = mem_cfg.get(feat, True)
                active = master and val
                note = "" if active else f"  {Colors.GRAY}(inactive — {'master off' if not master else 'feature off'}){Colors.RESET}"
                extra = ""
                if feat == "fuzzy_dedup" and val:
                    extra = f"  (threshold={mem_cfg.get('fuzzy_dedup_threshold', 0.78):.2f})"
                print(f"  {feat:<24}: {_bool_str(val)}{note}{extra}")
            print(f"  {'summarizer_model':<24}: {mem_cfg.get('summarizer_model', 'summarizer-anthropic')}")
            # Aging config
            age_on = mem_cfg.get("auto_memory_age", True)
            print(f"\n  {Colors.BOLD}Background Aging:{Colors.RESET}")
            print(f"  {'auto_memory_age':<28}: {_bool_str(age_on)}")
            def _timer_str(val: int) -> str:
                return f"{Colors.GRAY}disabled{Colors.RESET}" if val == -1 else f"{val} min"
            print(f"  {'memory_age_entrycount':<28}: {mem_cfg.get('memory_age_entrycount', 50)} rows")
            print(f"  {'memory_age_count_timer':<28}: {_timer_str(mem_cfg.get('memory_age_count_timer', 60))}")
            print(f"  {'memory_age_trigger_minutes':<28}: {mem_cfg.get('memory_age_trigger_minutes', 2880)} min ({mem_cfg.get('memory_age_trigger_minutes', 2880)//60}h) staleness threshold")
            print(f"  {'memory_age_minutes_timer':<28}: {_timer_str(mem_cfg.get('memory_age_minutes_timer', 360))}")

        elif action in ("enable", "disable"):
            new_val = (action == "enable")
            if feature == "all":
                mem_cfg["enabled"] = new_val
                label = "Master switch"
                self._save_plugins_enabled()
            elif feature == "tool_call_log":
                _set_tool_call_log_default(new_val)
                label = "tool_call_log (global default)"
            elif feature in FEATURES:
                mem_cfg[feature] = new_val
                label = feature
                self._save_plugins_enabled()
            else:
                print(f"{Colors.RED}✗ Unknown feature '{feature}'. "
                      f"Valid: all, {', '.join(FEATURES)}{Colors.RESET}")
                return
            state_str = "enabled" if new_val else "disabled"
            print(f"{Colors.GREEN}✓{Colors.RESET} {label}: {state_str} (persisted)")
            print(f"  Takes effect immediately — no server restart needed.")

        elif action == "set":
            key = feature  # reuse 'feature' positional arg as key name
            value = args[2] if len(args) > 2 else ""
            if key == "fuzzy_dedup_threshold":
                try:
                    t = float(value)
                    if not (0.0 < t <= 1.0):
                        raise ValueError
                    mem_cfg["fuzzy_dedup_threshold"] = t
                    self._save_plugins_enabled()
                    print(f"{Colors.GREEN}✓{Colors.RESET} fuzzy_dedup_threshold set to {t:.2f} (persisted)")
                    print(f"  Takes effect immediately (no restart needed).")
                except (ValueError, TypeError):
                    print(f"{Colors.RED}✗ Invalid threshold '{value}'. Must be a float between 0.0 and 1.0.{Colors.RESET}")
            elif key == "summarizer_model":
                if not value:
                    print(f"{Colors.RED}✗ Provide a model key, e.g.: memory set summarizer_model nuc11Localtokens{Colors.RESET}")
                    return
                from routes import LLM_REGISTRY
                if value not in LLM_REGISTRY:
                    available = ", ".join(LLM_REGISTRY.keys())
                    print(f"{Colors.RED}✗ Unknown model '{value}'.{Colors.RESET}\n  Available: {available}")
                    return
                mem_cfg["summarizer_model"] = value
                self._save_plugins_enabled()
                print(f"{Colors.GREEN}✓{Colors.RESET} summarizer_model set to '{value}' (persisted)")
                print(f"  Takes effect on next !reset.")
            elif key in ("memory_age_entrycount", "memory_age_count_timer",
                         "memory_age_trigger_minutes", "memory_age_minutes_timer"):
                try:
                    v = int(value)
                    if key == "memory_age_entrycount" and v < 1:
                        raise ValueError("must be >= 1")
                    if key != "memory_age_entrycount" and v != -1 and v < 1:
                        raise ValueError("must be >= 1 or -1 (disable)")
                    mem_cfg[key] = v
                    self._save_plugins_enabled()
                    note = " (disabled)" if v == -1 else f" min"
                    print(f"{Colors.GREEN}✓{Colors.RESET} {key} set to {v}{note} (persisted)")
                    print(f"  Takes effect on next timer cycle (no restart needed).")
                except (ValueError, TypeError) as exc:
                    print(f"{Colors.RED}✗ Invalid value '{value}': {exc}{Colors.RESET}")
            elif key == "auto_memory_age":
                if value.lower() in ("true", "1", "yes", "on"):
                    mem_cfg["auto_memory_age"] = True
                    self._save_plugins_enabled()
                    print(f"{Colors.GREEN}✓{Colors.RESET} auto_memory_age enabled (persisted)")
                elif value.lower() in ("false", "0", "no", "off"):
                    mem_cfg["auto_memory_age"] = False
                    self._save_plugins_enabled()
                    print(f"{Colors.GREEN}✓{Colors.RESET} auto_memory_age disabled (persisted)")
                else:
                    print(f"{Colors.RED}✗ Invalid value '{value}'. Use: true/false{Colors.RESET}")
            else:
                settable = (
                    "fuzzy_dedup_threshold, summarizer_model, auto_memory_age, "
                    "memory_age_entrycount, memory_age_count_timer, "
                    "memory_age_trigger_minutes, memory_age_minutes_timer"
                )
                print(f"{Colors.RED}✗ Unknown key '{key}'.{Colors.RESET}\n  Settable: {settable}")

        elif action == "test":
            self._memory_test()

        else:
            print(f"{Colors.RED}✗ Unknown action '{action}'. "
                  f"Valid: status, enable, disable, set, test{Colors.RESET}")

    def _memory_test(self):
        """
        Runtime test: toggle memory master switch and verify the live server honours it.

        Steps:
          1. Record current master-switch state.
          2. DISABLE memory → call !memstats via the API → confirm 'enabled: OFF'.
          3. ENABLE memory  → call !memstats via the API → confirm 'enabled: on'.
          4. Restore original state.

        Requires the agent-mcp API plugin to be running (default port 8767).
        """
        import urllib.request
        import urllib.error
        import time

        # Discover API port from config
        api_cfg = self.config.get("plugin_config", {}).get("plugin_client_api", {})
        api_port = api_cfg.get("api_port", 8767)
        api_host = api_cfg.get("api_host", "127.0.0.1")
        if api_host in ("0.0.0.0", ""):
            api_host = "127.0.0.1"
        base_url = f"http://{api_host}:{api_port}"

        def _api_send(text: str, timeout: int = 20) -> str | None:
            """POST to /api/v1/send with wait=true; return response text or None."""
            import json as _json
            payload = _json.dumps({"text": text, "wait": True, "timeout": timeout}).encode()
            req = urllib.request.Request(
                f"{base_url}/api/v1/submit",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
                    body = _json.loads(resp.read())
                    return body.get("text", "")
            except urllib.error.URLError as e:
                print(f"  {Colors.RED}✗ API call failed: {e}{Colors.RESET}")
                return None

        def _check_connectivity() -> bool:
            try:
                urllib.request.urlopen(f"{base_url}/health", timeout=3)
                return True
            except Exception:
                return False

        print(f"\n{Colors.BOLD}Memory Subsystem Runtime Test{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*55}{Colors.RESET}")

        # Check server reachable
        print(f"  Checking server at {base_url} ...", end=" ", flush=True)
        if not _check_connectivity():
            print(f"{Colors.RED}UNREACHABLE{Colors.RESET}")
            print(f"  Start the server first: python agent-mcp.py")
            return
        print(f"{Colors.GREEN}OK{Colors.RESET}")

        # Save current master-switch state
        mem_cfg = self.config.setdefault("plugin_config", {}).setdefault("memory", {})
        original_state = mem_cfg.get("enabled", True)

        def _set_master(val: bool):
            # Re-read config to get a fresh handle each time
            with open(self.config_path) as fh:
                import json as _json
                data = _json.load(fh)
            data.setdefault("plugin_config", {}).setdefault("memory", {})["enabled"] = val
            with open(self.config_path, "w") as fh:
                import json as _json
                _json.dump(data, fh, indent=2)
            # Update in-memory config too
            self.config = data

        def _confirm_state(expected_enabled: bool, response_text: str | None) -> bool:
            """Parse !memstats output to confirm enabled (master) state.
            Looks for:  'enabled (master)  : on'  or  ': OFF'  (added in routes.py).
            Falls back to verifying plugins-enabled.json directly if that line is absent.
            """
            if response_text is None:
                return False
            # New format: line explicitly shows master switch
            for line in (response_text or "").splitlines():
                if "enabled (master)" in line.lower():
                    if expected_enabled:
                        return ": off" not in line.lower()
                    else:
                        return ": off" in line.lower()
            # Fallback: !memstats responded (server alive) — verify JSON on disk
            try:
                import json as _json
                with open(self.config_path) as fh:
                    on_disk = _json.load(fh).get("plugin_config", {}).get("memory", {}).get("enabled", True)
                return on_disk == expected_enabled
            except Exception:
                return False

        passed = 0
        failed = 0

        # --- Step 1: Disable and verify ---
        print(f"\n  {Colors.BOLD}Step 1:{Colors.RESET} Disable memory master switch...", end=" ", flush=True)
        _set_master(False)
        time.sleep(0.3)
        print(f"{Colors.GREEN}done{Colors.RESET}")

        print(f"  {Colors.BOLD}Step 2:{Colors.RESET} Query !memstats (wait for response)...", end=" ", flush=True)
        resp = _api_send("!memstats", timeout=15)
        if resp is None:
            print(f"{Colors.RED}FAILED{Colors.RESET}")
            failed += 1
        elif _confirm_state(False, resp):
            has_line = any("enabled (master)" in l.lower() for l in (resp or "").splitlines())
            src = "!memstats master line" if has_line else "plugins-enabled.json (disk)"
            print(f"{Colors.GREEN}PASS — memory disabled, confirmed via {src}{Colors.RESET}")
            passed += 1
        else:
            snippet = next(
                (l.strip() for l in (resp or "").splitlines() if "enabled (master)" in l.lower()),
                "(enabled (master) line absent — disk confirm also failed)"
            )
            print(f"{Colors.RED}FAIL — expected disabled, got: {snippet!r}{Colors.RESET}")
            failed += 1

        # --- Step 2: Enable and verify ---
        print(f"\n  {Colors.BOLD}Step 3:{Colors.RESET} Enable memory master switch...", end=" ", flush=True)
        _set_master(True)
        time.sleep(0.3)
        print(f"{Colors.GREEN}done{Colors.RESET}")

        print(f"  {Colors.BOLD}Step 4:{Colors.RESET} Query !memstats (wait for response)...", end=" ", flush=True)
        resp = _api_send("!memstats", timeout=15)
        if resp is None:
            print(f"{Colors.RED}FAILED{Colors.RESET}")
            failed += 1
        elif _confirm_state(True, resp):
            has_line = any("enabled (master)" in l.lower() for l in (resp or "").splitlines())
            src = "!memstats master line" if has_line else "plugins-enabled.json (disk)"
            print(f"{Colors.GREEN}PASS — memory enabled, confirmed via {src}{Colors.RESET}")
            passed += 1
        else:
            snippet = next(
                (l.strip() for l in (resp or "").splitlines() if "enabled (master)" in l.lower()),
                "(enabled (master) line absent — disk confirm also failed)"
            )
            print(f"{Colors.RED}FAIL — expected enabled, got: {snippet!r}{Colors.RESET}")
            failed += 1

        # --- Restore ---
        if mem_cfg.get("enabled", True) != original_state:
            _set_master(original_state)
            state_label = "enabled" if original_state else "disabled"
            print(f"\n  Restored original state: {state_label}")

        # --- Summary ---
        print(f"\n{Colors.CYAN}{'='*55}{Colors.RESET}")
        total = passed + failed
        if failed == 0:
            print(f"  {Colors.GREEN}{Colors.BOLD}All {total}/{total} tests passed.{Colors.RESET}  "
                  f"Toggle is live — no server restart needed.")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}{failed}/{total} tests FAILED.{Colors.RESET}  "
                  f"Check server logs.")
        print()

    def show_help(self):
        """Print all available commands."""
        print(f"\n{Colors.BOLD}Plugin Commands:{Colors.RESET}")
        print("  list                              - List all plugins")
        print("  info <plugin>                     - Show plugin details")
        print("  enable <plugin>                   - Enable a plugin")
        print("  disable <plugin>                  - Disable a plugin")
        print(f"\n{Colors.BOLD}Model Commands:{Colors.RESET}")
        print("  models                            - List all models")
        print("  model-info <name>                 - Show model details")
        print("  model-add                         - Add a new model (interactive)")
        print("  model <name>                      - Set default model")
        print(f"\n{Colors.BOLD}Port Configuration:{Colors.RESET}")
        print("  port-list                         - Show listening ports for all client plugins")
        print("  port-set <plugin> <port>          - Set listening port for a plugin")
        print(f"\n{Colors.BOLD}Tmux Plugin Commands:{Colors.RESET}")
        print("  tmux-exec-timeout <seconds>               - Set exec read timeout (default 10)")
        print(f"\n{Colors.BOLD}History Management Commands:{Colors.RESET}")
        print("  history-list                              - Show history config and chain")
        print("  history-chain-add <plugin_name>          - Add plugin_history_*.py to chain")
        print("  history-chain-remove <plugin_name>       - Remove plugin from chain")
        print("  history-chain-move <plugin_name> <pos>   - Move plugin to position in chain")
        print(f"\n{Colors.BOLD}Unified Resource Commands:{Colors.RESET}")
        print("  llm-tools list|read|write|delete|add [name] [tools]")
        print("  model-cfg list|read|write|copy|delete|enable|disable [name] [field] [value]")
        print("    field 'llm_tools_gates': comma-separated gate entries, e.g. 'db_query,model_cfg write'")
        print("  limits list|read|write [key] [value]")
        print(f"\n{Colors.BOLD}Memory Commands:{Colors.RESET}")
        print("  memory status                             - Show memory feature on/off state")
        print("  memory enable [feature]                   - Enable memory (or a specific feature)")
        print("  memory disable [feature]                  - Disable memory (or a specific feature)")
        print(f"    features: context_injection, reset_summarize, post_response_scan, fuzzy_dedup, vector_search_qdrant")
        print("  memory set fuzzy_dedup_threshold <0.0-1.0>   - Set similarity threshold (default 0.78)")
        print("  memory set summarizer_model <model_key>       - Set model used on !reset summarization")
        print("  memory set auto_memory_age <true|false>       - Enable/disable background aging")
        print("  memory set memory_age_entrycount <n>          - Max short-term rows before count aging")
        print("  memory set memory_age_count_timer <min|-1>    - Count-pressure check interval (min)")
        print("  memory set memory_age_trigger_minutes <min>   - Staleness threshold in minutes")
        print("  memory set memory_age_minutes_timer <min|-1>  - Staleness check interval (min)")
        print("  memory test                                   - Runtime test: toggle enable/disable vs live server")
        print(f"\n{Colors.BOLD}Other:{Colors.RESET}")
        print("  help                              - Show this command list")
        print("  quit                              - Exit plugin manager")

    def interactive_menu(self):
        """Run interactive menu."""
        self.show_help()

        while True:
            try:
                cmd = input(f"\n{Colors.CYAN}agentctl>{Colors.RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

            if not cmd:
                continue

            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if action == "quit" or action == "exit":
                break
            elif action == "help":
                self.show_help()
            elif action == "list":
                self.list_plugins()
            elif action == "info":
                if not arg:
                    print(f"{Colors.RED}Usage: info <plugin_name>{Colors.RESET}")
                else:
                    self.show_plugin_info(arg)
            elif action == "enable":
                if not arg:
                    print(f"{Colors.RED}Usage: enable <plugin_name>{Colors.RESET}")
                else:
                    self.enable_plugin(arg)
            elif action == "disable":
                if not arg:
                    print(f"{Colors.RED}Usage: disable <plugin_name>{Colors.RESET}")
                else:
                    self.disable_plugin(arg)
            elif action == "model":
                if not arg:
                    print(f"{Colors.RED}Usage: model <model_name>{Colors.RESET}")
                else:
                    self.set_default_model(arg)
            elif action == "models":
                self.list_models()
            elif action == "model-info":
                if not arg:
                    print(f"{Colors.RED}Usage: model-info <model_name>{Colors.RESET}")
                else:
                    self.show_model_info(arg)
            elif action == "model-add":
                self._interactive_add_model()
            elif action == "tmux-exec-timeout":
                if not arg:
                    print(f"{Colors.RED}Usage: tmux-exec-timeout <seconds>{Colors.RESET}")
                else:
                    try:
                        self.tmux_exec_timeout(float(arg))
                    except ValueError:
                        print(f"{Colors.RED}Timeout must be a number (seconds){Colors.RESET}")
            elif action == "port-list":
                self.port_list()
            elif action == "port-set":
                args = arg.split()
                if len(args) != 2:
                    print(f"{Colors.RED}Usage: port-set <plugin_name> <port>{Colors.RESET}")
                else:
                    try:
                        self.port_set(args[0], int(args[1]))
                    except ValueError:
                        print(f"{Colors.RED}Port must be an integer{Colors.RESET}")
            elif action == "history-list":
                self.history_list()
            elif action == "history-chain-add":
                if not arg:
                    print(f"{Colors.RED}Usage: history-chain-add <plugin_name>{Colors.RESET}")
                else:
                    self.history_chain_add(arg)
            elif action == "history-chain-remove":
                if not arg:
                    print(f"{Colors.RED}Usage: history-chain-remove <plugin_name>{Colors.RESET}")
                else:
                    self.history_chain_remove(arg)
            elif action == "history-chain-move":
                args2 = arg.split()
                if len(args2) < 2:
                    print(f"{Colors.RED}Usage: history-chain-move <plugin_name> <position>{Colors.RESET}")
                else:
                    try:
                        self.history_chain_move(args2[0], int(args2[1]))
                    except ValueError:
                        print(f"{Colors.RED}Position must be an integer{Colors.RESET}")
            elif action == "llm-tools":
                self.llm_tools_cmd(arg.split() if arg else [])
            elif action == "model-cfg":
                self.model_cfg_cmd(arg.split() if arg else [])
            elif action == "limits":
                self.limits_cfg_cmd(arg.split() if arg else [])
            elif action == "memory":
                self.memory_cmd(arg.split() if arg else [])
            else:
                print(f"{Colors.RED}Unknown command: {action}{Colors.RESET}")

    def _interactive_add_model(self):
        """Interactive wizard to add a new model."""
        print(f"\n{Colors.BOLD}Add New Model{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")

        try:
            name = input("Model name (e.g., grok-4, openai): ").strip()
            if not name:
                print(f"{Colors.RED}Model name cannot be empty{Colors.RESET}")
                return

            model_id = input("Model ID (e.g., gpt-5.2, grok-4-1-fast-reasoning): ").strip()
            if not model_id:
                print(f"{Colors.RED}Model ID cannot be empty{Colors.RESET}")
                return

            print("\nModel type:")
            print("  1. OPENAI (OpenAI-compatible API)")
            print("  2. GEMINI (Google Gemini API)")
            type_choice = input("Choose (1-2): ").strip()

            if type_choice == "1":
                model_type = "OPENAI"
            elif type_choice == "2":
                model_type = "GEMINI"
            else:
                print(f"{Colors.RED}Invalid choice{Colors.RESET}")
                return

            host = input(f"API host (optional, press Enter for default): ").strip() or None
            env_key = input(f"Environment variable for API key (optional): ").strip() or None

            max_context_str = input("Max context messages (default: 50): ").strip()
            max_context = int(max_context_str) if max_context_str else 50

            description = input("Description (optional): ").strip() or ""

            self.add_model(name, model_id, model_type, host, env_key, max_context, description)

        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.YELLOW}Cancelled{Colors.RESET}")
        except ValueError as e:
            print(f"{Colors.RED}Invalid input: {e}{Colors.RESET}")


    # ------------------------------------------------------------------
    # History chain management
    # ------------------------------------------------------------------

    def _get_history_cfg(self) -> dict:
        """Return plugin_history_default config block (creates if missing)."""
        self.config.setdefault("plugin_config", {}).setdefault("plugin_history_default", {})
        return self.config["plugin_config"]["plugin_history_default"]

    def _save_plugins_enabled(self) -> bool:
        """Persist plugins-enabled.json."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"{Colors.RED}✗ Could not save plugins-enabled.json: {e}{Colors.RESET}")
            return False

    def _discover_history_plugins(self) -> list[str]:
        """Find all plugin_history_*.py files in the same directory."""
        import glob as _glob
        pattern = os.path.join(os.path.dirname(self.config_path), "plugin_history_*.py")
        files = _glob.glob(pattern)
        return sorted([os.path.splitext(os.path.basename(f))[0] for f in files])

    def history_list(self):
        """List available and active history plugins."""
        available = self._discover_history_plugins()
        cfg = self._get_history_cfg()
        chain = cfg.get("chain", ["plugin_history_default"])
        agent_max_ctx = cfg.get("agent_max_ctx", 200)
        max_users = self.config.get("max_users", 50)
        timeout = self.config.get("session_idle_timeout_minutes", 60)

        tool_preview_length = self.config.get("tool_preview_length", 500)
        if tool_preview_length == -1:
            tpl_str = "unlimited (-1)"
        elif tool_preview_length == 0:
            tpl_str = "tags only, no content (0)"
        else:
            tpl_str = f"{tool_preview_length} chars"
        tool_suppress = self.config.get("tool_suppress", False)

        print(f"\n{Colors.BOLD}History Configuration:{Colors.RESET}")
        print(f"  agent_max_ctx            : {agent_max_ctx} messages")
        print(f"  max_users                : {max_users} simultaneous sessions")
        timeout_str = f"{timeout} minutes" if timeout > 0 else "disabled"
        print(f"  session_idle_timeout     : {timeout_str}")
        print(f"  tool_preview_length      : {tpl_str}")
        print(f"  tool_suppress            : {'enabled' if tool_suppress else 'disabled'}")
        print(f"\n{Colors.BOLD}Active chain (in order):{Colors.RESET}")
        for i, name in enumerate(chain):
            marker = f"{Colors.GREEN}✓{Colors.RESET}" if name in available else f"{Colors.RED}✗ MISSING{Colors.RESET}"
            print(f"  [{i}] {name}  {marker}")
        print(f"\n{Colors.BOLD}Available history plugins:{Colors.RESET}")
        for name in available:
            in_chain = name in chain
            status = f"{Colors.GREEN}in chain{Colors.RESET}" if in_chain else "not in chain"
            print(f"  {name}  ({status})")

    def history_chain_add(self, plugin_name: str) -> bool:
        """Append a history plugin to the chain.

        If the plugin module defines DEFAULT_CONFIG (dict), scaffold it into
        plugin_config under the plugin's name (only if no entry exists yet).
        """
        available = self._discover_history_plugins()
        if plugin_name not in available:
            print(f"{Colors.RED}✗ '{plugin_name}' not found. Available: {available}{Colors.RESET}")
            return False
        cfg = self._get_history_cfg()
        chain = cfg.get("chain", ["plugin_history_default"])
        if plugin_name in chain:
            print(f"{Colors.YELLOW}'{plugin_name}' is already in the chain.{Colors.RESET}")
            return True
        chain.append(plugin_name)
        cfg["chain"] = chain

        # Scaffold default config if plugin provides DEFAULT_CONFIG
        plugin_cfg = self.config.setdefault("plugin_config", {})
        if plugin_name not in plugin_cfg:
            try:
                import importlib
                mod = importlib.import_module(plugin_name)
                defaults = getattr(mod, "DEFAULT_CONFIG", None)
                if defaults and isinstance(defaults, dict):
                    plugin_cfg[plugin_name] = dict(defaults)
                    print(f"{Colors.GREEN}✓ Scaffolded default config for '{plugin_name}'.{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Could not load defaults from '{plugin_name}': {e}{Colors.RESET}")

        # Ensure enabled flag is set
        if plugin_name in plugin_cfg:
            plugin_cfg[plugin_name]["enabled"] = True

        if self._save_plugins_enabled():
            print(f"{Colors.GREEN}✓ Added '{plugin_name}' to history chain (position {len(chain)-1}).{Colors.RESET}")
            return True
        return False

    def history_chain_remove(self, plugin_name: str) -> bool:
        """Remove a history plugin from the chain (cannot remove plugin_history_default)."""
        if plugin_name == "plugin_history_default":
            print(f"{Colors.RED}✗ Cannot remove plugin_history_default — it must always be first.{Colors.RESET}")
            return False
        cfg = self._get_history_cfg()
        chain = cfg.get("chain", ["plugin_history_default"])
        if plugin_name not in chain:
            print(f"{Colors.RED}✗ '{plugin_name}' is not in the chain.{Colors.RESET}")
            return False
        chain.remove(plugin_name)
        cfg["chain"] = chain

        # Set enabled=false in plugin config
        plugin_cfg = self.config.get("plugin_config", {})
        if plugin_name in plugin_cfg:
            plugin_cfg[plugin_name]["enabled"] = False

        if self._save_plugins_enabled():
            print(f"{Colors.GREEN}✓ Removed '{plugin_name}' from history chain.{Colors.RESET}")
            return True
        return False

    def history_chain_move(self, plugin_name: str, new_pos: int) -> bool:
        """Move a history plugin to a specific position (0 = first, but default is always 0)."""
        if plugin_name == "plugin_history_default" and new_pos != 0:
            print(f"{Colors.RED}✗ plugin_history_default must always be at position 0.{Colors.RESET}")
            return False
        cfg = self._get_history_cfg()
        chain = cfg.get("chain", ["plugin_history_default"])
        if plugin_name not in chain:
            print(f"{Colors.RED}✗ '{plugin_name}' is not in the chain.{Colors.RESET}")
            return False
        if new_pos == 0 and plugin_name != "plugin_history_default":
            print(f"{Colors.RED}✗ Position 0 is reserved for plugin_history_default.{Colors.RESET}")
            return False
        chain.remove(plugin_name)
        chain.insert(new_pos, plugin_name)
        cfg["chain"] = chain
        if self._save_plugins_enabled():
            print(f"{Colors.GREEN}✓ Moved '{plugin_name}' to position {new_pos}.{Colors.RESET}")
            return True
        return False


def main():
    """Main entry point."""
    manager = PluginManager()

    if not manager.load_files():
        print("Failed to load manifest or config files")
        return 1

    # If arguments provided, run command mode
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()

        # Help
        if cmd == "help":
            manager.show_help()

        # Plugin commands
        elif cmd == "list":
            manager.list_plugins()
        elif cmd == "info" and len(sys.argv) > 2:
            manager.show_plugin_info(sys.argv[2])
        elif cmd == "enable" and len(sys.argv) > 2:
            manager.enable_plugin(sys.argv[2])
        elif cmd == "disable" and len(sys.argv) > 2:
            manager.disable_plugin(sys.argv[2])

        # Model commands
        elif cmd == "models":
            manager.list_models()
        elif cmd == "model-info" and len(sys.argv) > 2:
            manager.show_model_info(sys.argv[2])
        elif cmd == "model-add":
            manager._interactive_add_model()
        elif cmd == "model" and len(sys.argv) > 2:
            manager.set_default_model(sys.argv[2])

        elif cmd == "port-list":
            manager.port_list()
        elif cmd == "port-set" and len(sys.argv) > 3:
            try:
                manager.port_set(sys.argv[2], int(sys.argv[3]))
            except ValueError:
                print(f"{Colors.RED}✗ Port must be an integer{Colors.RESET}")
                return 1

        # Tmux plugin commands
        elif cmd == "tmux-exec-timeout" and len(sys.argv) > 2:
            try:
                manager.tmux_exec_timeout(float(sys.argv[2]))
            except ValueError:
                print(f"{Colors.RED}✗ Timeout must be a number (seconds){Colors.RESET}")
                return 1

        # History management commands
        elif cmd == "history-list":
            manager.history_list()
        elif cmd == "history-chain-add" and len(sys.argv) > 2:
            if not manager.history_chain_add(sys.argv[2]):
                return 1
        elif cmd == "history-chain-remove" and len(sys.argv) > 2:
            if not manager.history_chain_remove(sys.argv[2]):
                return 1
        elif cmd == "history-chain-move" and len(sys.argv) > 3:
            try:
                manager.history_chain_move(sys.argv[2], int(sys.argv[3]))
            except ValueError:
                print(f"{Colors.RED}✗ Position must be an integer{Colors.RESET}")
                return 1

        # Unified resource commands
        elif cmd == "llm-tools":
            manager.llm_tools_cmd(sys.argv[2:])
        elif cmd == "model-cfg":
            manager.model_cfg_cmd(sys.argv[2:])
        elif cmd == "limits":
            manager.limits_cfg_cmd(sys.argv[2:])
        elif cmd == "memory":
            manager.memory_cmd(sys.argv[2:])

        else:
            print(f"{Colors.RED}Unknown command: {cmd}{Colors.RESET}")
            manager.show_help()
            return 1
    else:
        # Interactive mode
        manager.interactive_menu()

    return 0


if __name__ == "__main__":
    sys.exit(main())
