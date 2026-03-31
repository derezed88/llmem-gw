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
load_dotenv(override=True)


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

            print(f"\n{Colors.CYAN}Detailed help:{Colors.RESET} {Colors.BOLD}python plugin-manager.py info <plugin_name>{Colors.RESET}")

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
            print(f"   {Colors.BOLD}python llmem-gw.py{Colors.RESET}")

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
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
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
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
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
        print(f"  Change port: {Colors.BOLD}python plugin-manager.py port-set <plugin> <port>{Colors.RESET}")
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
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
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

                toolsets = config.get('llm_tools', [])
                ts_marker = f" {Colors.GREEN}[{','.join(toolsets)}]{Colors.RESET}" if toolsets else ""
                print(f"  {status_symbol} {Colors.BOLD}{name}{Colors.RESET}{default_marker}{ts_marker}")
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
        print(f"  {Colors.GREEN}[toolsets]{Colors.RESET} Toolsets assigned to model")
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

        toolsets = model.get('llm_tools', [])
        print(f"LLM Tools:        {', '.join(toolsets) if toolsets else '(none)'}")
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
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def remove_model(self, name: str):
        """Remove a model from the registry."""
        models = self.models.get('models', {})

        if name not in models:
            print(f"{Colors.RED}Model '{name}' not found{Colors.RESET}")
            return False

        # Prevent removing default model
        if name == self.models.get('default_model'):
            print(f"{Colors.RED}Cannot remove default model. Set a different default first.{Colors.RESET}")
            return False

        del models[name]
        self.models['models'] = models

        if self.save_models():
            print(f"{Colors.GREEN}✓ Removed model: {name}{Colors.RESET}")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def enable_model(self, name: str):
        """Enable a model."""
        models = self.models.get('models', {})

        if name not in models:
            print(f"{Colors.RED}Model '{name}' not found{Colors.RESET}")
            return False

        if models[name].get('enabled', True):
            print(f"{Colors.YELLOW}Model '{name}' is already enabled{Colors.RESET}")
            return True

        models[name]['enabled'] = True
        self.models['models'] = models

        if self.save_models():
            print(f"{Colors.GREEN}✓ Enabled model: {name}{Colors.RESET}")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def disable_model(self, name: str):
        """Disable a model."""
        models = self.models.get('models', {})

        if name not in models:
            print(f"{Colors.RED}Model '{name}' not found{Colors.RESET}")
            return False

        # Prevent disabling default model
        if name == self.models.get('default_model'):
            print(f"{Colors.RED}Cannot disable default model. Set a different default first.{Colors.RESET}")
            return False

        if not models[name].get('enabled', True):
            print(f"{Colors.YELLOW}Model '{name}' is already disabled{Colors.RESET}")
            return True

        models[name]['enabled'] = False
        self.models['models'] = models

        if self.save_models():
            print(f"{Colors.GREEN}✓ Disabled model: {name}{Colors.RESET}")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
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
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def set_max_context(self, model_name: str, max_context: int):
        """Set max_context (max history size) for a model."""
        models = self.models.get('models', {})

        if model_name not in models:
            print(f"{Colors.RED}Model '{model_name}' not found{Colors.RESET}")
            print(f"\nAvailable models:")
            for key in models.keys():
                print(f"  - {key}")
            return False

        # Validate max_context
        if max_context < 1:
            print(f"{Colors.RED}max_context must be at least 1{Colors.RESET}")
            return False

        if max_context > 100000:
            print(f"{Colors.YELLOW}Warning: max_context={max_context} is very large!{Colors.RESET}")
            print(f"{Colors.YELLOW}This may exceed the model's actual context window.{Colors.RESET}")
            try:
                confirm = input(f"Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Cancelled")
                    return False
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled")
                return False

        old_value = models[model_name].get('max_context', 50)
        models[model_name]['max_context'] = max_context
        self.models['models'] = models

        if self.save_models():
            print(f"{Colors.GREEN}✓ Updated max_context for {model_name}{Colors.RESET}")
            print(f"  Old: {old_value} messages")
            print(f"  New: {max_context} messages")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def rename_model(self, old_name: str, new_name: str):
        """Rename a model (change the user-facing key, preserve backend model_id)."""
        models = self.models.get('models', {})

        if old_name not in models:
            print(f"{Colors.RED}Model '{old_name}' not found{Colors.RESET}")
            print(f"\nAvailable models:")
            for key in models.keys():
                print(f"  - {key}")
            return False

        if new_name in models:
            print(f"{Colors.RED}Model name '{new_name}' already exists{Colors.RESET}")
            return False

        # Validate new name (no spaces, reasonable length)
        if not new_name or ' ' in new_name or len(new_name) > 50:
            print(f"{Colors.RED}Invalid model name. Must be non-empty, no spaces, max 50 chars{Colors.RESET}")
            return False

        # Copy model data to new key
        models[new_name] = models[old_name]

        # Update default model if needed
        if self.models.get('default_model') == old_name:
            self.models['default_model'] = new_name
            print(f"{Colors.CYAN}Updated default model reference to: {new_name}{Colors.RESET}")

        # Remove old key
        del models[old_name]
        self.models['models'] = models

        if self.save_models():
            print(f"{Colors.GREEN}✓ Renamed model: {old_name} → {new_name}{Colors.RESET}")
            print(f"  Backend model_id: {models[new_name].get('model_id', 'N/A')}")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def set_llm_call_timeout(self, model_name: str, timeout: int):
        """Set llm_call_timeout for a model (seconds)."""
        models = self.models.get('models', {})

        if model_name not in models:
            print(f"{Colors.RED}Model '{model_name}' not found{Colors.RESET}")
            return False

        if timeout < 1:
            print(f"{Colors.RED}Timeout must be at least 1 second{Colors.RESET}")
            return False

        old = models[model_name].get('llm_call_timeout', 60)
        models[model_name]['llm_call_timeout'] = timeout
        self.models['models'] = models

        if self.save_models():
            print(f"{Colors.GREEN}✓ llm_call_timeout for '{model_name}': {old}s → {timeout}s{Colors.RESET}")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

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
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def ratelimit_list(self):
        """List all rate limit configurations from plugins-enabled.json."""
        rate_limits = self.config.get('rate_limits', {})
        if not rate_limits:
            print(f"{Colors.YELLOW}No rate limits configured (using defaults){Colors.RESET}")
            return

        print(f"\n{Colors.BOLD}Rate Limit Configuration{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        print(f"{'Tool Type':<12}  {'Calls':>6}  {'Window':>8}  {'Auto-Disable'}")
        print(f"{'-'*12}  {'-'*6}  {'-'*8}  {'-'*12}")

        for tool_type, cfg in sorted(rate_limits.items()):
            calls = cfg.get('calls', 0)
            window = cfg.get('window_seconds', 0)
            auto_dis = cfg.get('auto_disable', False)

            calls_str = "unlimited" if calls == 0 else str(calls)
            window_str = "n/a" if window == 0 else f"{window}s"
            auto_str = f"{Colors.YELLOW}YES{Colors.RESET}" if auto_dis else "no"
            print(f"{tool_type:<12}  {calls_str:>6}  {window_str:>8}  {auto_str}")
        print()

    def ratelimit_set(self, tool_type: str, calls: int, window_seconds: int):
        """Set rate limit for a tool type in plugins-enabled.json."""
        valid_types = {"llm_call", "search", "extract", "drive", "db", "system"}
        if tool_type not in valid_types:
            print(f"{Colors.RED}Unknown tool type '{tool_type}'{Colors.RESET}")
            print(f"Valid types: {', '.join(sorted(valid_types))}")
            return False

        if calls < 0:
            print(f"{Colors.RED}calls must be >= 0 (0 = unlimited){Colors.RESET}")
            return False
        if window_seconds < 0:
            print(f"{Colors.RED}window_seconds must be >= 0 (0 = unlimited){Colors.RESET}")
            return False

        if 'rate_limits' not in self.config:
            self.config['rate_limits'] = {}
        if tool_type not in self.config['rate_limits']:
            self.config['rate_limits'][tool_type] = {}

        old_calls = self.config['rate_limits'][tool_type].get('calls', '?')
        old_window = self.config['rate_limits'][tool_type].get('window_seconds', '?')
        self.config['rate_limits'][tool_type]['calls'] = calls
        self.config['rate_limits'][tool_type]['window_seconds'] = window_seconds

        if self.save_config():
            calls_str = "unlimited" if calls == 0 else f"{calls} calls"
            window_str = "n/a" if window_seconds == 0 else f"/ {window_seconds}s"
            print(f"{Colors.GREEN}✓ Rate limit for '{tool_type}': {calls_str} {window_str}{Colors.RESET}")
            print(f"  Old: {old_calls} calls / {old_window}s")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def ratelimit_auto_disable(self, tool_type: str, value: bool):
        """Set auto_disable flag for a tool type rate limit."""
        valid_types = {"llm_call", "search", "extract", "drive", "db", "system"}
        if tool_type not in valid_types:
            print(f"{Colors.RED}Unknown tool type '{tool_type}'{Colors.RESET}")
            print(f"Valid types: {', '.join(sorted(valid_types))}")
            return False

        if 'rate_limits' not in self.config:
            self.config['rate_limits'] = {}
        if tool_type not in self.config['rate_limits']:
            self.config['rate_limits'][tool_type] = {}

        self.config['rate_limits'][tool_type]['auto_disable'] = value

        if self.save_config():
            status = f"{Colors.YELLOW}ENABLED{Colors.RESET}" if value else f"{Colors.GREEN}disabled{Colors.RESET}"
            print(f"{Colors.GREEN}✓{Colors.RESET} auto_disable for '{tool_type}': {status}")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
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
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    # ------------------------------------------------------------------
    # Depth limit management
    # ------------------------------------------------------------------

    _LIMIT_KEYS: dict[str, str] = {
        "max_at_llm_depth":    "Max nested llm_call(history=caller) hops before rejection (1 = no recursion)",
        "max_agent_call_depth": "Max nested agent_call hops before rejection (1 = no recursion)",
    }

    def _load_limits(self) -> dict:
        """Load limits section from llm-models.json."""
        try:
            with open(self.models_path, "r") as f:
                data = json.load(f)
            return data.get("limits", {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_limit(self, key: str, value: int) -> bool:
        """Persist a single limit key to llm-models.json."""
        try:
            with open(self.models_path, "r") as f:
                data = json.load(f)
            data.setdefault("limits", {})[key] = value
            with open(self.models_path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"{Colors.RED}Error saving limit: {e}{Colors.RESET}")
            return False

    def limit_list(self):
        """List all depth/iteration limits from llm-models.json."""
        limits = self._load_limits()

        print(f"\n{Colors.BOLD}Depth / Iteration Limits{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        print(f"  {'Key':<26} {'Value':>6}  Description")
        print(f"  {'-'*26} {'-'*6}  {'-'*40}")

        for key, desc in sorted(self._LIMIT_KEYS.items()):
            val = limits.get(key, "(not set — using default: 1)")
            print(f"  {key:<26} {str(val):>6}  {Colors.GRAY}{desc}{Colors.RESET}")

        print()
        print(f"  Set: {Colors.BOLD}limit-set <key> <value>{Colors.RESET}")
        print(f"  {Colors.CYAN}Changes take effect after restarting llmem-gw.py{Colors.RESET}")
        print()

    def limit_set(self, key: str, value: int) -> bool:
        """Set a depth limit in llm-models.json."""
        if key not in self._LIMIT_KEYS:
            print(f"{Colors.RED}Unknown limit key '{key}'{Colors.RESET}")
            print(f"  Valid keys: {', '.join(sorted(self._LIMIT_KEYS.keys()))}")
            return False

        if value < 0:
            print(f"{Colors.RED}Value must be >= 0{Colors.RESET}")
            return False

        limits = self._load_limits()
        old = limits.get(key, 1)

        if self._save_limit(key, value):
            print(f"{Colors.GREEN}✓ {key}: {old} → {value}{Colors.RESET}")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

    def gate_reset(self) -> bool:
        """Reset all gate defaults to gated (false) — delete gate-defaults.json."""
        if self._save_gate_defaults({"db": {}, "tools": {}}):
            print(f"{Colors.GREEN}✓ All gate defaults reset — everything will be gated on next restart{Colors.RESET}")
            print(f"{Colors.CYAN}Restart llmem-gw.py for changes to take effect{Colors.RESET}")
            return True
        return False

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
        print("  model-remove <name>               - Remove a model")
        print("  model-enable <name>               - Enable a model")
        print("  model-disable <name>              - Disable a model")
        print("  model-context <name> <n>          - Set max history size (messages)")
        print("  model-rename <old> <new>          - Rename a model")
        print("  model-desc <name> <desc>          - Set model description")
        print("  model-host <name> <url>           - Set model host URL")
        print("  model <name>                      - Set default model")
        print("  model-timeout <name> <secs>       - Set llm_call_timeout for model")
        print(f"\n{Colors.BOLD}Port Configuration:{Colors.RESET}")
        print("  port-list                         - Show listening ports for all client plugins")
        print("  port-set <plugin> <port>          - Set listening port for a plugin")
        print(f"\n{Colors.BOLD}Rate Limit Commands:{Colors.RESET}")
        print("  ratelimit-list                            - Show all rate limit settings")
        print("  ratelimit-set <type> <calls> <window>     - Set rate limit (calls=0 = unlimited)")
        print("  ratelimit-autodisable <type> <true|false> - Set auto_disable flag")
        print(f"  Valid types: llm_call, search, extract, drive, db, system")
        print(f"\n{Colors.BOLD}Tmux Plugin Commands:{Colors.RESET}")
        print("  tmux-exec-timeout <seconds>               - Set exec read timeout (default 10)")
        print(f"\n{Colors.BOLD}Depth Limit Commands:{Colors.RESET}")
        print("  limit-list                        - Show depth/iteration limits")
        print("  limit-set <key> <value>           - Set a depth limit")
        print(f"  Valid keys: {', '.join(sorted(PluginManager._LIMIT_KEYS.keys()))}")
        print(f"\n{Colors.BOLD}Gate Default Commands:{Colors.RESET}")
        print("  gate-list                                         - Show all gate defaults")
        print("  gate-set db <table|*> <read|write> <true|false>  - Set DB gate default")
        print("  gate-set <tool> <read|write> <true|false>         - Set tool gate default")
        print("  gate-reset                                        - Reset all gates to gated (false)")
        print(f"  Valid tools: {', '.join(sorted(self._GATE_TOOLS.keys()))}")
        print(f"\n{Colors.BOLD}Other:{Colors.RESET}")
        print("  help                              - Show this command list")
        print("  quit                              - Exit plugin manager")

    def interactive_menu(self):
        """Run interactive menu."""
        self.show_help()

        while True:
            try:
                cmd = input(f"\n{Colors.CYAN}plugin-manager>{Colors.RESET} ").strip()
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
            elif action == "model-remove":
                if not arg:
                    print(f"{Colors.RED}Usage: model-remove <model_name>{Colors.RESET}")
                else:
                    self.remove_model(arg)
            elif action == "model-enable":
                if not arg:
                    print(f"{Colors.RED}Usage: model-enable <model_name>{Colors.RESET}")
                else:
                    self.enable_model(arg)
            elif action == "model-disable":
                if not arg:
                    print(f"{Colors.RED}Usage: model-disable <model_name>{Colors.RESET}")
                else:
                    self.disable_model(arg)
            elif action == "model-context":
                if not arg:
                    print(f"{Colors.RED}Usage: model-context <model_name> <max_context>{Colors.RESET}")
                else:
                    args = arg.split(maxsplit=1)
                    if len(args) != 2:
                        print(f"{Colors.RED}Usage: model-context <model_name> <max_context>{Colors.RESET}")
                    else:
                        try:
                            max_ctx = int(args[1])
                            self.set_max_context(args[0], max_ctx)
                        except ValueError:
                            print(f"{Colors.RED}max_context must be a number{Colors.RESET}")
            elif action == "model-rename":
                if not arg:
                    print(f"{Colors.RED}Usage: model-rename <old_name> <new_name>{Colors.RESET}")
                else:
                    args = arg.split(maxsplit=1)
                    if len(args) != 2:
                        print(f"{Colors.RED}Usage: model-rename <old_name> <new_name>{Colors.RESET}")
                    else:
                        self.rename_model(args[0], args[1])
            elif action == "model-desc":
                if not arg:
                    print(f"{Colors.RED}Usage: model-desc <model_name> <description>{Colors.RESET}")
                else:
                    args = arg.split(maxsplit=1)
                    if len(args) != 2:
                        print(f"{Colors.RED}Usage: model-desc <model_name> <description>{Colors.RESET}")
                    else:
                        self.set_model_description(args[0], args[1])
            elif action == "model-host":
                if not arg:
                    print(f"{Colors.RED}Usage: model-host <model_name> <host_url>{Colors.RESET}")
                else:
                    args = arg.split(maxsplit=1)
                    if len(args) != 2:
                        print(f"{Colors.RED}Usage: model-host <model_name> <host_url>{Colors.RESET}")
                    else:
                        self.set_model_host(args[0], args[1])
            elif action == "model-timeout":
                if not arg:
                    print(f"{Colors.RED}Usage: model-timeout <model_name> <seconds>{Colors.RESET}")
                else:
                    args = arg.split(maxsplit=1)
                    if len(args) != 2:
                        print(f"{Colors.RED}Usage: model-timeout <model_name> <seconds>{Colors.RESET}")
                    else:
                        try:
                            self.set_llm_call_timeout(args[0], int(args[1]))
                        except ValueError:
                            print(f"{Colors.RED}Timeout must be a number (seconds){Colors.RESET}")
            elif action == "ratelimit-list":
                self.ratelimit_list()
            elif action == "ratelimit-set":
                args = arg.split()
                if len(args) != 3:
                    print(f"{Colors.RED}Usage: ratelimit-set <type> <calls> <window_seconds>{Colors.RESET}")
                    print(f"  Example: ratelimit-set llm_call 3 20")
                    print(f"  Use calls=0 for unlimited")
                else:
                    try:
                        self.ratelimit_set(args[0], int(args[1]), int(args[2]))
                    except ValueError:
                        print(f"{Colors.RED}calls and window_seconds must be integers{Colors.RESET}")
            elif action == "ratelimit-autodisable":
                args = arg.split()
                if len(args) != 2 or args[1].lower() not in ("true", "false", "1", "0", "yes", "no"):
                    print(f"{Colors.RED}Usage: ratelimit-autodisable <type> <true|false>{Colors.RESET}")
                else:
                    self.ratelimit_auto_disable(args[0], args[1].lower() in ("true", "1", "yes"))
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
            elif action == "limit-list":
                self.limit_list()
            elif action == "limit-set":
                args = arg.split()
                if len(args) != 2:
                    print(f"{Colors.RED}Usage: limit-set <key> <value>{Colors.RESET}")
                    print(f"  Valid keys: {', '.join(sorted(self._LIMIT_KEYS.keys()))}")
                else:
                    try:
                        self.limit_set(args[0], int(args[1]))
                    except ValueError:
                        print(f"{Colors.RED}Value must be an integer{Colors.RESET}")
            elif action == "gate-list":
                self.gate_list()
            elif action == "gate-set":
                self.gate_set(arg.split())
            elif action == "gate-reset":
                self.gate_reset()
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
        elif cmd == "model-remove" and len(sys.argv) > 2:
            manager.remove_model(sys.argv[2])
        elif cmd == "model-enable" and len(sys.argv) > 2:
            manager.enable_model(sys.argv[2])
        elif cmd == "model-disable" and len(sys.argv) > 2:
            manager.disable_model(sys.argv[2])
        elif cmd == "model" and len(sys.argv) > 2:
            manager.set_default_model(sys.argv[2])
        elif cmd == "model-context" and len(sys.argv) > 3:
            try:
                max_ctx = int(sys.argv[3])
                manager.set_max_context(sys.argv[2], max_ctx)
            except ValueError:
                print(f"{Colors.RED}✗ max_context must be a number{Colors.RESET}")
                return 1
        elif cmd == "model-rename" and len(sys.argv) > 3:
            manager.rename_model(sys.argv[2], sys.argv[3])
        elif cmd == "model-desc" and len(sys.argv) > 3:
            # Join all remaining args as description (allows spaces)
            description = " ".join(sys.argv[3:])
            manager.set_model_description(sys.argv[2], description)
        elif cmd == "model-host" and len(sys.argv) > 3:
            manager.set_model_host(sys.argv[2], sys.argv[3])
        elif cmd == "model-timeout" and len(sys.argv) > 3:
            try:
                manager.set_llm_call_timeout(sys.argv[2], int(sys.argv[3]))
            except ValueError:
                print(f"{Colors.RED}✗ Timeout must be a number (seconds){Colors.RESET}")
                return 1
        elif cmd == "port-list":
            manager.port_list()
        elif cmd == "port-set" and len(sys.argv) > 3:
            try:
                manager.port_set(sys.argv[2], int(sys.argv[3]))
            except ValueError:
                print(f"{Colors.RED}✗ Port must be an integer{Colors.RESET}")
                return 1

        elif cmd == "ratelimit-list":
            manager.ratelimit_list()
        elif cmd == "ratelimit-set" and len(sys.argv) > 4:
            try:
                manager.ratelimit_set(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
            except ValueError:
                print(f"{Colors.RED}✗ calls and window_seconds must be integers{Colors.RESET}")
                return 1
        elif cmd == "ratelimit-autodisable" and len(sys.argv) > 3:
            val = sys.argv[3].lower()
            if val not in ("true", "false", "1", "0", "yes", "no"):
                print(f"{Colors.RED}✗ Value must be true or false{Colors.RESET}")
                return 1
            manager.ratelimit_auto_disable(sys.argv[2], val in ("true", "1", "yes"))

        # Tmux plugin commands
        elif cmd == "tmux-exec-timeout" and len(sys.argv) > 2:
            try:
                manager.tmux_exec_timeout(float(sys.argv[2]))
            except ValueError:
                print(f"{Colors.RED}✗ Timeout must be a number (seconds){Colors.RESET}")
                return 1

        # Depth limit commands
        elif cmd == "limit-list":
            manager.limit_list()
        elif cmd == "limit-set" and len(sys.argv) > 3:
            try:
                manager.limit_set(sys.argv[2], int(sys.argv[3]))
            except ValueError:
                print(f"{Colors.RED}✗ Value must be an integer{Colors.RESET}")
                return 1

        # Gate commands
        elif cmd == "gate-list":
            manager.gate_list()
        elif cmd == "gate-set":
            # gate-set db <table> <read|write> <true|false>
            # gate-set <tool> <read|write> <true|false>
            if not manager.gate_set(sys.argv[2:]):
                return 1
        elif cmd == "gate-reset":
            manager.gate_reset()

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
