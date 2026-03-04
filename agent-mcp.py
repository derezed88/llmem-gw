#!/usr/bin/env python3
"""
MCP Agent with Plugin Architecture

Modular MCP agent server that loads plugins dynamically.

Core features (always enabled):
- System info (get_system_info tool)
- System prompt read/write (read_system_prompt, update_system_prompt tools)
- Session management (!session, !reset commands)
- LLM model management (!model command)

Pluggable features:
- Client interfaces (shell.py, llama proxy)
- Data access tools (MySQL, Google Drive, Google Search)

Usage:
    python agent-mcp.py [--help]

Configuration:
    - plugins-enabled.json - Which plugins to load
    - plugin-manifest.json - Plugin metadata
    - .env - Credentials and environment variables
"""

import uvicorn
import argparse
import asyncio
import socket
import sys
from starlette.applications import Starlette
from starlette.routing import Route

from config import log
from plugin_loader import PluginLoader
from tools import get_core_tools
import tools as tools_module
import agents as agents_module
from tools import register_plugin_commands
import plugin_sec_airs_cmd  # registers !airs command at import time


def _check_port_available(host: str, port: int) -> bool:
    """Return True if the port is free to bind, False if already in use."""
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((bind_host, port))
            return True
        except OSError:
            return False


async def run_agent(host: str = "0.0.0.0"):
    """Run MCP agent with plugins."""
    import asyncio
    from uvicorn import Config, Server

    # Load plugins
    log.info("="*70)
    log.info("MCP Agent starting with plugin system")
    log.info("="*70)

    loader = PluginLoader()
    plugins = loader.load_all_enabled()

    if not plugins:
        log.warning("No plugins loaded - agent will have limited functionality")

    # Get default model from configuration
    default_model = loader.get_default_model()
    log.info(f"Default LLM model: {default_model}")

    # Collect all routes from plugins
    all_routes = []
    client_plugins = []

    for plugin_name, plugin in plugins.items():
        if plugin.PLUGIN_TYPE == "client_interface":
            routes = plugin.get_routes()
            all_routes.extend(routes)
            client_plugins.append(plugin)
            log.info(f"  + {plugin_name}: {len(routes)} routes")
            # client_interface plugins may also expose tools and commands
            tool_defs = plugin.get_tools()
            if tool_defs.get('lc'):
                tools_module.register_plugin_tools(plugin_name, tool_defs)
                log.info(f"    {plugin_name}: +{len(tool_defs.get('lc', []))} tools")
            commands = plugin.get_commands()
            if commands:
                register_plugin_commands(plugin_name, commands, plugin.get_help())
        elif plugin.PLUGIN_TYPE == "data_tool":
            tool_defs = plugin.get_tools()
            # Register tools dynamically
            tools_module.register_plugin_tools(plugin_name, tool_defs)
            log.info(f"  + {plugin_name}: {len(tool_defs.get('lc', []))} tools")
            # Register !command handlers
            commands = plugin.get_commands()
            if commands:
                register_plugin_commands(plugin_name, commands, plugin.get_help())

    # Update agents module with dynamic tools
    agents_module.update_tool_definitions()

    # Create Starlette app with all routes.
    # Sort so specific paths come before wildcard catch-alls (e.g. llama's /{path:path}),
    # otherwise the wildcard swallows requests meant for specific routes.
    def _route_specificity(route):
        path = getattr(route, 'path', '')
        return (1 if '{' in path else 0, path)
    all_routes.sort(key=_route_specificity)
    app = Starlette(routes=all_routes)

    # Determine which servers to run
    servers_to_run = []

    # Check all ports before starting any server — fail fast with a clear message
    port_conflicts = []
    seen_ports: dict = {}
    for plugin in client_plugins:
        plugin_config = plugin.get_config()
        port = plugin_config.get('port')
        host = plugin_config.get('host', '0.0.0.0')
        name = plugin_config.get('name', 'unknown')

        if port is None:
            continue  # routes-only plugin, no port to bind

        if port in seen_ports:
            port_conflicts.append(
                f"  Port {port}: claimed by both '{seen_ports[port]}' and '{name}'"
            )
        else:
            seen_ports[port] = name

        if not _check_port_available(host, port):
            port_conflicts.append(
                f"  Port {port} ({name}): already in use — "
                f"another process is listening on {host}:{port}"
            )

    if port_conflicts:
        log.error("=" * 70)
        log.error("STARTUP ABORTED — port conflict(s) detected:")
        for msg in port_conflicts:
            log.error(msg)
        log.error("")
        log.error("Fix options:")
        log.error("  1. Stop the process already using the port")
        log.error("  2. Change the port:  python agentctl.py port-set <plugin> <new_port>")
        log.error("  3. List configured ports:  python agentctl.py port-list")
        log.error("=" * 70)
        sys.exit(1)

    for plugin in client_plugins:
        plugin_config = plugin.get_config()
        port = plugin_config.get('port')
        host = plugin_config.get('host', '0.0.0.0')
        name = plugin_config.get('name', 'unknown')

        if port is None:
            log.info(f"  + {plugin.PLUGIN_NAME}: routes-only (no dedicated port)")
            continue

        log.info(f"Starting {name} on {host}:{port}")
        log.info(f"  - Plugin: {plugin.PLUGIN_NAME}")

        config = Config(app, host=host, port=port, log_level="info")
        server = Server(config)
        servers_to_run.append(server.serve())

    if not servers_to_run:
        log.error("No client interface plugins enabled - cannot start any servers")
        return

    log.info("")
    log.info("="*70)
    log.info("Server startup complete!")
    log.info("="*70)

    for plugin in client_plugins:
        plugin_config = plugin.get_config()
        port = plugin_config.get('port')
        host = plugin_config.get('host', '0.0.0.0')
        name = plugin_config.get('name', 'unknown')
        if port is None:
            log.info(f"  {name}: routes-only (shared server)")
        else:
            log.info(f"  {name}: http://{host}:{port}")

    log.info("="*70)
    log.info("")

    # Ensure session-history directory exists for persisted histories
    from state import SESSION_HISTORY_DIR
    import os as _os
    _os.makedirs(SESSION_HISTORY_DIR, exist_ok=True)
    log.info(f"Session history directory: {SESSION_HISTORY_DIR}")

    # Session idle-timeout reaper: periodically evict sessions that have been
    # inactive longer than session_idle_timeout_minutes.  A timeout of 0 disables.
    async def _session_reaper():
        import time
        from state import sessions, remove_shorthand_mapping, save_history
        while True:
            await asyncio.sleep(60)  # check every minute
            try:
                from routes import get_session_idle_timeout
                timeout_minutes = get_session_idle_timeout()
                if timeout_minutes <= 0:
                    continue
                cutoff = time.time() - timeout_minutes * 60
                stale = [
                    cid for cid, data in list(sessions.items())
                    if data.get("last_active", 0) < cutoff
                ]
                for cid in stale:
                    data = sessions.pop(cid, None)
                    if data:
                        save_history(cid, data.get("history", []))
                    remove_shorthand_mapping(cid)
                    log.info(f"Session reaped (idle timeout): {cid}")
            except Exception as e:
                log.warning(f"Session reaper error: {e}")

    # Background memory aging tasks
    async def _age_count_task():
        """Count-pressure aging: runs every memory_age_count_timer minutes."""
        from memory import age_by_count, _age_cfg
        while True:
            try:
                cfg = _age_cfg()
                if not cfg["auto_memory_age"]:
                    await asyncio.sleep(300)
                    continue
                timer_min = cfg["memory_age_count_timer"]
                if timer_min == -1:
                    # Disabled — sleep a long time and re-check config periodically
                    await asyncio.sleep(3600)
                    continue
                await age_by_count()
            except Exception as e:
                log.warning(f"_age_count_task error: {e}")
            try:
                cfg = _age_cfg()
                sleep_sec = max(60, cfg["memory_age_count_timer"]) * 60
            except Exception:
                sleep_sec = 3600
            await asyncio.sleep(sleep_sec)

    async def _age_minutes_task():
        """Staleness aging: runs every memory_age_minutes_timer minutes."""
        from memory import age_by_minutes, _age_cfg
        while True:
            try:
                cfg = _age_cfg()
                if not cfg["auto_memory_age"]:
                    await asyncio.sleep(300)
                    continue
                timer_min = cfg["memory_age_minutes_timer"]
                if timer_min == -1:
                    await asyncio.sleep(3600)
                    continue
                await age_by_minutes(trigger_minutes=cfg["memory_age_trigger_minutes"])
            except Exception as e:
                log.warning(f"_age_minutes_task error: {e}")
            try:
                cfg = _age_cfg()
                sleep_sec = max(60, cfg["memory_age_minutes_timer"]) * 60
            except Exception:
                sleep_sec = 3600
            await asyncio.sleep(sleep_sec)

    # Run all servers, reaper, and memory aging tasks concurrently
    await asyncio.gather(
        *servers_to_run,
        _session_reaper(),
        _age_count_task(),
        _age_minutes_task(),
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MCP Agent with Plugin Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent-mcp.py                 # Start with plugins from plugins-enabled.json
  python agentctl.py list       # List available plugins
  python agentctl.py enable plugin_llama_proxy  # Enable llama proxy

Configuration Files:
  plugins-enabled.json    - Which plugins to enable
  plugin-manifest.json    - Plugin metadata
  .env                    - Environment variables and credentials
        """
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )

    args = parser.parse_args()

    # Run agent with asyncio
    try:
        asyncio.run(run_agent(host=args.host))
    except KeyboardInterrupt:
        log.info("\nShutting down...")
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
