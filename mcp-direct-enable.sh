#!/bin/bash
# Enable/disable the llmem-gw MCP Direct server in Claude Code
#
# Usage:
#   ./mcp-direct-enable.sh          # Add MCP server (local scope)
#   ./mcp-direct-enable.sh remove   # Remove MCP server
#   ./mcp-direct-enable.sh status   # Check if configured
#
# This connects Claude Code to llmem-gw's direct data layer:
# memory, goals, plans, beliefs, DB, Drive, calendar — no LLM routing.
#
# The server must be running (llmem-gw with plugin_mcp_direct enabled).

PORT="${MCP_DIRECT_PORT:-8769}"
SERVER_NAME="llmem-gw-direct"
URL="http://localhost:${PORT}/sse"

case "${1:-add}" in
  add|enable)
    echo "Adding MCP server: ${SERVER_NAME} → ${URL}"
    claude mcp add --transport sse "${SERVER_NAME}" "${URL}"
    echo ""
    echo "Done. Start a new Claude Code session to use it."
    echo "Tools available: memory_save, memory_recall, goal_create, step_create, db_query, etc."
    ;;
  remove|disable)
    echo "Removing MCP server: ${SERVER_NAME}"
    claude mcp remove "${SERVER_NAME}"
    echo "Done."
    ;;
  status)
    echo "Configured MCP servers:"
    claude mcp list 2>/dev/null | grep -A2 "${SERVER_NAME}" || echo "(not configured)"
    echo ""
    echo "Server health check:"
    curl -s "http://localhost:${PORT}/sse" --max-time 2 -o /dev/null -w "HTTP %{http_code}" 2>/dev/null || echo "not reachable"
    ;;
  *)
    echo "Usage: $0 [add|remove|status]"
    exit 1
    ;;
esac
