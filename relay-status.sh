#!/bin/bash
# Poll voice relay status for all GED subjects + samaritan-work
# Usage:
#   ./relay-status.sh          # one-shot
#   ./relay-status.sh watch    # refresh every 10s

MCP_PORT=8769
CHANNELS=("ged-math" "ged-reading" "ged-writing" "ged-science" "ged-social" "default")
LABELS=("Math" "Reading" "Writing" "Science" "Social" "samaritan-work")

_poll() {
    printf "%-16s %-9s %s\n" "CHANNEL" "STATUS" "DETAILS"
    printf "%-16s %-9s %s\n" "───────" "──────" "───────"
    for i in "${!CHANNELS[@]}"; do
        ch="${CHANNELS[$i]}"
        lbl="${LABELS[$i]}"
        data=$(curl -sf "http://localhost:${MCP_PORT}/voice_relay/status?channel=$ch" 2>/dev/null)
        if [ -z "$data" ]; then
            printf "%-16s %-9s %s\n" "$lbl" "?" "unreachable"
            continue
        fi
        enabled=$(echo "$data" | python3 -c "import sys,json; print('ON' if json.load(sys.stdin).get('enabled') else 'off')" 2>/dev/null)
        inbox=$(echo "$data" | python3 -c "import sys,json; print(json.load(sys.stdin).get('inbox_count',0))" 2>/dev/null)
        polls=$(echo "$data" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('polls_remaining','?'); e=d.get('empty_polls','?'); print(f'idle {e}/{d.get(\"idle_limit\",\"?\")}  remaining {r}')" 2>/dev/null)
        if [ "$enabled" = "ON" ]; then
            printf "%-16s \e[32m%-9s\e[0m inbox=%s  %s\n" "$lbl" "$enabled" "$inbox" "$polls"
        else
            printf "%-16s \e[90m%-9s\e[0m\n" "$lbl" "$enabled"
        fi
    done
    echo ""
    echo "$(date '+%H:%M:%S')"
}

case "${1:-once}" in
    watch)
        interval="${2:-10}"
        while true; do
            clear
            _poll
            sleep "$interval"
        done
        ;;
    help|-h|--help)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (none)       One-shot status check"
        echo "  watch [N]    Auto-refresh every N seconds (default 10)"
        echo "  help         Show this help"
        ;;
    *)
        _poll
        ;;
esac
