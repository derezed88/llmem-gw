# Systemd Service Setup for Agent MCP

## Option 1: Systemd Services (Recommended for Production)

### 1. Create MCP Service

```bash
sudo tee /etc/systemd/system/llmem-gw.service > /dev/null << 'EOF'
[Unit]
Description=Agent MCP Server
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/projects/llmem-gw
ExecStart=/home/YOUR_USER/projects/llmem-gw/venv/bin/python llmem-gw.py
Restart=always
RestartSec=10
EnvironmentFile=/home/YOUR_USER/projects/llmem-gw/.env
StandardOutput=append:/var/log/llmem-gw.log
StandardError=append:/var/log/llmem-gw.log

[Install]
WantedBy=multi-user.target
EOF
```

### 2. (Optional) Create SSH Tunnel Service

If you expose the agent remotely via an SSH tunnel (e.g. Pinggy, ngrok), create a corresponding service that starts your tunnel script:

```bash
sudo tee /etc/systemd/system/ssh-tunnel.service > /dev/null << 'EOF'
[Unit]
Description=SSH Tunnel to Agent MCP
After=network.target llmem-gw.service
Requires=llmem-gw.service

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/projects/llmem-gw
ExecStart=/bin/bash /home/YOUR_USER/projects/llmem-gw/start_tunnel.sh
Restart=always
RestartSec=5
StandardOutput=append:/var/log/ssh-tunnel.log
StandardError=append:/var/log/ssh-tunnel.log

[Install]
WantedBy=multi-user.target
EOF
```

### 3. Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable llmem-gw.service

# Start services now
sudo systemctl start llmem-gw.service

# Check status
sudo systemctl status llmem-gw.service

# View logs
sudo journalctl -u llmem-gw.service -f
```

### 4. Management Commands

```bash
# Restart services
sudo systemctl restart llmem-gw.service

# Stop services
sudo systemctl stop llmem-gw.service

# Disable autostart
sudo systemctl disable llmem-gw.service

# View recent logs
sudo journalctl -u llmem-gw.service --since "1 hour ago"
```

## Option 2: Tmux Session (Quick Setup)

### Start in a persistent tmux session:

```bash
# Create tmux session
tmux new-session -d -s mcp

# Window 0: MCP Server
tmux send-keys -t mcp:0 "cd /home/YOUR_USER/projects/llmem-gw" C-m
tmux send-keys -t mcp:0 "source venv/bin/activate && python llmem-gw.py" C-m

# Attach to see it
tmux attach -t mcp
```

### Tmux Quick Commands:

- `tmux attach -t mcp` - Attach to session
- `Ctrl+b, d` - Detach from session
- `Ctrl+b, 0` - Switch to window 0 (server)
- `tmux kill-session -t mcp` - Kill entire session

## Option 3: Screen Session (Alternative)

```bash
# Start detached screen
screen -dmS mcp bash -c "cd /home/YOUR_USER/projects/llmem-gw && source venv/bin/activate && python llmem-gw.py"

# List screens
screen -ls

# Attach to screen
screen -r mcp

# Detach: Ctrl+a, d

# Kill screen
screen -X -S mcp quit
```

## Remote Access via SSH Tunnel

The agent listens on port **8765** (MCP/shell.py) and optionally a configurable port for the Ollama-compatible proxy (default `11434`, set via `llama_port` in `plugins-enabled.json`).
To expose these ports remotely, use any SSH tunnel service:

- **Pinggy** — `ssh -p 443 -R0:localhost:8765 a.pinggy.io`
- **ngrok** — `ngrok tcp 8765`
- **Cloudflare Tunnel** — persistent, recommended for production

### Cloudflare Tunnel Example (Recommended):

```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Login and configure
cloudflared tunnel login
cloudflared tunnel create llmem-gw
cloudflared tunnel route dns llmem-gw mcp.yourdomain.com

# Create config
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << EOF
tunnel: <tunnel-id>
credentials-file: /home/YOUR_USER/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: mcp.yourdomain.com
    service: http://localhost:8765
  - service: http_status:404
EOF

# Run tunnel
cloudflared tunnel run llmem-gw
```
