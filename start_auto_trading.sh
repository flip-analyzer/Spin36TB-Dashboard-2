#!/bin/bash

# Spin36TB Auto PRADO Daemon Startup Script
# Makes it easy to start/stop/manage the automated trading system

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_SCRIPT="$SCRIPT_DIR/auto_prado_daemon.py"
PID_FILE="$SCRIPT_DIR/logs/auto_prado.pid"
LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

case "$1" in
    start)
        echo "🚀 Starting Auto PRADO Daemon..."
        
        # Check if already running
        if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
            echo "⚠️  Daemon is already running (PID: $(cat "$PID_FILE"))"
            exit 1
        fi
        
        # Start the daemon in background
        cd "$SCRIPT_DIR"
        nohup python3 "$DAEMON_SCRIPT" > "$LOG_DIR/daemon_output.log" 2>&1 &
        echo $! > "$PID_FILE"
        
        echo "✅ Auto PRADO Daemon started (PID: $!)"
        echo "📁 Logs: $LOG_DIR/"
        echo "📊 View live output: tail -f $LOG_DIR/daemon_output.log"
        ;;
        
    stop)
        echo "🛑 Stopping Auto PRADO Daemon..."
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "⏳ Sending graceful shutdown signal..."
                kill -TERM $PID
                
                # Wait for graceful shutdown (up to 60 seconds)
                for i in {1..60}; do
                    if ! ps -p $PID > /dev/null 2>&1; then
                        echo "✅ Daemon stopped gracefully"
                        break
                    fi
                    sleep 1
                done
                
                # Force kill if still running
                if ps -p $PID > /dev/null 2>&1; then
                    echo "⚠️  Force killing daemon..."
                    kill -KILL $PID
                fi
            else
                echo "⚠️  Daemon not running"
            fi
            rm -f "$PID_FILE"
        else
            echo "⚠️  PID file not found - daemon may not be running"
        fi
        ;;
        
    status)
        if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
            PID=$(cat "$PID_FILE")
            echo "✅ Auto PRADO Daemon is running (PID: $PID)"
            
            # Show recent logs
            if [ -f "$LOG_DIR/daemon_output.log" ]; then
                echo ""
                echo "📊 Recent activity:"
                tail -5 "$LOG_DIR/daemon_output.log"
            fi
        else
            echo "❌ Auto PRADO Daemon is not running"
            if [ -f "$PID_FILE" ]; then
                rm -f "$PID_FILE"  # Clean up stale PID file
            fi
        fi
        ;;
        
    restart)
        echo "🔄 Restarting Auto PRADO Daemon..."
        $0 stop
        sleep 3
        $0 start
        ;;
        
    logs)
        if [ -f "$LOG_DIR/daemon_output.log" ]; then
            echo "📊 Live daemon output:"
            tail -f "$LOG_DIR/daemon_output.log"
        else
            echo "❌ No daemon output log found"
        fi
        ;;
        
    install-service)
        echo "⚙️  Installing as system service..."
        
        # Create systemd service file (Linux)
        if command -v systemctl >/dev/null 2>&1; then
            SERVICE_FILE="/etc/systemd/system/spin36tb-auto-prado.service"
            
            sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Spin36TB Auto PRADO Trading Daemon
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $DAEMON_SCRIPT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
            
            sudo systemctl daemon-reload
            sudo systemctl enable spin36tb-auto-prado.service
            echo "✅ Service installed. Use: sudo systemctl start spin36tb-auto-prado"
            
        # Create LaunchAgent (macOS)
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            PLIST_FILE="$HOME/Library/LaunchAgents/com.spin36tb.autoprado.plist"
            
            cat > "$PLIST_FILE" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.spin36tb.autoprado</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$DAEMON_SCRIPT</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/launchd_output.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/launchd_error.log</string>
</dict>
</plist>
EOF
            
            launchctl load "$PLIST_FILE"
            echo "✅ LaunchAgent installed. Use: launchctl start com.spin36tb.autoprado"
        else
            echo "❌ Unsupported system for service installation"
        fi
        ;;
        
    *)
        echo "🤖 Spin36TB Auto PRADO Daemon Control"
        echo "=================================="
        echo "Usage: $0 {start|stop|restart|status|logs|install-service}"
        echo ""
        echo "Commands:"
        echo "  start           - Start the automated trading daemon"
        echo "  stop            - Stop the daemon gracefully"
        echo "  restart         - Restart the daemon"
        echo "  status          - Check if daemon is running"
        echo "  logs            - View live daemon output"
        echo "  install-service - Install as system service (auto-start)"
        echo ""
        echo "Logs location: $LOG_DIR/"
        exit 1
        ;;
esac