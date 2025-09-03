# 🤖 Spin36TB Automated PRADO Trading System

**Fully automated 24/7 trading with continuous learning and adaptive improvements.**

## 🚀 Quick Start

### 1. Start Automated Trading
```bash
./start_auto_trading.sh start
```

### 2. Monitor Live Trading
```bash
python monitor_auto_trading.py
```

### 3. Check Status
```bash
./start_auto_trading.sh status
```

### 4. Stop Trading
```bash
./start_auto_trading.sh stop
```

## 📋 Features

✅ **Automated Market Detection**
- Starts automatically when forex markets open
- Stops during market closure (weekends)
- Handles time zone changes automatically

✅ **24/7 Operation**
- Runs continuously during trading hours
- Daily restart at 5 PM EST (after NY close)
- Automatic error recovery

✅ **Comprehensive Logging**
- All activity logged to daily files
- Trade history preserved
- Error tracking and recovery logs

✅ **Adaptive Learning Integration**
- Continuous learning from trade outcomes
- Dynamic position sizing based on performance
- Model updates after each trade

✅ **PRADO Enhancements**
- López de Prado meta-labeling
- Dynamic triple barriers
- Market regime detection

## 📁 File Structure

```
spin36TB/
├── auto_prado_daemon.py          # Main daemon
├── start_auto_trading.sh          # Control script
├── monitor_auto_trading.py        # Live monitoring
├── auto_prado_config.json         # Configuration
├── logs/                          # All log files
│   ├── auto_prado_YYYYMMDD.log   # Daily trading logs
│   ├── daemon_output.log          # System logs
│   └── auto_prado.pid            # Process ID
└── README_AUTO_TRADING.md         # This file
```

## ⚙️ Configuration

Edit `auto_prado_config.json` to customize:

```json
{
  "starting_capital": 25000,
  "max_session_hours": 24,
  "restart_hour": 17,
  "market_check_interval": 300,
  "error_retry_delay": 60,
  "max_consecutive_errors": 5
}
```

## 🎮 Control Commands

### Basic Controls
- `./start_auto_trading.sh start` - Start daemon
- `./start_auto_trading.sh stop` - Stop daemon  
- `./start_auto_trading.sh restart` - Restart daemon
- `./start_auto_trading.sh status` - Check status
- `./start_auto_trading.sh logs` - View live logs

### System Service (Optional)
- `./start_auto_trading.sh install-service` - Install as system service

## 📊 Monitoring

### Real-time Dashboard
```bash
python monitor_auto_trading.py
```

### Log Files
- **Daily Trading**: `logs/auto_prado_YYYYMMDD.log`
- **System Output**: `logs/daemon_output.log`
- **Trade Sessions**: `logs/live_prado_session_*.json`

### Key Metrics Tracked
- Capital and returns
- Trade win/loss rates
- PRADO enhancement statistics
- Learning system performance
- Market regime detection
- Error rates and recovery

## 🔧 Troubleshooting

### Daemon Won't Start
1. Check if already running: `./start_auto_trading.sh status`
2. Check logs: `tail logs/daemon_output.log`
3. Verify dependencies: All required Python packages installed

### No Trades Being Executed  
1. Check market hours (daemon pauses on weekends)
2. Verify OANDA connection in logs
3. Check if learning system is filtering trades

### High Error Rates
1. Check network connectivity
2. Verify OANDA API credentials
3. Review error logs in daily log files

## 📈 Performance Optimization

### Learning System
- Records every trade outcome
- Adapts position sizing based on success patterns
- Filters poor-quality signals over time
- Updates models daily

### PRADO Enhancements  
- Dynamic profit targets based on market volatility
- Adaptive stop losses using meta-labeling
- Time-based exits to prevent overnight exposure

## 🚨 Safety Features

### Risk Management
- Maximum position size caps
- Daily loss limits (configurable)
- Emergency stop mechanisms
- Automatic system recovery

### Market Protection
- Weekend shutdown (forex markets closed)
- Holiday detection and pause
- Network failure handling
- Data feed backup systems

## 📞 Support

### Log Analysis
All issues are automatically logged with timestamps. Check:
1. `logs/auto_prado_YYYYMMDD.log` for trading issues
2. `logs/daemon_output.log` for system issues

### Manual Override
- Stop: `./start_auto_trading.sh stop`
- Emergency kill: `kill -9 $(cat logs/auto_prado.pid)`

---

🤖 **Fully automated, continuously learning, 24/7 PRADO-enhanced trading system**