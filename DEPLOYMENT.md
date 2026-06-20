# Deployment Guide - CPU Machine Setup

**Target:** 8 CPU / 32GB RAM / 500GB disk cloud VM  
**Purpose:** 24/7 data collection, paper trading, live trading  
**Last Updated:** 2026-06-21

---

## 🎯 Machine Specifications

### Recommended: 8 CPU / 32GB RAM / 500GB Disk

**Cloud Provider Options:**

| Provider | Instance Type | vCPU | RAM | Cost/Month | Notes |
|----------|--------------|------|-----|------------|-------|
| **AWS EC2** | m6i.2xlarge | 8 | 32GB | ~$250 | Spot: ~$75, best for production |
| **GCP Compute** | n2-standard-8 | 8 | 32GB | ~$240 | Preemptible: ~$70 |
| **DigitalOcean** | CPU-Optimized 8GB | 8 | 32GB | ~$168 | Simple pricing, easy setup |
| **Hetzner** | CCX33 | 8 | 32GB | ~$60 | **Cheapest**, EU-based |
| **Vultr** | 8 vCPU / 32GB | 8 | 32GB | ~$144 | Good global locations |

**Recommendation:** Start with **Hetzner CCX33** (~$60/month) or **DigitalOcean** (~$168/month) for simplicity.

### Why 8 CPU (not 4 CPU)?

- **Data collection:** 2-3 cores for WebSocket streams (trade, depth, mark price, liquidations)
- **Disk I/O:** 1-2 cores for Parquet compression/writes
- **Strategy execution:** 2-3 cores for real-time feature computation + model inference
- **Monitoring:** 1 core for health checks, logging, alerts
- **Headroom:** Handle spikes without dropping data

---

## 📋 Deployment Steps

### Step 1: Provision VM

#### Option A: Hetzner (Cheapest)

```bash
# 1. Sign up at https://www.hetzner.com/cloud
# 2. Create project
# 3. Add server:
#    - Location: Nuremberg (EU) or Ashburn (US)
#    - Type: CCX33 (8 vCPU, 32GB RAM)
#    - Image: Ubuntu 22.04 LTS
#    - Volume: 500GB
#    - SSH key: Upload your public key

# 4. Note the public IP
```

#### Option B: DigitalOcean (Easiest)

```bash
# 1. Sign up at https://www.digitalocean.com
# 2. Create Droplet:
#    - Plan: CPU-Optimized, 8 vCPU / 32GB RAM
#    - Region: Choose closest to you
#    - Image: Ubuntu 22.04 LTS
#    - Add Block Storage: 500GB volume
#    - SSH key: Upload your public key

# 3. Note the public IP
```

#### Option C: AWS EC2 (Production-grade)

```bash
# 1. AWS Console → EC2 → Launch Instance
# 2. Configure:
#    - Name: btc-trading-cpu
#    - AMI: Ubuntu Server 22.04 LTS
#    - Instance type: m6i.2xlarge
#    - Key pair: Select or create
#    - Storage: 500GB gp3 EBS
#    - Security group: Allow SSH (22) from your IP
#    - Consider spot instance for 70% savings

# 3. Launch and note public IP
```

---

### Step 2: Initial Server Setup

```bash
# SSH into server
ssh root@<your-server-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
  python3.11 \
  python3.11-venv \
  python3.11-dev \
  git \
  tmux \
  htop \
  curl \
  wget \
  build-essential

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Verify installation
uv --version
python3.11 --version
```

---

### Step 3: Clone Repository & Install

```bash
# Create project directory
mkdir -p ~/projects
cd ~/projects

# Clone repository
git clone <your-repo-url> quanthack
cd quanthack

# Install project dependencies
uv sync

# Verify installation
uv run intraday --help
```

---

### Step 4: Configure Data Directories

```bash
# Create data storage structure
mkdir -p data/raw/binance/{klines,trades,depth,funding,open_interest,mark_price,liquidations}
mkdir -p data/processed
mkdir -p data/checkpoints
mkdir -p logs

# Set permissions
chmod -R 755 data/
chmod -R 755 logs/

# Verify structure
tree -L 3 data/
```

---

### Step 5: Download Historical Data

```bash
# Create tmux session for downloads
tmux new -s downloads

# Download 12 months historical data
# This will take ~20-30 minutes total

# 5-minute klines (for features)
uv run intraday data download \
  --kind klines_5m \
  --start 2024-01-01 \
  --end 2024-12-31

# 1-minute klines (for fine-grained analysis)
uv run intraday data download \
  --kind klines_1m \
  --start 2024-01-01 \
  --end 2024-12-31

# Funding rate history
uv run intraday data download \
  --kind funding \
  --start 2024-01-01 \
  --end 2024-12-31

# Open interest history
uv run intraday data download \
  --kind open_interest \
  --start 2024-01-01 \
  --end 2024-12-31

# Verify downloads
uv run intraday data summary
uv run intraday data checkpoint

# Detach from tmux (Ctrl+B, then D)
# Reattach later: tmux attach -t downloads
```

---

### Step 6: Start Live Data Capture

```bash
# Create persistent tmux session
tmux new -s data-capture

# Start live capture (runs indefinitely)
uv run intraday data live-capture \
  --streams trade,depth,mark_price,liquidations

# You should see:
# [INFO] Starting live capture...
# [INFO] Connected to Binance WebSocket
# [INFO] Streaming: trade, depth, mark_price, liquidations
# [INFO] Writing to: data/raw/binance/

# Detach from tmux: Ctrl+B, then D
# The capture will keep running in background

# Reattach later to check status:
# tmux attach -t data-capture
```

---

### Step 7: Set Up Monitoring

```bash
# Create monitoring script
cat > ~/monitor_data.sh <<'EOF'
#!/bin/bash
echo "==============================================="
echo "Data Collection Status - $(date)"
echo "==============================================="
echo ""

cd ~/projects/quanthack

echo "=== Data Summary ==="
uv run intraday data summary
echo ""

echo "=== Checkpoint Status ==="
uv run intraday data checkpoint
echo ""

echo "=== Disk Usage ==="
df -h ~/projects/quanthack/data
echo ""

echo "=== Process Status ==="
ps aux | grep "intraday data live-capture" | grep -v grep
echo ""

echo "=== Memory Usage ==="
free -h
echo ""

echo "=== CPU Load ==="
uptime
echo ""
EOF

chmod +x ~/monitor_data.sh

# Test monitoring script
~/monitor_data.sh

# Add daily email report (optional, requires mail setup)
# (crontab -l 2>/dev/null; echo "0 9 * * * ~/monitor_data.sh | mail -s 'BTC Data Status' your-email@example.com") | crontab -
```

---

### Step 8: Set Up Health Checks

```bash
# Create health check script
cat > ~/healthcheck.sh <<'EOF'
#!/bin/bash

# Check if live capture is running
if ! pgrep -f "intraday data live-capture" > /dev/null; then
    echo "[ERROR] Live capture not running! Restarting..."
    cd ~/projects/quanthack
    tmux send-keys -t data-capture C-c
    sleep 2
    tmux send-keys -t data-capture "uv run intraday data live-capture --streams trade,depth,mark_price,liquidations" Enter
    echo "[INFO] Live capture restarted at $(date)" >> ~/logs/restarts.log
fi

# Check disk space (alert if <10% free)
DISK_USAGE=$(df -h ~/projects/quanthack/data | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "[WARN] Disk usage at ${DISK_USAGE}% - cleanup needed!" >> ~/logs/alerts.log
fi
EOF

chmod +x ~/healthcheck.sh

# Add to cron (check every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * ~/healthcheck.sh") | crontab -

# Verify cron job added
crontab -l
```

---

### Step 9: Configure Firewall (Security)

```bash
# Install UFW (Uncomplicated Firewall)
sudo apt install -y ufw

# Allow SSH
sudo ufw allow 22/tcp

# Allow only from your IP (recommended)
sudo ufw allow from <your-home-ip> to any port 22

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

---

### Step 10: Test Everything

```bash
# Run Phase 1 tests
cd ~/projects/quanthack
uv run pytest tests/phase_01/ -v

# Check data summary
uv run intraday data summary

# Expected output:
# ✅ 12 months klines_5m (52,560 rows)
# ✅ 12 months klines_1m (262,800 rows)
# ✅ Funding rate history
# ✅ Open interest history
# ✅ Live capture running (X hours captured)

# Check tmux sessions
tmux ls
# Should show: data-capture: 1 windows

# Attach to live capture to verify
tmux attach -t data-capture
# Should see real-time data streaming
# Detach: Ctrl+B, D

# Check logs
tail -f logs/*.log
```

---

## 📊 Monitoring Commands

```bash
# Quick status check
~/monitor_data.sh

# Detailed data summary
uv run intraday data summary

# Checkpoint status
uv run intraday data checkpoint

# Disk usage
df -h ~/projects/quanthack/data

# Memory usage
free -h

# CPU usage
htop

# Check live capture process
ps aux | grep "intraday data live-capture"

# View live capture logs
tmux attach -t data-capture

# View health check logs
tail -f ~/logs/restarts.log
tail -f ~/logs/alerts.log
```

---

## 🔄 Maintenance Tasks

### Daily

```bash
# Check monitoring script output
~/monitor_data.sh

# Verify live capture running
tmux attach -t data-capture  # Quick peek
```

### Weekly

```bash
# Check disk usage trends
du -sh ~/projects/quanthack/data/raw/binance/*

# Verify data quality
uv run pytest tests/phase_01/test_schema.py -v

# Update system packages
sudo apt update && sudo apt upgrade -y
```

### After 4-6 Weeks

```bash
# Verify sufficient tick data collected
uv run intraday data summary
# Should show ≥4-6 weeks of trade, depth, mark_price data

# Create backup before starting development
tar -czf ~/backups/data-backup-$(date +%Y%m%d).tar.gz ~/projects/quanthack/data/

# Ready to start Phase 2-7 MVP development!
```

---

## 🚨 Troubleshooting

### Live Capture Stopped

```bash
# Check if process running
ps aux | grep "intraday data live-capture"

# Check tmux session
tmux ls
tmux attach -t data-capture

# Restart if needed
tmux send-keys -t data-capture C-c
sleep 2
tmux send-keys -t data-capture "cd ~/projects/quanthack" Enter
tmux send-keys -t data-capture "uv run intraday data live-capture --streams trade,depth,mark_price,liquidations" Enter
```

### Disk Full

```bash
# Check what's using space
du -sh ~/projects/quanthack/data/raw/binance/*

# Compress old data (if needed)
cd ~/projects/quanthack/data/raw/binance/trades/
gzip *.parquet  # Compress older files

# Or delete test data (if any)
rm -rf ~/projects/quanthack/data/test_*
```

### High Memory Usage

```bash
# Check memory consumers
ps aux --sort=-%mem | head -10

# Restart live capture (clears buffers)
tmux send-keys -t data-capture C-c
sleep 5
tmux send-keys -t data-capture "uv run intraday data live-capture --streams trade,depth,mark_price,liquidations" Enter
```

### API Rate Limits

```bash
# Check download checkpoint
uv run intraday data checkpoint

# Resume from last checkpoint (automatic)
uv run intraday data download --kind klines_5m

# Downloads will auto-retry with backoff
```

---

## 🎯 Next Steps

**After deployment complete:**
1. ✅ Verify all historical data downloaded
2. ✅ Verify live capture running
3. ⏰ Set calendar reminder for 4-6 weeks from now
4. 📖 Read `idea/phases/02_features.md` to prepare for Phase 2

**After 4-6 weeks:**
1. Verify ≥4-6 weeks tick data collected
2. Start Phase 2-7 MVP development (local or on this machine)
3. Test entire pipeline on small dataset
4. Retrain on full 12-month data for production
5. Deploy production models to this machine for paper trading

---

## 📝 Checklist

- [ ] VM provisioned (8 CPU / 32GB RAM / 500GB disk)
- [ ] SSH access working
- [ ] System updated and dependencies installed
- [ ] Repository cloned and uv installed
- [ ] Data directories created
- [ ] Historical data downloaded (12 months)
- [ ] Live capture started in tmux
- [ ] Monitoring script set up
- [ ] Health check cron job configured
- [ ] Firewall configured
- [ ] Tests passing
- [ ] Calendar reminder set for 4-6 weeks

---

**Estimated Setup Time:** 2-3 hours  
**Monthly Cost:** $60-250 (depending on provider)  
**Data Collection:** 4-6 weeks (hands-off)
