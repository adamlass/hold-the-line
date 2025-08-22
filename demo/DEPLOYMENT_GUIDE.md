# Deployment Guide for Whisper Titans Demo on Raspberry Pi 4

## Prerequisites

### On your Raspberry Pi:
- Raspberry Pi 4 with at least 4GB RAM (8GB recommended)
- Raspberry Pi OS (64-bit recommended for better Docker performance)
- Static IP address configured
- SSH access enabled

## Step 1: Initial Raspberry Pi Setup

### 1.1 Update your Raspberry Pi
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Install Docker and Docker Compose
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version

# Install Docker Compose
sudo apt install -y docker-compose
```

### 1.3 Configure Docker for ARM architecture
```bash
# Enable Docker to start on boot
sudo systemctl enable docker
sudo systemctl start docker
```

## Step 2: Deploy the Application

### 2.1 Clone the repository on your Raspberry Pi
```bash
cd ~
git clone https://github.com/yourusername/whisper-titans.git
cd whisper-titans/demo
```

### 2.2 Copy training data
```bash
# Create data directory structure (important: maintain the exact structure)
mkdir -p data

# Copy your training data from your main machine to the Pi
# From your main machine (run this from the whisper-titans directory):
scp -r data/trainings pi@your-pi-ip:~/whisper-titans/demo/data/

# Or use rsync for better performance with large files:
rsync -avz --progress data/trainings/ pi@your-pi-ip:~/whisper-titans/demo/data/trainings/

# The final structure on your Pi should be:
# ~/whisper-titans/demo/
#   ├── data/
#   │   └── trainings/
#   │       ├── training_debug_20250807_103321/
#   │       │   ├── metadata.json
#   │       │   └── episodes/
#   │       └── ... (other training folders)
```

### 2.3 Create necessary directories
```bash
mkdir -p logs ssl nginx-cache certbot-www
```

### 2.4 Build and start the application
```bash
# Build the Docker image
docker-compose build

# Start the application (without SSL first)
docker-compose up -d whisper-demo
```

## Step 3: Domain and Network Configuration

### 3.1 Configure your router
1. **Port Forwarding**: Forward these ports to your Raspberry Pi's static IP:
   - Port 80 (HTTP)
   - Port 443 (HTTPS)
   - Port 8000 (Optional: direct app access for testing)

2. **Example for common routers**:
   - Access router admin panel (usually 192.168.1.1 or 192.168.0.1)
   - Navigate to Port Forwarding/Virtual Server section
   - Add rules:
     ```
     External Port: 80  → Internal IP: [Pi IP] → Internal Port: 80
     External Port: 443 → Internal IP: [Pi IP] → Internal Port: 443
     ```

### 3.2 Configure DNS
1. **Log into your domain registrar** (where you bought holdtheline.ai)
2. **Update DNS records**:
   ```
   Type: A     Name: @              Value: [Your Home Static IP]    TTL: 300
   Type: A     Name: www            Value: [Your Home Static IP]    TTL: 300
   Type: CNAME Name: *.holdtheline  Value: holdtheline.ai          TTL: 300
   ```

3. **Verify DNS propagation** (may take up to 48 hours):
   ```bash
   nslookup holdtheline.ai
   dig holdtheline.ai
   ```

## Step 4: SSL Certificate Setup

### 4.1 Initial HTTP setup (required for Let's Encrypt)
```bash
# Start nginx without SSL first
docker-compose up -d nginx
```

### 4.2 Obtain SSL certificates
```bash
# Edit docker-compose.ssl.yml and replace your-email@example.com with your actual email

# Run certbot to get certificates
docker-compose -f docker-compose.yml -f docker-compose.ssl.yml run --rm certbot

# After successful certificate generation, copy certificates to the ssl directory
sudo cp -r /etc/letsencrypt/live/holdtheline.ai/* ./ssl/
sudo chown -R $USER:$USER ./ssl/
```

### 4.3 Restart with SSL enabled
```bash
docker-compose restart nginx
```

### 4.4 Setup automatic certificate renewal
```bash
# Add cron job for renewal
(crontab -l 2>/dev/null; echo "0 2 * * * cd ~/whisper-titans/demo && docker-compose -f docker-compose.yml -f docker-compose.ssl.yml run --rm certbot renew && docker-compose restart nginx") | crontab -
```

## Step 5: Production Optimizations

### 5.1 Enable swap (for limited RAM)
```bash
# Check current swap
free -h

# Create swap file if needed (2GB example)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 5.2 Monitor resources
```bash
# Install monitoring tools
sudo apt install -y htop iotop

# Monitor Docker containers
docker stats

# Check logs
docker-compose logs -f whisper-demo
docker-compose logs -f nginx
```

### 5.3 Setup log rotation
```bash
# Already configured in docker-compose.yml with json-file logging driver
# Logs are automatically rotated when they reach 10MB, keeping last 3 files
```

## Step 6: Maintenance

### 6.1 Backup data
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/pi/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/whisper_data_$DATE.tar.gz data/
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x backup.sh

# Add to cron for daily backups
(crontab -l 2>/dev/null; echo "0 3 * * * /home/pi/whisper-titans/demo/backup.sh") | crontab -
```

### 6.2 Update application
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 6.3 View logs
```bash
# Application logs
docker-compose logs -f --tail=100 whisper-demo

# Nginx logs
docker-compose logs -f --tail=100 nginx

# All services
docker-compose logs -f
```

## Step 7: Troubleshooting

### Common Issues:

1. **Port forwarding not working**:
   ```bash
   # Test from outside network
   curl http://your-external-ip
   
   # Check if ports are open
   sudo netstat -tlnp | grep -E ':(80|443|8000)'
   ```

2. **Docker build fails on ARM**:
   ```bash
   # Ensure you're using ARM-compatible base images
   # The Dockerfile already uses python:3.11-slim-bookworm which supports ARM
   ```

3. **Out of memory**:
   ```bash
   # Reduce worker count in Dockerfile CMD
   # Change from workers=2 to workers=1
   
   # Or adjust memory limits in docker-compose.yml
   ```

4. **DNS not resolving**:
   ```bash
   # Flush DNS cache on your machine
   # Windows: ipconfig /flushdns
   # Mac: sudo dscacheutil -flushcache
   # Linux: sudo systemd-resolve --flush-caches
   ```

5. **SSL certificate issues**:
   ```bash
   # Check certificate status
   docker-compose exec nginx nginx -t
   
   # Manually renew
   docker-compose -f docker-compose.yml -f docker-compose.ssl.yml run --rm certbot renew
   ```

## Step 8: Security Hardening

### 8.1 Firewall setup
```bash
# Install and configure UFW
sudo apt install -y ufw

# Allow SSH (be careful not to lock yourself out!)
sudo ufw allow 22/tcp

# Allow web traffic
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw --force enable
```

### 8.2 Fail2ban for brute force protection
```bash
# Install fail2ban
sudo apt install -y fail2ban

# Configure for Docker and Nginx
sudo nano /etc/fail2ban/jail.local
# Add:
# [nginx-http-auth]
# enabled = true
# [nginx-noscript]
# enabled = true
```

### 8.3 Regular updates
```bash
# Create update script
cat > update_system.sh << 'EOF'
#!/bin/bash
sudo apt update
sudo apt upgrade -y
docker-compose pull
docker system prune -af
EOF

chmod +x update_system.sh
```

## Alternative: Quick Start (Without SSL)

For testing purposes, you can quickly start without SSL:

```bash
# Simple start (HTTP only on port 8000)
docker-compose up -d whisper-demo

# Access at http://your-pi-ip:8000
```

Then configure domain and SSL later.

## Monitoring Dashboard (Optional)

Consider adding Portainer for easy Docker management:

```bash
docker run -d \
  -p 9000:9000 \
  --name=portainer \
  --restart=unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:linux-arm64
```

Access at: http://your-pi-ip:9000

## Performance Tips

1. **Use SSD**: Boot from SSD instead of SD card for better performance
2. **Overclock** (optional): Edit `/boot/config.txt` for mild overclocking
3. **Optimize Docker**: 
   ```bash
   # Limit Docker logs
   echo '{"log-driver":"json-file","log-opts":{"max-size":"10m","max-file":"3"}}' | sudo tee /etc/docker/daemon.json
   sudo systemctl restart docker
   ```

## Support

- Check application health: `curl http://localhost:8000/`
- Container status: `docker-compose ps`
- Resource usage: `docker stats`
- Application logs: `docker-compose logs whisper-demo`

Remember to replace placeholder values:
- `your-email@example.com` with your actual email for Let's Encrypt
- `your-pi-ip` with your Raspberry Pi's IP address
- `your-external-ip` with your home's static IP address
