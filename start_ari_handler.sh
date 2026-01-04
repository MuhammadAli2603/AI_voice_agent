#!/bin/bash
#
# Startup script for Asterisk ARI Voice Agent Handler
# Starts the ARI handler that manages incoming calls via Asterisk REST Interface
#

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/venv"
ARI_SCRIPT="${SCRIPT_DIR}/telephony/asterisk_ari.py"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/ari_handler.pid"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "${LOG_DIR}"

echo -e "${GREEN}Starting Asterisk ARI Voice Agent Handler...${NC}"

# Check if virtual environment exists
if [ ! -d "${VENV_PATH}" ]; then
    echo -e "${RED}Error: Virtual environment not found at ${VENV_PATH}${NC}"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if ARI script exists
if [ ! -f "${ARI_SCRIPT}" ]; then
    echo -e "${RED}Error: ARI script not found at ${ARI_SCRIPT}${NC}"
    exit 1
fi

# Check if already running
if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if ps -p ${PID} > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: ARI handler already running (PID: ${PID})${NC}"
        echo "Use ./stop_ari_handler.sh to stop it first"
        exit 1
    else
        echo -e "${YELLOW}Removing stale PID file${NC}"
        rm "${PID_FILE}"
    fi
fi

# Check if Asterisk is running
if ! pgrep -x asterisk > /dev/null; then
    echo -e "${RED}Error: Asterisk is not running${NC}"
    echo "Please start Asterisk first: sudo systemctl start asterisk"
    exit 1
fi

# Check if ARI is enabled in Asterisk
echo "Checking ARI configuration..."
if ! sudo asterisk -rx "ari show status" | grep -q "ARI Status:"; then
    echo -e "${YELLOW}Warning: Could not verify ARI status${NC}"
    echo "Make sure ari.conf and http.conf are configured"
fi

# Load environment variables
if [ -f "${SCRIPT_DIR}/.env" ]; then
    echo "Loading environment variables from .env"
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
else
    echo -e "${YELLOW}Warning: .env file not found, using defaults${NC}"
fi

# Set default environment variables if not set
export ARI_URL="${ARI_URL:-http://localhost:8088}"
export ARI_USERNAME="${ARI_USERNAME:-voice_agent}"
export ARI_PASSWORD="${ARI_PASSWORD:-voice_agent_secret_2024}"
export ARI_APP_NAME="${ARI_APP_NAME:-voice_agent}"

echo "ARI Configuration:"
echo "  URL: ${ARI_URL}"
echo "  Username: ${ARI_USERNAME}"
echo "  App Name: ${ARI_APP_NAME}"

# Activate virtual environment and start ARI handler
echo "Starting ARI handler..."

source "${VENV_PATH}/bin/activate"

# Start in background
nohup python "${ARI_SCRIPT}" \
    > "${LOG_DIR}/ari_handler.log" 2>&1 &

# Save PID
echo $! > "${PID_FILE}"
PID=$(cat "${PID_FILE}")

# Wait a moment to check if it started successfully
sleep 2

if ps -p ${PID} > /dev/null 2>&1; then
    echo -e "${GREEN}✓ ARI handler started successfully (PID: ${PID})${NC}"
    echo "Logs: tail -f ${LOG_DIR}/ari_handler.log"
    echo "Stop: ./stop_ari_handler.sh"
else
    echo -e "${RED}✗ Failed to start ARI handler${NC}"
    echo "Check logs: ${LOG_DIR}/ari_handler.log"
    rm "${PID_FILE}"
    exit 1
fi
