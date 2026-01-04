#!/bin/bash
#
# Stop script for Asterisk ARI Voice Agent Handler
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${LOG_DIR}/ari_handler.pid"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "Stopping Asterisk ARI Voice Agent Handler..."

if [ ! -f "${PID_FILE}" ]; then
    echo -e "${YELLOW}Warning: PID file not found${NC}"
    echo "ARI handler may not be running"
    exit 1
fi

PID=$(cat "${PID_FILE}")

if ps -p ${PID} > /dev/null 2>&1; then
    echo "Stopping ARI handler (PID: ${PID})..."
    kill ${PID}

    # Wait for process to stop
    for i in {1..10}; do
        if ! ps -p ${PID} > /dev/null 2>&1; then
            echo -e "${GREEN}✓ ARI handler stopped successfully${NC}"
            rm "${PID_FILE}"
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo -e "${YELLOW}Process did not stop gracefully, forcing...${NC}"
    kill -9 ${PID}
    rm "${PID_FILE}"
    echo -e "${GREEN}✓ ARI handler forcefully stopped${NC}"
else
    echo -e "${YELLOW}ARI handler not running (stale PID file)${NC}"
    rm "${PID_FILE}"
fi
