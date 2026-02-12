#!/bin/bash
# Helper script to run evalscope evaluation on HiFP8 fake-quantized models
#
# Usage:
#   ./scripts/run_evalscope_evaluation.sh /path/to/quantized_model [port] [model_name]
#
# Example:
#   ./scripts/run_evalscope_evaluation.sh /home/data/quantized_qwen3_0.6b 8000 qwen3-0.6b-hifp8

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
MODEL_PATH="${1:-/home/data/quantized_qwen3_0.6b}"
PORT="${2:-8000}"
MODEL_NAME="${3:-qwen3-hifp8}"

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_PATH${NC}"
    echo "Usage: $0 /path/to/quantized_model [port] [model_name]"
    exit 1
fi

# Find project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================================================"
echo "HiFP8 Evalscope Evaluation"
echo "========================================================================"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Model Name: $MODEL_NAME"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Step 1: Start API server
echo -e "${YELLOW}[1/5] Starting API server...${NC}"

SERVER_LOG="$PROJECT_ROOT/server_$PORT.log"
SERVER_PID_FILE="$PROJECT_ROOT/server_$PORT.pid"

# Kill existing server on this port if any
if [ -f "$SERVER_PID_FILE" ]; then
    OLD_PID=$(cat "$SERVER_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Killing existing server (PID: $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$SERVER_PID_FILE"
fi

# Start server in background
cd "$PROJECT_ROOT"
python scripts/start_vllm_hifp8_server.py \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --model-name "$MODEL_NAME" \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$SERVER_PID_FILE"
echo "Server started (PID: $SERVER_PID)"
echo "Server logs: $SERVER_LOG"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ -f "$SERVER_PID_FILE" ]; then
        PID=$(cat "$SERVER_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping server (PID: $PID)"
            kill "$PID" 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if ps -p "$PID" > /dev/null 2>&1; then
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
        rm -f "$SERVER_PID_FILE"
    fi
    echo "Cleanup complete"
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Step 2: Wait for server to be ready
echo ""
echo -e "${YELLOW}[2/5] Waiting for server to be ready...${NC}"

MAX_WAIT=120  # Maximum wait time in seconds
WAIT_TIME=0
SLEEP_INTERVAL=2

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Server is ready!${NC}"
        break
    fi

    # Check if server process is still running
    if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
        echo -e "${RED}Error: Server process died${NC}"
        echo "Last 20 lines of server log:"
        tail -n 20 "$SERVER_LOG"
        exit 1
    fi

    echo "Waiting... ($WAIT_TIME/$MAX_WAIT seconds)"
    sleep $SLEEP_INTERVAL
    WAIT_TIME=$((WAIT_TIME + SLEEP_INTERVAL))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo -e "${RED}Error: Server failed to start within $MAX_WAIT seconds${NC}"
    echo "Last 30 lines of server log:"
    tail -n 30 "$SERVER_LOG"
    exit 1
fi

# Step 3: Test API connection
echo ""
echo -e "${YELLOW}[3/5] Testing API connection...${NC}"

# Test health endpoint
HEALTH_RESPONSE=$(curl -s "http://localhost:$PORT/health")
echo "Health check: $HEALTH_RESPONSE"

# Test models endpoint
MODELS_RESPONSE=$(curl -s "http://localhost:$PORT/v1/models")
echo "Models endpoint: $MODELS_RESPONSE"

# Test simple completion
echo "Testing completion endpoint..."
COMPLETION_RESPONSE=$(curl -s "http://localhost:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"prompt\": \"Hello\",
    \"max_tokens\": 10,
    \"temperature\": 0.7
  }")

if echo "$COMPLETION_RESPONSE" | grep -q "choices"; then
    echo -e "${GREEN}API test successful!${NC}"
else
    echo -e "${RED}API test failed!${NC}"
    echo "Response: $COMPLETION_RESPONSE"
    exit 1
fi

# Step 4: Check if evalscope is installed
echo ""
echo -e "${YELLOW}[4/5] Checking evalscope installation...${NC}"

if ! command -v evalscope &> /dev/null; then
    echo -e "${RED}Error: evalscope not found${NC}"
    echo "Install with: pip install evalscope"
    exit 1
fi

echo "Evalscope version:"
evalscope --version || true

# Step 5: Run evalscope evaluation
echo ""
echo -e "${YELLOW}[5/5] Running evalscope evaluation...${NC}"
echo ""

# Check if config file exists
CONFIG_FILE="$PROJECT_ROOT/examples/evalscope_config.yaml"

if [ -f "$CONFIG_FILE" ]; then
    echo "Using config file: $CONFIG_FILE"
    echo ""

    # Update config file with current settings
    # Note: This is a simple sed replacement, may not work for all YAML structures
    # For production, use a proper YAML parser

    evalscope eval \
        --model "$MODEL_NAME" \
        --api-base "http://localhost:$PORT/v1" \
        --datasets mmlu ceval \
        --num-fewshot 5 \
        --output-dir "$PROJECT_ROOT/evalscope_results"
else
    echo "Config file not found, using CLI arguments"
    echo ""

    evalscope eval \
        --model "$MODEL_NAME" \
        --api-base "http://localhost:$PORT/v1" \
        --datasets mmlu ceval \
        --num-fewshot 5 \
        --output-dir "$PROJECT_ROOT/evalscope_results"
fi

# Results summary
echo ""
echo "========================================================================"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo "========================================================================"
echo "Results saved to: $PROJECT_ROOT/evalscope_results"
echo "Server logs: $SERVER_LOG"
echo ""
echo "Server will be stopped automatically"
