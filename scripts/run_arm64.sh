#!/bin/bash
set -e
echo "Downloading MSFT data..."
python3.10 scripts/download_data.py
echo "Building Docker image..."
docker build -f docker/Dockerfile.arm64 -t flag-trader:arm64-0.3 .
echo "Running FLAG-TRADER pilot..."
docker run --rm --memory="8g" \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/logs:/workspace/logs \
    -v $(pwd)/figures:/workspace/figures \
    flag-trader:arm64-0.3
echo "Results saved in logs/demo_msft.csv and figures/"
