#!/bin/bash

set -e  # Exit on error

# STEP 1: Install system dependencies
echo "🔧 Installing system dependencies..."
apt update && apt install -y wget curl git make build-essential \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
  libsqlite3-dev libncursesw5-dev libgdbm-dev liblzma-dev \
  libffi-dev uuid-dev libdb-dev libexpat1-dev libmpdec-dev \
  libgmp-dev tk-dev libcrypt-dev  # 🔧 FIX: added libcrypt-dev

# STEP 2: Install Python 3.12
echo "🐍 Installing Python 3.12.0..."
cd /usr/src
rm -rf Python-3.12.0 Python-3.12.0.tgz  # 🧹 clean up if rerunning
wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz
tar xzf Python-3.12.0.tgz
cd Python-3.12.0
./configure --enable-optimizations
make -j$(nproc)
make altinstall
cd ~

# STEP 3: Clone your AI Travel Planner repo
if [ ! -d "ai-travel-planner-api" ]; then
  echo "📦 Cloning your repo..."
  git clone https://github.com/BapanBigData/ai-travel-planner-api.git
fi

cd ai-travel-planner-api

# STEP 4: Set up Python virtual environment
echo "📦 Setting up Python environment..."
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# STEP 5: Load .env file (if it exists)
if [ -f ".env" ]; then
  echo "🔐 Loading environment variables from .env..."
  export $(grep -v '^#' .env | xargs)
fi

# STEP 6: Start FastAPI server
echo "🚀 Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
