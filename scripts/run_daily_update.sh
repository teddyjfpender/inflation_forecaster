#!/bin/bash
# run_daily_update.sh - Script to run daily inflation model updates
# This can be run manually or scheduled via cron:
# 0 6 * * * /path/to/inflation_forecaster/scripts/run_daily_update.sh >> /path/to/logs/model_update.log 2>&1

set -e  # Exit immediately if a command exits with non-zero status

# Change to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    echo "Loading environment variables..."
    export $(grep -v '^#' .env | xargs)
fi

# Update repository (if running from a repo)
if [ -d ".git" ]; then
    echo "Updating repository..."
    git pull origin main
fi

# Install/update dependencies
echo "Updating dependencies..."
pip install -r requirements.txt

# Run the inflation model
echo "Running inflation model..."
python run_inflation_model.py

# Check for changes
if [ -z "$(git status --porcelain)" ]; then
    echo "No changes detected."
    exit 0
fi

# Create a new branch
TODAY=$(date +'%Y-%m-%d')
BRANCH_NAME="model-update-${TODAY}"
echo "Creating branch: ${BRANCH_NAME}"
git checkout -b "${BRANCH_NAME}"

# Commit changes
echo "Committing changes..."
git add data/ src/saved_models/ docs/images/
git commit -m "Daily model update ${TODAY}" -m "Automated model update with latest economic data"

# Push changes
echo "Pushing changes..."
git push -u origin "${BRANCH_NAME}"

# Create a PR (this part would need a GitHub token and API call)
echo "To create a PR, visit:"
echo "https://github.com/yourusername/inflation_forecaster/pull/new/${BRANCH_NAME}"

echo "Daily update completed successfully!" 