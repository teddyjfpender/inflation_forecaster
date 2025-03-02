FROM python:3.9-slim

WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for logs and output
RUN mkdir -p logs

# Set up environment for git
RUN git config --global user.name "Docker Bot" && \
    git config --global user.email "docker@example.com"

# Set default environment variables
ENV BLS_API_KEY=""
ENV FRED_API_KEY=""
ENV GITHUB_TOKEN=""
ENV GITHUB_REPO_OWNER=""
ENV GITHUB_REPO_NAME="inflation_forecaster"

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "daily" ]; then\n\
    python scripts/daily_update.py "$@"\n\
elif [ "$1" = "cron" ]; then\n\
    echo "0 6 * * * cd /app && python scripts/daily_update.py --auto-merge" > /etc/cron.d/model-cron\n\
    chmod 0644 /etc/cron.d/model-cron\n\
    crontab /etc/cron.d/model-cron\n\
    cron -f\n\
else\n\
    python run_inflation_model.py "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (if no argument provided)
CMD ["daily"] 