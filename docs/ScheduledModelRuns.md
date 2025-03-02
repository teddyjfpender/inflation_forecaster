# Scheduled Model Runs Guide

This document explains the different ways to set up scheduled runs of the inflation forecasting models.

## Table of Contents

1. [GitHub Actions (Recommended)](#github-actions-recommended)
2. [Python Script](#python-script)
3. [Shell Script](#shell-script)
4. [Docker Container](#docker-container)
5. [Systemd Service (Linux)](#systemd-service-linux)
6. [Windows Task Scheduler](#windows-task-scheduler)
7. [Cloud Options](#cloud-options)

## GitHub Actions (Recommended)

GitHub Actions provides a managed environment for running scheduled tasks directly within your GitHub repository.

### Setup

1. The workflow is already set up in `.github/workflows/daily_model_runs.yml`
2. Add required secrets to your GitHub repository:
   - Go to Settings > Secrets and Variables > Actions
   - Add these repository secrets:
     - `BLS_API_KEY`: Your Bureau of Labor Statistics API key
     - `FRED_API_KEY`: Your FRED API key

### How It Works

- The workflow runs automatically at 6 AM UTC every day
- It checks out the repository, installs dependencies, and runs the model
- If there are changes, it creates a new branch, makes a PR, and auto-merges it
- You can also trigger it manually from the Actions tab

## Python Script

The `scripts/daily_update.py` script provides a flexible way to run the update with detailed logging and error handling.

### Setup

1. Make sure the script is executable:
   ```bash
   chmod +x scripts/daily_update.py
   ```

2. Set up a cron job:
   ```bash
   # Edit crontab
   crontab -e
   
   # Add this line to run at 6 AM UTC
   0 6 * * * cd /path/to/inflation_forecaster && ./scripts/daily_update.py --auto-merge
   ```

### Options

```
usage: daily_update.py [-h] [--auto-merge] [--repo-owner REPO_OWNER]
                        [--repo-name REPO_NAME] [--github-token GITHUB_TOKEN]
                        [--no-git]

optional arguments:
  -h, --help            show this help message and exit
  --auto-merge          Automatically merge the PR after creation
  --repo-owner REPO_OWNER
                        GitHub repository owner
  --repo-name REPO_NAME
                        GitHub repository name
  --github-token GITHUB_TOKEN
                        GitHub personal access token
  --no-git              Skip Git operations (just run the model)
```

## Shell Script

The `scripts/run_daily_update.sh` script provides a simpler alternative for basic cron setups.

### Setup

1. Make sure the script is executable:
   ```bash
   chmod +x scripts/run_daily_update.sh
   ```

2. Set up a cron job:
   ```bash
   # Edit crontab
   crontab -e
   
   # Add this line to run at 6 AM UTC
   0 6 * * * /path/to/inflation_forecaster/scripts/run_daily_update.sh >> /path/to/logs/model_update.log 2>&1
   ```

## Docker Container

The Docker setup provides an isolated, portable environment for running the model update.

### Setup

1. Build the Docker image:
   ```bash
   docker build -t inflation-forecaster .
   ```

2. Run once (without Git operations):
   ```bash
   docker run --env-file .env -v "$(pwd)/data:/app/data" \
      -v "$(pwd)/docs/images:/app/docs/images" \
      -v "$(pwd)/src/saved_models:/app/src/saved_models" \
      inflation-forecaster
   ```

3. Schedule with Docker Compose:
   ```bash
   # Start the scheduler
   docker-compose up -d model-scheduler
   
   # Check status
   docker-compose ps
   
   # Stop the scheduler
   docker-compose down
   ```

Ensure your `.env` file contains all required API keys and GitHub credentials.

## Systemd Service (Linux)

On Linux systems with systemd, you can set up a service and timer.

### Setup

1. Edit the service file at `scripts/systemd/inflation-model.service` to set the correct paths, username, and tokens.

2. Install the service and timer:
   ```bash
   sudo cp scripts/systemd/inflation-model.service /etc/systemd/system/
   sudo cp scripts/systemd/inflation-model.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable inflation-model.timer
   sudo systemctl start inflation-model.timer
   ```

3. Check status:
   ```bash
   sudo systemctl status inflation-model.timer
   sudo systemctl list-timers
   ```

## Windows Task Scheduler

For Windows environments, you can use Task Scheduler.

### Setup

1. Open Task Scheduler
2. Create a new task:
   - General: Name it "Inflation Model Update"
   - Triggers: New Trigger > Daily > Start at 6:00 AM > Synchronize across time zones
   - Actions: New Action > Start a program:
     - Program/script: `C:\Path\to\Python\python.exe`
     - Arguments: `C:\Path\to\inflation_forecaster\scripts\daily_update.py --no-git`
     - Start in: `C:\Path\to\inflation_forecaster`
   - Conditions: Start only if network is available
   - Settings: Allow task to be run on demand

## Cloud Options

### AWS Lambda with EventBridge

For a serverless solution, you can use AWS Lambda with EventBridge:

1. Create a Lambda function that runs your model script
2. Set up an EventBridge rule to trigger the function daily at 6 AM UTC
3. Use AWS CodeBuild for Git operations or AWS SDK to update an S3 bucket with the results

### Google Cloud Functions with Cloud Scheduler

Similar to AWS, you can use Google Cloud Functions with Cloud Scheduler:

1. Create a Cloud Function that runs your model script
2. Set up a Cloud Scheduler job to trigger the function daily at 6 AM UTC
3. Use Google Cloud Build for Git operations or Google Cloud Storage for results

## Troubleshooting

### Common Issues

- **API rate limits**: If you hit API rate limits, implement exponential backoff and retry logic
- **Git authentication**: Ensure your GitHub token has the correct permissions
- **Missing predictions**: Check logs for errors in data fetching or model training
- **Disk space**: Ensure there's enough disk space for model files and predictions

### Logs

- GitHub Actions: Available in the Actions tab of your repository
- Python script: Logs are saved to the `logs/` directory
- Docker: View logs with `docker logs <container_id>` or `docker-compose logs model-scheduler`
- Systemd: View logs with `journalctl -u inflation-model.service` 