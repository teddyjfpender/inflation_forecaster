#!/usr/bin/env python3
"""
Daily Inflation Model Update Script

This script:
1. Runs the inflation forecasting models
2. Commits changes to a new branch
3. Creates a pull request
4. Optionally auto-merges the PR

Can be run manually or scheduled with:
- cron
- systemd timer
- Windows Task Scheduler
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
import traceback
import json
import requests
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"model_update_{datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("daily_update")

def run_command(command, cwd=None):
    """Run a shell command and return the output"""
    logger.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            cwd=cwd
        )
        if result.stdout:
            logger.info(result.stdout)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        raise

def create_github_pr(repo_owner, repo_name, branch, title, body, github_token):
    """Create a PR using GitHub API"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body,
        "head": branch,
        "base": "main"
    }
    
    logger.info(f"Creating PR from {branch} to main")
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code in (200, 201):
        pr_data = response.json()
        logger.info(f"PR created successfully: {pr_data['html_url']}")
        return pr_data
    else:
        logger.error(f"Failed to create PR: {response.status_code}")
        logger.error(response.text)
        return None

def auto_merge_pr(repo_owner, repo_name, pr_number, github_token):
    """Auto-merge a PR using GitHub API"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}/merge"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "merge_method": "squash"
    }
    
    logger.info(f"Auto-merging PR #{pr_number}")
    response = requests.put(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        logger.info("PR merged successfully")
        return True
    else:
        logger.error(f"Failed to merge PR: {response.status_code}")
        logger.error(response.text)
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run daily inflation model updates")
    parser.add_argument(
        "--auto-merge", 
        action="store_true", 
        help="Automatically merge the PR after creation"
    )
    parser.add_argument(
        "--repo-owner",
        type=str,
        default=os.environ.get("GITHUB_REPO_OWNER", ""),
        help="GitHub repository owner"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=os.environ.get("GITHUB_REPO_NAME", "inflation_forecaster"),
        help="GitHub repository name"
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub personal access token"
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip Git operations (just run the model)"
    )
    return parser.parse_args()

def main():
    """Main function to run the daily update"""
    try:
        args = parse_args()
        
        # Determine project root directory
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        os.chdir(project_root)
        
        logger.info(f"Starting daily model update at {datetime.now()}")
        
        # Run the model
        logger.info("Running inflation forecasting model...")
        run_command("python run_inflation_model.py")
        logger.info("Model run completed successfully")
        
        if args.no_git:
            logger.info("Skipping Git operations as requested")
            return 0
        
        # Check for changes
        changes = run_command("git status --porcelain")
        if not changes:
            logger.info("No changes detected. Exiting.")
            return 0
        
        # Create a branch with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        branch_name = f"model-update-{today}"
        run_command(f"git checkout -b {branch_name}")
        
        # Commit changes
        run_command("git add data/ src/saved_models/ docs/images/")
        commit_message = f"Daily model update {today}\n\nAutomated model update with latest economic data"
        run_command(f'git commit -m "{commit_message}"')
        
        # Push changes
        run_command(f"git push -u origin {branch_name}")
        
        # Create a PR if GitHub token is provided
        if args.github_token and args.repo_owner:
            pr_title = f"Daily model update {today}"
            pr_body = f"""# Automated Daily Model Update

This PR contains the latest inflation forecasts generated on {today}.

## Changes include:
- Updated model predictions in data/predictions/
- Saved model files in src/saved_models/
- Updated visualization charts in docs/images/

Automatically generated by the daily model update script.
"""
            pr_data = create_github_pr(
                args.repo_owner,
                args.repo_name,
                branch_name,
                pr_title,
                pr_body,
                args.github_token
            )
            
            # Auto-merge if requested
            if pr_data and args.auto_merge:
                auto_merge_pr(
                    args.repo_owner,
                    args.repo_name,
                    pr_data["number"],
                    args.github_token
                )
        else:
            logger.info("Skipping PR creation - GitHub token or repo owner not provided")
            logger.info(f"Branch '{branch_name}' has been pushed. Create PR manually.")
        
        logger.info("Daily update completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error in daily update: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 