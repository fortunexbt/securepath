#!/usr/bin/env python3
"""
SecurePath Bot Setup Script
Automated setup for the refactored SecurePath Discord bot
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path


class Colors:
    """Terminal colors for pretty output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")


def check_python_version():
    """Check if Python version is compatible"""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False


def check_git_branch():
    """Check if we're on the correct branch"""
    print_info("Checking Git branch...")
    try:
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True)
        branch = result.stdout.strip()
        
        if branch == 'refactor/modular-architecture':
            print_success(f"On correct branch: {branch}")
            return True
        else:
            print_warning(f"Currently on branch: {branch}")
            print_info("Expected branch: refactor/modular-architecture")
            response = input("Switch to refactor branch? (y/n): ")
            if response.lower() == 'y':
                subprocess.run(['git', 'checkout', 'refactor/modular-architecture'])
                print_success("Switched to refactor/modular-architecture")
                return True
            return False
    except Exception as e:
        print_error(f"Git error: {e}")
        return False


def create_virtual_environment():
    """Create Python virtual environment"""
    print_info("Setting up virtual environment...")
    
    venv_path = Path('venv')
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        response = input("Recreate virtual environment? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree('venv')
        else:
            return True
    
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print_success("Virtual environment created")
        
        # Get activation command based on OS
        if sys.platform == 'win32':
            activate_cmd = 'venv\\Scripts\\activate'
        else:
            activate_cmd = 'source venv/bin/activate'
            
        print_info(f"To activate: {activate_cmd}")
        return True
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def install_dependencies():
    """Install required Python packages"""
    print_info("Installing dependencies...")
    
    # Determine pip command
    if sys.platform == 'win32':
        pip_cmd = 'venv\\Scripts\\pip'
    else:
        pip_cmd = 'venv/bin/pip'
    
    # Check if we're in venv
    if not Path(pip_cmd).exists():
        pip_cmd = 'pip3'
        print_warning("Not in virtual environment, using system pip")
    
    try:
        # Upgrade pip first
        subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True)
        
        # Install requirements
        subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'], check=True)
        print_success("All dependencies installed")
        return True
    except Exception as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def create_env_file():
    """Create .env file from template"""
    print_info("Setting up environment configuration...")
    
    env_path = Path('.env')
    env_example_path = Path('.env.example')
    
    # Create .env.example if it doesn't exist
    if not env_example_path.exists():
        print_info("Creating .env.example template...")
        env_template = """# Discord Configuration
DISCORD_TOKEN=your_discord_bot_token_here
BOT_PREFIX=!
OWNER_ID=your_discord_user_id_here

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Database (PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/securepath

# Optional: Logging
LOG_LEVEL=INFO
LOG_CHANNEL_ID=your_log_channel_id_here

# Optional: Specific Channels
SUMMARY_CHANNEL_ID=
CHARTIST_CHANNEL_ID=
NEWS_CHANNEL_ID=
NEWS_BOT_USER_ID=

# Optional: API Settings
USE_PERPLEXITY_API=True
PERPLEXITY_TIMEOUT=30
API_RATE_LIMIT_MAX=100
API_RATE_LIMIT_INTERVAL=60
"""
        env_example_path.write_text(env_template)
        print_success("Created .env.example")
    
    if env_path.exists():
        print_warning(".env file already exists")
        return True
    else:
        # Copy from example
        shutil.copy('.env.example', '.env')
        print_success("Created .env from template")
        print_warning("Please edit .env and add your API keys and tokens")
        return True


def validate_configuration():
    """Validate the configuration"""
    print_info("Validating configuration...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        
        # Check critical settings
        issues = []
        
        if not settings.discord_token or settings.discord_token == 'your_discord_bot_token_here':
            issues.append("DISCORD_TOKEN not configured")
            
        if not settings.perplexity_api_key or settings.perplexity_api_key == 'your_perplexity_api_key_here':
            issues.append("PERPLEXITY_API_KEY not configured")
            
        if settings.owner_id == 0:
            issues.append("OWNER_ID not configured")
            
        if issues:
            print_error("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            print_warning("Please edit .env file and configure required values")
            return False
        else:
            print_success("Configuration validated")
            return True
            
    except Exception as e:
        print_error(f"Failed to validate configuration: {e}")
        return False


def test_imports():
    """Test that all modules can be imported"""
    print_info("Testing module imports...")
    
    modules_to_test = [
        'src.config.settings',
        'src.bot.client',
        'src.ai.ai_manager',
        'src.database.connection',
        'src.services.rate_limiter',
        'src.utils.validators',
    ]
    
    failed = []
    for module in modules_to_test:
        try:
            __import__(module)
            print_success(f"Imported {module}")
        except Exception as e:
            print_error(f"Failed to import {module}: {e}")
            failed.append(module)
    
    if failed:
        print_error(f"Failed to import {len(failed)} modules")
        return False
    else:
        print_success("All modules imported successfully")
        return True


def setup_database():
    """Setup database tables"""
    print_info("Setting up database...")
    
    try:
        from src.database import db_manager
        import asyncio
        
        async def init_db():
            connected = await db_manager.connect()
            if connected:
                print_success("Database connected and tables initialized")
                await db_manager.disconnect()
                return True
            else:
                print_error("Failed to connect to database")
                print_info("Make sure DATABASE_URL is configured in .env")
                return False
        
        return asyncio.run(init_db())
        
    except Exception as e:
        print_error(f"Database setup failed: {e}")
        print_info("Database is optional for basic functionality")
        return True  # Don't fail setup if database is not available


def create_run_scripts():
    """Create convenient run scripts"""
    print_info("Creating run scripts...")
    
    # Create run.sh for Unix
    run_sh = """#!/bin/bash
# Run the refactored SecurePath bot

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the bot
echo "Starting SecurePath Bot (Refactored)..."
python main_new.py
"""
    
    # Create run.bat for Windows
    run_bat = """@echo off
REM Run the refactored SecurePath bot

REM Activate virtual environment if it exists
if exist venv\\Scripts\\activate (
    call venv\\Scripts\\activate
)

REM Run the bot
echo Starting SecurePath Bot (Refactored)...
python main_new.py
"""
    
    # Write scripts
    Path('run.sh').write_text(run_sh)
    Path('run.bat').write_text(run_bat)
    
    # Make run.sh executable on Unix
    if sys.platform != 'win32':
        os.chmod('run.sh', 0o755)
    
    print_success("Created run scripts (run.sh / run.bat)")
    return True


def main():
    """Main setup process"""
    print_header("SecurePath Bot Setup")
    print_info("Setting up the refactored SecurePath Discord bot\n")
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Git Branch Check", check_git_branch),
        ("Virtual Environment", create_virtual_environment),
        ("Install Dependencies", install_dependencies),
        ("Environment Configuration", create_env_file),
        ("Configuration Validation", validate_configuration),
        ("Module Import Test", test_imports),
        ("Database Setup", setup_database),
        ("Create Run Scripts", create_run_scripts),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n{Colors.BOLD}Step: {step_name}{Colors.ENDC}")
        print("-" * 40)
        
        try:
            success = step_func()
            if not success:
                failed_steps.append(step_name)
                response = input("\nContinue with setup? (y/n): ")
                if response.lower() != 'y':
                    break
        except Exception as e:
            print_error(f"Unexpected error in {step_name}: {e}")
            failed_steps.append(step_name)
    
    # Summary
    print_header("Setup Summary")
    
    if not failed_steps:
        print_success("All setup steps completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys and tokens")
        print("2. Run the bot with: ./run.sh (Unix) or run.bat (Windows)")
        print("3. Or directly with: python main_new.py")
    else:
        print_warning(f"Setup completed with {len(failed_steps)} issues:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease resolve these issues before running the bot")
    
    print(f"\n{Colors.BOLD}Happy coding!{Colors.ENDC} 🚀")


if __name__ == "__main__":
    main()