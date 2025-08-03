import subprocess
import time
import sys
import os
import logging

# Configure basic logging for the launcher script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_app_in_background(script_name, port, app_module=None):
    """
    Runs a FastAPI application in a separate subprocess.
    If app_module is provided, it assumes a Uvicorn command.
    Otherwise, it assumes the script itself starts Uvicorn.
    """
    cmd = []
    if app_module:
        cmd = [sys.executable, "-m", "uvicorn", f"{app_module}:app", "--host", "0.0.0.0", "--port", str(port)]
    else:
        # Assuming script_name itself contains uvicorn.run() call
        cmd = [sys.executable, script_name]

    logger.info(f"Launching {script_name} on port {port} with command: {' '.join(cmd)}")
    # Use preexec_fn for Unix-like systems to detach, or creationflags for Windows
    if sys.platform == "win32":
        # Detach process on Windows
        process = subprocess.Popen(cmd, creationflags=subprocess.DETACHED_PROCESS, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        # Detach process on Unix-like systems (Linux, macOS)
        process = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Log initial output for debugging
    # Note: For fully detached processes, capturing output immediately might not work
    # as the child process might close its stdout/stderr after detaching.
    # We mainly rely on logs from individual applications.
    try:
        # Give it a moment to start and produce some output
        time.sleep(2)
        # Check if process is still alive
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            logger.error(f"Error launching {script_name}. Exit code: {process.returncode}")
            logger.error(f"Stdout:\n{stdout.decode()}")
            logger.error(f"Stderr:\n{stderr.decode()}")
        else:
            logger.info(f"{script_name} launched successfully with PID: {process.pid}")
            logger.info(f"You can monitor logs from {script_name} directly in its own log file or console if configured.")
    except subprocess.TimeoutExpired:
        logger.info(f"{script_name} appears to be running in background (PID: {process.pid}).")
    except Exception as e:
        logger.error(f"An error occurred while checking {script_name} startup: {e}")

    return process

if __name__ == "__main__":
    processes = []
    logger.info("Starting NovaPress applications...")

    try:
        # Launch Scraper API (which internally calls uvicorn.run)
        # It's set to port 8000 in Scraper.py, so we reflect that.
        scraper_process = run_app_in_background("Scraper.py", 8000)
        processes.append(scraper_process)
        time.sleep(5) # Give scraper some time to initialize AI models and MongoDB

        # Launch LoginModal API using uvicorn command
        login_process = run_app_in_background("LoginModal", 8001, app_module="LoginModal")
        processes.append(login_process)
        time.sleep(2) # Give login app some time to start

        logger.info("\nAll applications launched. Check the respective ports (8000 and 8001).")
        logger.info("This launcher script will now exit. The applications will continue to run in the background.")
        logger.info("To stop them, you will need to manually kill their processes (e.g., using 'kill PID' on Linux/macOS or Task Manager on Windows).")
        logger.info("The PID for Scraper is: %s", scraper_process.pid)
        logger.info("The PID for LoginModal is: %s", login_process.pid)


    except Exception as e:
        logger.error(f"An error occurred during launching: {e}")
        for p in processes:
            if p.poll() is None: # If still running
                p.terminate() # Try to terminate
        logger.info("Attempted to terminate launched processes due to error.")
        sys.exit(1)

    sys.exit(0) # Exit the launcher script