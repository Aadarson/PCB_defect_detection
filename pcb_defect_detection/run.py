import os
import sys
import time
import webbrowser
import subprocess

def main():
    print("=============================================")
    print("   PCB Defect Detection System Launcher")
    print("=============================================")
    
    # 1. Determine paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(current_dir, "frontend", "index.html")
    venv_python = os.path.join(current_dir, "venv", "Scripts", "python.exe")
    
    if not os.path.exists(venv_python):
        print("[!] Error: Virtual environment not found. Please setup the venv first.")
        sys.exit(1)

    # 2. Launch FastAPI Backend in a separate process
    print("[*] Starting FastAPI Backend Server...")
    backend_cmd = [venv_python, "-m", "uvicorn", "backend.app:app", "--host", "localhost", "--port", "8000"]
    backend_process = subprocess.Popen(backend_cmd)

    # 3. Wait briefly for the server to spin up
    print("[*] Waiting for backend AI compiler to fuse CPU layers (approx. 3-4 seconds)...")
    time.sleep(4)

    # 4. Open the Frontend UI in the default browser
    if os.path.exists(frontend_path):
        print(f"[*] Opening Frontend Dashboard: {frontend_path}")
        webbrowser.open(f"file:///{frontend_path.replace(chr(92), '/')}") # Replace Windows backslashes
    else:
        print("[!] Error: Frontend index.html not found.")

    print("\n[+] Successfully launched! The system is running.")
    print("[+] Press Ctrl+C in this terminal window to shut down the server safely.\n")

    # 5. Keep the script alive so the user can see the backend logs
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down PCB Defect Detection System...")
        backend_process.terminate()
        backend_process.wait()
        print("[*] Shutdown complete.")

if __name__ == "__main__":
    main()
