import subprocess
import sys

def run_script(script_name):
    print(f"Running {script_name}...")
    # subprocess to run scripts
    subprocess.Popen([sys.executable, script_name])

def main():
    # visualization files to run
    scripts_to_run = [
        "DatasetVisualizer.py",
        "DisasterDatasetTimelapseAnimator.py",
    ]
    
    for script in scripts_to_run:
        run_script(script)

if __name__ == "__main__":
    main()
