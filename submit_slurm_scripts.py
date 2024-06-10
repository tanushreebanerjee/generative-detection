import os
import subprocess
import argparse
import time

def submit_slurm_scripts(directory, delay):
    # Get a list of all files in the specified directory
    files = os.listdir(directory)
    
    # Filter out the files that end with .slurm
    slurm_files = [f for f in files if f.endswith('.slurm')]
    
    # Submit each slurm script
    for slurm_file in slurm_files:
        file_path = os.path.join(directory, slurm_file)
        try:
            result = subprocess.run(['sbatch', file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Submitted {file_path}: {result.stdout.decode().strip()}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit {file_path}: {e.stderr.decode().strip()}")
        time.sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Submit all SLURM scripts in a specified directory with a delay between submissions.')
    parser.add_argument('directory', nargs='?', default='scripts/slurm/train/jun10', help='Directory containing SLURM scripts')
    parser.add_argument('--delay', type=int, default=10, help='Delay in seconds between submissions (default: 10 seconds)')
    
    args = parser.parse_args()
    submit_slurm_scripts(args.directory, args.delay)
