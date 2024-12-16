import paramiko
from google.cloud import storage
from google.cloud import storage
from google.auth import default

import sys
import subprocess

project = "inner-analyst-398816"
instance_name = "ehren-era5-vm2"
zone = "us-central1-c"

def run_commands_on_vm(commands):
    """
    Runs a series of commands on a Google Cloud VM using gcloud SSH.
    
    Args:
        project (str): Google Cloud project ID.
        instance_name (str): Name of the VM instance.
        zone (str): Zone of the VM instance.
        commands (list): List of commands to execute sequentially on the VM.
    """
    try:
        for command in commands:
            # Construct the gcloud SSH command
            gcloud_command = [
                "gcloud", "compute", "ssh",
                instance_name,
                "--project", project,
                "--zone", zone,
                "--command", command
            ]
            
            # Run the command
            print(f"Running command on VM: {command}")
            result = subprocess.run(gcloud_command, capture_output=True, text=True)
            
            # Print the output
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error: {result.stderr}")
                break
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage
project = "inner-analyst-398816"
instance_name = "ehren-era5-vm2"
zone = "us-central1-c"

def run_process_on_vm_and_transfer(lat1, lon1, lat2, lon2, start_date, end_date):
    # Define commands to run on the VM
    commands = [
        f"./run_wind_cutout.sh {lat1} {lon1} {lat2} {lon2} {start_date} {end_date}"  # Placeholder for file creation (replace with real command)
    ]
    run_commands_on_vm(commands)

# Check if the correct number of arguments are provided
if len(sys.argv) != 7:
    raise ValueError("Please provide exactly 6 arguments: lat1, lon1, lat2, lon2, startdate, enddate")

# Assign command line arguments to variables
lat1 = float(sys.argv[1])
lon1 = float(sys.argv[2])
lat2 = float(sys.argv[3])
lon2 = float(sys.argv[4])


startdate = sys.argv[5]
enddate = sys.argv[6]

run_process_on_vm_and_transfer(lat1, lon1, lat2, lon2, startdate, enddate)