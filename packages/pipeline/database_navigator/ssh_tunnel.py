"""SSH Tunnel helper for connecting to RDS through a bastion host."""
import subprocess
import sys
import os
from pathlib import Path

# Configuration
BASTION_HOST = "54.175.205.16"  # Elastic IP of bastion host
BASTION_USER = "ec2-user"  # Amazon Linux 2023
BASTION_KEY = os.path.expanduser("~/.ssh/ramdass-bastion-temp.pem")
RDS_ENDPOINT = "dam-ramdass-io-rds-instance-1.c7ecmfdohgux.us-east-1.rds.amazonaws.com"
RDS_PORT = 5432
LOCAL_PORT = 5433  # Local port to forward to


def create_tunnel():
    """Create SSH tunnel to RDS through bastion."""
    print(f"Creating SSH tunnel to RDS...")
    print(f"  Bastion: {BASTION_USER}@{BASTION_HOST}")
    print(f"  RDS: {RDS_ENDPOINT}:{RDS_PORT}")
    print(f"  Local port: {LOCAL_PORT}")
    print()
    print(f"Once connected, use: localhost:{LOCAL_PORT} as your DATABASE_HOST")
    print()
    
    cmd = [
        "ssh",
        "-i", BASTION_KEY,
        "-N",  # Don't execute remote command
        "-L", f"{LOCAL_PORT}:{RDS_ENDPOINT}:{RDS_PORT}",
        f"{BASTION_USER}@{BASTION_HOST}"
    ]
    
    try:
        print("Press Ctrl+C to close the tunnel")
        print("=" * 60)
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTunnel closed")
    except subprocess.CalledProcessError as e:
        print(f"Error creating tunnel: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if not os.path.exists(BASTION_KEY):
        print(f"ERROR: SSH key not found at: {BASTION_KEY}")
        print("Please ensure the key file exists.")
        sys.exit(1)
    
    create_tunnel()
