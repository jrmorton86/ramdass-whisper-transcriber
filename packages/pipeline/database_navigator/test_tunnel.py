"""Test database connection through SSH tunnel."""
import subprocess
import time
import sys
import os
from pathlib import Path

# SSH tunnel configuration
BASTION_HOST = "54.175.205.16"
BASTION_USER = "ec2-user"
BASTION_KEY = os.path.expanduser("~/.ssh/ramdass-bastion-temp.pem")
RDS_ENDPOINT = "dam-ramdass-io-rds-instance-1.c7ecmfdohgux.us-east-1.rds.amazonaws.com"
RDS_PORT = 5432
LOCAL_PORT = 5433


def test_ssh_connection():
    """Test SSH connection to bastion."""
    print("Testing SSH connection to bastion...")
    cmd = [
        "ssh",
        "-i", BASTION_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{BASTION_USER}@{BASTION_HOST}",
        "echo 'SSH connection successful!'"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            print("✓ SSH connection successful!")
            return True
        else:
            print(f"✗ SSH connection failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ SSH connection timed out")
        return False
    except Exception as e:
        print(f"✗ SSH connection error: {e}")
        return False


def start_tunnel_background():
    """Start SSH tunnel in background."""
    print(f"\nStarting SSH tunnel in background...")
    print(f"  localhost:{LOCAL_PORT} -> {RDS_ENDPOINT}:{RDS_PORT}")
    
    cmd = [
        "ssh",
        "-i", BASTION_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-N",
        "-L", f"{LOCAL_PORT}:{RDS_ENDPOINT}:{RDS_PORT}",
        f"{BASTION_USER}@{BASTION_HOST}"
    ]
    
    # Start tunnel in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for tunnel to establish
    print("  Waiting for tunnel to establish...")
    time.sleep(3)
    
    # Check if process is still running
    if process.poll() is None:
        print("✓ Tunnel established!")
        return process
    else:
        _, stderr = process.communicate()
        print(f"✗ Tunnel failed to start: {stderr.decode()}")
        return None


def test_db_connection():
    """Test database connection through tunnel."""
    print("\nTesting database connection through tunnel...")
    
    import psycopg2
    from database_navigator.config import settings
    
    try:
        # Connect directly through localhost tunnel
        conn = psycopg2.connect(
            host='localhost',
            port=LOCAL_PORT,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            connect_timeout=10
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"✓ Database connection successful!")
            print(f"  PostgreSQL version: {version[:80]}...")
        conn.close()
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("SSH TUNNEL + DATABASE CONNECTION TEST")
    print("=" * 80)
    
    if not os.path.exists(BASTION_KEY):
        print(f"✗ SSH key not found: {BASTION_KEY}")
        return False
    
    # Test SSH connection first
    if not test_ssh_connection():
        print("\n⚠️  Cannot connect to bastion host via SSH")
        print("   Please wait a minute for the instance to fully start, then try again.")
        return False
    
    # Start tunnel
    tunnel_process = start_tunnel_background()
    if not tunnel_process:
        return False
    
    try:
        # Test database connection
        db_success = test_db_connection()
        
        if db_success:
            print("\n" + "=" * 80)
            print("SUCCESS! Everything is working!")
            print("=" * 80)
            print("\nTo use this connection in your scripts:")
            print(f"  1. Keep this tunnel running in background")
            print(f"  2. Set DATABASE_HOST=localhost in your .env")
            print(f"  3. Set DATABASE_PORT={LOCAL_PORT} in your .env")
            print("\nOr run database_navigator/ssh_tunnel.py to start tunnel manually")
        
        return db_success
        
    finally:
        # Clean up tunnel
        print("\nClosing tunnel...")
        tunnel_process.terminate()
        tunnel_process.wait(timeout=5)
        print("✓ Tunnel closed")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
