"""Diagnostic tool to test database connectivity and configuration."""
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    try:
        import psycopg2
        print("  ✓ psycopg2")
    except ImportError as e:
        print(f"  ✗ psycopg2: {e}")
        return False
    
    try:
        import boto3
        print("  ✓ boto3")
    except ImportError as e:
        print(f"  ✗ boto3: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("  ✓ python-dotenv")
    except ImportError as e:
        print(f"  ✗ python-dotenv: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from database_navigator.config import settings
        
        print(f"  DB Host: {settings.db_host}")
        print(f"  DB Port: {settings.db_port}")
        print(f"  DB Name: {settings.db_name}")
        print(f"  DB User: {settings.db_user}")
        print(f"  AWS Region: {settings.aws_region}")
        print(f"  Use Secrets Manager: {settings.use_secrets_manager}")
        print(f"  Secret Name: {settings.db_secret_name}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_secrets_manager():
    """Test AWS Secrets Manager access."""
    print("\nTesting AWS Secrets Manager access...")
    try:
        from database_navigator.config import settings
        
        if not settings.use_secrets_manager:
            print("  ⊘ Secrets Manager is disabled (USE_SECRETS_MANAGER=false)")
            return True
        
        print(f"  Attempting to retrieve secret: {settings.db_secret_name}")
        password = settings.db_password
        
        if password:
            print(f"  ✓ Successfully retrieved password (length: {len(password)})")
            return True
        else:
            print("  ✗ Password is empty")
            return False
            
    except Exception as e:
        print(f"  ✗ Error accessing Secrets Manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_network():
    """Test basic network connectivity to RDS host."""
    print("\nTesting network connectivity...")
    import socket
    
    try:
        from database_navigator.config import settings
        
        print(f"  Attempting to resolve {settings.db_host}...")
        ip = socket.gethostbyname(settings.db_host)
        print(f"  ✓ DNS resolved to: {ip}")
        
        print(f"  Attempting to connect to {settings.db_host}:{settings.db_port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((settings.db_host, settings.db_port))
        sock.close()
        
        if result == 0:
            print(f"  ✓ Port {settings.db_port} is reachable")
            return True
        else:
            print(f"  ✗ Port {settings.db_port} is NOT reachable (error code: {result})")
            print("\n  Possible issues:")
            print("    - Security group doesn't allow your IP")
            print("    - Need to be on VPN")
            print("    - RDS instance is not publicly accessible")
            return False
            
    except socket.gaierror as e:
        print(f"  ✗ DNS resolution failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Network test failed: {e}")
        return False


def test_database_connection():
    """Test actual database connection."""
    print("\nTesting database connection...")
    try:
        from database_navigator.db import get_connection
        
        print("  Attempting to connect to database...")
        conn = get_connection()
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"  ✓ Connected successfully!")
            print(f"  PostgreSQL version: {version}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


def main():
    """Run all diagnostic tests."""
    print("=" * 80)
    print("DATABASE CONNECTION DIAGNOSTICS")
    print("=" * 80)
    
    results = {
        "Imports": test_imports(),
        "Configuration": test_config(),
        "AWS Secrets Manager": test_secrets_manager(),
        "Network": test_network(),
        "Database Connection": test_database_connection()
    }
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to extract the schema.")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
