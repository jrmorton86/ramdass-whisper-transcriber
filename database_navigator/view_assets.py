"""View assets table columns and sample data."""
from database_navigator import get_connection
import json

def show_columns():
    """Display all columns in the assets table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get column information
            cur.execute("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_name = 'assets' 
                ORDER BY ordinal_position
            """)
            
            columns = cur.fetchall()
            
            print("\n" + "="*100)
            print("ASSETS TABLE COLUMNS")
            print("="*100)
            print(f"{'Column Name':<30} {'Data Type':<25} {'Nullable':<12} {'Default'}")
            print("-"*100)
            
            for col in columns:
                nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                default = str(col[3])[:40] if col[3] else ""
                print(f"{col[0]:<30} {col[1]:<25} {nullable:<12} {default}")
            
            print(f"\nTotal columns: {len(columns)}")
            
            # Get row count
            cur.execute("SELECT COUNT(*) FROM assets")
            count = cur.fetchone()[0]
            print(f"Total rows: {count:,}")
            
            # Show sample record
            print("\n" + "="*100)
            print("SAMPLE RECORD (first asset)")
            print("="*100)
            
            cur.execute("SELECT * FROM assets LIMIT 1")
            col_names = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            
            if row:
                for i, col_name in enumerate(col_names):
                    value = row[i]
                    # Format value for display
                    if value is None:
                        display_value = "NULL"
                    elif isinstance(value, str) and len(value) > 100:
                        display_value = value[:100] + "..."
                    else:
                        display_value = str(value)
                    
                    print(f"{col_name:<30}: {display_value}")


if __name__ == "__main__":
    show_columns()
