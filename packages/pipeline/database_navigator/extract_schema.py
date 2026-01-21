"""Extract complete database schema from RDS."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from .db import get_connection


def extract_schema() -> Dict[str, Any]:
    """
    Extract complete database schema including:
    - Tables with columns, data types, nullability
    - Primary keys and foreign keys
    - Indexes
    - Constraints
    - Views
    """
    conn = get_connection()
    schema_info = {
        "extracted_at": datetime.utcnow().isoformat(),
        "database": None,
        "tables": {},
        "views": {},
        "sequences": []
    }
    
    try:
        with conn.cursor() as cur:
            # Get database name
            cur.execute("SELECT current_database();")
            schema_info["database"] = cur.fetchone()[0]
            
            print(f"Connected to database: {schema_info['database']}")
            
            # Get all tables in public schema
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            print(f"\nFound {len(tables)} tables:")
            for table in tables:
                print(f"  - {table}")
                schema_info["tables"][table] = extract_table_info(cur, table)
            
            # Get all views
            cur.execute("""
                SELECT table_name, view_definition
                FROM information_schema.views 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            views = cur.fetchall()
            
            print(f"\nFound {len(views)} views:")
            for view_name, view_def in views:
                print(f"  - {view_name}")
                schema_info["views"][view_name] = {
                    "definition": view_def
                }
            
            # Get sequences
            cur.execute("""
                SELECT sequence_name, data_type, start_value, 
                       minimum_value, maximum_value, increment
                FROM information_schema.sequences
                WHERE sequence_schema = 'public'
                ORDER BY sequence_name;
            """)
            sequences = cur.fetchall()
            
            print(f"\nFound {len(sequences)} sequences:")
            for seq in sequences:
                seq_info = {
                    "name": seq[0],
                    "data_type": seq[1],
                    "start_value": seq[2],
                    "minimum_value": seq[3],
                    "maximum_value": seq[4],
                    "increment": seq[5]
                }
                print(f"  - {seq[0]}")
                schema_info["sequences"].append(seq_info)
                
    finally:
        conn.close()
    
    return schema_info


def extract_table_info(cur, table_name: str) -> Dict[str, Any]:
    """Extract detailed information about a specific table."""
    table_info = {
        "columns": [],
        "primary_key": [],
        "foreign_keys": [],
        "indexes": [],
        "constraints": [],
        "row_count": 0
    }
    
    # Get column information
    cur.execute("""
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            is_nullable,
            column_default,
            udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public' 
        AND table_name = %s
        ORDER BY ordinal_position;
    """, (table_name,))
    
    for row in cur.fetchall():
        col_info = {
            "name": row[0],
            "data_type": row[1],
            "max_length": row[2],
            "numeric_precision": row[3],
            "numeric_scale": row[4],
            "nullable": row[5] == 'YES',
            "default": row[6],
            "udt_name": row[7]
        }
        table_info["columns"].append(col_info)
    
    # Get primary key
    cur.execute("""
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = %s::regclass
        AND i.indisprimary
        ORDER BY a.attnum;
    """, (f'public.{table_name}',))
    
    table_info["primary_key"] = [row[0] for row in cur.fetchall()]
    
    # Get foreign keys
    cur.execute("""
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name,
            rc.update_rule,
            rc.delete_rule
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        JOIN information_schema.referential_constraints AS rc
            ON rc.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = 'public'
        AND tc.table_name = %s;
    """, (table_name,))
    
    for row in cur.fetchall():
        fk_info = {
            "column": row[0],
            "references_table": row[1],
            "references_column": row[2],
            "on_update": row[3],
            "on_delete": row[4]
        }
        table_info["foreign_keys"].append(fk_info)
    
    # Get indexes
    cur.execute("""
        SELECT
            i.relname AS index_name,
            a.attname AS column_name,
            ix.indisunique AS is_unique,
            ix.indisprimary AS is_primary
        FROM pg_class t
        JOIN pg_index ix ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
        WHERE t.relkind = 'r'
        AND t.relname = %s
        AND t.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        ORDER BY i.relname, a.attnum;
    """, (table_name,))
    
    indexes_dict = {}
    for row in cur.fetchall():
        idx_name = row[0]
        if idx_name not in indexes_dict:
            indexes_dict[idx_name] = {
                "name": idx_name,
                "columns": [],
                "unique": row[2],
                "primary": row[3]
            }
        indexes_dict[idx_name]["columns"].append(row[1])
    
    table_info["indexes"] = list(indexes_dict.values())
    
    # Get check constraints
    cur.execute("""
        SELECT
            con.conname AS constraint_name,
            pg_get_constraintdef(con.oid) AS constraint_definition
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
        WHERE nsp.nspname = 'public'
        AND rel.relname = %s
        AND con.contype = 'c';
    """, (table_name,))
    
    for row in cur.fetchall():
        constraint_info = {
            "name": row[0],
            "definition": row[1]
        }
        table_info["constraints"].append(constraint_info)
    
    # Get approximate row count
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        table_info["row_count"] = cur.fetchone()[0]
    except Exception:
        table_info["row_count"] = "Error counting rows"
    
    return table_info


def save_schema_to_file(schema_info: Dict[str, Any], output_dir: str = "schema_output"):
    """Save schema information to JSON and human-readable files."""
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full schema as JSON
    json_file = output_path / f"schema_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(schema_info, f, indent=2, default=str)
    
    print(f"\n✓ Saved full schema to: {json_file}")
    
    # Save human-readable summary
    summary_file = output_path / f"schema_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        write_schema_summary(f, schema_info)
    
    print(f"✓ Saved readable summary to: {summary_file}")
    
    return json_file, summary_file


def write_schema_summary(f, schema_info: Dict[str, Any]):
    """Write a human-readable schema summary."""
    f.write(f"Database Schema Summary\n")
    f.write(f"=" * 80 + "\n\n")
    f.write(f"Database: {schema_info['database']}\n")
    f.write(f"Extracted: {schema_info['extracted_at']}\n")
    f.write(f"Tables: {len(schema_info['tables'])}\n")
    f.write(f"Views: {len(schema_info['views'])}\n")
    f.write(f"Sequences: {len(schema_info['sequences'])}\n\n")
    
    # Write table details
    for table_name, table_info in schema_info["tables"].items():
        f.write(f"\n{'=' * 80}\n")
        f.write(f"TABLE: {table_name}\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Row Count: {table_info['row_count']}\n\n")
        
        # Columns
        f.write("COLUMNS:\n")
        f.write(f"  {'Name':<30} {'Type':<20} {'Nullable':<10} {'Default'}\n")
        f.write(f"  {'-'*30} {'-'*20} {'-'*10} {'-'*30}\n")
        for col in table_info["columns"]:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            default = col["default"] or ""
            f.write(f"  {col['name']:<30} {col['data_type']:<20} {nullable:<10} {default}\n")
        
        # Primary Key
        if table_info["primary_key"]:
            f.write(f"\nPRIMARY KEY: {', '.join(table_info['primary_key'])}\n")
        
        # Foreign Keys
        if table_info["foreign_keys"]:
            f.write("\nFOREIGN KEYS:\n")
            for fk in table_info["foreign_keys"]:
                f.write(f"  {fk['column']} -> {fk['references_table']}.{fk['references_column']}\n")
                f.write(f"    ON UPDATE: {fk['on_update']}, ON DELETE: {fk['on_delete']}\n")
        
        # Indexes
        if table_info["indexes"]:
            f.write("\nINDEXES:\n")
            for idx in table_info["indexes"]:
                idx_type = []
                if idx["primary"]:
                    idx_type.append("PRIMARY")
                if idx["unique"]:
                    idx_type.append("UNIQUE")
                type_str = f" ({', '.join(idx_type)})" if idx_type else ""
                f.write(f"  {idx['name']}{type_str}: {', '.join(idx['columns'])}\n")
        
        # Constraints
        if table_info["constraints"]:
            f.write("\nCONSTRAINTS:\n")
            for constraint in table_info["constraints"]:
                f.write(f"  {constraint['name']}: {constraint['definition']}\n")
    
    # Write view details
    if schema_info["views"]:
        f.write(f"\n\n{'=' * 80}\n")
        f.write("VIEWS\n")
        f.write(f"{'=' * 80}\n")
        for view_name in schema_info["views"].keys():
            f.write(f"\n- {view_name}\n")


if __name__ == "__main__":
    print("Extracting database schema...")
    print("=" * 80)
    
    try:
        schema = extract_schema()
        json_file, summary_file = save_schema_to_file(schema)
        
        print("\n" + "=" * 80)
        print("Schema extraction complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
