"""Query assets that need search vectors generated."""
from database_navigator import get_connection
from typing import List, Dict, Any
import json


def get_assets_without_search_vector(limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Get all assets where search_vector is NULL (needs to be generated).
    
    Args:
        limit: Maximum number of records to return (None = all)
        offset: Number of records to skip
        
    Returns:
        List of asset dictionaries
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    id,
                    bynder_id,
                    name,
                    description,
                    short_description,
                    asset_type,
                    asset_sub_type,
                    decade,
                    publication_date,
                    created_at,
                    updated_at
                FROM assets
                WHERE search_vector IS NULL
                ORDER BY created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cur.execute(query)
            
            columns = [desc[0] for desc in cur.description]
            results = []
            
            for row in cur.fetchall():
                asset = {}
                for i, col_name in enumerate(columns):
                    value = row[i]
                    # Convert to JSON-serializable types
                    if hasattr(value, 'isoformat'):  # datetime/date
                        value = value.isoformat()
                    asset[col_name] = value
                results.append(asset)
            
            return results


def count_assets_without_search_vector() -> int:
    """Count how many assets need search vectors."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM assets WHERE search_vector IS NULL")
            return cur.fetchone()[0]


def get_all_asset_columns_without_search_vector(limit: int = None) -> List[Dict[str, Any]]:
    """
    Get ALL columns for assets without search vectors.
    
    Args:
        limit: Maximum number of records to return (None = all)
        
    Returns:
        List of complete asset dictionaries with all 45 columns
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT *
                FROM assets
                WHERE search_vector IS NULL
                ORDER BY created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cur.execute(query)
            
            columns = [desc[0] for desc in cur.description]
            results = []
            
            for row in cur.fetchall():
                asset = {}
                for i, col_name in enumerate(columns):
                    value = row[i]
                    # Convert to JSON-serializable types
                    if hasattr(value, 'isoformat'):  # datetime/date
                        value = value.isoformat()
                    elif value is None:
                        value = None
                    else:
                        value = str(value) if not isinstance(value, (str, int, float, bool)) else value
                    asset[col_name] = value
                results.append(asset)
            
            return results


def export_assets_without_search_vector(output_file: str = "assets_needing_search_vectors.json"):
    """
    Export all assets without search vectors to a JSON file.
    
    Args:
        output_file: Filename for the JSON export
    """
    from pathlib import Path
    
    print("Counting assets without search vectors...")
    count = count_assets_without_search_vector()
    print(f"Found {count:,} assets without search vectors")
    
    if count == 0:
        print("All assets have search vectors!")
        return
    
    print(f"\nFetching {count:,} records...")
    assets = get_all_asset_columns_without_search_vector()
    
    output_path = Path(__file__).parent / output_file
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "count": len(assets),
            "exported_at": __import__('datetime').datetime.utcnow().isoformat(),
            "assets": assets
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported {len(assets):,} assets to: {output_path}")
    
    # Print summary
    print("\nSummary by asset type:")
    type_counts = {}
    for asset in assets:
        asset_type = asset.get('asset_type') or 'Unknown'
        type_counts[asset_type] = type_counts.get(asset_type, 0) + 1
    
    for asset_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset_type}: {count:,}")


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("ASSETS WITHOUT SEARCH VECTORS")
    print("="*80)
    
    # Check count
    count = count_assets_without_search_vector()
    total_count = 0
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM assets")
            total_count = cur.fetchone()[0]
    
    print(f"\nAssets without search vectors: {count:,} / {total_count:,}")
    print(f"Percentage: {(count/total_count*100):.1f}%")
    
    if count > 0:
        print("\nOptions:")
        print("  1. Show sample (10 records)")
        print("  2. Export all to JSON")
        print("  3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == "1":
            print("\nFetching sample (10 records)...\n")
            assets = get_assets_without_search_vector(limit=10)
            for i, asset in enumerate(assets, 1):
                print(f"{i}. {asset['name']}")
                print(f"   Type: {asset['asset_type']} | Decade: {asset['decade']}")
                print(f"   ID: {asset['id']}")
                print()
        
        elif choice == "2":
            export_assets_without_search_vector()
        
        else:
            print("Exiting.")
    else:
        print("\n✓ All assets have search vectors!")
