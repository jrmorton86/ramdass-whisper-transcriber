"""Get assets that have NO embeddings in the asset_embeddings table."""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_navigator.db import get_connection
from typing import List, Dict, Any, Optional
import json


def get_assets_without_any_embeddings(
    limit: Optional[int] = None, 
    offset: int = 0,
    asset_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get assets that have NO embeddings at all (cross-reference exclusion).
    
    This queries assets where the asset.id does NOT exist in asset_embeddings.resource_id
    
    Args:
        limit: Maximum number of records to return (None = all)
        offset: Number of records to skip
        asset_type: Filter by asset type (Audio, Video, etc.)
        
    Returns:
        List of asset dictionaries with all 45 columns
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT a.*
                FROM assets a
                WHERE a.id NOT IN (
                    SELECT DISTINCT resource_id 
                    FROM asset_embeddings
                )
            """
            
            params = []
            if asset_type:
                query += " AND a.asset_type = %s"
                params.append(asset_type)
            
            query += " ORDER BY a.created_at DESC"
            
            if limit:
                query += f" LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            
            cur.execute(query, params)
            
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
                        # Handle special types
                        value = str(value) if not isinstance(value, (str, int, float, bool)) else value
                    asset[col_name] = value
                results.append(asset)
            
            return results


def count_assets_without_any_embeddings(asset_type: Optional[str] = None) -> int:
    """
    Count how many assets have NO embeddings at all.
    
    Args:
        asset_type: Filter by asset type
        
    Returns:
        Count of assets without embeddings
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT COUNT(*)
                FROM assets a
                WHERE a.id NOT IN (
                    SELECT DISTINCT resource_id 
                    FROM asset_embeddings
                )
            """
            
            params = []
            if asset_type:
                query += " AND a.asset_type = %s"
                params.append(asset_type)
            
            cur.execute(query, params)
            return cur.fetchone()[0]


def get_embedding_coverage_by_type() -> Dict[str, Dict[str, int]]:
    """
    Get embedding coverage breakdown by asset type.
    
    Returns:
        Dictionary with counts for each asset type
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COALESCE(a.asset_type, 'Unknown') as asset_type,
                    COUNT(a.id) as total,
                    COUNT(ae.id) as with_embeddings,
                    COUNT(a.id) - COUNT(ae.id) as without_embeddings
                FROM assets a
                LEFT JOIN (
                    SELECT DISTINCT resource_id, id
                    FROM asset_embeddings
                    LIMIT 1
                ) ae ON a.id = ae.resource_id
                GROUP BY a.asset_type
                ORDER BY total DESC
            """)
            
            results = {}
            for row in cur.fetchall():
                asset_type = row[0]
                results[asset_type] = {
                    'total': row[1],
                    'with_embeddings': row[2],
                    'without_embeddings': row[3],
                    'coverage_percent': (row[2] / row[1] * 100) if row[1] > 0 else 0
                }
            
            return results


def export_assets_without_embeddings(
    output_file: str = "assets_without_embeddings.json",
    asset_type: Optional[str] = None
):
    """
    Export all assets without ANY embeddings to a JSON file.
    Saves to the parent 'transcriber' folder for use by batch_process_from_json.py
    
    Args:
        output_file: Filename for the JSON export
        asset_type: Optional filter by asset type
    """
    from pathlib import Path
    import datetime
    
    print("Counting assets without embeddings...")
    count = count_assets_without_any_embeddings(asset_type)
    
    filter_msg = f" of type '{asset_type}'" if asset_type else ""
    print(f"Found {count:,} assets{filter_msg} without ANY embeddings")
    
    if count == 0:
        print(f"\n✓ All assets{filter_msg} have embeddings!")
        return
    
    print(f"\nFetching {count:,} records...")
    assets = get_assets_without_any_embeddings(asset_type=asset_type)
    
    # Save to parent 'transcriber' folder (not database_navigator folder)
    output_path = Path(__file__).parent.parent / output_file
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "count": len(assets),
            "exported_at": datetime.datetime.utcnow().isoformat(),
            "filter": {"asset_type": asset_type} if asset_type else None,
            "assets": assets
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Exported {len(assets):,} assets to: {output_path}")
    
    # Print summary
    print("\nBreakdown by asset type:")
    type_counts = {}
    for asset in assets:
        atype = asset.get('asset_type') or 'Unknown'
        type_counts[atype] = type_counts.get(atype, 0) + 1
    
    for atype, cnt in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {atype}: {cnt:,}")


if __name__ == "__main__":
    print("="*80)
    print("ASSETS WITHOUT EMBEDDINGS (Cross-Reference Exclusion)")
    print("="*80)
    
    # Get total counts
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM assets")
            total_assets = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(DISTINCT resource_id) FROM asset_embeddings")
            assets_with_embeddings = cur.fetchone()[0]
    
    assets_without = count_assets_without_any_embeddings()
    
    print(f"\nTotal Assets: {total_assets:,}")
    print(f"Assets WITH embeddings: {assets_with_embeddings:,}")
    print(f"Assets WITHOUT embeddings: {assets_without:,}")
    print(f"Coverage: {(assets_with_embeddings/total_assets*100):.1f}%")
    
    # Show coverage by type
    print("\n" + "="*80)
    print("EMBEDDING COVERAGE BY ASSET TYPE")
    print("="*80)
    print(f"{'Asset Type':<20} {'Total':<10} {'With Emb':<12} {'Without':<12} {'Coverage'}")
    print("-"*80)
    
    coverage = get_embedding_coverage_by_type()
    for asset_type, stats in sorted(coverage.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"{asset_type:<20} {stats['total']:<10,} {stats['with_embeddings']:<12,} "
              f"{stats['without_embeddings']:<12,} {stats['coverage_percent']:>6.1f}%")
    
    # Interactive menu
    print("\n" + "="*80)
    print("OPTIONS")
    print("="*80)
    print("1. Show sample (10 assets without embeddings)")
    print("2. Export ALL assets without embeddings to JSON")
    print("3. Filter by asset type and export")
    print("4. Exit")
    
    choice = input("\nChoose an option (1-4): ").strip()
    
    if choice == "1":
        print("\nFetching sample (10 assets)...\n")
        assets = get_assets_without_any_embeddings(limit=10)
        for i, asset in enumerate(assets, 1):
            print(f"{i}. {asset['name']}")
            print(f"   Type: {asset['asset_type']} | Decade: {asset['decade']}")
            print(f"   ID: {asset['id']}")
            if asset.get('description'):
                desc = asset['description'][:100] + "..." if len(asset.get('description', '')) > 100 else asset.get('description', '')
                print(f"   Description: {desc}")
            print()
    
    elif choice == "2":
        export_assets_without_embeddings()
    
    elif choice == "3":
        print("\nAvailable asset types:")
        for i, (atype, stats) in enumerate(sorted(coverage.items(), key=lambda x: x[1]['without_embeddings'], reverse=True), 1):
            if stats['without_embeddings'] > 0:
                print(f"  {i}. {atype} ({stats['without_embeddings']:,} without embeddings)")
        
        asset_type = input("\nEnter asset type: ").strip()
        if asset_type:
            filename = f"assets_without_embeddings_{asset_type.lower().replace(' ', '_')}.json"
            export_assets_without_embeddings(output_file=filename, asset_type=asset_type)
    
    else:
        print("Exiting.")
