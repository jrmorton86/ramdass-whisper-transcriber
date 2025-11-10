"""Query and explore asset_embeddings table."""
from database_navigator import get_connection
from typing import List, Dict, Any, Optional
import json


def get_embedding_stats() -> Dict[str, Any]:
    """Get statistics about asset embeddings."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Total embeddings
            cur.execute("SELECT COUNT(*) FROM asset_embeddings")
            total_embeddings = cur.fetchone()[0]
            
            # Total assets
            cur.execute("SELECT COUNT(*) FROM assets")
            total_assets = cur.fetchone()[0]
            
            # Assets without embeddings
            cur.execute("""
                SELECT COUNT(*) 
                FROM assets a
                LEFT JOIN asset_embeddings ae ON a.id = ae.resource_id
                WHERE ae.id IS NULL
            """)
            assets_without_embeddings = cur.fetchone()[0]
            
            # Embeddings by content type
            cur.execute("""
                SELECT content_type, COUNT(*) 
                FROM asset_embeddings 
                GROUP BY content_type 
                ORDER BY COUNT(*) DESC
            """)
            by_content_type = dict(cur.fetchall())
            
            # Embeddings by model
            cur.execute("""
                SELECT embedding_model, COUNT(*) 
                FROM asset_embeddings 
                GROUP BY embedding_model 
                ORDER BY COUNT(*) DESC
            """)
            by_model = dict(cur.fetchall())
            
            return {
                "total_embeddings": total_embeddings,
                "total_assets": total_assets,
                "assets_with_embeddings": total_assets - assets_without_embeddings,
                "assets_without_embeddings": assets_without_embeddings,
                "coverage_percentage": ((total_assets - assets_without_embeddings) / total_assets * 100) if total_assets > 0 else 0,
                "by_content_type": by_content_type,
                "by_model": by_model
            }


def get_assets_without_embeddings(
    limit: Optional[int] = None, 
    offset: int = 0,
    asset_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get assets that don't have embeddings yet.
    
    Args:
        limit: Maximum number of records to return (None = all)
        offset: Number of records to skip
        asset_type: Filter by asset type (Audio, Video, etc.)
        
    Returns:
        List of asset dictionaries
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    a.id,
                    a.bynder_id,
                    a.name,
                    a.description,
                    a.asset_type,
                    a.asset_sub_type,
                    a.decade,
                    a.publication_date,
                    a.media_duration_seconds,
                    a.created_at
                FROM assets a
                LEFT JOIN asset_embeddings ae ON a.id = ae.resource_id
                WHERE ae.id IS NULL
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
                        # Handle numeric types
                        value = float(value) if isinstance(value, type(value)) and hasattr(value, '__float__') else value
                    asset[col_name] = value
                results.append(asset)
            
            return results


def get_embedding_details(resource_id: str) -> List[Dict[str, Any]]:
    """
    Get all embeddings for a specific asset.
    
    Args:
        resource_id: UUID of the asset
        
    Returns:
        List of embedding records
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    resource_id,
                    content_type,
                    embedding_model,
                    created_at,
                    updated_at
                FROM asset_embeddings
                WHERE resource_id = %s
                ORDER BY created_at DESC
            """, (resource_id,))
            
            columns = [desc[0] for desc in cur.description]
            results = []
            
            for row in cur.fetchall():
                embedding = {}
                for i, col_name in enumerate(columns):
                    value = row[i]
                    if hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    embedding[col_name] = value
                results.append(embedding)
            
            return results


def get_assets_with_embeddings_details(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get assets with their embedding information.
    
    Args:
        limit: Number of records to return
        
    Returns:
        List of assets with embedding details
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    a.id,
                    a.name,
                    a.asset_type,
                    a.decade,
                    ae.content_type,
                    ae.embedding_model,
                    ae.created_at as embedding_created_at
                FROM assets a
                INNER JOIN asset_embeddings ae ON a.id = ae.resource_id
                ORDER BY ae.created_at DESC
                LIMIT %s
            """, (limit,))
            
            columns = [desc[0] for desc in cur.description]
            results = []
            
            for row in cur.fetchall():
                item = {}
                for i, col_name in enumerate(columns):
                    value = row[i]
                    if hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    item[col_name] = value
                results.append(item)
            
            return results


def export_assets_without_embeddings(output_file: str = "assets_needing_embeddings.json"):
    """
    Export all assets without embeddings to a JSON file.
    
    Args:
        output_file: Filename for the JSON export
    """
    from pathlib import Path
    import datetime
    
    print("Fetching embedding statistics...")
    stats = get_embedding_stats()
    
    print(f"\nTotal assets: {stats['total_assets']:,}")
    print(f"Assets with embeddings: {stats['assets_with_embeddings']:,}")
    print(f"Assets without embeddings: {stats['assets_without_embeddings']:,}")
    print(f"Coverage: {stats['coverage_percentage']:.1f}%")
    
    if stats['assets_without_embeddings'] == 0:
        print("\n✓ All assets have embeddings!")
        return
    
    print(f"\nFetching {stats['assets_without_embeddings']:,} assets without embeddings...")
    assets = get_assets_without_embeddings()
    
    output_path = Path(__file__).parent / output_file
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "count": len(assets),
            "exported_at": datetime.datetime.utcnow().isoformat(),
            "statistics": stats,
            "assets": assets
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported {len(assets):,} assets to: {output_path}")
    
    # Print summary
    print("\nAssets without embeddings by type:")
    type_counts = {}
    for asset in assets:
        asset_type = asset.get('asset_type') or 'Unknown'
        type_counts[asset_type] = type_counts.get(asset_type, 0) + 1
    
    for asset_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset_type}: {count:,}")


if __name__ == "__main__":
    print("="*80)
    print("ASSET EMBEDDINGS ANALYSIS")
    print("="*80)
    
    # Get statistics
    stats = get_embedding_stats()
    
    print(f"\nTotal Assets: {stats['total_assets']:,}")
    print(f"Total Embeddings: {stats['total_embeddings']:,}")
    print(f"Assets with Embeddings: {stats['assets_with_embeddings']:,}")
    print(f"Assets without Embeddings: {stats['assets_without_embeddings']:,}")
    print(f"Coverage: {stats['coverage_percentage']:.1f}%")
    
    print("\nEmbeddings by Content Type:")
    for content_type, count in stats['by_content_type'].items():
        print(f"  {content_type}: {count:,}")
    
    print("\nEmbeddings by Model:")
    for model, count in stats['by_model'].items():
        print(f"  {model}: {count:,}")
    
    if stats['assets_without_embeddings'] > 0:
        print("\n" + "="*80)
        print("ASSETS WITHOUT EMBEDDINGS")
        print("="*80)
        
        print("\nOptions:")
        print("  1. Show sample (10 assets without embeddings)")
        print("  2. Export all to JSON")
        print("  3. Show assets WITH embeddings (sample)")
        print("  4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            print("\nFetching sample (10 assets)...\n")
            assets = get_assets_without_embeddings(limit=10)
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
            print("\nAssets WITH embeddings (recent 10)...\n")
            assets = get_assets_with_embeddings_details(limit=10)
            for i, asset in enumerate(assets, 1):
                print(f"{i}. {asset['name']}")
                print(f"   Type: {asset['asset_type']} | Content: {asset['content_type']}")
                print(f"   Model: {asset['embedding_model']}")
                print()
        
        else:
            print("Exiting.")
    else:
        print("\n✓ All assets have embeddings!")
        
        print("\nShowing sample of assets WITH embeddings (recent 10)...\n")
        assets = get_assets_with_embeddings_details(limit=10)
        for i, asset in enumerate(assets, 1):
            print(f"{i}. {asset['name']}")
            print(f"   Type: {asset['asset_type']} | Content: {asset['content_type']}")
            print(f"   Model: {asset['embedding_model']}")
            print()
