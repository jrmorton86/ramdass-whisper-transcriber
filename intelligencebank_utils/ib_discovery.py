"""Intelligence Bank V3 API asset discovery service."""
import httpx
from typing import List, Dict, Optional
from .ib_client import ib_client


class IBAssetDiscovery:
    """Service for discovering and listing assets from Intelligence Bank V3 API."""
    
    async def get_resource_by_uuid(self, uuid: str) -> Optional[Dict]:
        """
        Get a specific resource by UUID using POST /api/json endpoint.
        
        Args:
            uuid: Resource UUID (e.g., '0bf53ae5f0ac496c85166a247302ab51')
        
        Returns:
            Resource object or None
        """
        await ib_client.ensure_authenticated()
        headers = await ib_client.get_authenticated_headers()
        headers['Content-Type'] = 'application/json'
        
        # Build URL: POST {{apiV3url}}/api/json
        url = f"{ib_client.api_v3_url}/api/json"
        
        # Build POST payload with wrapped_conditions
        payload = {
            "method": "GET",
            "version": "3.0.0",
            "client": ib_client.client_id or "vJgXP3",
            "table": "resource.limit(1)",
            "query_params": {
                "searchParams": {
                    "ib_folder_s": "",
                    "isSearching": True,
                    "wrapped_conditions": [
                        [
                            {
                                "field": "ib_uuid",
                                "value": uuid,
                                "union": "and",
                                "type": "text",
                                "op": "equals"
                            }
                        ]
                    ]
                }
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # IB V3 API returns: {"response": {"count": N, "rows": [...]}}
            if isinstance(data, dict) and 'response' in data:
                resp = data['response']
                rows = resp.get('rows', [])
                # Return the resource or first item if it's a list
                if len(rows) > 0:
                    return rows[0]
                return None
            # Return the resource or first item if it's a list
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            elif isinstance(data, dict):
                return data
            return None
    
    async def list_folders(self, parent_id: Optional[str] = None) -> List[Dict]:
        """
        List folders from IB V3 API.
        
        Args:
            parent_id: Parent folder ID (None for root)
        
        Returns:
            List of folder objects
        """
        await ib_client.ensure_authenticated()
        headers = await ib_client.get_authenticated_headers()
        
        # Build URL - IB V3 folder listing endpoint
        base_url = ib_client.get_v3_base_url()
        url = f"{base_url}/folder"
        params = {}
        if parent_id:
            params['searchParams[parentId]'] = parent_id
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # IB API returns folders in different structures - adapt as needed
            if isinstance(data, dict) and 'folders' in data:
                return data['folders']
            elif isinstance(data, list):
                return data
            return []
    
    async def list_assets(self, folder_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> Dict:
        """
        List assets/resources from IB V3 API.
        
        Args:
            folder_id: Folder ID to list assets from (None for all)
            limit: Maximum number of assets to return (default 100)
            offset: Pagination offset
        
        Returns:
            Dict with 'assets' list and 'total' count
        """
        await ib_client.ensure_authenticated()
        headers = await ib_client.get_authenticated_headers()
        
        # Build URL: {{apiV3url}}/api/3.0.0/{{clientid}}/resource.limit(100).order(createTime:-1)?verbose=null
        base_url = ib_client.get_v3_base_url()
        url = f"{base_url}/resource.limit({limit}).order(createTime:-1)"
        
        params = {'verbose': 'null'}
        if folder_id:
            params['searchParams[folderId]'] = folder_id
        if offset:
            params['offset'] = offset
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # IB V3 API returns: {"response": {"count": N, "rows": [...]}}
            if isinstance(data, dict) and 'response' in data:
                resp = data['response']
                assets = resp.get('rows', [])
                total = resp.get('count', len(assets))
            elif isinstance(data, dict):
                assets = data.get('resources', data.get('assets', data.get('items', [])))
                total = data.get('total', data.get('totalCount', len(assets)))
            else:
                assets = data if isinstance(data, list) else []
                total = len(assets)
            
            return {
                'assets': assets,
                'total': total,
                'limit': limit,
                'offset': offset
            }
    
    async def get_asset_download_url(self, asset_id: str) -> str:
        """
        Get download URL for a specific asset.
        
        Args:
            asset_id: Asset UUID from IB
        
        Returns:
            Download URL string
        """
        await ib_client.ensure_authenticated()
        headers = await ib_client.get_authenticated_headers()
        
        # First get the resource to find download URL
        resource = await self.get_resource_by_uuid(asset_id)
        if resource:
            # Check for download URL in resource
            download_url = resource.get('downloadUrl', resource.get('url'))
            if download_url:
                return download_url
        
        # Fallback: try direct download endpoint
        base_url = ib_client.get_v3_base_url()
        url = f"{base_url}/resource/{asset_id}/download"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, allow_redirects=False)
            
            # Check if response contains download URL or is a redirect
            if response.status_code in [301, 302, 307, 308]:
                return response.headers.get('Location', url)
            
            try:
                data = response.json()
                download_url = data.get('downloadUrl', data.get('url', data.get('link', url)))
                return download_url
            except:
                return url
    
    async def discover_all_assets(self, folder_id: Optional[str] = None) -> List[Dict]:
        """
        Discover all assets, handling pagination.
        
        Args:
            folder_id: Optional folder to limit discovery to
        
        Returns:
            List of all discovered assets
        """
        all_assets = []
        offset = 0
        limit = 100
        
        while True:
            result = await self.list_assets(folder_id=folder_id, limit=limit, offset=offset)
            assets = result['assets']
            all_assets.extend(assets)
            
            # Check if we've retrieved all assets
            if len(assets) < limit or offset + len(assets) >= result['total']:
                break
            
            offset += limit
        
        return all_assets


# Singleton instance
asset_discovery = IBAssetDiscovery()
