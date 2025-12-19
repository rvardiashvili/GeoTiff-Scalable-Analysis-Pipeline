import os
import requests
import urllib.parse
import zipfile
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# --- Default Configuration (Can be overridden by CLI args) ---
# Credentials should ideally be env vars, but keeping defaults for fallback
DEFAULT_USERNAME = os.environ.get("CDSE_USERNAME", "vardiashvilirati33@gmail.com")
DEFAULT_PASSWORD = os.environ.get("CDSE_PASSWORD", "/C9&z2ha/e6VYnV")
# Venice, Italy (Approximate Bounding Box)
# 12.20 45.30 to 12.50 45.60
DEFAULT_AOI_WKT = "POLYGON((22.58 32.74, 22.84 32.74, 22.84 32.96, 22.58 32.96, 22.58 32.74))" 
DEFAULT_START_DATE = "2023-09-01T00:00:00.000Z"
DEFAULT_END_DATE = "2023-09-30T23:59:59.999Z"
DEFAULT_OUTPUT_DIR = "/home/rati/bsc_thesis/sentinel-2/"
DEFAULT_TIME_WINDOW_DAYS = 2
MAX_RETRIES = 3

def get_access_token(username, password):
    """Gets access token from Copernicus Dataspace."""
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            return response.json()["access_token"]
        except requests.exceptions.RequestException as e:
            print(f"Error getting access token (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                raise

def search_products(access_token, aoi_wkt, start_date, end_date, collection, extra_filters=""):
    """Searches for products in Copernicus Dataspace with pagination."""
    all_products = []
    odata_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        f"?$expand=Attributes"
    )
    filter_parts = [
        f"Collection/Name eq '{collection}'",
        f"ContentDate/Start ge {start_date} and ContentDate/End le {end_date}"
    ]
    
    # Only add spatial filter if aoi_wkt is provided
    if aoi_wkt:
        filter_parts.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')")
        
    if extra_filters:
        filter_parts.append(extra_filters)

    filter_str = " and ".join(filter_parts)
    encoded_filter = urllib.parse.quote(filter_str, safe="()'/;= ")
    
    paginated_url = f"{odata_url}&$filter={encoded_filter}&$top=1000"

    headers = {"Authorization": f"Bearer {access_token}"}
    
    while paginated_url:
        try:
            response = requests.get(paginated_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            all_products.extend(data['value'])
            paginated_url = data.get('@odata.nextLink')
        except requests.exceptions.RequestException as e:
            print(f"Error searching for products: {e}")
            raise
            
    return all_products

def download_product(access_token, product_id, product_name, output_dir):
    """Downloads a product, extracts it, and removes the zip file."""
    download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    safe_product_name = product_name.replace('/', '_')
    zip_path = os.path.join(output_dir, f"{safe_product_name}.zip")
    
    if os.path.exists(zip_path.replace('.zip', '.SAFE')):
        print(f"{product_name} already exists. Skipping download.")
        return

    headers = {"Authorization": f"Bearer {access_token}"}
    
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(download_url, headers=headers, stream=True) as dl:
                dl.raise_for_status()
                total_size = int(dl.headers.get('content-length', 0))
                
                with open(zip_path, "wb") as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {product_name}"
                ) as pbar:
                    for chunk in dl.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print(f"Saved to {zip_path}")

            print(f"Extracting {zip_path} to {output_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"Extraction complete.")

            if os.path.exists(zip_path):
                print(f"Removing {zip_path}...")
                os.remove(zip_path)
                print(f"Removed {zip_path}.")
            return 
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {product_name} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                print(f"Failed to download {product_name} after {MAX_RETRIES} attempts.")
                if os.path.exists(zip_path):
                    os.remove(zip_path) # Clean up partial download
                raise


def download_specific_product(access_token, product_name, collection, output_dir):
    """Downloads a specific product by its full name."""
    print(f"\nSearching for specific product: {product_name} in {collection} collection...")
    # Use very broad date range and AOI to ensure the product is found
    # The actual date and AOI are implicitly part of the product name.
    broad_aoi = "POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))" # Global AOI
    broad_start_date = "2010-01-01T00:00:00.000Z" # Start of Sentinel missions
    broad_end_date = "2030-01-01T00:00:00.000Z" # Far in the future
    
    name_filter = f"Name eq '{product_name}'"
    products = search_products(access_token, broad_aoi, broad_start_date, broad_end_date, collection, extra_filters=name_filter)
    
    if products:
        product = products[0]
        product_id = product['Id']
        print(f"Found product ID for {product_name}: {product_id}")
        
        # Create a directory named after the product for this specific download
        product_output_dir = os.path.join(output_dir, product_name.replace('.SAFE', ''))
        os.makedirs(product_output_dir, exist_ok=True)
        
        download_product(access_token, product_id, product_name, product_output_dir)
        print(f"✅ Successfully downloaded {product_name} to {product_output_dir}")
    else:
        print(f"❌ Product not found: {product_name}")


def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-1 and Sentinel-2 products.")
    parser.add_argument("--aoi", type=str, default=DEFAULT_AOI_WKT, help="Area of Interest (WKT Polygon) for pair search.")
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE, help="Start Date (ISO 8601) for pair search.")
    parser.add_argument("--end_date", type=str, default=DEFAULT_END_DATE, help="End Date (ISO 8601) for pair search.")
    parser.add_argument("--cloud_cover", type=float, default=5.0, help="Max Cloud Cover Percentage for Sentinel-2 in pair search.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base Output Directory for all downloads.")
    parser.add_argument("--username", type=str, default=DEFAULT_USERNAME, help="CDSE Username")
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD, help="CDSE Password")
    parser.add_argument("--time_window", type=int, default=DEFAULT_TIME_WINDOW_DAYS, help="Time window in days for finding matching Sentinel-1 products.")
    
    # New arguments for specific product downloads
    parser.add_argument("--s1_product_name", type=str, help="Full name of a specific Sentinel-1 product to download.")
    parser.add_argument("--s2_product_name", type=str, help="Full name of a specific Sentinel-2 product to download.")
    parser.add_argument("--download_one_s2", action="store_true", help="Download only one Sentinel-2 product (and its potential S1 pair).")
    parser.add_argument("--target_s2_name", type=str, help="Full name of a specific Sentinel-2 product to target for pair download.")
    parser.add_argument("--only_s1_for_s2_name", type=str, help="Skip S2 download; search and download S1s for this existing S2 product.")
    parser.add_argument("--s2_tile_id", type=str, help="Specify a Sentinel-2 tile ID (e.g., 32UQC) to use as the AOI.")
    parser.add_argument("--only_s2", action="store_true", help="Only download Sentinel-2 products, skip Sentinel-1 pair search.")
    
    # Multi-temporal arguments
    parser.add_argument("--multi_temporal", action="store_true", help="Download a time series of Sentinel-2 images.")
    parser.add_argument("--max_images", type=int, default=3, help="Maximum number of images to download for multi-temporal series.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        access_token = get_access_token(args.username, args.password)
        print("Access token obtained.")

        # --- Multi-Temporal Download Logic ---
        if args.multi_temporal:
            selected_products = []
            
            # Case A: Target S2 Name provided (Smart phenological search)
            if args.target_s2_name:
                print(f"Searching for reference product: {args.target_s2_name}...")
                name_filter = f"Name eq '{args.target_s2_name}'"
                # No AOI needed for name search
                ref_products = search_products(access_token, None, "2020-01-01T00:00:00.000Z", "2030-01-01T00:00:00.000Z", 'SENTINEL-2', extra_filters=name_filter)
                
                if not ref_products:
                    print(f"Reference product {args.target_s2_name} not found.")
                    return
                
                ref_prod = ref_products[0]
                ref_date = datetime.strptime(ref_prod['ContentDate']['Start'], '%Y-%m-%dT%H:%M:%S.%fZ')
                
                # Extract Tile ID from attributes if possible, or assume user provided it? 
                # Better to get from product attributes.
                tile_id = None
                for attr in ref_prod['Attributes']:
                    if attr['Name'] == 'mgrsTile':
                        tile_id = attr['Value']
                        break
                
                if not tile_id:
                    print("Could not determine Tile ID from reference product. Please provide --s2_tile_id manually if needed, or check product metadata.")
                    # Fallback to args.s2_tile_id if available
                    tile_id = args.s2_tile_id
                
                if not tile_id:
                    print("Aborting: Tile ID missing.")
                    return

                print(f"Reference Date: {ref_date.date()}, Tile: {tile_id}")
                print(f"Selecting {args.max_images} images spaced ~30 days apart...")
                
                # Always include the reference product first
                selected_products.append(ref_prod)
                
                # Find previous time steps
                for i in range(1, args.max_images):
                    target_date = ref_date - timedelta(days=30 * i)
                    s_start = (target_date - timedelta(days=10)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    s_end = (target_date + timedelta(days=10)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    
                    print(f"  Step {i}: Looking around {target_date.date()} ({s_start} to {s_end})...")
                    
                    # Simplify API Filter: Use Name contains logic instead of complex Attributes
                    # Filename format: S2A_MSIL2A_..._T32UNE_...
                    # We search for 'MSIL2A' (Product Type) and 'T{tile_id}' (Tile ID)
                    
                    s2_filter_parts = [
                        f"contains(Name, 'MSIL2A')",
                        f"contains(Name, 'T{tile_id}')"
                    ]
                    s2_filter = " and ".join(s2_filter_parts)
                    
                    # No AOI needed if filtering by Tile ID in Name
                    step_products = search_products(access_token, None, s_start, s_end, 'SENTINEL-2', extra_filters=s2_filter)
                    
                    # Client-side Cloud Filtering
                    def get_cloud(p):
                        for a in p['Attributes']:
                            if a['Name'] == 'cloudCover': return a['Value']
                        return 100.0

                    if step_products:
                        # Filter by max cloud cover
                        valid_products = [p for p in step_products if get_cloud(p) < args.cloud_cover]
                        
                        if valid_products:
                            # Sort by cloud cover (best first)
                            valid_products.sort(key=get_cloud)
                            best = valid_products[0]
                            print(f"    Found match: {best['Name']} (Cloud: {get_cloud(best)}%)")
                            selected_products.append(best)
                        else:
                            print(f"    Found {len(step_products)} images, but none met cloud criteria (< {args.cloud_cover}%).")
                    else:
                        print(f"    No L2A images found near {target_date.date()}. Skipping.")

                # Reverse to be chronological (Oldest -> Newest)
                selected_products.reverse()

            # Case B: General Search (Original Logic)
            elif args.s2_tile_id:
                print(f"Searching for Sentinel-2 time series for tile {args.s2_tile_id}...")
                
                s2_filter_parts = [
                    f"Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' and a/Value lt {args.cloud_cover})",
                    f"Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/Value eq 'S2MSI2A')",
                    f"Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'mgrsTile' and a/Value eq '{args.s2_tile_id}')"
                ]
                s2_filter = " and ".join(s2_filter_parts)
                
                # Search
                s2_products = search_products(access_token, args.aoi, args.start_date, args.end_date, 'SENTINEL-2', extra_filters=s2_filter)
                
                if not s2_products:
                    print("No matching Sentinel-2 products found.")
                    return
                    
                # Sort by Date
                s2_products.sort(key=lambda p: p['ContentDate']['Start'])
                
                print(f"Found {len(s2_products)} candidate images. Selecting {args.max_images}...")
                
                if len(s2_products) <= args.max_images:
                    selected_products = s2_products
                else:
                    selected_products = s2_products[:args.max_images]
            else:
                print("Error: Either --target_s2_name or --s2_tile_id must be provided for multi-temporal download.")
                return
                
            # Create Series Directory
            if not selected_products:
                print("No products selected.")
                return

            series_name = f"S2_Series_{selected_products[0]['ContentDate']['Start'][:10]}_{selected_products[-1]['ContentDate']['Start'][:10]}"
            # Append Tile ID if available
            tile_id_str = args.s2_tile_id if args.s2_tile_id else "Series"
            series_name = f"{tile_id_str}_Series_{selected_products[0]['ContentDate']['Start'][:10]}_{selected_products[-1]['ContentDate']['Start'][:10]}"
            
            series_dir = os.path.join(args.output_dir, series_name)
            os.makedirs(series_dir, exist_ok=True)
            
            print(f"Downloading series to: {series_dir}")
            
            for prod in selected_products:
                name = prod['Name']
                pid = prod['Id']
                date = prod['ContentDate']['Start']
                print(f"Downloading {name} ({date})...")
                try:
                    download_product(access_token, pid, name, series_dir)
                except Exception as e:
                    print(f"Failed to download {name}: {e}")
                    
            print(f"✅ Multi-temporal series download complete: {series_dir}")
            return

        # Handle specific product downloads if names are provided
        if args.s1_product_name:
            download_specific_product(access_token, args.s1_product_name, 'SENTINEL-1', args.output_dir)
        
        if args.s2_product_name:
            download_specific_product(access_token, args.s2_product_name, 'SENTINEL-2', args.output_dir)

        # If --only_s1_for_s2_name is provided, skip S2 download and only find/download S1s
        if args.only_s1_for_s2_name:
            print(f"\nSearching for metadata of existing S2 product: {args.only_s1_for_s2_name}...")
            # Use very broad date range and AOI to ensure the product is found
            broad_aoi = "POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))" # Global AOI
            broad_start_date = "2010-01-01T00:00:00.000Z"
            broad_end_date = "2030-01-01T00:00:00.000Z"
            name_filter = f"Name eq '{args.only_s1_for_s2_name}'"
            
            s2_products_meta = search_products(access_token, broad_aoi, broad_start_date, broad_end_date, 'SENTINEL-2', extra_filters=name_filter)
            
            if s2_products_meta:
                s2_product_meta = s2_products_meta[0]
                s2_name = s2_product_meta['Name']
                s2_id = s2_product_meta['Id']
                s2_date_str = s2_product_meta['ContentDate']['Start']
                s2_date = datetime.strptime(s2_date_str, '%Y-%m-%dT%H:%M:%S.%fZ')

                pair_dir = os.path.join(args.output_dir, s2_name + '.SAFE')
                if not os.path.isdir(pair_dir):
                    print(f"Error: Existing S2 product directory '{pair_dir}' not found. Please ensure the S2 product is already downloaded to the specified output directory.")
                    return

                print(f"Found existing S2 product metadata: {s2_name} from {s2_date}. Searching for S1s...")

                s1_start_date = (s2_date - timedelta(days=args.time_window)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                s1_end_date = (s2_date + timedelta(days=args.time_window)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                
                s1_filter = "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/Value eq 'GRD')"
                
                print(f"Searching for matching Sentinel-1 products between {s1_start_date} and {s1_end_date}...")
                s1_products = search_products(access_token, args.aoi, s1_start_date, s1_end_date, 'SENTINEL-1', extra_filters=s1_filter)
                print(f"Found {len(s1_products)} Sentinel-1 products.")

                if s1_products:
                    # Download all found S1 products into the specified pair_dir
                    for s1_prod in s1_products:
                        s1_name_iter = s1_prod['Name']
                        s1_id_iter = s1_prod['Id']
                        print(f"Downloading matching Sentinel-1 product: {s1_name_iter}")
                        download_product(access_token, s1_id_iter, s1_name_iter, pair_dir)
                    print(f"✅ Successfully downloaded all associated S1s to {pair_dir}")
                else:
                    print("No matching Sentinel-1 product found for this Sentinel-2 product within the specified time window and AOI.")
                return
            else:
                print(f"❌ Existing S2 product '{args.only_s1_for_s2_name}' metadata not found.")
                return

        # If no specific product names are provided, proceed with pair search logic
        if not args.s1_product_name and not args.s2_product_name:
            if args.s2_tile_id and args.aoi == DEFAULT_AOI_WKT: # Check if AOI is still default, implying user didn't provide
                print("Error: When --s2_tile_id is provided, --aoi must also be explicitly provided with the WKT for that tile.")
                return
            
            # Simplified Filter Construction
            s2_filter_l2a_parts = [
                f"contains(Name, 'MSIL2A')"
            ]
            
            # Use Name contains for Tile ID if provided (Robuster than Attribute filter)
            if args.s2_tile_id:
                s2_filter_l2a_parts.append(f"contains(Name, 'T{args.s2_tile_id}')")
                
            s2_filter = " and ".join(s2_filter_l2a_parts)
            
            print("Searching for Sentinel-2 products (pair search mode)...")
            
            # If filtering by Tile ID, AOI might be redundant/problematic for API if complex.
            # But general search usually needs AOI.
            # Let's try passing None for AOI if Tile ID is present, assuming Tile ID is sufficient spatial filter.
            search_aoi = None if args.s2_tile_id else args.aoi
            
            s2_products = search_products(access_token, search_aoi, args.start_date, args.end_date, 'SENTINEL-2', extra_filters=s2_filter)
            
            # Client-side Cloud Filtering
            def get_cloud_cover(product):
                for attr in product['Attributes']:
                    if attr['Name'] == 'cloudCover':
                        return attr['Value']
                return 100.0

            if s2_products:
                # Filter by cloud cover
                s2_products = [p for p in s2_products if get_cloud_cover(p) < args.cloud_cover]
                print(f"Found {len(s2_products)} Sentinel-2 products matching criteria.")
            else:
                print("No products found (before cloud filtering).")

            if not s2_products:
                print("No low-cloud Sentinel-2 products found for pair search.")
                return

            # Sort by cloud cover ascending
            s2_products.sort(key=get_cloud_cover)

            s2_products_to_process = []
            if args.target_s2_name:
                found_target_s2 = False
                for s2_prod in s2_products:
                    if s2_prod['Name'] == args.target_s2_name:
                        s2_products_to_process.append(s2_prod)
                        found_target_s2 = True
                        break
                if not found_target_s2:
                    print(f"Error: Target Sentinel-2 product '{args.target_s2_name}' not found within search criteria.")
                    return
            else:
                s2_products_to_process = s2_products


            for s2_product in s2_products_to_process:
                s2_name = s2_product['Name']
                s2_id = s2_product['Id']
                s2_date_str = s2_product['ContentDate']['Start']
                s2_date = datetime.strptime(s2_date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                s2_cloud = get_cloud_cover(s2_product)

                print(f"\nFound Sentinel-2 product: {s2_name} from {s2_date} (Cloud: {s2_cloud}%)")

                if args.only_s2:
                    print("Skipping Sentinel-1 search due to --only_s2 flag.")
                    # Create directory for S2 product
                    s2_output_dir = os.path.join(args.output_dir, s2_name + '.SAFE')
                    os.makedirs(s2_output_dir, exist_ok=True)
                    try:
                        download_product(access_token, s2_id, s2_name, s2_output_dir)
                        print(f"✅ Successfully downloaded {s2_name} to {s2_output_dir}")
                    except Exception as e:
                        print(f"Could not download {s2_name}. Reason: {e}")
                    if args.download_one_s2:
                        return # Exit after one successful S2 download if flag is set
                    continue # Continue to next S2 if not download_one_s2

                s1_start_date = (s2_date - timedelta(days=args.time_window)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                s1_end_date = (s2_date + timedelta(days=args.time_window)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                
                s1_filter = "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/Value eq 'GRD')"
                
                print(f"Searching for matching Sentinel-1 products between {s1_start_date} and {s1_end_date}...")
                s1_products = search_products(access_token, args.aoi, s1_start_date, s1_end_date, 'SENTINEL-1', extra_filters=s1_filter)
                print(f"Found {len(s1_products)} Sentinel-1 products.")

                # Corrected pair_dir calculation and multi-S1 download logic
                pair_dir = os.path.join(args.output_dir, s2_name + '.SAFE')
                os.makedirs(pair_dir, exist_ok=True)

                print("\nDownloading matching pair:")
                try:
                    download_product(access_token, s2_id, s2_name, pair_dir)
                    # Download all found S1 products into the same pair_dir
                    for s1_prod in s1_products:
                        s1_name_iter = s1_prod['Name']
                        s1_id_iter = s1_prod['Id']
                        print(f"Downloading matching Sentinel-1 product: {s1_name_iter}")
                        download_product(access_token, s1_id_iter, s1_name_iter, pair_dir)
                    print(f"✅ Successfully downloaded pair and all associated S1s to {pair_dir}")
                    if args.download_one_s2:
                        return # Exit after one successful pair download if flag is set
                except Exception as e:
                    print(f"Could not download pair for {s2_name}. Reason: {e}")
                
                if args.download_one_s2: # If only one S2 is requested, break after checking this one.
                    break

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if hasattr(e, 'response') and e.response:
            print(e.response.text)

if __name__ == "__main__":
    main()
