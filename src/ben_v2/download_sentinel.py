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
DEFAULT_AOI_WKT = "POLYGON((12.20 45.30, 12.50 45.30, 12.50 45.60, 12.20 45.60, 12.20 45.30))" 
DEFAULT_START_DATE = "2023-06-01T00:00:00.000Z"
DEFAULT_END_DATE = "2023-08-30T23:59:59.999Z"
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
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')",
        f"ContentDate/Start ge {start_date} and ContentDate/End le {end_date}"
    ]
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
    
    # New arguments for specific product downloads
    parser.add_argument("--s1_product_name", type=str, help="Full name of a specific Sentinel-1 product to download.")
    parser.add_argument("--s2_product_name", type=str, help="Full name of a specific Sentinel-2 product to download.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        access_token = get_access_token(args.username, args.password)
        print("Access token obtained.")

        # Handle specific product downloads if names are provided
        if args.s1_product_name:
            download_specific_product(access_token, args.s1_product_name, 'SENTINEL-1', args.output_dir)
        
        if args.s2_product_name:
            download_specific_product(access_token, args.s2_product_name, 'SENTINEL-2', args.output_dir)

        # If no specific product names are provided, proceed with pair search logic
        if not args.s1_product_name and not args.s2_product_name:
            s2_filter = f"Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' and a/Value lt {args.cloud_cover})"
            print("Searching for Sentinel-2 products (pair search mode)...")
            s2_products = search_products(access_token, args.aoi, args.start_date, args.end_date, 'SENTINEL-2', extra_filters=s2_filter)
            print(f"Found {len(s2_products)} Sentinel-2 products.")

            if not s2_products:
                print("No low-cloud Sentinel-2 products found for pair search.")
                return

            # Sort by cloud cover ascending
            def get_cloud_cover(product):
                for attr in product['Attributes']:
                    if attr['Name'] == 'cloudCover':
                        return attr['Value']
                return 100.0
                
            s2_products.sort(key=get_cloud_cover)

            for s2_product in s2_products:
                s2_name = s2_product['Name']
                s2_id = s2_product['Id']
                s2_date_str = s2_product['ContentDate']['Start']
                s2_date = datetime.strptime(s2_date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                s2_cloud = get_cloud_cover(s2_product)

                print(f"\nFound Sentinel-2 product: {s2_name} from {s2_date} (Cloud: {s2_cloud}%)")

                s1_start_date = (s2_date - timedelta(days=DEFAULT_TIME_WINDOW_DAYS)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                s1_end_date = (s2_date + timedelta(days=DEFAULT_TIME_WINDOW_DAYS)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                
                s1_filter = "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/Value eq 'GRD')"
                
                print(f"Searching for matching Sentinel-1 products between {s1_start_date} and {s1_end_date}...")
                s1_products = search_products(access_token, args.aoi, s1_start_date, s1_end_date, 'SENTINEL-1', extra_filters=s1_filter)
                print(f"Found {len(s1_products)} Sentinel-1 products.")

                if s1_products:
                    s1_product = s1_products[0]
                    s1_name = s1_product['Name']
                    s1_id = s1_product['Id']
                    print(f"Found matching Sentinel-1 product: {s1_name}")

                    pair_dir_name = f"{s2_name.split('_')[5]}_{s2_name.split('_')[2]}"
                    pair_dir = os.path.join(args.output_dir, pair_dir_name)
                    os.makedirs(pair_dir, exist_ok=True)

                    print("\nDownloading matching pair:")
                    try:
                        download_product(access_token, s2_id, s2_name, pair_dir)
                        download_product(access_token, s1_id, s1_name, pair_dir)
                        print(f"✅ Successfully downloaded pair to {pair_dir}")
                        return # Exit after one successful pair download
                    except Exception as e:
                        print(f"Could not download pair for {s2_name}. Reason: {e}")

                else:
                    print("No matching Sentinel-1 product found for this Sentinel-2 product.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if hasattr(e, 'response') and e.response:
            print(e.response.text)

if __name__ == "__main__":
    main()
