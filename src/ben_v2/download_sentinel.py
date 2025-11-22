import os
import requests
import pandas as pd
import urllib.parse
import zipfile
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# --- Configuration ---
USERNAME = "vardiashvilirati33@gmail.com"
PASSWORD = "/C9&z2ha/e6VYnV"
AOI_WKT = "POLYGON((8.715 53.11,8.91 53.11,8.91 53.01,8.715 53.01,8.715 53.11))"
START_DATE = "2023-06-01T00:00:00.000Z"
END_DATE = "2023-06-30T23:59:59.999Z"
OUTPUT_DIR = "/home/rati/bsc_thesis/sentinel-2/"
TIME_WINDOW_DAYS = 2  # Time window in days to search for matching images
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
    encoded_filter = urllib.parse.quote(filter_str, safe="()'/,;= ")
    
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


def main():
    """Main function to download matching Sentinel-1 and Sentinel-2 images."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        access_token = get_access_token(USERNAME, PASSWORD)
        print("Access token obtained.")

        s2_filter = "Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' and a/Value lt 10)"
        print("Searching for Sentinel-2 products...")
        s2_products = search_products(access_token, AOI_WKT, START_DATE, END_DATE, 'SENTINEL-2', extra_filters=s2_filter)
        print(f"Found {len(s2_products)} Sentinel-2 products.")

        if not s2_products:
            print("No low-cloud Sentinel-2 products found.")
            return

        for s2_product in s2_products:
            s2_name = s2_product['Name']
            s2_id = s2_product['Id']
            s2_date_str = s2_product['ContentDate']['Start']
            s2_date = datetime.strptime(s2_date_str, '%Y-%m-%dT%H:%M:%S.%fZ')

            print(f"\nFound Sentinel-2 product: {s2_name} from {s2_date}")

            s1_start_date = (s2_date - timedelta(days=TIME_WINDOW_DAYS)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            s1_end_date = (s2_date + timedelta(days=TIME_WINDOW_DAYS)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            s1_filter = "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/Value eq 'GRD')"
            
            print(f"Searching for matching Sentinel-1 products between {s1_start_date} and {s1_end_date}...")
            s1_products = search_products(access_token, AOI_WKT, s1_start_date, s1_end_date, 'SENTINEL-1', extra_filters=s1_filter)
            print(f"Found {len(s1_products)} Sentinel-1 products.")

            if s1_products:
                s1_product = s1_products[0]
                s1_name = s1_product['Name']
                s1_id = s1_product['Id']
                print(f"Found matching Sentinel-1 product: {s1_name}")

                pair_dir_name = f"{s2_name.split('_')[5]}_{s2_name.split('_')[2]}"
                pair_dir = os.path.join(OUTPUT_DIR, pair_dir_name)
                os.makedirs(pair_dir, exist_ok=True)

                print("\nDownloading matching pair:")
                try:
                    download_product(access_token, s2_id, s2_name, pair_dir)
                    download_product(access_token, s1_id, s1_name, pair_dir)
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