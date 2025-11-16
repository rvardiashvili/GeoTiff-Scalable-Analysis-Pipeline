import os
import requests
import pandas as pd
import urllib.parse

# --- Credentials ---
username = "vardiashvilirati33@gmail.com"
password = "/C9&z2ha/e6VYnV"

# --- Area of Interest and Date Range ---
aoi_wkt = "POLYGON((8.715 53.11,8.91 53.11,8.91 53.01,8.715 53.01,8.715 53.11))"
start_date = "2023-06-01T00:00:00.000Z"
end_date = "2023-06-30T23:59:59.999Z"
output_dir = "/home/rati/bsc_thesis/sentinel-2/"
os.makedirs(output_dir, exist_ok=True)

# --- Get Access Token ---
token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
data = {
    "grant_type": "password",
    "client_id": "cdse-public",
    "username": username,
    "password": password
}
access_token = requests.post(token_url, data=data).json()["access_token"]

# --- Filter Sentinel-2 data with cloud coverage below 10% ---
filter_str = (
    "Collection/Name eq 'SENTINEL-2' and "
    f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') and "
    f"ContentDate/Start ge {start_date} and ContentDate/End le {end_date} and "
    "Attributes/OData.CSC.DoubleAttribute/any(a:a/Name eq 'cloudCover' and a/Value lt 10)"
)

encoded_filter = urllib.parse.quote(filter_str, safe="()'/,;= ")
odata_url = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    f"?$top=5&$expand=Attributes&$filter={encoded_filter}"
)

headers = {"Authorization": f"Bearer {access_token}"}

# --- Send Request ---
r = requests.get(odata_url, headers=headers)
print("Status:", r.status_code)
if r.status_code != 200:
    print(r.text)
r.raise_for_status()

results = r.json()["value"]
print("Returned:", len(results))

if not results:
    print("No low-cloud products found.")
else:
    df = pd.DataFrame(results)
    cols = [c for c in ["Id", "Name", "ContentDate", "Attributes"] if c in df.columns]
    print(df[cols].head())

    # --- Download the first 5 products ---
    for _, row in df.iterrows():
        product_id = row["Id"]
        product_name = row["Name"]
        download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        print(f"Downloading {product_name} ...")
        dest = os.path.join(output_dir, f"{product_name}.zip")
        with requests.get(download_url, headers=headers, stream=True) as dl:
            dl.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in dl.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Saved to {dest}")
