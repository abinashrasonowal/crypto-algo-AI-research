import os
import requests

# Define the URLs and corresponding filenames
urls = {
    "TrxFees": "https://etherscan.io/chart/transactionfee?output=csv",
    "PendingTrx": "https://etherscan.io/chart/pendingtx?output=csv",
    "GasPrice": "https://etherscan.io/chart/gasprice?output=csv",
    "ActiveAddresses": "https://etherscan.io/chart/active-address?output=csv"
}

# Add headers to mimic a browser
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/90.0.4430.212 Safari/537.36"
    ),
    "Referer": "https://etherscan.io/"
}

# Create the folder if it doesn't exist
folder_name = "on_chain"
os.makedirs(folder_name, exist_ok=True)

# Try downloading with headers
for name, url in urls.items():
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        filepath = os.path.join(folder_name, f"{name}.csv")
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Saved {name}.csv")
    else:
        print(f"Failed to download {name} (status code: {response.status_code})")
