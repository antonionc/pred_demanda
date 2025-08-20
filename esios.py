import requests

def read_api_key(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip()

def download_esios_indicators(api_key_file):
    api_key = read_api_key(api_key_file)
    url = "https://api.esios.ree.es/indicators"
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": f"{api_key}"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Example usage:
indicators = download_esios_indicators('esios_api_key.txt')
print(indicators)