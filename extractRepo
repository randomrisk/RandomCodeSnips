# This script is to extract the contents of a Github repo, and combine into one file.

import requests
from bs4 import BeautifulSoup

# List of all filenames to extract
files = [
    "examplefiles",
]

# Base URL of the repository
base_url = "https://raw.githubusercontent.com/username/repo_name/main/"

# File to save the merged functions
output_file = ""

with open(output_file, "w") as outfile:
    for file in files:
        url = base_url + file
        response = requests.get(url)
        
        if response.status_code == 200:
            outfile.write(f"% {file}\n")
            outfile.write(response.text)
            outfile.write("\n\n")
        else:
            print(f"Failed to retrieve {file}")

print(f"Functions merged into {output_file}")