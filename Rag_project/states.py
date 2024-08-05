import os
import requests

# List of all US states
states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
          "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", 
          "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", 
          "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
          "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", 
          "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

# Create the wiki directory if it doesn't exist
output_dir = "/Users/richikghosh/Documents/Rag_project/wiki"
os.makedirs(output_dir, exist_ok=True)
print("Created directory wiki")

# Loop through each state and download its Wikipedia page
for state in states:
    formatted_state = state.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{formatted_state}"
    print(f"Downloading {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        with open(os.path.join(output_dir, f"{formatted_state}.txt"), 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"Successfully downloaded {state}")
    except requests.RequestException as e:
        print(f"Failed to download {state}. Error: {e}")

print("Download complete!")
