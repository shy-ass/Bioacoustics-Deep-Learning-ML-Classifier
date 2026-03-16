import pandas as pd
import subprocess
import os
import urllib.request

# 1. Download the Balanced Train CSV if you don't have it
csv_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
csv_file = "balanced_train_segments.csv"
if not os.path.exists(csv_file):
    print("Downloading AudioSet balanced train segments...")
    urllib.request.urlretrieve(csv_url, csv_file)

# 2. Paste the dictionary output from the first script here:
target_mids = {
    # Birds
    "Crow": "/m/04s8yn",
    "Owl": "/m/09d5_",
    "Duck": "/m/09ddx",
    "Pigeon, dove": "/m/0h0rv",
    "Chicken, rooster": "/m/09b5t", 
    
    # Domestic
    "Dog": "/m/0bt9lr",
    "Cat": "/m/01yrx",
    "Cow (Cattle)": "/m/01xq0k1",
    "Pig": "/m/068zj",
    "Horse": "/m/03k3r",
    
    # Wild
    "Lion/Tiger (Roaring cats)": "/m/0cdnk",
    "Snake": "/m/078jl",
    "Frog": "/m/09ld4",
    "Whale": "/m/032n05",
    "Wolf (Howl)": "/m/07qf0zm"
}

# Flip the dictionary for easy lookup: { '/m/0bt9lr': 'Dog' }
mid_to_name = {v: k for k, v in target_mids.items()}

# 3. Read the CSV (Google adds a few comment lines at the top, so we skip them)
df = pd.read_csv(csv_file, skiprows=2, skipinitialspace=True)

# AudioSet formats the labels as a string: "/m/01yrx,/m/0bt9lr". We need to find rows containing our MIDs.
os.makedirs("dataset", exist_ok=True)

def download_clip(yt_id, start_time, end_time, animal_name):
    # Create a folder for each animal
    save_dir = os.path.join("dataset", animal_name)
    os.makedirs(save_dir, exist_ok=True)
    
    output_filename = os.path.join(save_dir, f"{yt_id}_{start_time}.wav")
    
    if os.path.exists(output_filename):
        return True # Skip if already downloaded

    url = f"https://www.youtube.com/watch?v={yt_id}"
    
    # yt-dlp command to extract specifically the 10-second chunk as a WAV file
    command = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--download-sections", f"*{start_time}-{end_time}",
        "--force-keyframes-at-cuts",
        "-o", output_filename,
        url
    ]
    
    try:
        # We suppress output to keep the terminal clean, but it will take time!
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Success: {animal_name} - {yt_id}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed (Video likely deleted): {url}")
        return False

# 4. Loop through the dataset and download
print("Starting extraction... (This will take a while, grab a coffee!)")
for index, row in df.iterrows():
    labels = str(row['positive_labels'])
    
    # Check if any of our target MIDs are in this row's labels
    for mid, animal_name in mid_to_name.items():
        if mid in labels:
            download_clip(row['# YTID'], row['start_seconds'], row['end_seconds'], animal_name)
            break # Move to the next row once we find a match
