import requests
from tqdm import tqdm
from os.path import join

def download_files(downloads, location):
    """
    Downloads every file from the downloads dict and saves to location
    downloads = {
    "twitter-stream-2021-01-01.zip": "https://archive.org/download/archiveteam-twitter-stream-2021-01/twitter-stream-2021-01-01.zip",
    "twitter-stream-2021-01-02.zip": "https://archive.org/download/archiveteam-twitter-stream-2021-01/twitter-stream-2021-01-02.zip",
    "twitter-stream-2021-01-03.zip": "https://archive.org/download/archiveteam-twitter-stream-2021-01/twitter-stream-2021-01-03.zip",
    "twitter-stream-2021-01-04.zip": "https://archive.org/download/archiveteam-twitter-stream-2021-01/twitter-stream-2021-01-04.zip",
    "twitter-stream-2021-01-05.zip": "https://archive.org/download/archiveteam-twitter-stream-2021-01/twitter-stream-2021-01-05.zip",
    }
    location = "G:\My Drive\SSTP\data"
    download_files(downloads, location)
    """
    
    for key in downloads.keys():
        url = downloads[key]
        print(f"Downloading {key} from {url}")
        

        response = requests.get(url, stream=True)
        total_size_in_MiB= int(response.headers.get('content-length'))/8388608
        block_size = 8388608 #1 Mebibyte
        
        with open(join(location, key), 'wb') as file:
            for data in tqdm(response.iter_content(block_size), 
                            total=total_size_in_MiB, 
                            unit='MiB', unit_scale=True,
                            position=0, leave=True
                            ):
                file.write(data)

