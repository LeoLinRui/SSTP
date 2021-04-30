import os
import re
import csv
import numpy as np

from langdetect import detect
from p_tqdm import p_map
import tqdm
import multiprocessing


def filter_by_length(work_dir:str, save_dir:str, limit=50):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def process(tweets:list):
        content = []
        for tweet in tweets: # remove tqdm if doing multiprocessing
            if len(tweet) >= limit:
                content.append(tweet)
        return content

    with open(work_dir, encoding='utf-8') as f:
        reader = csv.reader(f)
        tweets = list(reader)[0]

    
    # use this for multiprocessing
    tweets = list(chunks(tweets, int(len(tweets)/100)))
    output = p_map(process, tweets, num_cpus=multiprocessing.cpu_count(), desc="Filtering")
    print("Recombining tweets list")
    output = sum(output, [])
    

    # output = process(tweets)
 
    with open(os.path.join(save_dir), 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        for tweet in tqdm.tqdm(output, desc="Saving csv", unit="rows"):
            writer.writerow([tweet])

if __name__ == "__main__":
    filter_by_length(
        work_dir=input("Path to source .csv:"),
        save_dir=input("Path to destination .csv:"), 
        limit=input("Length (lower) limit: "))