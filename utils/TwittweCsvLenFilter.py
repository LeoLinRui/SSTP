import os
import re
import csv
import numpy as np

from langdetect import detect
from p_tqdm import p_map
import tqdm


LEN_LIMIT = 100  # This is a lower limit


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process(tweets:list):
    content = []
    for tweet in tqdm.tqdm(tweets): # remove tqdm if doing multiprocessing
        if len(tweet) >= LEN_LIMIT:
            content.append(tweet)
    return content


if __name__ == '__main__':

    #work_dir = input("work directory (file):")
    #save_dir = input("save directory (not file):")

    work_dir = r"C:\Users\Leo's PC\Desktop\JS_Folder\28746tweets.csv"
    save_dir = r"C:\Users\Leo's PC\Desktop\JS_Folder"

    with open(work_dir, encoding='utf-8') as f:
        reader = csv.reader(f)
        tweets = list(reader)[0]

    '''
    use this for multiprocessing
    tweets = list(chunks(tweets, int(len(tweets)/10000)))
    output = p_map(process, tweets, num_cpus=multiprocessing.cpu_count() - 1)
    output = sum(output, [])
    '''

    output = process(tweets)
 
    with open(os.path.join(save_dir, str(len(output)) + "tweets_filtered.csv"), 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(output)