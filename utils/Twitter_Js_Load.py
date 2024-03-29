import json
import os
import re
import bz2
import csv
import pandas as pd

from langdetect import detect
from p_tqdm import p_map
import multiprocessing


ENGLISH_ONLY = True
NO_RT = True
LEN_LIMIT = 100  # This is a lower limit
REMOVE_EMOJI = True
UNFINISHED_LIMIT = True


def checkEng(text:str):
    if ENGLISH_ONLY:
        return detect(text) == 'en'
    else:
        return True


def checkRT(text:str):
    if NO_RT:
        return not text.startswith('RT')
    else:
        return True


def checkLen(text:str):
    if LEN_LIMIT is not None:
        return len(text) >= LEN_LIMIT
    else:
        return True


def checkLUnfinished(text:str):
    if UNFINISHED_LIMIT:
        return not text.endswith("…") and "… http" not in text
    else:
        return True


def removeEmojis(data):

    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)

    return re.sub(emoj, '', data)


def process(filedir:str):
    # print("processing", filedir)
    content = []
    
    if filedir.endswith(".bz2"):

        with bz2.open(filedir) as f:

            for line in f:
                tweet = json.loads(line)
                try:
                    tweet = tweet["text"]

                    try:
                        if checkLen(tweet) and checkRT(tweet) and checkLUnfinished(tweet) and checkEng(tweet):
                            if REMOVE_EMOJI:
                                tweet = removeEmojis(tweet)

                            content.append(tweet)
                    except: # handle No language feature error
                        pass
                    
                except KeyError:
                    pass     
        return content


if __name__ == '__main__':

    work_dir = input("work directory (source)")
    save_dir = input("save directory (not file):")

    dir_list = []

    for root, _, filenames in os.walk(work_dir):
        for filename in filenames:
            if filename.endswith("bz2"):
                dir_list.append(os.path.join(root, filename))

    print(len(dir_list), "files found")


    tweet_list = p_map(process, dir_list, num_cpus=multiprocessing.cpu_count() - 1) # Leave 1 thread for other processes
    tweet_list = sum(tweet_list, [])
    

    print(len(tweet_list), 'Tweets gathered\n\n Samples:\n')
    for tweet in tweet_list[:5]:
        print(tweet)

    
    '''
    with open(os.path.join(save_dir, str(len(tweet_list)) + "tweets.csv"), 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter="=")
        for tweet in tweet_list:
            writer.writerow([tweet])
    '''

    df = pd.DataFrame(tweet_list, columns=['tweet'])
    df.to_csv(os.path.join(save_dir, str(len(tweet_list)) + "tweets.csv"), index=False)