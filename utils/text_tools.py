import os
import json
import bz2
import csv
import glob
import pandas as pd
from math import ceil
from random import randint, choices
import string
from p_tqdm import p_map

def append_all_txts(in_path, out_path):
  """
  Appends all txts in a folder into one big txt separated by two line breaks
  """

  if out_path.endswith('.txt'):
    all = open(out_path, 'x')
  else:
    all = open(out_path+'.txt', 'x')
  
  for fp in os.listdir(in_path):
    if fp.endswith(".txt"):
        fp=os.path.join(in_path, fp)
        txt=open(fp, 'r').read()
        all.write(txt + '\n \n')

def extract_articles(
    in_path: str, out_path: str, publication="", chunksize=500000):
  """
  Extracts all articles from a certain news publisher.

  Extracts from every publisher if publisher==""
  You may run out of memory if you attempt to do that

  Outputs a single-column csv containing 1 article per cell
  Returns pandas.DataFrame of that csv

  Designed to work with Andrew Thompson's dataset at: 
  https://components.one/datasets/all-the-news-2-news-articles-dataset/

  params:
  chunksize: chunk size for parsing csv
  """
  # Type-checking and such
  assert type(publication)==str
  if not in_path.endswith(".csv"):
    raise TypeError("Input file is not csv or does not have .csv extension")
  if not out_path.endswith(".csv"):
    raise TypeError("Output must have .csv extension")

  # Parses the csv in chunks to limit memory usage
  iter_csv = pd.read_csv(
    in_path, usecols=["publication", "article"],
     iterator=True, chunksize=chunksize)
  df = pd.concat(
      [chunk[chunk['publication']==publication] for chunk in iter_csv])

  df.drop(["publication"], axis="columns", inplace=True) # Remove publication column
  df.to_csv(out_path, header=False, index=False) # Save as .csv
  return df

def split_csv(in_path: str, out_path="", no_of_files=5):
  """
  splits a large csv into multiple smaller csv's

  params:
  
  in_path: path and name of csv, i.e., data/things.csv
  out_path: path to output directory output/things/
  no_of_files: target number of csv's to split into

  returns: list of pandas.Dataframe objects
  """
  df = pd.read_csv(in_path, header=None)

  rows_per_file = ceil(len(df.index)/no_of_files)

  all = []
  for idx in range(no_of_files-1):
    all.append(df[idx*rows_per_file:(idx+1)*rows_per_file])

  all.append(df[(no_of_files-1)*rows_per_file:])

  if not os.path.isdir(out_path):
    os.mkdir(out_path)

  out_path = os.path.join(out_path, os.path.splitext(os.path.basename("drive/MyDrive/SSTP/data/CNN.csv"))[0])
  for idx, element in enumerate(all):
    element.to_csv(f"{out_path}_{idx:03}.csv", header=False, index=False)

  return all

def create_test_csv(fp: str, length = 100):

  data = []
  for idx in range(length):
    entry_length = randint(0,200)
    rand_text = ''.join(choices(string.ascii_uppercase + string.digits, k=entry_length))
    entry = f"{idx:04d}, {(entry_length+39):03d} characters (including this): " + rand_text
    data.append(entry)

  pd.DataFrame(data).to_csv(fp, header=False, index=False)
  