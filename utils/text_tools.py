import os
import pandas as pd

def extract_articles(in_path: str, out_path: str, publication="", chunksize=500000):
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
    "drive/MyDrive/SSTP/data/news.csv", usecols=["publication", "article"],
     iterator=True, chunksize=chunksize)
  df = pd.concat(
      [chunk[chunk['publication']==publication] for chunk in iter_csv])

  df.drop("publication", axis="columns", inplace=True) # Remove publication column
  df.to_csv(out_path, header=False, index=False) # Save as .csv
  return df

# I suppose the module name is a misnomer since it only contains one method
