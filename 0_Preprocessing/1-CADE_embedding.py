#################################################
#       WHAT YOU NEED IN ORDER TO EXECUTE       #
#       THIS SCRIPT:                            #
#       - Just-News/data_safe.csv : see 0th     #
#         script                                #
#       - Just-News/models : empty directory    #
#       - Just-News/corporas : empty directory  #
#################################################
#################################################
#       WHAT IS PRODUCED:                       #
#       - Just-News/models/*.model              #
#         Word2Vec trained embeddings           #
#################################################
import pandas as pd
from cade.cade import CADE
from gensim.models.word2vec import Word2Vec
import os
import numpy as np
from ast import literal_eval
import string
import numpy as np
from numpy.linalg import norm
import codecs
import tqdm
import glob
#function to remove punctuation and upper cases
def simple_preproc(text):
  """
  see: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
  """
  return text.translate(str.maketrans('', '', string.punctuation))

def clean_good(text):
    return (
            simple_preproc(text).lower()
                                .replace("-", "")
                                .replace('"', '')
                                .replace('“', '')
                                .replace('”', '')
                                .replace("'s", '')
                                .replace("’s", '')
                                .replace("—", '')
           )

def main():
    #loading the dataframe
    print("[INFO] Loading data frame Just-News/data_safe.csv")
    with open("./data_safe.csv") as file:
        df = pd.read_csv(file, engine='c')
    df["Paragraphs"] = df["Paragraphs"].apply(literal_eval)

    #creating a file that contains all the text,
    #without punctuation and uppercases
    print("[INFO] Creating Just-News/EVERYTHING.txt")
    EVERYTHING = ""
    for k in df["Paragraphs"]:
        EVERYTHING += " ".join(k)
    EVERYTHING = clean_good(EVERYTHING)
    with open("./EVERYTHING.txt", 'w') as file:
        file.write(EVERYTHING)

    #now training the compass matrix
    print("[INFO] Training compass matrix")
    aligner = CADE(size=100, workers=8)
    aligner.train_compass("./EVERYTHING.txt", overwrite=False)

    #creating a .txt file for each slice
    #without lower cases and punctuation
    print("[INFO] creating a .txt file for each newspaper")
    for i, Newssite in enumerate(df["Newssite"].unique()):
        print("[INFO] Progress:"+str(i+1)+"/"+str(len(df["Newssite"].unique())))
        newssite_to_text = ""
        for k in df[df["Newssite"] == Newssite]["Paragraphs"][:]:
            newssite_to_text += " ".join(k)

        newssite_to_text = clean_good(newssite_to_text)
        with open("./corpora/text_"+str(Newssite)+".txt", 'w') as file:
            file.write(newssite_to_text)

    #training slices and saving models in dictionary
    print("[INFO] Training slices...")
    slices = {
              Newssite: aligner.train_slice("./corpora/text_"+str(Newssite)+".txt", save=False)
              for Newssite in df["Newssite"].unique()
             }

    #saving models in Just-News/models
    print("[INFO] saving trained models in JustNews/models")
    for my_slice in slices:
        slices[my_slice].save("models/"+str(my_slice)+".model")

    #now removing all the unusefpul files
    for filename in glob.glob('./EVERYTHING.txt'):
        os.remove(filename)
    os.remove("./model/log.txt")
    os.rmdir("./model")



if __name__ == "__main__":
    main()
