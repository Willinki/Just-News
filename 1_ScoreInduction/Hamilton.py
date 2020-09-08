# General purpose
import pandas as pd
import os
import glob
import json
import numpy as np
import string
import matplotlib.pyplot as plt
from datetime import datetime
from ast import literal_eval
#word2vec
from gensim.models.word2vec import Word2Vec
#sklearn related
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

import embedding
import polarity_induction_methods as pi

def propagate(model = None, positive_seed = None, negative_seed = None, name = "Unknown"):
    print("[INFO] Model name:", name)
    return(pi.random_walk(embedding.Embedding(model.wv.vectors,
                                              list(model.wv.vocab.keys())),
                                      positive_seed,
                                      negative_seed))


def main():
    #setting the correct working directory
    # os.chdir("./1_ScoreInduction")

    # Open the pre-prepared lexicon
    with open("./lexicon/lexicon_refined.csv", 'r') as file:
        lexicon_refined = pd.read_csv(file, engine='c')

    # change column name
    lexicon_refined = lexicon_refined.rename(columns={"Unnamed: 0": "Words"})

    #loading models and selecting
    slices = {
              filename.split('/')[-1].replace(".model", "") :
              Word2Vec.load(filename)
              for filename in glob.glob('./models/*.model')
             }
    print("[INFO] Models loaded")
    ###########################################
    # DEFINING POSITIVE-NEGATIVE SEEDS        #
    ###########################################
    positive_seed = list(lexicon_refined[lexicon_refined["Valence"] == 1]["Words"])
    negative_seed = list(lexicon_refined[lexicon_refined["Valence"] == -1]["Words"])

    propagations = {sl: propagate(model = slices[sl], positive_seed = positive_seed, negative_seed = negative_seed, name = sl)
                  for sl in slices}

    for name in slices:
        with open("../propagations/propagation_"+str(name)+".csv", 'w') as file:
            pd.DataFrame({
                         "Words": [word
                                    for word in propagations[name]],
                         "Labels": [propagations[name][word]
                                    for word in propagations[name]]
                         }).to_csv(file)

if __name__ == "__main__":
    main()
