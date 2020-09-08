#########################################################
#       WHAT YOU NEED IN ORDER TO EXECUTE               #
#       THIS SCRIPT:                                    #
#       - Just-News/lexicon/subjectivityLexicon.csv     #
#         .csv containing the subjectivity lexicon      #
#########################################################
#########################################################
#       WHAT IS PRODUCED:                               #
#       Just-News/lexicon/clean_lexicon.csv             #
#       - Clean .csv file containing [Word, Subj_score] #
#########################################################
import pandas as pd
from cade.cade import CADE
from gensim.models.word2vec import Word2Vec
import os
import numpy as np
from ast import literal_eval
import string
from jupyterthemes import jtplot
# currently installed theme will be used to
# set plot style if no arguments provided
jtplot.style()

#function to encode csv file
def encode_subj(string_score = None):
    if string_score == "weaksubj":
        return(-1)
    elif string_score == "strongsubj":
        return(1)
    else:
        print('[ERROR] Some problems occurd.')
        return(-1)

def main():
    #opening the csv file
    print("[INFO] Processing lexicon")
    with open("./lexicon/subjectivityLexicon.csv", 'r') as file:
        lexicon = pd.read_csv(file, engine='c', sep=';', header=None)
    #renaming columns
    lexicon.columns = ["Word", "Subj_score"]
    #cleaning format
    lexicon["Word"] = lexicon["Word"].apply(lambda x: x.replace("word1=", ""))
    lexicon["Subj_score"] = lexicon["Subj_score"].apply(lambda x:
                                                        x.replace("type=", ""))
    lexicon["Subj_score"] = lexicon["Subj_score"].apply(encode_subj)

    #saving to /Just-News/lexicon/clean_lexicon.csv
    print("[INFO] Saving to /Just-News/lexicon/clean_lexicon.csv")
    with open("./lexicon/clean_lexicon.csv", 'w') as file:
        lexicon.to_csv(file)

if __name__ == "__main__":
    main()
