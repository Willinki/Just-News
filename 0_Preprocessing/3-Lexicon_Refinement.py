
#########################################################
#       WHAT YOU NEED IN ORDER TO EXECUTE               #
#       THIS SCRIPT:                                    #
#       - Just-News/lexicon/clean_lexicon.csv           #
#         .csv containing the subjectivity lexicon      #
#########################################################
#########################################################
#       WHAT IS PRODUCED:                               #
#       Just-News/lexicon/enriched_lexicon.csv          #
#       The refined and augmented lexicon               #
#########################################################
import pandas as pd
from cade.cade import CADE
from gensim.models.word2vec import Word2Vec
import os
import glob
import numpy as np
import string
from jupyterthemes import jtplot
# currently installed theme will be used to
import nicoli_utils 
# set plot style if no arguments provided
jtplot.style()

def main():
    # loading clean_lexicon
    print("[INFO] Loading Just-News/lexicon/clean_lexicon.csv")
    with open("./lexicon/clean_lexicon.csv", 'r') as file:
        lexicon = pd.read_csv(file, index_col=1)
    lexicon = lexicon.drop("Unnamed: 0", axis=1)
    lexicon = lexicon.to_dict()["Subj_score"]

    # loading models
    print("[INFO] Loading Just-News/lexicon/clean_lexicon.csv")
    slices = {filename.split('/')[-1].replace(".model", ""): 
              Word2Vec.load(filename)
              for filename in glob.glob('./models/*.model')}
    print("[WARNING] remember to set up models and corpora correctly")
    models_test = [slices["New York Times"], slices["Breitbart"]]
    corpora_test = ["./corporas/text_New York Times.txt", "./corporas/text_Breitbart.txt"]
    
    #refining lexicon
    print("[INFO] Applying lexicon refinement")
    lexicon_refined = nicoli_utils.lexicon_refinement(lex = lexicon, 
                                                      models = models_test, 
                                                      corpora = corpora_test, 
                                                      zipf_cutoff=5)

    #enriching lexicon
    print("[INFO] Applying lexicon augmentation")
    vectorized_lexicon, lexicon_labels = nicoli_utils.enrich(lex = lexicon_refined, 
                                                             models = models_test, 
                                                             n_target = 500, 
                                                             msteps = 200, 
                                                             return_words = False)
    
    #saving the enriched lexicon
    print("[INFO] Saving the enriched lexicon at Just-News/lexicons/enriched_lexicon.csv")
    print("\t Column names: Vectorized_words, Labels")
    with open("./lexicon/enriched_lexicon.csv", 'w') as file:
        pd.DataFrame({
                      "Vectorized_words": vectorized_lexicon.tolist(), 
                      "Labels": lexicon_labels
                     }).to_csv(file)

if __name__ == "__main__":
    main()