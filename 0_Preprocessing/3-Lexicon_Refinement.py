
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
# currently installed theme will be used to
from utils import *

def main():
        # Caricamento del lexicon
    print("[INFO] Loading cleaned lexicon")
    with open("./lexicon/clean_lexicon.csv", 'r') as file:
        lexicon = pd.read_csv(file, index_col=1)

    print("[INFO] Loading trained embeddings")
    # Caricamento dei modelli già addestrati
    slices = {filename.split('/')[-1].replace(".model", ""):
              Word2Vec.load(filename)
              for filename in glob.glob('./models/*.model')}

    lexicon = lexicon.drop("Unnamed: 0", axis=1).to_dict()["Subj_score"]

    # Lessico che si ottiene raffinando su tutti
    # gli embedding generati
    print("[INFO] Lexicon refinement")
    lexicon_refined = lexicon_refinement(lex = lexicon,
                                        models = [slices[sli]
                                                 for sli in slices],
                                        corpora = ["./corpora/text_"+str(sli)+".txt"
                                                    for sli in slices],
                                        zipf_cutoff=5)

    print("[INFO] Saving lexicon refined to Just-News/lexicon/lexicon_refined.csv")
    with open("./lexicon/lexicon_refined.csv", 'w') as file:
        pd.DataFrame(lexicon_refined).to_csv(file)

    print("[INFO] Data augmentation on lexicon")
    vectorized_lexicon, lexicon_labels, words = enrich(lex = lexicon_refined,
                                                       models = [slices[sli]
                                                                for sli in slices],
                                                       n_target = 300,
                                                       msteps = 200,
                                                       return_words = True)

    print("[INFO] Saving augmented data to Just-News/lexicon/enriched_lexicon.csv")
    with open("./lexicon/enriched_lexicon.csv", 'w') as file:
        pd.DataFrame({"Vectorized_words": vectorized_lexicon.tolist(),
                  "Labels": lexicon_labels, "Words": words}).to_csv(file)

if __name__ == "__main__":
    main()
