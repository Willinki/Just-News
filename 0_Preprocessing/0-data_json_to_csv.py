#################################################
#       WHAT YOU NEED IN ORDER TO EXECUTE       #
#       THIS SCRIPT:                            #
#       - Just-News/data_final : directory      #
#         containing json files. One for each   #
#         newspaper                             #
#################################################
#################################################
#       WHAT IS PRODUCED:                       #
#       Just-News/data_final.csv                #
#       a dataframe containing all the          #
#       articles and information                #
#################################################
import pandas as pd
import glob
import os
import json
import numpy as np
import string
from jupyterthemes import jtplot
# currently installed theme will be used to
# set plot style if no arguments provided
jtplot.style()

#function to extract newspaper from link
def create_newssite(link = None):
        if link is np.nan:
            return("Wikipedia")
        elif "slate.com" in link:
            return("Slate")
        elif "https://www.nytimes.com" in link:
            return("New York Times")
        elif "https://www.breitbart.com" in link:
            return("Breitbart")
        elif "https://www.cnn.com" in link:
            return("CNN")
        elif "abcnews" in link:
            return("ABC News")
        elif "https://thefederalist.com" in link:
            return("The Federalist")
        elif "https://www.newsmax.com" in link:
            return("News Max")
        else:
            return("Unknown")

def main():
    #first we create the csv containing all the data about
    #the articles
    print("[INFO] creating .csv file")
    df_list = pd.DataFrame()
    for filename in glob.glob('./data_final/*.json'):
        with open(filename, 'r') as f:
            json_load = json.loads(f.read())
            df_list = df_list.append(pd.DataFrame.from_records(json_load, index="_id"), 
                                     ignore_index = True)
    
    #we remove unnecessary columns
    print("[INFO] manipulating dataFrame")
    df_list = df_list[['Title', 
                       "Date", 
                       "Link", 
                       "Paragraphs", 
                       "Authors"]]
    
    #we apply the 'create_newssite' function and add a column
    df_list["Newssite"] = df_list["Link"].apply(create_newssite)
    
    #we save the dataFrame in "../data_safe.csv"
    print("[INFO] Saving in ---> Just-News/data_safe.csv")
    with open("./data_safe.csv", 'w') as file:
        df_list.to_csv(file)

if __name__ == "__main__":
    main()

    