import helper # own python file containing helper functions
import argparse
from scipy.stats import spearmanr
import os
import matplotlib.pyplot as plt

# CORRELATION MEASUREMENT ----------------------------
def get_correlation(values0,values1):
    """
    Calculate the Spearman correlation coefficient for two lists of values.
    Parameters:
        values0 (list): list of values, i.e. list of compositionality scores
        values1 (list): list of other values, i.e. list of compositionality ratings
    Returns:
        corr,p_val (tuple): Spearman's Rho correlation value (between the two values-lists) along with its associated p-value
    """
    corr,p_val = spearmanr(values0, values1)
    return corr,p_val

# SPECIFIC HELPER FUNCTiONS --------------------------

def extract_sorted_ratings(ratings1:dict,ratings2:dict) -> tuple:
    """
    Takes two dictionaries (structure: {(comp,const):rating,...}) sorts them according to their keys and returns lists containing the ratings.
    Parameters:
        ratings1 (dict): a dictionary containing ratings for (comp-const)-pairs 
        ratings2 (dict): another dictionary containing ratings for (comp-const)-pairs
    Returns:
        ratings1_values,ratings2_values: both are lists of ratings in specific order
    """
    if set(ratings1.keys()) != set(ratings2.keys()):
        print("problem: the comp-const pairs differ for the two ratings", len(set(ratings1.keys())),len(set(ratings2.keys())))
        return [],[]
    sorted_pairs = sorted(ratings1.keys())
    ratings1_values = [ratings1[pair] for pair in sorted_pairs]
    ratings2_values = [ratings2[pair] for pair in sorted_pairs]
    return ratings1_values,ratings2_values


# ANALYZE CORRELATIONS BETWEEN RATINGS ---------------

def analyze_correlation(ratings1_path:str,ratings2_path:str,analysis_dir:str) -> None:
    """
    Look at correlation between two files listing compositionality scores for compound-constituent pairs. 
    Reports and plots the correlation for head-ratings, modifier-ratings and all ratings.
    Parameters:
        ratings1_path (str): Path to file where ratings (or measured scores) are saved
        ratings2_path (str): Path to file where other ratings (or measured scores) are saved
        analysis_dir (str): Directory where results from correlation analysis between ratings1 and ratings2 are saved
    Returns: None (only used for early break if arguments were not passed correctly)
    """
    # get rating-names
    rating1_name = ratings1_path.split("/")[-1].split(".csv")[0]
    rating2_name = ratings2_path.split("/")[-1].split(".csv")[0]
    identifier = rating1_name + " vs " + rating2_name
    path_starter = analysis_dir+identifier+"/"
    # --- handle stupid file name length problems
    if len(path_starter)-len(analysis_dir) > 256:
        path_starter = path_starter.replace("compositionality_ratings","c")
        if len(path_starter)-len(analysis_dir) > 256 and "limited" in path_starter:
            path_starter = path_starter.replace("limited","lim")
        if len(path_starter)-len(analysis_dir) > 256 and "chatgpt_constituent_definitions" in path_starter:
            path_starter = path_starter.replace("chatgpt_constituent_definitions","chatgpt_const_defs")
        if len(path_starter)-len(analysis_dir) > 256 and "chatgpt_compound_definition" in path_starter:
            path_starter = path_starter.replace("chatgpt_compound_definition","chatgpt_comp_def")
        if len(path_starter)-len(analysis_dir) > 256:
            print("can't create a correlation-folder for ",path_starter, " as the name would be too long")
            return None
    # --- end of handling stupid file name length problems
    if os.path.exists(path_starter): # this correlation-value already was calculated
        # print(path_starter," already exists")
        return None
    
    # load ratings from file -> structure: {"heads":{(comp,const):rating,...}, "modifiers":{(comp,const):rating,...}, "all":{(comp,const):rating,...}}
    try:
        ratings1 = helper.load_ratings(ratings1_path)
        ratings2 = helper.load_ratings(ratings2_path)
    except:
        print("couldn't load the ratings from file")
        return None

    # check if both ratings cover the same comp-const-pairs, and extract ratings (sorted wrt. same keys) 
    head_ratings1,head_ratings2 = extract_sorted_ratings(ratings1["heads"],ratings2["heads"])
    mod_ratings1,mod_ratings2 = extract_sorted_ratings(ratings1["modifiers"],ratings2["modifiers"])
    all_ratings1,all_ratings2 = extract_sorted_ratings(ratings1["all"],ratings2["all"])
    if not all([len(rat)>0 for rat in [head_ratings1,head_ratings2,mod_ratings1,mod_ratings2,all_ratings1,all_ratings2]]):
        print("couldn't extract all ratings")
        return None

    # calculate correlations
    head_corr,head_p = get_correlation(head_ratings1,head_ratings2)
    mod_corr,mod_p = get_correlation(mod_ratings1,mod_ratings2)
    all_corr,all_p = get_correlation(all_ratings1,all_ratings2)
    
    # write correlations to file
    if not os.path.exists(path_starter):
        os.makedirs(path_starter, exist_ok=True) # allow to create directory with parent directory
    with open(path_starter+"correlations.csv","w") as outfile:
        outfile.write("set,corr,p\n")
        outfile.write("heads,"+str(head_corr)+","+str(head_p)+"\n")
        outfile.write("modifiers,"+str(mod_corr)+","+str(mod_p)+"\n")
        outfile.write("all,"+str(all_corr)+","+str(all_p)+"\n")

    return None

# main function for py file
def main():
    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        '-r1',
        '--ratings1_path',
        help='Path to file where ratings (or measured scores) are saved.'
    )
    parsers.add_argument(
        '-r2',
        '--ratings2_path',
        help='Path to file where ratings (or measured scores) are saved.'
    )
    parsers.add_argument(
        '-a',
        '--analysis_dir',
        help='Directory where results from correlation analysis between ratings1 and ratings2 are saved.'
    )
    args = parsers.parse_args()

    analyze_correlation(ratings1_path=args.ratings1_path,ratings2_path=args.ratings2_path,analysis_dir=args.analysis_dir)


if __name__ == "__main__":
    main()