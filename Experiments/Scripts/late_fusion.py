import argparse
import helper # own python file containing helper functions
import os

cache_directory = "/path/to/cache/" # TODO need to adapt it
alpha_values = [round(i*0.1,2) for i in range(11)] # set weighting factors: 0.0, 0.1, 0.2, ..., 1.0

# LATE FUSION ----------------------------------------
# (= combine predictions of different modalities into multimodal prediction)

def combine_scores(score_t,score_v,alpha) -> float:
    """
    Takes two different scores and combines them into one score using a scaling factor.
    Parameters:
        score_t: one score (e.g. from text modality)
        score_v: other score (e.g. from vision modality)
        alpha: scaling factor 
    Returns: 
        combined score (float)
    """
    # make sure the values are of type float!
    score_t = float(score_t)
    score_v = float(score_v)
    alpha = float(alpha)
    # calculate and return combined score
    return (alpha * score_t) + ((1-alpha)*score_v)

def extract_sorted_ratings_with_pairs(ratings1:dict,ratings2:dict) -> tuple:
    """
    Takes two dictionaries (structure: {(comp,const):rating,...}) sorts them according to their keys and returns lists containing the ratings and the pairs.
    Parameters:
        ratings1 (dict): a dictionary containing ratings for (comp-const)-pairs 
        ratings2 (dict): another dictionary containing ratings for (comp-const)-pairs
    Returns:
        ratings1_values,ratings2_values,pairs: first two are lists of ratings in specific order, last contains the matching pairs in order
    """
    if set(ratings1.keys()) != set(ratings2.keys()):
        print("problem: the comp-const pairs differ for the two ratings", len(set(ratings1.keys())),len(set(ratings2.keys())))
        return [],[]
    sorted_pairs = sorted(ratings1.keys())
    ratings1_values = [ratings1[pair] for pair in sorted_pairs]
    ratings2_values = [ratings2[pair] for pair in sorted_pairs]
    return ratings1_values,ratings2_values, sorted_pairs

def perform_late_fusion(compound_file,scores_t_path,scores_v_path,output_dir):
    """
    Take compositionality scores from two files and combine them. This is done for head- and modifier ratings separately.
    Parameters:
        compound_file (str): CSV file with one compound per line for which we want to combine the scores from different modalities
        scores_t_path (str): Path to file where (text-based) ratings (or measured scores) are saved
        scores_v_path (str): Path to file where (vision-based) ratings (or measured scores) are saved
        output_dir (str): Path to directory where combined scores are saved
    Returns: None (only used for early break if arguments were not passed correctly)
    """
    # load targets (= compounds)
    if os.path.exists(compound_file): 
        csv_data = helper.load_csv(compound_file)
        targets = set() # use set as we don't want to extract duplicate embeddings
        for target in csv_data:
            targets.add(target["noun_compound"])
        print("csv file is processed")
    else: 
        print(compound_file, " is not a valid file")
        return None

    # get rating-names
    text_scores_name = scores_t_path.split("/")[-1].split(".csv")[0]
    vision_scores_name = scores_v_path.split("/")[-1].split(".csv")[0]
    identifier = text_scores_name + " + " + vision_scores_name
    path_starter = output_dir+identifier+"/"
    # --- handle stupid file name length problems
    if len(path_starter)-len(output_dir) > 256:
        path_starter = path_starter.replace("compositionality_ratings","c")
        if len(path_starter)-len(output_dir) > 256 and "limited" in path_starter:
            path_starter = path_starter.replace("limited","lim")
        if len(path_starter)-len(output_dir) > 256 and "chatgpt_constituent_definitions" in path_starter:
            path_starter = path_starter.replace("chatgpt_constituent_definitions","chatgpt_const_defs")
        if len(path_starter)-len(output_dir) > 256 and "chatgpt_compound_definition" in path_starter:
            path_starter = path_starter.replace("chatgpt_compound_definition","chatgpt_comp_def")
        if len(path_starter)-len(output_dir) > 256:
            print("can't create a correlation-folder for ",path_starter, " as the name would be too long")
            return None
    print("file directory: ",path_starter)
    # --- end of handling stupid file name length problems
    if not os.path.exists(path_starter): # need to create the directory
        os.makedirs(path_starter, exist_ok=True) # allow to create directory with parent directory
    # if os.path.exists(path_starter): 
        # print(path_starter," already exists -> no further combination is calculated")
        # return None
    
    # load ratings from file -> structure: {"heads":{(comp,const):rating,...}, "modifiers":{(comp,const):rating,...}, "all":{(comp,const):rating,...}}
    try:
        text_scores = helper.load_ratings(scores_t_path)
        vision_scores = helper.load_ratings(scores_v_path)
    except:
        print("couldn't load the ratings from file")
        return None

    # check if both ratings cover the same comp-const-pairs, and extract ratings (sorted wrt. same keys) 
    head_ratings_text,head_ratings_vision,head_pairs = extract_sorted_ratings_with_pairs(text_scores["heads"],vision_scores["heads"])
    mod_ratings_text,mod_ratings_vision,mod_pairs = extract_sorted_ratings_with_pairs(text_scores["modifiers"],vision_scores["modifiers"])
    if not all([len(rat)>0 for rat in [head_ratings_text,head_ratings_vision,mod_ratings_text,mod_ratings_vision,head_pairs,mod_pairs]]): 
        print("couldn't extract all ratings")
        return None

    # calculate combined scores
    for alpha in alpha_values:
        out_file_path = path_starter + str(alpha) + ".csv"
        if os.path.exists(out_file_path): # we already calculated this score earlier
            return None 
        
        combined_scores = {comp:{"modifier":None,"head":None} for comp in targets}
        for pair, t_score, v_score in zip(head_pairs,head_ratings_text,head_ratings_vision):
            combined_scores[pair[0]]["head"] = combine_scores(t_score,v_score,alpha)
        for pair, t_score, v_score in zip(mod_pairs,mod_ratings_text,mod_ratings_vision):
            combined_scores[pair[0]]["modifier"] = combine_scores(t_score,v_score,alpha)
        
        # write scores to file
        outfile = open(out_file_path,"w")
        outfile.write("compound,modifier,modifier_score,head,head_score\n") # header line
        for comp,scores in combined_scores.items():
            modifier,head = comp.split()
            outfile.write(comp+","+modifier+","+str(combined_scores[comp]["modifier"])+","+head+","+str(combined_scores[comp]["head"])+"\n")
        outfile.close()

        print("saved all compositionality scores for alpha = ",str(alpha)," to file")

    return None

# main function for py file
def main():

    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        '-c',
        '--compound_file',
        help='Link to file with one compound per line.'
    )
    parsers.add_argument(
        '-st',
        '--scores_text',
        help='Path to csv file where text-based compositionality predictions are saved.'
    )
    parsers.add_argument(
        '-sv',
        '--scores_vision',
        help='Path to csv file where vision-based compositionality predictions are saved.'
    )
    parsers.add_argument(
        '-o',
        '--output_dir',
        help='Directory where combined scores based on a combination of scores_text and scores_vision are saved.'
    )
    args = parsers.parse_args()

    perform_late_fusion(compound_file=args.compound_file,scores_t_path=args.scores_text,scores_v_path=args.scores_vision,output_dir=args.output_dir)

if __name__ == "__main__":
    main()
