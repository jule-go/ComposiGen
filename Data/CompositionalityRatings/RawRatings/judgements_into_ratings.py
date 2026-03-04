import pandas as pd
import os
import statistics

path_to_ratings = "/path/to/repo/CompositionalityRatings/"


# ----- helper functions -----

def get_result_from_file(filename:str) -> pd.DataFrame:
    """
    Returns the content (= annotation result) of a csv-file as a dataframe with the help of pandas.
    Parameters: 
        filename (str): path of the csv file that is used for creating the dataframe
    Returns: 
        _: dataframe containing the content from the file
    """
    return pd.read_csv(filename)

def combine_results_to_df(directory:str) -> pd.DataFrame:
    """
    Takes csv files in a directory, converts them into separate dataframes and then merges them into one dataframe.
    Parameters:
        directory (str): path to folder that contains csv-files whose content should be returned within a dataframe
    Returns: 
        _: dataframe containing the content from all csv-files in directory
    """
    files = os.listdir(directory)
    dfs = [get_result_from_file(directory+filename) for filename in files if filename.startswith("form")]
    result = pd.concat(dfs)
    result = result.reset_index(drop=True)
    return result

# function that given a list of counts, looks at different statistical measures, further information: see https://realpython.com/python-statistics/

def analyze_list(what_to_analyze:str,counts:list):
    '''
    Given some list of integers return values of some measures in a dictionary.
    Parameters:
        what_to_analyze (str): States what the counts represent
        counts (list): List of integers
    Returns: dict with following keys: "mean", "median", "mode", "variance", "std-dev", "max", "min", "range"
    '''
    if len(counts)==0: # list of counts doesn't contain any data point
        print("Trying to analyze ",what_to_analyze," went wrong as the counts-list is empty.")
        return {"information":"---","mean":"---","median":"---","mode":"---","variance":"---","std-dev":"---","max":"---","min":"---","range":"---"}
    
    elif len(counts)==1: # list contains exactly one data point
        print("Trying to analyze ",what_to_analyze," went wrong as the counts-list only contains one data point: ",counts)
        return {"information":"---","mean":"---","median":"---","mode":"---","variance":"---","std-dev":"---","max":"---","min":"---","range":"---"}

    # some measures of central tendency
    list_mean = statistics.mean(counts) # sample (arithmetic) mean or average; = arithmetic average of all items in dataset
    list_median = statistics.median(counts) # sample median; = middle element of a sorted dataset
    list_mode = statistics.multimode(counts) # sample mode; = value in the dataset that occurs most frequently; use multimode as there might be more than one modal value

    # some measures of variability
    list_variance = statistics.variance(counts) # sample variance; = shows numerically how far the data points are from the mean
    list_standard_deviation = statistics.stdev(counts) # = positive square root of sample variance
    list_max_val = max(counts)
    list_min_val = min(counts)
    list_range = list_max_val - list_min_val # = difference between maximum and minimum element in the dataset

    # collect information (e.g. to be printed later)
    information = "\t{:<15} \tAverage: {:<10} \tMin. value: {:<10} \tMax. value: {:<10} \tStd. deviation: {:<10} \tMedian: {:<10} \tModus: {:<10}\n".format(what_to_analyze+":",round(list_mean,3),list_min_val,list_max_val,round(list_standard_deviation,3),round(list_median,3),', '.join(str(m) for m in list_mode))
    
    return {"information":information,"mean":list_mean,"median":list_median,"mode":list_mode, "variance":list_variance, "std-dev":list_standard_deviation, "max":list_max_val, "min":list_min_val, "range":list_range}

# -----

# merge all studies into one dataframe
judgements = combine_results_to_df(path_to_ratings+"RawRatings/")
print("amount of judgements: ",len(judgements.index)," [check: 200 compounds * 15 annotators * 2 constituents to rate for = 6000 judgements]")

ratings = dict() # format: keys = (compound,constituent), {"all": [rating1, rating2, ..., rating15], "mean": val, "stdev": val, ...}

# bring the annotations into better readable format
for rating_id in judgements.index:
    worker_id = judgements["workerID"][rating_id]
    compound = judgements["compound"][rating_id]
    constituent = judgements["const"][rating_id]
    rating = judgements["judgement"][rating_id]
    if (compound,constituent) in ratings:
        ratings[(compound,constituent)]["all"] += [rating]
    else:
        ratings[(compound,constituent)] = {"all": [rating]}
print(ratings)

# calculate mean and stdev
for pair,info in ratings.items():
    temp_ratings = [float(val) for val in info["all"]]
    analysis = analyze_list("compound ratings",temp_ratings)
    info["mean"] = analysis["mean"]
    info["stdev"] = analysis["std-dev"]
    info["median"] = analysis["median"]
    info["mode"] = analysis["mode"]
print(ratings)

# write merged ratings to file
with open(path_to_ratings+"compositionality_ratings.csv","w") as outfile:
    outfile.write("compound,const,mean,stand_dev\n")
    for pair,info in ratings.items():
        outfile.write(pair[0]+","+pair[1]+","+str(info["mean"])+","+str(info["stdev"])+"\n")