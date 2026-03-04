import torch
import argparse
import helper # own python file containing helper functions
import os
from tqdm import tqdm
import json

embeddings_path_template = "{input_dir}/{target}.pt" # TODO change if you use a different structure for saving the embeddings

# SIMILARITY MEASUREMENT -----------------------------

def get_cosine_similarity(vector1:torch.Tensor, vector2:torch.Tensor, dim:int = 0):
    """
    Calculate the cosine similarity between two vectors.
    Parameters:
        vector1: vector1 we want to use for the similarity measurement
        vector2: vector2 we want to use for the similarity measurement
    Returns:
        _: cosine similarity (value)
    """
    return torch.cosine_similarity(vector1, vector2, dim=dim).item()

# CALCULATE COMPOSITIONALITY SCORES ------------------

def calculate_scores(compound_file:str,constituent_dir:str,compound_dir:str,output_dir:str,compound_limitation:str=None) -> None:
    """
    Calculate the cosine similarity between compound-constituent embeddings, which will serve as compositionality scores.
    Parameters:
        compound_file (str): Path to csv file where compounds are saved
        compound_dir (str): A directory where the embeddings (= feature vectors) for the compounds are stored, between which the cosine similarity will be calculated wrt. to the constituent vectors
        constituent_dir (str): A directory where the embeddings (= feature vectors) for the constituens are stored, between which the cosine similarity will be calculated wrt. to the compounds
        output_dir (str): Directory where similarity scores will be stored
        compound_limitation (str): (optional) Specifies which image-specific embeddings should be used for measurements. E.g. only compounds generated with denoising strength 0.9 should be considered when using "_FROM_0.9_", or specify path to txt file that contains images to consider
        # c_format (str): Format of compound, either "underscore", "hyphenated", "open", or "closed"
    Return: None (only used for early break if arguments were not passed correctly)
    """
    # combine the filenames appropriately!
    constituent_identifier = constituent_dir.split("Embeddings/")[1].strip("/").replace("/","_")
    compound_identifier = compound_dir.split("Embeddings/")[1].strip("/").replace("/","_")
    out_file_path = output_dir + constituent_identifier + "__" + compound_identifier 
    if compound_limitation:
        if compound_limitation.endswith(".json"):
            limitation_identifier = "_"+compound_limitation.replace(".json","").split("/")[-1]
            # load to dict: key = compound, value = image to be considered
            with open(compound_limitation) as loader_file: 
                limitation_file = json.load(loader_file)
        else: 
            limitation_identifier = compound_limitation
        out_file_path += "__limited" + limitation_identifier
    out_file_path += ".csv"
    # --- handle stupid file name length problems
    if len(out_file_path)-len(output_dir) >= 256:
        if "clip-concatenation" in out_file_path:
            out_file_path = out_file_path.replace("clip-concatenation","clip-concat")
        if "flava-average-last" in out_file_path:
            out_file_path = out_file_path.replace("flava-average-last","flava-average-la")
        if len(out_file_path)-len(output_dir) >= 256 and "limited" in out_file_path:
            out_file_path = out_file_path.replace("limited","lim")
        if len(out_file_path)-len(output_dir) >= 256 and "chatgpt_compound_definition_prompts__direct_approach__00" in out_file_path:
            out_file_path = out_file_path.replace("chatgpt_compound_definition_prompts__direct_approach","chatgpt_comp_def__direct")
        if len(out_file_path)-len(output_dir) >= 256 and "chatgpt_constituent_definitions" in out_file_path:
            out_file_path = out_file_path.replace("chatgpt_constituent_definitions","chatgpt_const_def")
        if len(out_file_path)-len(output_dir) >= 256 and "wordnet_constituent_sense-specific" in out_file_path:
            out_file_path = out_file_path.replace("wordnet_constituent_sense-specific","wordnet_const")
        if "+" in out_file_path: # done some early fusion -> long name -> assume info is encoded in the output directory
            out_file_path = output_dir + constituent_dir.split("/")[-2] + ".csv"
            if not os.path.exists(output_dir): # need to create the directory
                os.makedirs(output_dir, exist_ok=True) # allow to create directory with parent directory
            combiner = " "
        if len(out_file_path)-len(output_dir) >= 256:
            print("couldn't create a score-file for ",out_file_path, " as the file name would be too long")
            return None
     # --- end of handling stupid file name length problems
    # if os.path.exists(out_file_path): # we already calculated this score earlier
    #     return None 
    outfile = open(out_file_path,"w")
    outfile.write("compound,modifier,modifier_score,head,head_score\n") # header line

    # handle format 
    if not combiner: 
        if "underscore" in compound_identifier: 
            combiner = "_"
        elif "hyphenated" in compound_identifier:
            combiner = "-"
        elif "closed" in compound_identifier:
            combiner = ""
        else:
            combiner = " "

    # make sure to load the targets correctly 
    if os.path.exists(compound_file): 
        csv_data = helper.load_csv(compound_file)
        targets = set() # use set as we don't want to extract duplicate embeddings
        for target in csv_data:
            targets.add(target["noun_compound"])
        print("csv file is processed")
    else: 
        print(compound_file, " is not a valid file")
        return None
    
    # iterate over compound-constituent pairs
    for compound in tqdm(targets):
        modifier,head = compound.strip().split(" ")
        combined_compound = modifier+combiner+head

        # load according embeddings
        compound_embedding_path = embeddings_path_template.format(input_dir=compound_dir,target=combined_compound)
        if constituent_dir.endswith("/"):
            constituent_dir = constituent_dir[:-1]
        if compound_limitation: # use specific embedding that matches the limitation
            if compound_limitation.startswith("_FROM"): 
                specific_compound_paths = os.listdir(compound_dir+"/image-specific")
                matching_path = [path for path in specific_compound_paths if path.startswith(combined_compound) and compound_limitation in path][0]
                compound_embedding_path = embeddings_path_template.format(input_dir=compound_dir+"image-specific",target=matching_path.replace(".pt",""))
            if os.path.exists(compound_limitation):
                compound_embedding_path = limitation_file[compound]
        elif not os.path.exists(compound_embedding_path):
            compound_embedding_path = embeddings_path_template.format(input_dir=compound_dir,target=combined_compound+"_avg")
        modifier_embedding_path = embeddings_path_template.format(input_dir=constituent_dir,target=modifier)
        if not os.path.exists(modifier_embedding_path):
            modifier_embedding_path = embeddings_path_template.format(input_dir=constituent_dir,target=modifier+"_avg")
        head_embedding_path = embeddings_path_template.format(input_dir=constituent_dir,target=head)
        if not os.path.exists(head_embedding_path):
            head_embedding_path = embeddings_path_template.format(input_dir=constituent_dir,target=head+"_avg")
        if all(os.path.exists(file) for file in [compound_embedding_path,modifier_embedding_path,head_embedding_path]):
            compound_embedding = helper.load_vector(compound_embedding_path)
            modifier_embedding = helper.load_vector(modifier_embedding_path)
            head_embedding = helper.load_vector(head_embedding_path)

        else:
            print("not all embeddings are available for ",compound)
            return None

        # calculate similarities between pairs
        try:
            head_score = get_cosine_similarity(head_embedding,compound_embedding)
            modifier_score = get_cosine_similarity(modifier_embedding,compound_embedding)
        except:
            print("couldn't calculate cosine similarities for embeddings of ",compound)
            return None

        # save similarity scores
        outfile.write(modifier+" "+head+","+modifier+","+str(modifier_score)+","+head+","+str(head_score)+"\n")

    outfile.close()
    print("saved all compositionality scores to file")
    return None


# main function for py file
def main():
    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        '-c',
        '--compound_file',
        help='Link to file with one compound per line'
    )
    parsers.add_argument(
        '-p',
        '--part_dir',
        help='Directory where embeddings of constituents are stored.'
    )
    parsers.add_argument(
        '-w',
        '--whole_dir',
        help='Directory where embeddings of compounds are stored.'
    )
    parsers.add_argument(
        '-o',
        '--output_dir',
        help='Output directory where the measured scores shall be saved.'
    )
    parsers.add_argument(
        '-l',
        '--limitation_for_compound',
        help='Specify whether e.g. only compounds generated with denoising strength 0.9 should be considered. E.g. _FROM_0.9_',
        nargs="?", # allow it to be an optional argument
        default=None # set default to None as we want to consider the general (averaged) compound vector version
    )
    args = parsers.parse_args()

    calculate_scores(compound_file=args.compound_file,constituent_dir=args.part_dir,compound_dir=args.whole_dir,output_dir=args.output_dir,compound_limitation=args.limitation_for_compound)#,c_format=args.format_of_compound)


if __name__ == "__main__":
    main()
