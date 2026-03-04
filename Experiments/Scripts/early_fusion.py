import argparse
import helper # own python file containing helper functions
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

cache_directory = "/path/to/cache/" # TODO need to adapt it
combination_types = ["concat","mean","weighted-0.2","weighted-0.4","weighted-0.6","weighted-0.8"]
compound_types = {"underscore":("_",""),"closed":("",""),"open":(" ",""),"hyphenated":("-",""),"sentence":(" ",""),"multi":(" ","_avg")}

# EARLY FUSION ---------------------------------------
# (= combine features of different modalities into multimodal features, proceed with compositionality prediction as before but with these features)

def pad_to_same_size(embedding_t:torch.tensor,embedding_v:torch.tensor) -> (torch.tensor,torch.tensor):
    """
    Takes two different embeddings and brings them into the same size by padding the smaller vector to the size of the larger one.
    Parameters:
        embedding_t: one vector (e.g. encoding information from text modality)
        embedding_v: other vector (e.g. encoding information from vision modality)
    Returns: padded versions of two embeddings
    """
    len_t = embedding_t.size(0)
    len_v = embedding_v.size(0)
    max_len = max(len_t,len_v)
    embedding_t_padded = F.pad(embedding_t, (0,max_len-len_t))
    embedding_v_padded = F.pad(embedding_v, (0,max_len-len_v))
    return embedding_t_padded,embedding_v_padded


def combine_embeddings(embedding_t:torch.tensor,embedding_v:torch.tensor,combiner:str):
    """
    Takes two different embeddings and combines them into one using the specified combiner as approach.
    Parameters:
        embedding_t: one embedding saved as vector (e.g. encoding information from text modality)
        embedding_v: other embedding saved as vector (e.g. encoding information from vision modality)
        combiner: defines how the embeddings are combined into one embedding
    Returns: 
        combined embedding: resulting embedding (saved as vector)
    """
    if combiner == "concat":
        result = torch.cat((embedding_t, embedding_v), dim=-1)
    else: # use padded versions for embeddings!
        padded_embedding_t,padded_embedding_v = pad_to_same_size(embedding_t,embedding_v)
        if "weighted" in combiner:
            weight = float(combiner.split("-")[1])
            result = (weight * padded_embedding_t) + ((1 - weight) * padded_embedding_v)
        elif combiner == "mean":
            result = (padded_embedding_t + padded_embedding_v) / 2
        else:
            print(combiner," is not a valid combination type")
            return None
    return result


def perform_early_fusion(compound_file,embeddings_text_constituents,embeddings_text_compounds,embeddings_vision_constituents,embeddings_vision_compounds,output_dir):
    """
    Takes embeddings from constituents and compounds from two different sources and combines them. 
    Parameters:
        compound_file (str): CSV file with one compound per line for which we want to combine the scores from different modalities
        embeddings_text_constituents (str): Path to directory where (text-based) constituent embeddings are saved as vectors
        embeddings_text_compounds (str): Path to directory where (text-based) compound embeddings are saved as vectors
        embeddings_vision_constituents (str): Path to directory where (vision-based) constituent embeddings are saved as vectors
        embeddings_vision_compounds (str): Path to directory where (vision-based) compound embeddings are saved as vectors
        output_dir (str): Path to directory where combined scores are saved
    Returns: None (only used for early break if arguments were not passed correctly)
    """
    # load targets (= compounds)
    if os.path.exists(compound_file): 
        csv_data = helper.load_csv(compound_file)
        compounds = set() # use set as we don't want to extract duplicate embeddings
        constituents = set()
        for target in csv_data:
            compounds.add(target["noun_compound"])
            const1, const2 = target["noun_compound"].split()
            constituents.add(const1)
            constituents.add(const2)
        print("csv file is processed")
    else: 
        print(compound_file, " is not a valid file")
        return None

    # handle file names -> note: I don't combine the names rather assume that the output directory path contains valuable information
    if not os.path.exists(output_dir): # need to create the directory
        for combination_type in combination_types:
            os.makedirs(output_dir+"constituents/"+combination_type+"/", exist_ok=True) # allow to create directory with parent directory
            os.makedirs(output_dir+"compounds/"+combination_type+"/", exist_ok=True) # allow to create directory with parent directory

    # load embeddings
    constituent_embeddings = {const:{"text":None,"vision":None} for const in constituents}
    compound_embeddings = {comp:{"text":None,"vision":None} for comp in compounds}
    
    for const in constituents:
        text_const_path = embeddings_text_constituents+const+".pt"
        vision_const_path = embeddings_vision_constituents+const+"_avg.pt"
        if all(os.path.exists(file) for file in [text_const_path,vision_const_path]):
            constituent_embeddings[const]["text"] = helper.load_vector(text_const_path)
            constituent_embeddings[const]["vision"] = helper.load_vector(vision_const_path)
        else:
            print("for constituent ",const," not all necessary embeddings are available")
            return None
    print("constituent embeddings are loaded")
    
    for comp in compounds:
        modifier, head = comp.split()
        text_combiner1,text_combiner2 = compound_types[embeddings_text_compounds.split("/")[-2]]
        text_comp_path = embeddings_text_compounds+modifier+text_combiner1+head+text_combiner2+".pt"
        vision_comp_path = embeddings_vision_compounds+comp+"_avg.pt"
        if all(os.path.exists(file) for file in [text_comp_path,vision_comp_path]):
            compound_embeddings[comp]["text"] = helper.load_vector(text_comp_path)
            compound_embeddings[comp]["vision"] = helper.load_vector(vision_comp_path)
        else:
            print("for compound ",comp," not all necessary embeddings are available")
            return None
    print("compound embeddings are loaded")

    # combine the embeddings appropriately and save them to file
    for combiner in tqdm(combination_types):
        for const in constituents:
            combined_const_path = output_dir+"constituents/"+combiner+"/"+const+".pt" 
            if not os.path.exists(combined_const_path): # only do combination if we haven't done it before
                combined_const_embedding = combine_embeddings(constituent_embeddings[const]["text"],constituent_embeddings[const]["vision"],combiner)
                helper.save_vector(vector=combined_const_embedding,file_path=combined_const_path)

        for comp in compounds:
            combined_comp_path = output_dir+"compounds/"+combiner+"/"+comp+".pt"
            if not os.path.exists(combined_comp_path): # only do combination if we haven't done it before
                combined_comp_embedding = combine_embeddings(compound_embeddings[comp]["text"],compound_embeddings[comp]["vision"],combiner)
                helper.save_vector(vector=combined_comp_embedding,file_path=combined_comp_path)

    print("saved all combined embeddings to file")

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
        '-etp',
        '--embeddings_text_parts',
        help='Path to directory that contains text embeddings of constituents.'
    )
    parsers.add_argument(
        '-etc',
        '--embeddings_text_compounds',
        help='Path to directory that contains text embeddings of compounds.'
    )
    parsers.add_argument(
        '-evp',
        '--embeddings_vision_parts',
        help='Path to directory that contains visual embeddings of constituents.'
    )
    parsers.add_argument(
        '-evc',
        '--embeddings_vision_compounds',
        help='Path to directory that contains visual embeddings of compounds.'
    )
    parsers.add_argument(
        '-o',
        '--output_dir',
        help='Directory where combined embeddings for compounds and constituents are saved.'
    )
    args = parsers.parse_args()

    perform_early_fusion(compound_file=args.compound_file,
                         embeddings_text_constituents=args.embeddings_text_parts,
                         embeddings_text_compounds=args.embeddings_text_compounds,
                         embeddings_vision_constituents=args.embeddings_vision_parts,
                         embeddings_vision_compounds=args.embeddings_vision_compounds,
                         output_dir=args.output_dir)

if __name__ == "__main__":
    main()