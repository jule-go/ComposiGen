from gensim.models import Word2Vec
from transformers import BertTokenizerFast, BertModel 
import torch
import argparse
from abc import ABC
from abc import abstractmethod
import os
from tqdm import tqdm
import helper # own python file containing helper functions
from collections import defaultdict

cache_directory = "/path/to/cache/" # TODO need to adapt it
device = None
BERT_sentences = ["/path/to/repo/Data/NounDefinitions/chatgpt_compound_definitions.tsv",
                 "/path/to/repo/Data/NounDefinitions/chatgpt_constituent_definitions.tsv"] # TODO need to adapt

# FEATURE EXTRACTION MODELS --------------------------

class FeatureExtractor(ABC): 
    @abstractmethod
    def __init__(self) -> None:
        """
        Initializes the model (along with its components) that will be used for feature extraction
        """
        pass

    @abstractmethod
    def extract_embedding(self, target:str, input_text:str=None) -> torch.Tensor:
        """
        Extract an embedding of a textual representation as torch vector.
        Parameter:
            target (str): string of word to extract embedding for
            input_text (str): (optional) string of sentence used as context for extracting the word embedding
        (Later) Returns:
            _ : embedding as torch Tensor
        """
        pass

    
class SkipGram(FeatureExtractor):
    def __init__(self) -> None:
        model_path = "/path/to/repo/Experiments/Skip-gram/checkpoint.model" # TODO first train the model and then load according checkpoint
        self.model = Word2Vec.load(model_path)

    def extract_embedding(self, target:str) -> torch.Tensor:
        return torch.tensor(self.model.wv[target])
    
class BERT(FeatureExtractor):
    def __init__(self,method:str,layer:int) -> None:
        bert_model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name,cache_dir=cache_directory)
        self.model = BertModel.from_pretrained(bert_model_name,cache_dir=cache_directory)
        self.model.eval() # use in inference mode
        self.model = self.model.to(device)
        self.method = method # "avg", "sum", "cls"
        self.layer = layer # number between 0 and 12 (12 represents the last hidden state), or None for the pool approach

    def extract_embedding(self, target:str, input_text:str) -> torch.Tensor:
        # note: target = constituent or compound, input_text = text used as context for extracting the embedding -> either sentence with compound / constituent, or compound / constituent only
        
        if self.method == "cls":
            add_cls = True
        else:
            add_cls = False

        input_ids = self.tokenizer(input_text, add_special_tokens=add_cls, return_tensors='pt', return_offsets_mapping=True)
        
        # identify token indices that belong to the target -> note: this needs to be done this way as the BERT tokenizer might split a token into multiple ones
        offsets = input_ids.pop("offset_mapping")[0].tolist() # extract the offsets and remove this information from the BERT input
        target_start = input_text.lower().find(target)
        target_end = target_start + len(target)
        target_indices = [i for i,(start,end) in enumerate(offsets) if start >= target_start and end <= target_end and start != end]
        
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in input_ids.items()}
            outputs = self.model(**inputs,output_hidden_states=True)
            
            if self.layer:
                hidden_state = outputs.hidden_states[self.layer]  # shape: [1 (= batch size), sequence_length (of input_text!), 768 (= BERTs hidden_dim)]
                hidden_state = hidden_state[:,target_indices,:] # shape: [1 (= batch size), sequence length (of target!), 768 (= BERTs hidden dim)]
                vector = hidden_state.squeeze(0) # remove "empty" dimensions -> new shape: [sequence_length (of target!), 768 (= BERTs hidden_dim)]
            else: # use pooler output
                vector = outputs.pooler_output # shape: [1 (= batch size), 768 (= BERTs hidden_dim)]

            if self.method == "avg":
                if vector.size()[0] > 1:
                    mean_vector = torch.mean(vector,dim=0).detach().cpu() # take the mean values
                    return mean_vector
                return vector.squeeze(0).detach().cpu()
            elif self.method == "sum":
                if vector.size()[0] > 1:
                    sum_vector = torch.sum(vector,dim=0).detach().cpu() # take the summed values
                    return sum_vector
                return vector.squeeze(0).detach().cpu()
            elif self.method == "cls" or self.method == "pool":
                return vector[0].detach().cpu()
                # note: in cls case it's the embedding of the cls token (as special tokens were added)
            else:
                print(self.method, " is not an available method")
                return None

# EXTRACT FEATURES AS EMBEDDINGS ---------------------

def get_text_embeddings(input_file:str,output_dir:str,model:str,c_format:str) -> None:
    """
    Uses feature extraction method to extract embeddings for the compound / constituent representations and save them to file.
    Parameters:
        input_file (str): Input path where file is stored that lists compounds and constituents
        output_dir (str): Output directory where the embeddings shall be stored
        model (str): Model / approach used for extracting the features
        c_format (str): Format of compound, either "underscore", "hyphenated", "open", "closed", "sentence", or "multi"
    Returns: None (only used for early break if arguments were not passed correctly)
    """

    possible_models = ["spacy","skipgram","fasttext","bert-pool"]
    possible_models += ["bert-avg-"+str(i) for i in range(12)]
    possible_models += ["bert-sum-"+str(i) for i in range(12)]
    possible_models += ["bert-cls-"+str(i) for i in range(12)]

    # first make sure to load the correct model
    model = model.lower().strip()
    if model not in possible_models:
        print(model, " is not available, please choose from ",list(possible_models.keys()))
        return None
    elif model == "skipgram":
        extractor = SkipGram()
    else: # some variation of the bert model
        if "pool" in model:
            extractor = BERT("pool",None)
        else:
            _,combi,layer = model.split("-")
            extractor = BERT(combi,int(layer))

    print(model, " model is loaded")

    # handle format 
    if c_format == "underscore": 
        combiner = "_"
    elif c_format == "hyphenated":
        combiner = "-"
    elif c_format == "open":
        combiner = " "
    elif c_format == "closed":
        combiner = ""
    elif c_format == "sentence":
        combiner = " "
        sentences = helper.load_sentences(BERT_sentences,"one") 
    elif c_format == "multi":
        combiner = " "
        sentences = helper.load_sentences(BERT_sentences,"multi")
    else:
        print(c_format + " is not among the available formats")
        return None

    # make sure to load the targets correctly 
    if os.path.exists(input_file): 
        csv_data = helper.load_csv(input_file)
        compound_targets = set() # use set as we don't want to extract duplicate embeddings
        constituent_targets = set()
        for target in csv_data:
            # targets.add(target["noun_compound"])
            mod = target["component_1"]
            head = target["component_2"]
            constituent_targets.add(mod)
            constituent_targets.add(head)
            compound_targets.add(mod+combiner+head) # also add compound
        print("csv file is processed")
    else: 
        print(input_file, " is not a valid file")
        return None
    
    # extract vector embedding for each target and save to file
    print("start extracting embeddings")
    goal_path_compounds = output_dir+model+"/compounds/"+c_format+"/"
    goal_path_constituents = output_dir+model+"/constituents/"
    if c_format == "sentence":
        goal_path_constituents += c_format+"/"
    elif c_format == "multi":
        goal_path_constituents += c_format+"/specific/"
        goal_path_compounds += "specific/"
    else:
        goal_path_constituents += "basic/"
    paths_to_consider = [goal_path_constituents,goal_path_compounds]
    for path in paths_to_consider:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True) # allow to create directory with parent directory
            print("created the directory: ",path)
    for target in tqdm(compound_targets):
        if "bert" in model.lower().strip():
            sentence = [target] # in case no sentence is provided, simply use the target (as context as well) for extracting the embedding
            if c_format == "sentence" or c_format == "multi":
                sentence = sentences[target]
                sentence = [sent if " "+target+" " in sent else target+" "+sent for sent in sentence] # manually add it e.g. with "dog's dinner" we wouldn't find a match else
            for sent_id,sent in enumerate(sentence):
                tensor_id = "" if len(sentence)==1 else str("_"+str(sent_id))
                helper.save_vector(vector=extractor.extract_embedding(target=target,input_text=sent),file_path=goal_path_compounds+target+tensor_id+".pt")
        else:
            helper.save_vector(vector=extractor.extract_embedding(target),file_path=goal_path_compounds+target+".pt")
    for target in tqdm(constituent_targets):
        constituent_path = goal_path_constituents+target+".pt"
        if not os.path.exists(constituent_path):
            if "bert" in model.lower().strip():
                sentence = [target] # in case no sentence is provided, simply use the target (as context as well) for extracting the embedding
                if c_format == "sentence" or c_format == "multi":
                    sentence = sentences[target]
                    sentence = [sent if " "+target+" " in sent else target+" "+sent for sent in sentence] # manually add it e.g. with "dog's dinner" we wouldn't find a match else
                for sent_id,sent in enumerate(sentence):
                    tensor_id = "" if (len(sentence)==1) else str("_"+str(sent_id))
                    if not os.path.exists(goal_path_constituents+target+tensor_id+".pt"):
                        helper.save_vector(vector=extractor.extract_embedding(target=target,input_text=sent),file_path=goal_path_constituents+target+tensor_id+".pt")
            else:
                helper.save_vector(vector=extractor.extract_embedding(target),file_path=constituent_path)
    print("all embeddings have been extracted and saved to file")

    # average files in case of multi setup
    if c_format == "multi":
        compound_file_groups = defaultdict(list)
        for compound_emb in os.listdir(goal_path_compounds):
            target = compound_emb.split("_")[0]
            compound_file_groups[target].append(os.path.join(goal_path_compounds,compound_emb)) 
        for target, tensor_files in compound_file_groups.items():
            tensors = [torch.load(t) for t in tensor_files]
            avg_vector = torch.mean(torch.stack(tensors),dim=0)
            torch.save(avg_vector,os.path.join(goal_path_compounds[:-9],target+"_avg.pt"))
        constituent_file_groups = defaultdict(list)
        for const_emb in os.listdir(goal_path_constituents):
            target = const_emb.split("_")[0]
            constituent_file_groups[target].append(os.path.join(goal_path_constituents,const_emb))
        for target, tensor_files in constituent_file_groups.items():
            tensors = [torch.load(t) for t in tensor_files]
            avg_vector = torch.mean(torch.stack(tensors),dim=0)
            torch.save(avg_vector,os.path.join(goal_path_constituents[:-9],target+"_avg.pt"))
        print("also saved averaged versions")

    return None

# main function for py file
def main():

    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        '-i',
        '--input_file',
        help='Input directory where images are stored / file that lists compounds and constituents.'
    )
    parsers.add_argument(
        '-o',
        '--output_dir',
        help='Output directory where the images shall be stored.'
    )
    parsers.add_argument(
        '-m',
        '--model',
        help='Model / approach to use for extracting the features: "skipgram", "bert-avg-num", "bert-sum-num", "bert-cls-num", "bert-pool".'
    )
    parsers.add_argument(
        '-f',
        '--format_of_compound',
        help='Format of the compound: "underscore", "hyphenated", "open", "closed", or "sentence"'
    )
    parsers.add_argument(
        '-g',
        '--gpu',
        help='GPU device on which the feature extraction process should run. E.g. 0',
        nargs="?", # allow it to be an optional argument
        default=None # set default to "cpu" -> run model on cpu
    )
    args = parsers.parse_args()

    if (str(args.gpu) != "cpu") and (torch.cuda.is_available()):
        device = torch.device("cuda:"+str(args.gpu))
    else:
        device = "cpu"
    print("Set ",device," as the device to run the feature extraction process on")

    get_text_embeddings(output_dir=args.output_dir,input_file=args.input_file,model=args.model,c_format=args.format_of_compound)


if __name__ == "__main__":
    main()