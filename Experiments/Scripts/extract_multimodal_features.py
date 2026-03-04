import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from transformers import FlavaProcessor, FlavaModel
import json
import argparse
from abc import ABC
from abc import abstractmethod
import os
from tqdm import tqdm
import helper # own python file containing helper functions
from collections import defaultdict

cache_directory = "/path/to/cache/" # TODO need to adapt it
device = None

# FEATURE EXTRACTION MODELS --------------------------

class FeatureExtractor(ABC): 
    @abstractmethod
    def __init__(self) -> None:
        """
        Initializes the model (along with components) that will be used for feature extraction
        """
        pass

    @abstractmethod
    def extract_embedding(self, image_path: str, text: str, method:str=None) -> torch.Tensor:
        """
        Extract an embedding of a visual representation as torch vector.
        Parameter:
            image_path (str): link to file where image is stored
            text (str): string representation used for textual feature extraction, e.g. "ballet"
            method (str): if embeddings are retrieved for modalities separately this defines how they should be combined
        (Later) Returns:
            _ : embedding as torch Tensor
        """
        pass

class CLIP(FeatureExtractor):

    def __init__(self,method:str,version:str) -> None:
        # load pretrained model (along with other needed components)
        if version == "16":
            model_version = "openai/clip-vit-base-patch16"
        elif version == "32":
            model_version = "openai/clip-vit-base-patch32" 
        else:
            print("model version "+version+" is not available")
        self.clip_model = CLIPModel.from_pretrained(model_version,cache_dir=cache_directory).to(device)
        self.clip_model.eval() # use in inference mode
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(model_version,cache_dir=cache_directory)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_version,cache_dir=cache_directory)
        self.method = method

    def get_separate_embeddings(self,image_path:str,text:str) -> tuple:
        # load image from file and preprocess it
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        with torch.no_grad():
            preprocessed_img = self.clip_image_processor(image,return_tensors="pt").to(device)
            
            # preprocess tokens
            tokens = self.clip_tokenizer(text,truncation=True,return_tensors="pt").to(device)

            # extract features
            image_embeddings = self.clip_model.get_image_features(**preprocessed_img).squeeze(0).float() # before squeeze of size [1,512]
            text_embeddings = self.clip_model.get_text_features(**tokens).squeeze(0).float() # before squeeze of size [1,512]
            
            # normalize the embeddings
            image_embeddings /= image_embeddings.norm(dim=-1,keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1,keepdim=True)   

        return image_embeddings,text_embeddings

    def extract_embedding(self, image_path:str, text:str) -> torch.Tensor:
        image_embeddings,text_embeddings = self.get_separate_embeddings(image_path,text)
        if self.method == "concatenation":
            return torch.cat([image_embeddings, text_embeddings]).detach().cpu()
        elif self.method == "average":
            return torch.stack([image_embeddings,text_embeddings]).mean(dim=0).detach().cpu()
        elif self.method == "sum":
            sum_vector = image_embeddings+text_embeddings
            return sum_vector.detach().cpu()
        else:
            print(self.method," is not a valid method")
            return None
        
class FLAVA(FeatureExtractor):

    def __init__(self,method:str) -> None:
        # load pretrained model (along with other needed components)
        model_version = "facebook/flava-full"
        self.flava_model = FlavaModel.from_pretrained(model_version,cache_dir=cache_directory).to(device)
        self.flava_model.eval() # use in inference mode
        self.flava_processor = FlavaProcessor.from_pretrained(model_version,cache_dir=cache_directory)
        self.method = method

    def extract_embedding(self, image_path:str, text:str):
        img = Image.open(image_path)

        inputs = self.flava_processor(text=[text],images=[img],return_tensors="pt")

        layer_outputs = {}

        def save_output_hook(name, mod, inp, out):
            layer_outputs[name] = out
        # register hooks on layers in multimodal encoder to capture the output of intermediate layers
        for i, layer in enumerate(self.flava_model.multimodal_model.encoder.layer):
            layer.register_forward_hook(lambda mod, inp, out, i=i: save_output_hook(f"multimodal_layer_{i}", mod, inp, out))

        with torch.no_grad():
            outputs = self.flava_model(**inputs.to(device),output_hidden_states=True)

            if self.method == "pool":
                vector = outputs["multimodal_output"]["pooler_output"] # of size [batch_size,hidden_size]
                return vector.squeeze(0).detach().cpu()
            else:
                if "-" in self.method:
                    combination,layer = self.method.split("-")
                else:
                    print(self.method, " is not an available method")
                    return None
                if layer == "last":
                    vector = outputs["multimodal_output"]["last_hidden_state"] # of size [batch_size,num_image_patches+text_len+3,hidden_size]
                elif "multimodal_layer_"+layer in layer_outputs:
                    vector = layer_outputs["multimodal_layer_"+layer][0] # of size [batch_size,num_image_patches+text_len+3,hidden_size]
                else:
                    print(layer," is not an available layer")
                vector = vector.squeeze(0) # remove "empty" dimensions -> new shape: [num_image_patches+text_len+3,hidden_size]
                if combination == "average":
                    mean_vector = torch.mean(vector,dim=0).detach().cpu() # take the mean values
                    return mean_vector
                elif combination == "sum":
                    sum_vector = torch.sum(vector,dim=0).detach().cpu() # take the summed values
                    return sum_vector
                else:
                    print(combination," is not an available combination")
                    return None


# EXTRACT FEATURES AS EMBEDDINGS ---------------------

def compute_averaged_embeddings(tensor_folder_path:str): 
    """
    Load image tensors from the specified folder path and calculate the averaged tensor per target.
    Parameters:
        tensor_folder_path (str): Directory where image-specific tensors are saved, e.g. "/path/to/repo/Experiments/temp/Embeddings/vit/pixartsigma__chatgpt_definitions__direct_approach/image-specific/"
    """
    file_groups = defaultdict(list)
    if os.path.exists(tensor_folder_path):
        for image_tensor_file in os.listdir(tensor_folder_path):
            if image_tensor_file.endswith(".pt"):
                target = image_tensor_file.split("_")[0] # can be either compound or constituent
                file_groups[target].append(os.path.join(tensor_folder_path,image_tensor_file))

        for target, tensor_files in file_groups.items():
            tensors = [torch.load(t) for t in tensor_files]
            avg_vector = torch.mean(torch.stack(tensors),dim=0)
            torch.save(avg_vector,os.path.join(tensor_folder_path[:-18],target+"_avg.pt"))
        print("saved averaged tensor versions of embeddings")
    else:
        print("the path ",tensor_folder_path," doesn't exist")

def get_multimodal_embeddings(input_file:str,output_dir:str,model_name:str) -> None:
    """
    Uses feature extraction method to extract embeddings for the compound / constituent representations and save them to file.
    Parameters:
        input_file (str): Input file having saved the list of text-image-pairs
        output_dir (str): Output directory where the embeddings shall be stored
        model (str): Model / approach used for extracting the features
    Returns: None (only used for early break if arguments were not passed correctly)
    """
    # first make sure to load the correct model
    model_name = model_name.lower().strip()
    if "clip-" in model_name:
        _,model_method,model_version = model_name.strip().split("-") # e.g. "clip-sum-16"
        extractor = CLIP(method=model_method,version=str(model_version))
    elif "flava-" in model_name:
        extractor = FLAVA(method=model_name.replace("flava-",""))
    else:
        print(model_name+" is not available as feature extraction model")
        return None
    print(model_name, " model is loaded")

    # make sure to load the targets correctly 
    if os.path.exists(input_file): 
        with open(input_file) as loader_file: 
            targets = json.load(loader_file)  
            print("targets have been loaded from file")          
    else: 
        print(input_file, " is not a valid file")
        return None
    
    if "__def_form.json" in input_file: 
        target_form = "def"
    else:
        target_form = "token"
    
    # extract vector embedding for each target and save to file
    print("start extracting embeddings")
    # note: the following three commands are quite specific and only match to the version I saved the files!
    input_file_name_info = input_file.split("/")[-1].replace(".json","")
    target_type = input_file_name_info.split("__")[0] # says "constituents" or "compounds"
    image_identifier = input_file_name_info.replace(target_type+"__","") # contains model, prompts, and experiment of generated images
    goal_path = output_dir+model_name+"/"+target_form+"/"+target_type+"/"+image_identifier+"/instance-specific/"
    if not os.path.exists(goal_path):
        os.makedirs(goal_path, exist_ok=True) # allow to create directory with parent directory
        print("created the directory: ",goal_path)
    for target in tqdm(targets):
        target_file_identifier = target[0].split("/")[-1].replace(".jpg","")
        if target_file_identifier.startswith("chatgpt_"):
            target_file_identifier = target_file_identifier.replace("chatgpt_","",1)
        helper.save_vector(vector=extractor.extract_embedding(image_path=target[0],text=target[1]),file_path=goal_path+target_file_identifier+".pt")
    print("all embeddings have been extracted and saved to file")

    # calculate the averaged tensors per target
    compute_averaged_embeddings(goal_path)
    return None


# main function for py file
def main():
    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        '-i',
        '--input_file',
        help='Input file that lists text-image pairs to retrieve embeddings from.'
    )
    parsers.add_argument(
        '-o',
        '--output_dir',
        help='Output directory where the embeddings shall be stored.'
    )
    parsers.add_argument(
        '-m',
        '--model',
        help='Model / approach to use for extracting the features: "clip-sum", "clip-average", "clip-concatenation".'
    )
    parsers.add_argument(
        '-g',
        '--gpu',
        help='GPU device on which the feature extraction process should run. E.g. 0',
        nargs="?", # allow it to be an optional argument
        default=None # set default to "cpu" -> run model on cpu
    )
    args = parsers.parse_args()

    global device
    if (str(args.gpu) != "cpu") and (torch.cuda.is_available()):
        device = torch.device("cuda:"+str(args.gpu))
    else:
        device = "cpu"
    print("Set ",device," as the device to run the feature extraction process on")

    get_multimodal_embeddings(output_dir=args.output_dir,input_file=args.input_file,model_name=args.model)
    

if __name__ == "__main__":
    main()
