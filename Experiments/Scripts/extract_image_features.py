import torch
from PIL import Image
import torchvision
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
    def extract_embedding(self, image_path: str) -> torch.Tensor:
        """
        Extract an embedding of a visual representation as torch vector.
        Parameter:
            image_path (str): link to file where image is stored
        (Later) Returns:
            _ : embedding as torch Tensor
        """
        pass


class ViT(FeatureExtractor):

    def __init__(self,version:str) -> None:
        # set cache_dir for downloading pretrained model
        torch.hub.set_dir(cache_directory) 
        # load pretrained model
        if version == "h14":
            self.model = torchvision.models.vit_h_14(weights=torchvision.models.ViT_H_14_Weights.DEFAULT)
            resize_size = (518,518) # is crop- and resize size
        elif version == "b16":
            self.model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
            resize_size = (224,224) # actually is the crop size
        else:
            print("model version ",version," is not available")
        # remove last layer (= classification head) of model to be able to access features directly
        self.model.heads = torch.nn.Sequential(*list(self.model.heads.children())[:-1])
        # set to evaluation mode and put to cuda
        self.model.eval()
        self.model = self.model.to(device)
        # after manually checking the default preprocessing via print(torchvision.models.ViT_H_14_Weights.DEFAULT.transforms()) and the other model versions define image transformations
        self.transformations = torchvision.transforms.Compose([   
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)), # (mean),(std)
            torchvision.transforms.Resize(resize_size)])

    def extract_embedding(self, image_path:str):
        # load image from file
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # extract features
        preprocessed_image = self.transformations(image).float().unsqueeze_(0).to(device)
        with torch.no_grad():
            embedding = self.model(preprocessed_image).squeeze(0).detach().cpu()
        return embedding 


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
            torch.save(avg_vector,os.path.join(tensor_folder_path[:-15],target+"_avg.pt"))
        print("saved averaged tensor versions of images")
    else:
        print("the path ",tensor_folder_path," doesn't exist")

def get_image_embeddings(input_dir:str,output_dir:str,model_name:str,limitation:str=None) -> None:
    """
    Uses feature extraction method to extract embeddings for the compound / constituent representations and save them to file.
    Parameters:
        input_dir (str): Input directory where images of constituents / compounds are stored
        output_dir (str): Output directory where the embeddings shall be stored
        model_name (str): Model / approach used for extracting the features
        limitation (str): (optional) allows to reduce the amount of images to look at by specifying a pattern that should be true for the images
    Returns: None (only used for early break if arguments were not passed correctly)
    """
    # first make sure to load the correct model
    model_name = model_name.lower().strip()
    if "vit" in model_name: # e.g. "vit-h14", "vit-b16"
        extractor = ViT(model_name.split("-")[1])
    else:
        print(model_name+" is not available as feature extraction model")
        return None
    print(model_name, " model is loaded")
    # print("model is loaded on ",next(extractor.model.parameters()).device) # TODO check

    # make sure to load the targets correctly 
    if os.path.exists(input_dir): 
        image_files = os.listdir(input_dir)
        targets = set() 
        for img_file in image_files:
            if limitation: # make sure to e.g. only consider images generated with seed=0,promptID=0 if limitation=00
                if "_FROM_" in img_file:
                    if not img_file.split("_FROM_")[0].endswith(limitation):
                        continue
                else:
                    if not img_file.endswith(limitation+".jpg"):
                        continue
            img_raw_name = img_file.split(".jpg")[0] # e.g. only "ballet_01" instead of "ballet_01.jpg"
            if img_raw_name.startswith("chatgpt_"):
                img_raw_name = img_raw_name.replace("chatgpt_","")
            targets.add((input_dir+img_file,img_raw_name))
        print("files in directory are listed")
    else: 
        print(input_dir, " is not a valid directory")
        return None
    
    # extract vector embedding for each target and save to file
    print("start extracting embeddings")
    image_identifier = input_dir[:-1].split("Images/")[1].replace("/","__") #"__".join(input_dir.split("/")[-4:-1]) # contains rq-setup, model, prompts, and experiment of generated images
    if limitation: 
        image_identifier += "__"+limitation
    target_type = input_dir.split("/")[6].replace("Image","").lower() # either "compounds", or "constituents"
    goal_path = output_dir+model_name+"/"+target_type+"/"+image_identifier+"/image-specific/"
    if not os.path.exists(goal_path):
        os.makedirs(goal_path, exist_ok=True) # allow to create directory with parent directory
        print("created the directory: ",goal_path)
    for target in tqdm(targets):
        helper.save_vector(vector=extractor.extract_embedding(target[0]),file_path=goal_path+target[1]+".pt")
    print("all embeddings have been extracted and saved to file")

    # calculate the averaged tensors per target
    compute_averaged_embeddings(goal_path)

    return None

# main function for py file
def main():
    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        '-i',
        '--input_dir',
        help='Input directory where images are stored.'
    )
    parsers.add_argument(
        '-o',
        '--output_dir',
        help='Output directory where the embeddings shall be stored.'
    )
    parsers.add_argument(
        '-m',
        '--model',
        help='Model / approach to use for extracting the features: "vit", "resnet".'
    )
    parsers.add_argument(
        '-g',
        '--gpu',
        help='GPU device on which the feature extraction process should run. E.g. 0',
        nargs="?", # allow it to be an optional argument
        default=None # set default to "cpu" -> run model on cpu
    )
    parsers.add_argument(
        '-l',
        '--limitation',
        help='Limit the images to look at by specifying this pattern. E.g. "00" to say we only want to consider target_00.jpg images' ,
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

    get_image_embeddings(output_dir=args.output_dir,input_dir=args.input_dir,model_name=args.model,limitation=args.limitation)

if __name__ == "__main__":
    main()
