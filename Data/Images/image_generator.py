from abc import ABC
from abc import abstractmethod
import torch
import argparse
from diffusers import PixArtSigmaPipeline, FluxPipeline, FluxImg2ImgPipeline
from tqdm import tqdm
import os
import json
from PIL import Image

# ----- SETUP -----

cache_directory = "/path/to/cache/" # TODO need to adapt it
image_savings_folder = "/path/to/repo/Images/CompoundImages/" # TODO might want to adapt it
dir_path_template = "{model}/{prompts}/{experiment_identifier}/" # TODO change if you use a different structure for saving the images
file_path_template = "{compound}_{seed}{prompt_id}.jpg" # TODO change if you use a different structure for saving the images
manual_seed = 0 # work with manual seed in order to generate images that are reproducible
number_images_per_prompt = 1 # TODO change depending on the amount of images you want to generate per prompt
image_height = 1024
image_width = 1024
denoise_strengths = [round(0.8 + i*0.02,2) for i in range(11)] # [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0] # TODO adapt if you want to see images from different scale
device = None
x2i = ""

# ----- IMAGE GENERATION MODELS -----

class ImageGenerator(ABC): 
    @abstractmethod
    def __init__(self) -> None:
        pass

    def generate_image(self, target:str, prompt:str, file_path:str, seed:int, starter_paths:list=None, negative_prompt:str=None) -> None:
        """
        Generate an image for the specified prompt, starting from the specified image. 
        Also saves the generated image to the specified file.
        (Is the same method for all different generation models) 
            target (str): target we want to create an image of
            prompt (str): prompt used for prompting the image-generation model
            file_path (str): path were the generated images should be saved
            seed (int): seed used for generating the image -> to get different images for same prompt
            starter_images (list): list of paths to image that should be used as starting point (when doing image-to-image)
            negative_prompt (str): negative prompt for image-generation model
        """
        generator = torch.Generator(device=device).manual_seed(seed)

        if starter_paths != [None]:
            # load images
            starter_images = [Image.open(img_path).resize((image_width,image_height)) for img_path in starter_paths]
            # adapt filepaths
            filepaths = dict() 
            new_id = file_path.split("/")[-1][:-4]
            for st_img in starter_paths:
                head_id = st_img.split("/")[-1][:-4]
                filepaths[st_img] = "/".join(file_path.split("/")[:-1])+"/"+new_id+"_FROM_{strength}_"+head_id+".jpg"

        generation_args = {"prompt":prompt,
                           "num_inference_steps":self.num_steps,
                           "generator":generator,
                           "width":image_width,
                           "height":image_height}
        
        if self.guidance_scale:
            generation_args["guidance_scale"] = self.guidance_scale

        negatives = ""
        if self.general_negatives:
            negatives += self.general_negatives
        if negative_prompt:
            negatives += negative_prompt
        if len(negatives) > 0:
            if self.identifier in ["pixartsigma"]: 
                generation_args["negative_prompt"] = negatives
            elif self.identifier == "flux":
                if starter_paths == [None]: 
                    generation_args["negative_prompt"] = negatives
            
        # do the generation
        with torch.no_grad():
            if starter_paths != [None]:
                for st_img,st_file_path in zip(starter_images,starter_paths):
                    if not os.path.exists(filepaths[st_file_path].format(strength="1.0")): # TODO delete this line again
                        if self.identifier in ["pixartsigma","flux"]:
                            generation_args["image"] = st_img
                        for denoise_strength in denoise_strengths:
                            if self.identifier in ["flux"]:
                                strength_dict = {"strength": denoise_strength}
                                generated_image = self.i2i_pipeline(**dict(**generation_args,**strength_dict)).images[0]
                                generated_image.save(filepaths[st_file_path].format(strength=str(denoise_strength))) # save image to file
                            else:
                                generated_image = self.i2i_pipeline(**dict(**generation_args)).images[0]
                                generated_image.save(filepaths[st_file_path].replace("{strength}_","")) # save image to file
                                break

            else:
                generated_image = self.t2i_pipeline(**dict(generation_args)).images[0]
                generated_image.save(file_path) # save image to file


class PixArtSigma(ImageGenerator):

    def __init__(self) -> None:
        pixartsigma_pipe = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",torch_dtype=torch.float16,use_safetensors=True,cache_dir=cache_directory).to(device)
        pixartsigma_pipe.transformer = torch.compile(pixartsigma_pipe.transformer, mode="reduce-overhead", fullgraph=True) # improve inference speed
        self.t2i_pipeline = pixartsigma_pipe
        self.identifier = "pixartsigma"
        self.num_steps = 30
        # define variables (that other models need) as None
        self.guidance_scale = None
        self.general_negatives = None
        self.i2i_pipeline = None
        
class Flux(ImageGenerator):
    def __init__(self) -> None:
        if x2i == "i2i":
            flux_i2i_pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.float16,use_safetensors=True,cache_dir=cache_directory).to(device)
            self.i2i_pipeline = flux_i2i_pipe
            self.t2i_pipeline = None
        elif x2i == "t2i":
            flux_t2i_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.float16,use_safetensors=True,cache_dir=cache_directory).to(device)
            self.t2i_pipeline = flux_t2i_pipe
            self.i2i_pipeline = None
        self.identifier = "flux"
        self.num_steps = 30
        self.general_negatives = "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality, over-exposure, under-exposure, saturated, duplicate, cropped, jpeg artifacts, morbid, mutilated, out of frame, blurry, " # use general negatives
        self.guidance_scale = None # define variables (that other models need) as None

# ----- HELPER FUNCTIONS -----
def load_prompts(filepath: str) -> dict:
    """
    Loads prompts listed in file into a dictionary.
    Parameter:
        filepath (str): link to where file is stored, lines in file are in the format "target \t prompt (\t prompt)", and for each target a separate line; at least one prompt, but multiple ones are possible
    Returns:
        prompts (dict): dictionary with targets as keys and prompts (list) as values
    """
    prompts = dict()
    with open(filepath,"r") as infile:
        for line in infile:
            prompt_info = line.split("\t")
            target = prompt_info[0]
            target_prompts = prompt_info[1:]
            if target in prompts:
                print("check the file again for the target '"+target+"', as it occurs multiple times")
            prompts[target] = target_prompts
    return prompts

def dict_to_file(filepath:str,data:dict,indent:int):
    """
    Saves content of a python dictionary to a json file.
    Parameters:
        filepath (str): filepath specifying where the txt should be saved
        data (dict): data that should be saved to file
        indent (int): indentation representing the spacing used when "writing" into the file
    """
    with open(filepath,"w") as saver_file:
        json.dump(data,saver_file,indent=indent)

def file_to_dict(filepath:str) -> dict:
    """
    Loads json file content into a python dictionary.
    Parameters:
        filepath (str): filepath specifying where the txt should be saved
    Returns:    
        _ (dict): dictionary with content that was saved to file
    """
    with open(filepath) as loader_file: 
        return json.load(loader_file)

# ----- IMAGE GENERATION -----

def generate_images(model_name:str, prompt_file:str, experiment_identifier:str, starter_images:str=None, negative_prompt_file:str=None):
    """
    Loads a file with prompts and generates images using the specified model (starting with the starter images).
    Parameters:
        model_name (str): identifier of image generation model
        prompt_file (str): tsv file listing image genartion prompts, format is target \t prompt1 (\tpromptx)*
        experiment_identifier (str): string used for identifying the setup when running this image generation process
        starter_images (str): (optional) if not provided, generation process starts with random noise, else, uses images listed in file of this path as starting signal
        negative_prompt_file (str): (optional) if not provided, prompts are taken from prompt file only, else, additionally use negative prompts listed in this file
    Returns:
        None
    """
    # load prompt file
    prompts = load_prompts(prompt_file)
    if negative_prompt_file:
        negative_prompts = load_prompts(negative_prompt_file)
    else:
        negative_prompts = None
    print("prompts are loaded")

    global x2i
    if starter_images:
        x2i = "i2i"
    else:
        x2i = "t2i"

    # instantiate model
    if model_name.lower() == "pixartsigma":
        model = PixArtSigma()
    elif model_name.lower() == "flux":
        model = Flux()
    else:
        print(model_name+" is not available as image generation model")
    print("image generation model "+model_name+" is loaded")

    # load starter images
    if starter_images: # generation type is image-to-image
        const_image_mapping = file_to_dict(starter_images)

    # create folder(s) if necessary
    image_dir = image_savings_folder+dir_path_template.format(model=model.identifier,
                                                       prompts=prompt_file.split("/")[-1][:-4],
                                                       experiment_identifier=experiment_identifier)
    if not os.path.exists(image_dir): # directory should be image_savings_folder+"{generation_type}/{model}/{prompts}/{experiment_identifier}/"
        os.makedirs(image_dir, exist_ok=True) # allow to create directory with parent directory
        print("created the directory ",image_dir)

    # generate the images
    for target in tqdm(prompts):
        # target_prompts = prompts[target]
        target_prompts = [target_prompts[0]] # for now I only want to generate images for one prompt

        if starter_images:
            # start_images = const_image_mapping[target] # list of starter images for the generation process
            start_images = [start_images[0]] # for now I only want to generate images for one prompt
        else:
            start_images = [None]
        if negative_prompts and target in negative_prompts:
            neg_prompt = negative_prompts[target]
        else:
            neg_prompt = None

        for prompt_id, prompt in enumerate(target_prompts):
            for seed_num in range(number_images_per_prompt):
                file_name = file_path_template.format(compound=target,seed=str(manual_seed+seed_num),prompt_id=prompt_id) # remember: file structure is  "{compound}_{seed}{prompt_id}.jpg"
                if not os.path.exists(image_dir+file_name):
                    model.generate_image(target,prompt,image_dir+file_name,manual_seed+seed_num,start_images,neg_prompt)

# main function for py file
def main():
    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        '-g',
        '--gpu',
        help='GPU device on which the image generation process should run.'
    )
    parsers.add_argument(
        '-m',
        '--model',
        help='Model used for generating images, e.g. "pixartsigma".'
    )
    parsers.add_argument(
        '-p',
        '--prompts',
        help='Path to tsv file where prompts are saved. Each line consists of target along with the prompts (tab-separated).'
    )
    parsers.add_argument(
        '-i',
        '--identifier',
        help='Identifier of the experiment / image generation run.'
    )
    parsers.add_argument(
        '-s',
        '--starter_images',
        help='Path to json file that lists where starter images are saved.',
        nargs="?", # allow it to be an optional argument
        default=None # set default to None -> start generation process from noise
    )
    parsers.add_argument(
        '-n',
        '--negative_prompts',
        help='Path to tsv file where negative prompts are saved. Each line consists of target along with the prompts.',
        nargs="?", # allow it to be an optional argument
        default=None # set default to None -> standard = generation process without negative prompts
    )    
    parsers.add_argument(
        '-d',
        '--directory_to_images',
        help='Specify the directory where the compound images should be saved',
        nargs="?", # allow it to be an optional argument
        default=None # set default to None -> standard = generation process without negative prompts
    ) 
    args = parsers.parse_args()
    print(args)
    
    # set GPU device
    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # "4,5" # set possible GPU devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu") 
    print("Set ",device," as the device to run the generation process on")

    # evtl. change image saving directory
    if args.directory_to_images:
        global image_savings_folder
        image_savings_folder = args.directory_to_images
    
    # generate images for all prompts in the specified file using the specified starter images
    generate_images(model_name=args.model,prompt_file=args.prompts,experiment_identifier=args.identifier,starter_images=args.starter_images,negative_prompt_file=args.negative_prompts)


# main function for py file
if __name__ == "__main__":
    main()