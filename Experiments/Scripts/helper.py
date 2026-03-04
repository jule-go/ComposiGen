from PIL import Image
import csv
import torch

# HELPER FUNCTIONS -----------------------------------

def load_image(file_path: str) -> Image.Image:
    image = Image.open(file_path)
    return image

def save_image(image: Image.Image, file_path: str) -> None:
    image.save(file_path)

def load_vector(file_path: str) -> torch.Tensor:
    vector = torch.load(file_path)
    return vector

def save_vector(vector: torch.Tensor, file_path: str) -> None:
    torch.save(vector, file_path)

def load_csv(file_path: str) -> list:
    data = []
    with open(file_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        header = csv_reader.fieldnames
        for row in csv_reader:
            data.append(row)
    return data

def load_ratings(file_path:str) -> dict:
    csv_data = load_csv(file_path)
    head_ratings = dict() # structure: key = (compound,head), val = mean rating
    modifier_ratings = dict() # structure: key = (compound,modifier), val = mean rating
    all_ratings = dict() # structure: key = (compound,constituent), val = mean rating
    for rating in csv_data:
        compound = rating["compound"]
        modifier,head = compound.split()
        if "const" in rating: # file with human annotations
            if rating["const"] == modifier:
                modifier_ratings[(compound,modifier)] = rating["mean"]
            else:
                head_ratings[(compound,head)] = rating["mean"]
            all_ratings[(compound,rating["const"])] = rating["mean"]
        else: # file with measured scores between embeddings
            modifier_ratings[(compound,modifier)] = rating["modifier_score"]
            head_ratings[(compound,head)] = rating["head_score"]
            all_ratings[(compound,modifier)] = rating["modifier_score"]
            all_ratings[(compound,head)] = rating["head_score"]
    return {"heads":head_ratings,"modifiers":modifier_ratings,"all":all_ratings}

def load_sentences(file_paths:list,approach:str) -> dict:
    sentences = dict() # structure: key = target (e.g. compound or constituent), val = sentence containing this target
    for file_path in file_paths:
        with open(file_path,"r") as infile:
            for line in infile:
                sentence_info = line.split("\t")
                target = sentence_info[0].strip()
                if target in sentences:
                    print("check the file again for the target '"+target+"', as it occurs multiple times")
                if approach == "one":
                    target_sentence = [sentence_info[1].strip()] # if want to use more than one prompt: " ".join(sentence_info[1:])
                elif approach == "multi":
                    target_sentence = [sent.strip() for sent in sentence_info[1:]]
                sentences[target] = target_sentence
    return sentences