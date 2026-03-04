# Images
This folder represents a placeholder for images of compounds and constituents generated using diffusion models, along with their automatic and human evaluation scores. 
In order to get access to the images, please reach out to us via email.

## Image generation with `image_generator.py`
Noun definitions used as textual guidance are saved in ``Data/NounDefinitions/``. 

The script requires four (+ three optional) command-line arguments:

1. **GPU device** (`-g` or `--gpu`)
   - GPU device on which the image generation process should run.
   - Example:  `--gpu 0` 

2. **Image generation model** (`-m` or `--model`)
   - Example:  `--model flux` 

3. **Prompts for textual guidance** (`-p` or `--prompts`)
   - Path to tsv file where prompts are saved. Each line consists of target along with the prompts (tab-separated).
   - Example: `--prompts /path/to/repo/Data/NounDefinitions/chatgpt_compound_definitions.tsv`

4. **Experiment identifier** (`-i` or `--identifier`)
   - Identifier of the experiment / image generation run.
   - Example: `--identifier head2img`

5. **Starter images** [optional] (`-s` or `--starter_images`)
   - This is only needed when doing image-to-image generation and the according start images need to be specified.
   - Path to json file that lists where starter images are saved.
   - Example: `--starter_images /path/to/repo/Data/Images/CompoundImages/head2img/mapping_compound2ChatGPTdefinitionHeadImg.json`

6. **Negative prompts** [optional] (`-n` or `--negative_prompts`)
   - We didn't use this but in theory one can use negative prompts during generations.
   - Path to tsv file where negative prompts are saved. Each line consists of target along with the negative prompts.
   - Example: `--negative_prompts ...`

7. **Experiment identifier** [optional] (`-d` or `--directory_to_images`)
   - Directory where the generated images should be saved.
   - Example: `--directory_to_images "/path/to/repo/Data/Images/CompoundImages/head2img/flux/chatgpt_compound_definition_prompts/direct approach/"`   

**Example command**: 
```bash
python image_generator.py --gpu 0 --model flux --prompts /path/to/repo/Data/NounDefinitions/chatgpt_compound_definitions.tsv --identifier head2img --starter_images /path/to/repo/Data/Images/CompoundImages/head2img/mapping_compound2ChatGPTdefinitionHeadImg.json --directory_to_images "/path/to/repo/Data/Images/CompoundImages/head2img/flux/chatgpt_compound_definition_prompts/direct approach/"
```
or in another script:
```bash
import image_generator
import sys
sys.argv = ['image_generator.py', '--gpu', "0", '--model', "flux", '--prompts', "/path/to/repo/Data/NounDefinitions/chatgpt_compound_definitions.tsv", '--identifier', "head2img", '--starter_images', "/path/to/repo/Data/Images/CompoundImages/head2img/mapping_compound2ChatGPTdefinitionHeadImg.json", '--directory_to_images', "/path/to/repo/Data/Images/CompoundImages/head2img/flux/chatgpt_compound_definition_prompts/direct approach/"]
image_generator.main()
```
---


## Overview
* ``CompoundImages``: Contains generated compound images from all three approaches (noise2img, head2img (= head-to-compound), modifier2img (= modifier-to-compound)). For the noise2img approach it's one image per compound, for the other two approaches it's 11 images per compound each - the used denoising strength is encoded in the according filename.
* ``ConstituentImages``: Contains generated constituent images. One image per constituent. 
* ``ImageEvaluations``: Contains VQAscores and jupyter notebook that shows how to retrieve them.
* ``ImageRatings``: Contains human ratings for noise-to-image generated images.