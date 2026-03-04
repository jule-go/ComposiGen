# Compositionality Prediction Pipeline
The Python scripts cover the traditional feature-based compositionality prediction pipeline, including the extraction of features for constituents and compounds, the calculation of compositionality scores by looking at the cosine similarity, and finally correlating those scores with the human compositionality ratings. 
Results for the different steps are not solved within this repository. 

## Scripts Overview

### `helper.py`
Contains some helper functions useful for the other scripts.

### `create_mapping_files.ipynb`
Extracts list of (img,target)-pairs when looking at directories of generated images. These pairs are later used as input to multimodal embedding models.

### `extract_text_features.py`
Extracts features for compound / constituent representation from strings. Several approaches / models can be used: BERT, SkipGram.

The script requires four command-line arguments. There is one additional argument that is optional:

1. **Input File** (`-i` or `--input_file`)
   - A csv file containing the compounds along with its constituents.
   - Format of csv file: One compound per line. Headers are: ID,noun_compound,component_1,component_2
   - Example file structure:
     ```
     0,ballet shoe,ballet,shoe
     1,dog dinner,dog,dinner
     ...
     ```
   - Example:  `--input_file "/path/to/repo/Data/compounds.csv"` 

2. **Output Directory** (`-o` or `--output_dir`)
   - Directory where embeddings will be stored.
   - Example: `--output_dir "/path/to/repo/Experiments/temp/Embeddings/"`

3. **Model** (`-m` or `--model`)
   - Model / approach to use for extracting the features. 
   - Available models are: "skipgram", "bert-avg-num" (with num between 0 and 12), "bert-sum-num" (with num between 0 and 12), "bert-cls-num" (with num between 0 and 12), "bert-pool"
   - Example: `--model skipgram` will use the skipgram model for extracting textual features.

4. **Compound format** (`-f` or `--format_of_compound`)
   - Format of the compound is either "underscore", "hyphenated", "open", "closed", "sentence", or "multi"
   - Note: in the case of "sentence", ChatGPT's first definition of the according target is used as context when extracting the target embedding. If you want to change the origin of where the sentence is taken from: Specify a different file in the script.
   - Example: `--format_of_compound "underscore"`

5. **GPU device** (`-g`or `--gpu`)
   - GPU device on which the feature extraction process should run
   - (optional argument) if left out "cpu" is selected
   - Example: `--gpu 0` will use GPU 0 when extracting textual features.


**Example command**: 
```bash
python extract_text_features.py -i /path/to/repo/Data/compounds.csv -o /path/to/repo/Experiments/temp/Embeddings/ -m skipgram -f underscore -g 0
```
or in another script:
```bash
import extract_text_features
import sys
sys.argv = ['extract_text_features.py', '--i', "/path/to/repo/Data/compounds.csv", '--o', "/path/to/repo/Experiments/temp/Embeddings/", '--m', 'skipgram', '--f', "underscore", '--g', '0']
extract_text_features.main()
```
---

### `extract_image_features.py`
Extracts features for compound / constituent representation from images. The following model can be used: ViT.

The script requires three command-line arguments. There is one additional argument that is optional:

1. **Input Directory** (`-i` or `--input_dir`)
   - A directory where the constituent or compound images are stored. (Note that the script needs to be run for each directory separately.)
   - Example: `--input_dir "/path/to/repo/Data/Images/ConstituentImages/pixartsigma/chatgpt_constituent_definitions/"`

2. **Output Directory** (`-o` or `--output_dir`)
   - Directory where embeddings will be stored.
   - Example: `--output_dir "/path/to/repo/Experiments/temp/Embeddings/"`

3. **Model** (`-m` or `--model`)
   - Model / approach to use for extracting the features. 
   - Available models are: "vit", "resnet"
   - Example: `--model vit` will use the Vision Transformer model for extracting image features.

4. **GPU device** (`-g`or `--gpu`)
   - GPU device on which the feature extraction process should run
   - (optional argument) if left out "cpu" is selected
   - Example: `--gpu 0` will use GPU 0 when extracting textual features.

5. **Limitation** (`-l`or `--limitation`)
   - Limit the images to look at by specifying this pattern. 
   - (optional argument) if not specified all images in the input directory will be considered

**Example command**: 
```bash
python extract_image_features.py -i /path/to/repo/Data/Images/ConstituentImages/pixartsigma/chatgpt_constituent_definitions/ -o /path/to/repo/Experiments/temp/Embeddings/ -m vit -g 0
```
or in another script:
```bash
import extract_image_features
import sys
sys.argv = ['extract_image_features.py', '--i', "/path/to/repo/Data/Images/ConstituentImages/pixartsigma/chatgpt_constituent_definitions/", '--o', "/path/to/repo/Experiments/temp/Embeddings/", '--m', 'vit', '--g', '0']
extract_image_features.main()
```
---

### `extract_multimodal_features.py`
Extracts features for compound / constituent representation from images and text. Several approaches / models can be used: CLIP, FLAVA (taking average or concatenation of embeddings).

The script requires three command-line arguments. There is one additional argument that is optional:

1. **Input File** (`-i` or `--input_file`)
   - A json file where the constituent or compounds along with their image-paths are stored.
   - Example file structure:
     ```
     [["/link/to/image1","target1"],
      ["/link/to/image2","target2"],
     ...]
     ```
   - Example: `--input_file "/path/to/repo/Experiments/temp/Mappings/constituent_pixart_mappings.json"` 

2. **Output Directory** (`-o` or `--output_dir`)
   - Directory where embeddings will be stored.
   - Example: `--output_dir "/path/to/repo/temp/Embeddings/"`

3. **Model** (`-m` or `--model`)
   - Model / approach to use for extracting the features. 
   - Available models are: "clip-average", "clip-concatenation", "clip-sum", "flava-pool", "flava-average-last", "flava-average-num" (with num between 0 and 5), "flava-sum-last", "flava-sum-num" (with num between 0 and 5)
   - Example: `--model clip-average` will use the CLIP model (merging separate embedding by taking the average vector) for extracting multimodal features.

4. **GPU device** (`-g`or `--gpu`)
   - GPU device on which the feature extraction process should run
   - (optional argument) if left out "cpu" is selected
   - Example: `--gpu 0` will use GPU 0 when extracting multimodal features.


**Example command**: 
```bash
python extract_multimodal_features.py -i /path/to/repo/Experiments/temp/Mappings/constituent_pixart_mappings.json -o /path/to/repo/Experiments/temp/Embeddings/ -m clip-average -g 0
```
or in another script:
```bash
import extract_multimodal_features
import sys
sys.argv = ['extract_multimodal_features.py', '--i', "/path/to/repo/Experiments/temp/Mappings/constituent_pixartsigma_mappings.json", '--o', '/path/to/repo/Experiments/temp/Embeddings/', '--m', 'clip-average','--g', '0']
extract_multimodal_features.main()
```
---

### `early_fusion.py`
Uses unimodal extracted features and fuses them into multimodal features.

The script requires six command-line arguments:

1. **Compound file** (`-c` or `--compound_file`)
   - A csv file containing the compounds along with its constituents.
   - Format of csv file: One compound per line. Headers are: ID,noun_compound,component_1,component_2
   - Example file (`/path/to/repo/Data/compounds.csv`):
     ```
     0,ballet shoe,ballet,shoe
     1,dog dinner,dog,dinner
     ...
     ```
   - Example:  `--compound_file "/path/to/repo/Data/compounds.csv"` 

2. **Directory with text-based constituent embeddings** (`-etp` or `--embeddings_text_parts`)
   - Path to directory that contains text embeddings of constituents.
   - Example:  `--embeddings_text_parts "/path/to/repo/Experiments/temp/Embeddings/skip-gram/constituents/"`

3. **Directory with text-based compound embeddings** (`-etc` or `--embeddings_text_compounds`)
   - Path to directory that contains text embeddings of compounds.
   - Example:  `--embeddings_text_compounds "/path/to/repo/Experiments/temp/Embeddings/skip-gram/compounds/underscore/"`

4. **Directory with vision-based constituent embeddings** (`-evp` or `--embeddings_vision_parts`)
   - Path to directory that contains visual embeddings of constituents.
   - Example:  `--embeddings_vision_parts "/path/to/repo/Experiments/temp/Embeddings/vit/constituents/"`

5. **Directory with vision-based compound embeddings** (`-evc` or `--embeddings_vision_compounds`)
   - Path to directory that contains visual embeddings of compounds.
   - Example:  `--embeddings_vision_compounds "/path/to/repo/Experiments/temp/Embeddings/vit/compounds/head2img/flux/"`

6. **Output directory** (`-o` or `--output_dir`)
   - Path to directory where the embeddings from the early fusion should be saved to.
   - Example:  `--output_dir "/path/to/repo/Experiments/temp/Embeddings/early-fusion/"`

**Example command**: 
```bash
python early_fusion.py --compound_file "/path/to/repo/Data/compounds.csv" --embeddings_text_parts "/path/to/repo/Experiments/temp/Embeddings/skip-gram/constituents/" --embeddings_text_compounds "/path/to/repo/Experiments/temp/Embeddings/skip-gram/compounds/underscore/" --embeddings_vision_parts "/path/to/repo/Experiments/temp/Embeddings/vit/constituents/" --embeddings_vision_compounds "/path/to/repo/Experiments/temp/Embeddings/vit/compounds/head2img/flux/" --output_dir "/path/to/repo/Experiments/temp/Embeddings/early-fusion/"
```
or in another script:
```bash
import early_fusion
import sys
sys.argv = ['early_fusion.py', '--compound_file', "/path/to/repo/Data/compounds.csv", '--embeddings_text_parts', "/path/to/repo/Experiments/temp/Embeddings/skip-gram/constituents/", '--embeddings_text_compounds', "/path/to/repo/Experiments/temp/Embeddings/skip-gram/compounds/underscore/", '--embeddings_vision_parts', "/path/to/repo/Experiments/temp/Embeddings/vit/constituents/", '--embeddings_vision_compounds', "/path/to/repo/Experiments/temp/Embeddings/vit/compounds/head2img/flux/", '--output_dir', "/path/to/repo/Experiments/temp/Embeddings/early-fusion/"]
early_fusion.main()
```
---


### `measure_similarity.py`
Calculates similarity between two embeddings (= feature vectors). Several feature types can be used: textual, visual, ... (depending on the model chosen for feature extraction)

The script requires four command-line arguments:

1. **Path to compound file** (`-c` or `--compound_file`)
   - Path to csv file where compounds are saved, one per line.` Headers of file are: ID,noun_compound,component_1,component_2
   - Example file structure:
     ```
     0,ballet shoe,ballet,shoe
     1,dog dinner,dog,dinner
     ...
     ```
   - Example:  `--compound_file "/path/to/repo/Data/compounds.csv"` 

2. **Constituent Directory** (`-p` or `--part_dir`)
   - A directory where the embeddings (= feature vectors) of constituents are stored, between which the cosine similarity will be calculated.
   - Example:  `--part_dir "/path/to/repo/Experiments/temp/Embeddings/skip-gram/constituents/"` 

3. **Compound Directory** (`-w` or `--whole_dir`)
   - A directory where the embeddings (= feature vectors) of compounds are stored, between which the cosine similarity will be calculated.
   - Example:  `--whole_dir "path/to/repo/Experiments/temp/Embeddings/skip-gram/compounds/underscore/"` 

4. **Output Directory** (`-o` or `--output_dir`)
   - Directory where similarity scores will be stored.
   - Example: `--output_dir "/path/to/repo/Experiments/temp/Scores/"`

5. **Compound Limitation** (`-l`or `--limitation_for_compound`)
   - Specifies which image-specific embeddings should be used for measurements.
   - (optional argument)
   - Example: `--limitation_for_compound _FROM_0.9_` will only consider embeddings generated with denoising strength 0.9 when calculating the similarities

**Example command**: 
```bash
python measure_similarity.py -c /path/to/repo/Data/compounds.csv -p /path/to/repo/Experiments/temp/Embeddings/skip-gram/constituents/ -w path/to/repo/Experiments/temp/Embeddings/skip-gram/compounds/underscore/ -o /path/to/repo/Experiments/temp/Scores/
```
or in another script:
```bash
import measure_similarity
import sys
sys.argv = ['measure_similarity.py', '--c', "/path/to/repo/Data/compounds.csv", '--p', "/path/to/repo/Experiments/temp/Embeddings/skip-gram/constituents/", '--w', "path/to/repo/Experiments/temp/Embeddings/skip-gram/compounds/underscore/", '--o', "/path/to/repo/Experiments/temp/Scores/"]
measure_similarity.main()
```
---

### `late_fusion.py`
Uses unimodal compositionality scores and combines them into a multimodal score.

The script requires six command-line arguments:

1. **Compound file** (`-c` or `--compound_file`)
   - A csv file containing the compounds along with its constituents.
   - Format of csv file: One compound per line. Headers are: ID,noun_compound,component_1,component_2
   - Example file (`/path/to/repo/Data/compounds.csv`):
     ```
     0,ballet shoe,ballet,shoe
     1,dog dinner,dog,dinner
     ...
     ```
   - Example:  `--compound_file "/path/to/repo/Data/compounds.csv"` 

2. **Directory with text-based scores** (`-st` or `--scores_text`)
   - Path to csv file where text-based compositionality predictions are saved.
   - Example:  `--scores_text "/path/to/repo/Experiments/temp/Scores/skip-gram_constituents-basic_compounds-underscore.csv"`

3. **Directory with vision-based scores** (`-sv` or `--scores_vision`)
   - Path to csv file where vision-based compositionality predictions are saved.
   - Example:  `--scores_vision "/path/to/repo/Experiments/temp/Scores/vit_constituents-pixartsigma-chatgpt-definitions_compounds-noise2img-flux-chatgpt-definitions.csv"`

6. **Output directory** (`-o` or `--output_dir`)
   - Path to directory where the scores from the late fusion should be saved to.
   - Example:  `--output_dir "/path/to/repo/Experiments/temp/Embeddings/late-fusion/"`

**Example command**: 
```bash
python late_fusion.py --compound_file "/path/to/repo/Data/compounds.csv" --scores_text "/path/to/repo/Experiments/temp/Scores/skip-gram_constituents-basic_compounds-underscore.csv" --scores_vision "/path/to/repo/Experiments/temp/Scores/vit_constituents-pixartsigma-chatgpt-definitions_compounds-noise2img-flux-chatgpt-definitions.csv" --output_dir "/path/to/repo/Experiments/temp/Embeddings/late-fusion/"
```
or in another script:
```bash
import late_fusion
import sys
sys.argv = ['late_fusion.py', '--compound_file', "/path/to/repo/Data/compounds.csv", '--embeddings_text_parts', '--scores_text'. "/path/to/repo/Experiments/temp/Scores/skip-gram_constituents-basic_compounds-underscore.csv", '--scores_vision', "/path/to/repo/Experiments/temp/Scores/vit_constituents-pixartsigma-chatgpt-definitions_compounds-noise2img-flux-chatgpt-definitions.csv", '--output_dir', "/path/to/repo/Experiments/temp/Embeddings/early-fusion/"]
late_fusion.main()
```
---

###  `check_correlation.py`
Compares the measured compositionality scores with the human-elicited compositionality ratings by looking at Spearman's rank correlation. 
Writes results to file.

The script requires three command-line arguments:

1. **Path to first rating file** (`-r1` or `--ratings1_path`)
   - Path to file where ratings (or measured scores) are saved.
   - CSV file is of structure: compound,modifier,modifier_score,head,head_score  
   - Alternative structure (in case of human ratings): compound,const,mean,stand_dev
   - Example:  `--ratings1_path "/path/to/repo/Data/CompositionalityRatings/compositionality_ratings.csv"` 

2. **Input Directory** (`-r2` or `--ratings2_path`)
   - Path to file where other ratings (or measured scores) are saved.
   - CSV file is of structure: compound,modifier,modifier_score,head,head_score  
   - Alternative structure (in case of human ratings): compound,const,mean,stand_dev
   - Example:  `--input_dir "/path/to/repo/Experiments/temp/Scores/skipgram_constituents__skipgram_compounds_underscore.csv"` 

3. **Output Directory** (`-a` or `--analysis_dir`)
   - Directory where results from correlation analysis between ratings1 and ratings2 are saved.
   - Example: `--analysis_dir "/path/to/repo/Experiments/temp/Correlations/"`

**Example command**: 
```bash
python check_correlation.py -r1 /path/to/repo/Data/CompositionalityRatings/compositionality_ratings.csv -r2 /path/to/repo/Experiments/temp/Scores/skipgram_constituents__skipgram_compounds_underscore.csv -a /path/to/repo/Experiments/temp/Correlations/
```
or in another script:
```bash
import check_correlation 
import sys
sys.argv = ['check_correlation.py', '--ratings1_path', "/path/to/repo/Data/CompositionalityRatings/compositionality_ratings.csv", '--ratings2_path', "/path/to/repo/Experiments/temp/Scores/skipgram_constituents__skipgram_compounds_underscore.csv", '--a', "/path/to/repo/Experiments/temp/Correlations/"]
check_correlation.main()
```
---


## Script Sequence
In order to run through the compositionality prediction pipeline run the scripts in the following sequence.
0. (create img-target-mappings with `create_mapping_files.ipynb`)
1. `extract_text_features.py`, `extract_image_features.py`, `extract_multimodal_features.py`. `early_fusion.py`: extract features from text and/or images, and save the vectors to file
2. `measure_similarity.py`: get compositionality scores by measuring cosine similarity between compound and each constituent
3. `late_fusion.py`: get multimodal score by combining unimodal scores
And for evaluation of the scores / ratings:
4. `check_correlation.py`: correlate the measured compositionality scores with the compositionality ratings

## Skip-gram
Is a folder that contains scripts from training our own Skip-gram model.