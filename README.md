# ComposiGen and compositionality predictions 

This repository contains the code and data for our paper accepted at LREC 2026!

**Fruitcakes and Cupcakes Emerging from Noise: The ComposiGen Dataset of Compounds and their Compositionality**

## *ComposiGen* dataset 📂

This folder contains files representing the *ComposiGen* dataset. Also scripts used to create the dataset are included. 

Broad folder structure:
* ``CompositionalityRatings``: human ratings of compositionality
* ``Images``: generated images (via text-guided text-to-image generation and text-guided image-to-image generation) along with their (automatic and human-elicited) evaluation scores
* ``NounDefinitions``: generated noun definitions for constituents and compounds
* ``Subsets``: subsets of the 200 compounds used for the subset investigations
* ``compounds.csv``: list of 200 noun-noun compounds

Please note that we did not upload the generated images. If you want to get access to our full dataset, please reach out to us via email.

## Compositionality Prediction Experiments 🔎

Broad folder structure:
* ``ChatGPT``: few-shot ratings of compositionality using ChatGPT
* ``Scripts``: scripts written for the compositionality prediction task

Please note that the intermediate steps and results of the compositionality prediction pipeline are not included in this repository. Predicted scores, however, are included in the ``all.json`` file (see below).

## ``all.json`` 🥳
We summarized our work in one file! It contains our 200 compounds along with additional information (such as concreteness or paths to images), human-elicited compositionality ratings and predicted scores of compositionality.

File structure:

```
{ "id":
    {
        "compound": "tokens",
        "frequency": val, 
        "concreteness": "cat",
        "constituents": {
            "modifier": {
                "token": "token",
                "concreteness": val,
                "definitions": ["def1", "def2", "def3"],
                "images": [("/path/to/image_dir/...",VQAScore,human_rating)]
            },
            "head": {
                "token": "token",
                "concreteness": val,
                "definitions": ["def1", "def2", "def3"],
                "images": [("/path/to/image_dir/...",VQAScore,human_rating)]
            },
        },        
        "definitions": ["def1", "def2", "def3"],
        "images": {
            "n2i": [("/path/to/image_dir/...",VQAScore,human_rating)],
            "m2i": [("/path/to/image_dir/...",VQAScore), ...],
            "h2i": [("/path/to/image_dir/...",VQAScore), ...],
        },
        "compositionality scores": {
            "head-specific": {
                "human": rating,
                "predictions": {
                    "feature-based": {
                        "Skip-gram": score,
                        "BERT": score,
                        "ViT": score,
                        "early fusion": score,
                        "late fusion": score,
                        "CLIP": score,
                        "FLAVA": score,
                        },
                    "parameter-based": score,
                    "ChatGPT": score,
                    }
            },
            "modifier-specific: {
                "human": rating,
                "predictions": {
                    "feature-based": {
                        "Skip-gram": score,
                        "BERT": score,
                        "ViT": score,
                        "early fusion": score,
                        "late fusion": score,
                        "CLIP": score,
                        "FLAVA": score,
                        },
                    "parameter-based": score,
                    "ChatGPT": score,
                    }
            }
        }
    }
}
```

## Installation and Running 💻

In order to run the scripts in this repository you need to install python and create two virtual environments (one for our code and one is needed when wanting to replicate the VQAScores). Further details are provided in the ``instructions.md``-file. 

Depending on where you save this repository, you still need to replace ``/path/to/repo/`` by the according paths.
Also, for some scripts you need to set a cache directory by replacing ``/path/to/cache/``. In some files ``/path/to/data/`` refers to existing other data resources. 


## Citations

If you find our work useful for your research, please use the following BibTex entry.

```
@TODO{}
```