# Compositionality ratings from ChatGPT
This folder contains compositionality ratings directly extracted from ChatGPT. We prompted OpenAI's ChatGPT-4o model via the free web version on 27th of August 2025. The instructions are similar to those we gave to the human annotators.

## Prompting

The prompting happens in five phases:
1. **Preparation phase**: The objective is explained and the task is embedded within a broader context.
2. **Instruction phase**: The task is specified in detail by providing concrete instructions. 
3. **Example phase**: An example is provided to illustrate how the task should be completed.
4. **Clarification phase**: ChatGPT is asked to confirm that it understood the task and is ready to proceed.
5. **Query phase**: Now the actual prompting starts. One compound is queried after the other always asking for the modifier rating first. We queried ChatGPT in batches of 25 compounds. 

Actual wording:
* Our prompt:
    ```
    Hi! You are an expert annotator for linguistic tasks. This time your task involves complex English expressions, in particular noun-noun compounds. Your task is to evaluate the extent to which the overall meaning of a compound can be related to the meaning of its parts. Here are some example compounds with different degrees of compositionality:
        •  The meaning of search engine is related to the meanings of search and engine.
        •  The meaning of blackmail is not related to the meanings of black or mail.
        •  The meaning of strawberry is not related to the meaning of straw, but it is related to the meaning of berry.

    I will provide you with a list of noun-noun compounds. Every compound is written in a separate line. Its constituents are separated with whitespace.
    !!! Your task is to understand the meaning of the compound as well as the meanings of its constituents and then to provide constituent-specific ratings on compositionality. !!! 
    The compositionality ratings should be on a scale between 0 (definitely opaque, i.e. low compositionality) and 5 (definitely transparent, i.e. high compositionality). Feel free to use the whole range. For each compound you are expected to provide two compositionality ratings: one with respect to the first constituent and one with respect to the second constituent. Please provide the ratings in the format “compound,constituent,rating”, using a separate line for each compound-constituent combination, as if writing to a txt-file. Please refer to the example below for the desired format. 

    ----------- Example: my input -----------
    flea market
    spelling bee
    graduate student
    ...

    ----------- Example: your output --------	
    flea market,flea,0.379
    flea market,market,4.714
    spelling bee,spelling,4.815
    spelling bee,bee,0.517
    graduate student,graduate,4.700
    graduate student,student,5.000
    ...

    Please let me know if we can start or if you have questions that need further clarification!
    ```
* ChatGPT's answer:
    ```
    Got it! ✅ Thanks for the detailed explanation and the clear example — that helps a lot.
    I’ll be evaluating each compound by considering how much the meaning of the whole compound depends on each of its constituents, then giving a score from 0 (totally opaque) to 5 (fully transparent). The output format is perfectly clear as well.

    I’m ready to start whenever you provide the list of noun-noun compounds. Would you like to send me the first batch now?
    ```
* [then we started providing our compounds, e.g. example excerpt showing two compounds only]
    ``` 
    Here are the compounds:
    ------------------- 
    concept album 
    family album 
    ```

Afterwards we formatted the raw outputs into a csv-file without changing the content.

## Overview
* ``chatgpt_compound_definitions.tsv``: Contains compositionality ratings from prompting ChatGPT.