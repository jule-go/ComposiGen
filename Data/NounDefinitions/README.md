# Noun definitions
This folder contains noun definitions for constituents and compounds generated with ChatGPT. Note that three definitions per target have been generated while only one definition is used as part of our experiments. We prompted OpenAI's ChatGPT-4o model via the free web version on 13/14th of January 2025.

## Prompting

The prompting happens in five phases:
1. **Preparation phase**: The objective is explained and the task is embedded within a broader context.
2. **Instruction phase**: The task is specified in detail by providing concrete instructions. 
3. **Example phase**: An example is provided to illustrate how the task should be completed.
4. **Clarification phase**: ChatGPT is asked to confirm that it understood the task and is ready to proceed.
5. **Query phase**: Now the actual prompting starts. Noun definitions are generated in batches of 25 targets. Here, a compound together with its two constituents is referred to as target. We don't ask for duplicate definitions.

Actual wording:
* Our prompt:
    ```
    Hi! You are an intelligent machine, generating prompts that are suitable inputs for image generation models. In order to generate good images, it is necessary to have prompts with a fine-grained level of detail that are of high quality. You are the expert for generating such image generation prompts!

    I will provide you with a list of targets. Those can be unigrams or bigrams. Every target is written in a separate line. 
    !!! Your task is to first understand the targets, and then to list three different noun definitions of this target you consider suitable as image generation prompt !!! 
    The definition prompts can consist of tags or natural language sentences. They should span a maximum of 75 tokens each. Every prompt should be highly informative on itself. Don’t include duplicate prompts, the three definition prompts should differ from each other. Please list the definition prompts as if writing to a txt-file. Please refer to the example below for the desired format.  

    ----------- Example: my input -----------
    couch potato
    couch
    potato
    ...

    ----------- Example: your output --------	
    Definitions for "couch potato":
    1. ‘A person who spends a significant amount of time sitting or lying down, typically watching television or engaging in sedentary activities.‘
    2. ‘A term describing someone who leads a sedentary lifestyle, preferring indoor activities such as watching TV or playing video games.‘
    3. ‘An informal term for a person who is inactive or lazy, often spending leisure time on a couch or sofa.‘

    Definitions for "couch":
    1. ‘A piece of furniture designed for seating two or more people, typically with a back and armrests.‘
    2. ‘A long upholstered piece of furniture for reclining or sitting, often found in living rooms or lounges.‘
    3. ‘A sofa or settee, usually with cushions and upholstered arms and back, used for relaxation or casual seating.‘

    Definitions for "potato":
    1. ‘An edible tuber that is a staple food in many cultures, typically underground and harvested from the Solanum tuberosum plant.‘
    2. ‘A starchy vegetable with a variety of culinary uses, such as boiling, baking, frying, or mashing.‘
    3. ‘The plant itself, Solanum tuberosum, which belongs to the nightshade family and produces tubers that vary in size, shape, and color.‘
    ...

    Please let me know if we can start or if you have questions that need further clarification!
    ```
* ChatGPT's answer:
    ```
    Got it! I understand the task perfectly. You provide a list of targets, and I will generate three distinct, high-quality noun definitions for each target, suitable for image-generation prompts. Let me know when you're ready to start and share the targets!
    ```
* [then we started providing our compounds and constituents as targets, e.g. example excerpt showing two targets only]
    ``` 
    Here are the targets:
    ------------------- 
    concept album 
    concept 
    album 
    family album 
    family 
    music 
    album
    ```

Afterwards we formatted the raw outputs into according tsv-files without changing their content.

## Overview
* ``chatgpt_compound_definitions.tsv``: Contains three noun definitions for each compound generated using ChatGPT. 
* ``chatgpt_constituent_definitions.tsv``: Contains three noun definitions for each constituent generated using ChatGPT. 