# Image-Word Correspondence Ratings
This folder contains files from annotating text-to-image generated images for 200 English noun-noun compounds and their 172 different constituents. The ratings have been collected from February 2026 to March 2026 using the Prolific platform. The set of images has been split into 12 batches. For each target image at least 3 annotators provided a rating on a scale between 0 (= image depicts word not at all) to 4 (= image depicts word perfectly).  
For anonymity reasons, Prolific worker-ids are replaced by new ids.

## Survey instructions
* Participants were asked to judge how well the image depicts the given word mentioned in the question.
* For each image a separate page is shown. 

Example for "chamber music":

<img src="./example-question_chambermusic.png" alt="Example question: chamber music" width="300" height="200">

## Participant criteria
* 1st language and earliest language in life: English
* One of the following participant locations: UK, Ireland, USA, Australia, Canada
* Approval rate: 95-100

## Response quality assurance
* In order to maintain response quality, participants also need to pass certain control instances. 
* Control instances: Here participants are instructed to select the listed value on the scale.
* For each rejected participant, a replacement participant was added to complete the survey.

## Result
* Results are processed into a tsv file with one averaged rating per line.
```bash
token	0	1	2	3	mean	std	category	conccat	VQA	batch
bike	4.0	4.0	4.0		4.0	0.0	modifier	C	0.9924596548080444	0
...
```

## Overview
* ``image_ratings.tsv``: Contains the image ratings, along with their mean and standard deviation. (One image rating per line, for all 372 target images.)
* ``example-question_artcritic.png``: Example question from google form when asking for compound-constituent ratings related to the compound "art critic".