Create virtual environment 'env' for the compositionality predictions:
 * [we suggest using python 3.11]
 * set up the environment by running the following commands in the activated environment:
    ```
    python -m pip install jupyter==1.1.1
    python -m pip install ipykernel==6.29.5
    python -m pip install transformers==4.49.0
    python -m pip install numpy==1.26.4
    python -m pip install diffusers==0.32.2
    python -m pip install gensim==4.3.3
    python -m pip install fasttext==0.9.3
    python -m pip install torch==2.5.1
    python -m pip install torchvision==0.20.1
    python -m pip install datasets==3.1.0
    python -m pip install bing-image-downloader==1.1.2
    python -m pip install nltk==3.9.1
    python -m pip install matplotlib==3.10.1
    python -m pip install scikit-learn==1.6.0
    python -m pip install sentencepiece==0.2.0
    python -m pip install ftfy==6.3.1
    python -m pip install natsort==8.4.0
    python -m pip install accelerate==1.1.1
    python -m pip install spacy==3.8.2
    python -m pip install numpy==1.26.4
    python -m spacy download en_core_web_lg
    python -m ipykernel install --user --name=env
    python -m pip install protobuf==6.30.2
    python -m pip install adjustText==1.3.0 


Create virtual environment 'img_env' for the automatic image evaluation scores:
* [we suggest using python 3.10]
* set up the environment by running the following commands in the activated environment:
    ```
    python -m pip install jupyter
    python -m pip install ipykernel
    python -m ipykernel install --user --name=img_env
    git clone https://github.com/linzhiqiu/t2v_metrics
    python -m pip install torch torchvision torchaudio
    python -m pip install git+https://github.com/openai/CLIP.git
    cd t2v_metrics	
    python -m pip install -e .
    python -m pip upgrade transformers==4.37.2		
* in case of problems, please refer to the original instructions in the repository