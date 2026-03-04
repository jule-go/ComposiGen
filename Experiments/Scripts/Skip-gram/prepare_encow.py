from tqdm import tqdm

# load the set of compounds
compounds = []
with open("/path/to/repo/Experiments/Scripts/Skip-gram/targets.txt","r") as f:
    for line in f:
        compounds += [line.strip()]
print("targets are loaded")

# define replacement mapping scheme
replacement_map = dict() # key = old form (open,closed,hyphenated), new form (with underscore)
for compound in compounds:
    mod,head = compound.split()
    forms = [mod+head,mod+" "+head,mod+"-"+head] # consider closed, open, hyphenated occurrences (ignore underscore format here -> no need to mention as we keep this structure)
    for form in forms:
        replacement_map[form] = mod+"_"+head
print("replacement map is created")

# use lemmatised sentence version of encow-corpus 
available_file_numbers = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]
file_path_start = "/path/to/data/encow/lemma-"
file_path_end = ".gz"

# specify where to save the processed version
save_directory = "/path/to/data/encow/processed/"

# preprocess the corpus by replacing occurrences of compounds (in closed/open/hyphenated/underscored format) with underscore form in sentences
for file_num in available_file_numbers:
    print("Processing slice ",file_num,"/",str(len(available_file_numbers)),":")
    current_file = file_path_start+file_num+file_path_end

    # read lemmatized ENCOW subset and save sentences into list
    sentences = []
    with open(current_file,"r") as infile: 
        lines = infile.readlines()
    sentences = ["".join(line) for line in lines] # convert lines to strings (representing sentences)

    # now replace occurrences in sentences
    sentences_processed = []
    print("\tProcessing sentences...")
    for sentence in tqdm(sentences):
        sentence_processed = sentence
        for form,replacment in replacement_map.items():
            if form in sentence:
                sentence_processed = sentence_processed.replace(form,replacement_map[form]) # replace the occurrence with underscore format
        sentences_processed += [sentence_processed]

    # save sentences to file
    print("\tSave processed sentences to file...")
    with open(save_directory+"processed-"+file_num+".txt","w") as outfile:
        for sentence in sentences_processed:
            outfile.write(sentence+"\n")