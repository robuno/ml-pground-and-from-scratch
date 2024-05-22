import requests
import os
import pickle
import time
import math

def merge_sentences_pickle(sentences, pickle_file=None):
    merged_text = ' '.join(sentences)
    if pickle_file:
        with open(pickle_file, 'wb') as file:
            pickle.dump(merged_text, file)
        print(f"Merged text pickled to: {pickle_file}")

        file_size = os.path.getsize(pickle_file)
        print(f"Size of the pickled file: {file_size / (1024 * 1024)} MB with {len(sentences)} sentences!")

    return merged_text

def download_treebank(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded treebank to {output_path}")

def extract_sentences(conllu_file, print_info=False):
    sentences = []
    with open(conllu_file, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(' '.join(sentence))
                    sentence = []
            elif not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) > 1:
                    sentence.append(parts[1])
        if sentence:
            sentences.append(' '.join(sentence))
        
    
    if print_info == True:
        print(f"Number of sentences: {len(sentences)}")
    return sentences


def check_dirs(directories):

    for directory in directories:
        if os.path.exists('./'+directory):
            print(f"The directory '{'./'+directory}' exists.")
        else:
            os.makedirs(directory)
            print(f"The directory '{'./'+directory}' does not exist.")


def main():

    req_directories = ["texts/train/", "texts/test/", "texts_pickles/train/", "texts_pickles/test/"]
    check_dirs(req_directories)


    treebank_train_urls = [
    "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/master/tr_kenet-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/master/nl_alpino-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-FicTree/master/cs_fictree-ud-train.conllu",
    # "https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/master/sv_talbanken-ud-train.conllu",
    # "https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master/ro_rrt-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master/es_gsd-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-TDT/master/fi_tdt-ud-train.conllu",
    # "https://raw.githubusercontent.com/UniversalDependencies/UD_Croatian-SET/master/hr_set-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Slovenian-SSJ/master/sl_ssj-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-LFG/master/pl_lfg-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu",
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-GSD/master/pt_gsd-ud-train.conllu"
    ]

    treebank_test_urls = [
        "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/master/tr_kenet-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Dutch-Alpino/master/nl_alpino-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-FicTree/master/cs_fictree-ud-test.conllu",
        # "https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/master/sv_talbanken-ud-test.conllu",
        # "https://raw.githubusercontent.com/UniversalDependencies/UD_Romanian-RRT/master/ro_rrt-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master/es_gsd-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Finnish-TDT/master/fi_tdt-ud-test.conllu",
        # "https://raw.githubusercontent.com/UniversalDependencies/UD_Croatian-SET/master/hr_set-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Slovenian-SSJ/master/sl_ssj-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Polish-LFG/master/pl_lfg-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-test.conllu",
        "https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-GSD/master/pt_gsd-ud-test.conllu"
    ]


    treebank_train_info = {}

    for url_tree in treebank_train_urls:
        file_name = url_tree.split('/')[-1]
        output_file = "texts/train/"+  file_name
        pickle_file = "texts_pickles/train/" + file_name.replace(".conllu", ".pkl")

        download_treebank(url_tree, output_file)
        sentences = extract_sentences(output_file, print_info=True)
        merge_sentences_pickle(sentences, pickle_file=pickle_file)

        treebank_train_info[file_name.split(".")[0]] = len(sentences)
        
        for i, sentence in enumerate(sentences[:2]): 
            print(f"Sentence {i+1}: {sentence}")
        print("-"*20+"\n")



    treebank_test_info = {}

    for url_tree in treebank_test_urls:
        file_name = url_tree.split('/')[-1]
        output_file = "texts/test/"+  file_name
        pickle_file = "texts_pickles/test/" + file_name.replace(".conllu", ".pkl")

        download_treebank(url_tree, output_file)

        sentences = extract_sentences(output_file, print_info=True)
        print(type(sentences))
        with open(pickle_file, 'wb') as file:
            pickle.dump(sentences, file)
            print(f"Sentences pickled to: {pickle_file}")


        treebank_test_info[file_name.split(".")[0]] = len(sentences)
        
        for i, sentence in enumerate(sentences[:2]): 
            print(f"Sentence {i+1}: {sentence}")
        print("-"*20+"\n")


if __name__ == "__main__":
    main()