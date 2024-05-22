import requests
import os
import pickle
import time
import math

def get_dict_texts_train():
    dict_texts_train = {}

    for pickle_name in os.listdir("texts_pickles/train/"):

        lang = pickle_name.split("_")[0]
        pickle_path = "texts_pickles/train/" + pickle_name

        with open(pickle_path, 'rb') as file:
            dict_texts_train[lang] = pickle.load(file)

    print("Train languages:",list(dict_texts_train.keys()))
    return dict_texts_train


def get_dict_texts_test():
    dict_texts_test = {}

    for pickle_name in os.listdir("texts_pickles/test/"):

        lang = pickle_name.split("_")[0]
        pickle_path = "texts_pickles/test/" + pickle_name

        with open(pickle_path, 'rb') as file:
            dict_texts_test[lang] = pickle.load(file)

    print("Test languages:",list(dict_texts_test.keys()))
    return dict_texts_test


def tokenize(text, n):
    return [text[i:i+n] for i in range(len(text)-n+1)]

def create_token_freq_table(text, n):

    # 1) create tokens for given text
    tokens = tokenize(text, n)
    # print(tokens[:500])

    # 2) create character frequency table
    token_freqs = {char: 0 for char in set(tokens)}
    # print(len(token_freqs))
    # print(token_freqs)


    total_tokens = 0
    # 3) set character frequencies and normalize them
    for token in tokens:
        if token in token_freqs:
            token_freqs[token] += 1
            total_tokens += 1

    for token in token_freqs:
        token_freqs[token] /= total_tokens  # normalize frequencies
    
    return token_freqs

def detect_lang_distances(text, lang_token_freqs, n):
    lang_dists = {}
    for lang, lang_token_freq in lang_token_freqs.items():
        # 1) create tokens and token-freq table for given text
        tokens = tokenize(text, n)
        text_token_freq = create_token_freq_table(text, n)

        # 2) compare distances of tokens of the text with other languages' tokens
        distance = 0.0
        for tok in tokens:
            text_token = text_token_freq.get(tok, 0)
            lang_token = lang_token_freq.get(tok, 0)
            distance +=  (text_token - lang_token) ** 2

        lang_dists[lang] = distance

    closest_lang = min(lang_dists, key=lang_dists.get)  
    return closest_lang, lang_dists




def detect_lang_likelihood(text, lang_token_freqs, n, penalizer_val=1e-5):
    lang_scores = {}
    for lang, lang_token_freq in lang_token_freqs.items():

        # 1) create tokens for given text
        tokens = tokenize(text, n)

        # 2) calculate likelihood of languages with the given text
        lang_score = 0
        for token in tokens:
            if token in lang_token_freq:
                lang_score += math.log(lang_token_freq[token])
            else:
                lang_score += math.log(penalizer_val)  

        lang_scores[lang] = lang_score
        
    closest_lang = max(lang_scores, key=lang_scores.get)
    return closest_lang, lang_scores

def main():

    dict_texts_train = get_dict_texts_train()

    N_GRAM_VAL = 2

    # 1) Create language representations = token frequency tables
    lang_token_freqs = {}
    for lang, text in dict_texts_train.items():
        lang_token_freqs[lang] = create_token_freq_table(text,
                                                         n=N_GRAM_VAL)
    
    # 2) Set your text
    YOUR_TEXT = "Merhaba, ben Unat! Bilgisayar mühendisiyim!"
    print("YOUR TEXT: ",YOUR_TEXT)

    # 3) Detection methods
    # 3.1) Likelihood Method

    detected_lang, likelihoods = detect_lang_likelihood(YOUR_TEXT, 
                                                        lang_token_freqs, 
                                                        n=N_GRAM_VAL,
                                                        penalizer_val=1e-12)


    print("Likelihood method:")    
    print(f'The detected language is: {detected_lang}')
    print(likelihoods)

    print("-"*70)


    # 3.2) Distance Method

    detected_lang, distances = detect_lang_distances(YOUR_TEXT, 
                                                     lang_token_freqs, 
                                                     n=N_GRAM_VAL)

    print("Distance method:")    
    print(f'The detected language is: {detected_lang}')
    print(distances)


if __name__ == "__main__":
    main()
