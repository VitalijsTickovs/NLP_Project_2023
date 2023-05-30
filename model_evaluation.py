from cgi import test
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import numpy as np
from collections import Counter
import re

def build_id2title_dictionary(catalog):
    return dict(zip(catalog['id'], catalog['title']))

def get_elements_from_keys(mapping_dictionary, key_list):
    """ get a list of entries using a dictionary and a key list """
    
    elements = [mapping_dictionary.get(key) for key in key_list]
    return elements

def calculate_bleu(predicted_titles, target_titles):
    # create a corpus
    wrapped_target_titles = [[tt] for tt in target_titles]
    print('target titles are now ready for the bleu')
    
    # calculate corpus_bleu
    print('calculating bleu')
    return corpus_bleu(wrapped_target_titles, predicted_titles)
    
    
def get_titles(ids, catalog):
    # building mapping dictionaries
    id2title_dictionary = build_id2title_dictionary(catalog)
    
    # create predicted titles
    titles = get_elements_from_keys(id2title_dictionary, ids)
    
    return titles

## CATALOG PART ##
def _init_catalog():
    def clean(text,freq_threshold=1):
        text = re.findall(r'\b(?:[a-zA-Z]+|\d+)\b', text)
        new_text = []
        for word in text:
            word = word.lower()
            if word_counter[word] > freq_threshold:
                new_text.append(word)
        text = new_text
        text = " ".join(text)
        
        return text

    products = pd.read_csv('products_train.csv')
    products_eng = products[products['locale']=='UK']

    titles = np.array(products_eng['title'])
    titles = " ".join(titles)

    words = re.findall(r'\w+', titles)
    words = [w.lower() for w in words]
    word_counter = Counter(words)
    word_frequencies = np.array(list(word_counter.values()))

    products_eng['title'] = products_eng['title'].apply(clean)
    catalog = products_eng[['id','title']]
    
    print('Catalog data loaded')
    return catalog
## ----------------##

## TRAIN/TEST SESSIONS ##
def _init_sessions(n=None):
    def clean_items(text):

        text = text[1:-1]
        text = re.findall(r"'([^']*)'", text)

        return text
    
    test_sessions = pd.read_csv('test_sessions.csv')
    test_sessions['prev_items'] = test_sessions['prev_items'].apply(clean_items)
    test_sessions['last_item'] = test_sessions['prev_items'].apply(lambda x: x[-1])
    test_sessions = test_sessions.drop(columns='Unnamed: 0')
    
    print('Test data loaded')
    if n is None:
        return test_sessions
    else:
        return test_sessions.sample(n)
## ----------------##

## EXAMPLE ##
catalog = _init_catalog()
test_sessions = _init_sessions(2)
predicted_titles = ['Hello wolld', 'How are you?']
target_titles = get_titles(test_sessions['next_item'].values, catalog)

print(calculate_bleu(predicted_titles, target_titles))
    