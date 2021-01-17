
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import nltk
from nltk import word_tokenize

def return_similar(in_text):
    
    processed=pd.read_pickle(r"F:\NUS-NLP Project\combined_balanced_100k_text_processed_2610.pkl")
    check_back=processed['reviewText']

    model= Doc2Vec.load(r"F:\NUS-NLP Project\d2v500e50V.model")
    #to find the vector of a document which is not in training data
    test_data = word_tokenize(in_text.lower())
    v1 = model.infer_vector(test_data)
    #print("V1_infer", v1)
    
    # to find most similar doc using tags
    similar_doc = model.docvecs.most_similar([v1])
    #print(similar_doc)

    return(check_back[int((similar_doc[0][0]))])
    # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
    #print(model.docvecs['1'])
