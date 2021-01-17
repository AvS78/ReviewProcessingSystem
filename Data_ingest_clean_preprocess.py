
import json
import csv
import random
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec

data_new=pd.read_pickle("combined_balanced_10k_text_processed_2510.pkl")
data_new.head()

# Create a stopword list from the standard list of stopwords available in nltk
stop = stopwords.words('english')

porter=PorterStemmer() # porter stemmer for word stemming
wnl=nltk.WordNetLemmatizer()

def check_word(text):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', text) is not None)

def process_text_review(review):
    processed_review = []
    review = review.lower()
    review = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', review)
    review = re.sub(r'\.{2,}', ' ', review)
    review = review.strip(' "\'')
    review = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', 'POS_FEELING', review)
    review = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', 'POS_FEELING', review)
    review = re.sub(r'(<3|:\*)', 'POS_FEELING', review)
    review = re.sub(r'(;-?\)|;-?D|\(-?;)', 'POS_FEELING', review)
    review = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', 'NEG_FEELING', review)
    review = re.sub(r'(:,\(|:\'\(|:"\()', 'NEG_FEELING', review)
    
    review = re.sub(r'\s+', ' ', review)
    review=review.replace(".", " ")
    text = review.split()
    for word in text:

        word = word.strip('\'"?!,.():;')
        word = re.sub(r'(.)\1+', r'\1\1', word)
        word = re.sub(r'(-|\')', '', word)
        if check_word(word):
            if word not in stop:
                # process only non stop words
                if stemming:
                    word = str(porter.stem(word))
                if lemmatize:
                    word=wnl.lemmatize(word)
    processed_review.append(word)

    return ' '.join(processed_review)

# organise datasets separately
a_clothing_file=r"E:\NLP Project\Amz Dataset\Clothing_Shoes_and_Jewelry_5.json"
a_movies_file=r"E:\NLP Project\Amz Dataset\Movies_and_TV_5.json"
a_beauty_file=r"E:\NLP Project\Amz Dataset\Beauty_5.json"
a_cell_file=r"E:\NLP Project\Amz Dataset\Cell_Phones_and_Accessories_5.json"
a_electronics_file=r"E:\NLP Project\Amz Dataset\Electronics_5.json"
a_home_file=r"E:\NLP Project\Amz Dataset\Home_and_Kitchen_5.json"
a_pet_file=r"E:\NLP Project\Amz Dataset\Pet_Supplies_5.json"
a_toys_file=r"E:\NLP Project\Amz Dataset\Toys_and_Games_5.json"
a_health_file=r"E:\NLP Project\Amz Dataset\Health_and_Personal_Care_5.json"
a_grocery_file=r"E:\NLP Project\Amz Dataset\Grocery_and_Gourmet_Food_5.json"

df_beauty=pd.read_json(a_beauty_file, lines=True)
df_clothing=pd.read_json(a_clothing_file, lines=True)
df_movies=pd.read_json(a_movies_file, lines=True)
df_cell=pd.read_json(a_cell_file, lines=True)
df_electronics=pd.read_json(a_electronics_file, lines=True)
df_home=pd.read_json(a_home_file, lines=True)
df_pet=pd.read_json(a_pet_file, lines=True)
df_grocery=pd.read_json(a_cell_file, lines=True)
df_health=pd.read_json(a_health_file, lines=True)
df_toys=pd.read_json(a_toys_file, lines=True)


print(df_beauty.shape)
print(df_clothing.shape)
print(df_movies.shape)
print(df_cell.shape)
print(df_electronics.shape)
print(df_home.shape)
print(df_pet.shape)
print(df_cell.shape)
print(df_grocery.shape)
print(df_health.shape)


df_beauty=df_beauty[['reviewText', 'overall']]
df_beauty['dept']='beauty'
df_clothing=df_clothing[['reviewText', 'overall']]
df_clothing['dept']='clothing'
df_movies=df_movies[['reviewText', 'overall']]
df_movies['dept']='movies'
df_cell=df_cell[['reviewText', 'overall']]
df_cell['dept']='cell'
df_electronics=df_electronics[['reviewText', 'overall']]
df_electronics['dept']='electronics'
df_home=df_home[['reviewText', 'overall']]
df_home['dept']='home'
df_pet=df_pet[['reviewText', 'overall']]
df_pet['dept']='pet'
df_grocery=df_grocery[['reviewText', 'overall']]
df_grocery['dept']='grocery'
df_health=df_health[['reviewText', 'overall']]
df_health['dept']='health'
df_toys=df_toys[['reviewText', 'overall']]
df_toys['dept']='toys'


df_beauty.to_pickle("a_beauty.pkl")
df_clothing.to_pickle("a_clothing.pkl")
df_movies.to_pickle("a_movies.pkl")
df_electronics.to_pickle("a_electronics.pkl")
df_home.to_pickle("a_home.pkl")
df_pet.to_pickle("a_pet.pkl")
df_grocery.to_pickle("a_grocery.pkl")
df_health.to_pickle("a_health.pkl")
df_toys.to_pickle("a_toys.pkl")
df_cell.to_pickle("a_cell.pkl")


# select a small sample from each dataset
sample_size=10000
df_electronics=df_electronics.sample(n = sample_size) 
df_movies=df_movies.sample(n = sample_size) 
df_home=df_home.sample(n=sample_size)
df_beauty=df_beauty.sample(n=sample_size)
df_clothing=df_clothing.sample(n=sample_size)
df_pet=df_pet.sample(n=sample_size)
df_grocery=df_grocery.sample(n=sample_size)
df_toys=df_toys.sample(n=sample_size)
df_cell=df_cell.sample(n=sample_size)
df_health=df_health.sample(n=sample_size)

print(df_electronics.shape)
print(df_movies.shape)
print(df_home.shape)
print(df_health.shape)
print(df_beauty.shape)
print(df_clothing.shape)
print(df_pet.shape)
print(df_grocery.shape)
print(df_toys.shape)
print(df_cell.shape)


combined_df=df_beauty
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_cell,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_clothing,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_electronics,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_grocery,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_health,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_home,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_movies,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_pet,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())
combined_df=combined_df.append(df_toys,ignore_index=True)
print("shape", combined_df.shape, "head", combined_df.tail())


combined_df.groupby(['dept']).agg(['count'])


combined_df.to_pickle("combined_balanced_10k.pkl")

data=pd.read_pickle("combined_balanced_10k.pkl")
print(data.info())

# examine number of reviews in each dept to see dataset is reasonably balancced


data['sentiment'] = np.where(data['overall']<=3, "negative", "positive")
data['normalized_overall']=(data['overall'])/5
data.groupby(['sentiment']).agg(['count'])



stemming=False
lemmatize=False
processed_text = data['reviewText'].apply(lambda x : process_text_review(x))
data['processed_text']=processed_text
stemming=True
lemmatize=False
processed_with_stem = data['reviewText'].apply(lambda x : process_text_review(x))
data['processed_with_stem']=processed_with_stem
stemming=False
lemmatize=True
processed_with_lemmatize = data['reviewText'].apply(lambda x : process_text_review(x))
data['processed_with_lemmatize']=processed_with_lemmatize
data.to_pickle("combined_balanced_10k_text_processed.pkl")


