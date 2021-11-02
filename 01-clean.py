#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from keras import backend as K
import keras
from keras import preprocessing
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


# In[2]:


random.seed(77)


# In[3]:


# read in all books as raw text
books = []
original_books_path = './books/'
for filename in os.listdir(original_books_path):
    if filename.endswith(".txt"):
        print(filename)
        with open(os.path.join(original_books_path, filename), 'rb') as f:
            contents = f.read().decode('utf-8').rstrip()
            books.append(contents)


# In[4]:


len(books[0])


# In[5]:


# only get book contents in between
# *** START OF THE PROJECT GUTENBERG
# and
# *** END OF THE PROJECT GUTENBERG
truncated_books = []
for each_book in books:
    start_of_book = re.search(r"START OF THE PROJECT GUTENBERG", each_book).start()
    end_of_book  = re.search(r"END OF THE PROJECT GUTENBERG", each_book).start()
    print(end_of_book)
    truncated_books.append(each_book[start_of_book+48:end_of_book])


# In[6]:


# def split_whole_book_into_trunks_of_texts(book):
#     # take a long string of a whole book and divide a book into a list of strings
#     # only keep lines that are not empty
#     final_trunks = []
#     whole_trunks = book.split('.')
#     for each_line in whole_trunks:
#         if each_line != '':
#             each_line = each_line.splitlines()
#             final_trunks.append(each_line)
#     return final_trunks


# In[7]:


for each_book in truncated_books:
    each_book = each_book.lower()


# In[8]:


# truncated_books[0] = truncated_books[0].lower()
# ulysses_trunks = nltk.sent_tokenize(truncated_books[0])

# tokenizer = RegexpTokenizer(r'\w+')
# tokens = tokenizer.tokenize(str(a))
# words = [word for word in tokens if word.isalpha()]


# In[9]:


a = clean_text(truncated_books[0])
a


# In[12]:


ulysses_trunks = nltk.sent_tokenize(truncated_books[0])
emma_trunks = nltk.sent_tokenize(truncated_books[1])
dubliners_trunks = nltk.sent_tokenize(truncated_books[2])
prideAndPrejudice_trunks = nltk.sent_tokenize(truncated_books[3])
len(emma_trunks)


# In[13]:


len(prideAndPrejudice_trunks)


# In[14]:


len(dubliners_trunks)


# In[15]:


# ulysses_trunks = split_whole_book_into_trunks_of_texts(truncated_books[0])
# emma_trunks = split_whole_book_into_trunks_of_texts(truncated_books[1])
# dubliners_trunks = split_whole_book_into_trunks_of_texts(truncated_books[2])
# peterpan_trunks = split_whole_book_into_trunks_of_texts(truncated_books[3])


# In[ ]:





# In[16]:


def sample_from_lists(book_trunks):
    sampled_trunks = random.sample(book_trunks, 3500)
    return sampled_trunks


# In[17]:


ulysses_sampled_trunks = sample_from_lists(ulysses_trunks)
emma_sampled_trunks = sample_from_lists(emma_trunks)
dubliners_sampled_trunks = sample_from_lists(dubliners_trunks)
prideAndPrejudice_sampled_trunks = sample_from_lists(prideAndPrejudice_trunks)


# In[18]:


Joyce_label = ['Joyce'] * 7000
Austen_label = ['Austen'] * 7000


# In[19]:


labels = Joyce_label+ Austen_label
data = ulysses_sampled_trunks + emma_sampled_trunks + dubliners_sampled_trunks + prideAndPrejudice_sampled_trunks


# In[20]:


# data_flat = [item for sublist in data for item in sublist]
# corpus = []


# In[21]:


def clean_text(text):
    text_data = re.sub('[^a-zA-Z]', ' ', text)
    text_data = text_data.lower()
    text_data = text_data.split()
#     text_data = [word in text_data if not word in set(stopwords.words('english'))]
    text_data = ' '.join(text_data)
    return text_data


# In[22]:


corpus = []
for each_sentence in data:
    cleaned_sentence = clean_text(each_sentence)
    corpus.append(cleaned_sentence)


# In[23]:


len(corpus)


# In[ ]:


# def clean_text(text):
#     """
#         text: a string

#         return: modified initial string
#     """
#     text = text.lower() # lowercase text
#     text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
#     text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
#     text = text.replace('x', '')
# #    text = re.sub(r'\W+', '', text)
#     text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
#     return text
# print(nltk.tokenize.sent_tokenize(text))


# In[ ]:





# In[ ]:


# x[2].shape


# In[ ]:


# to do:
# 1. combine x and y
# 2. shuffle together x and y
# 3. encode y
# train test split a and y


# In[24]:


# prepare 2D training data
vocab_size = 15000
encoded_docs = [keras.preprocessing.text.one_hot(d, vocab_size) for d in corpus]
# print(encoded_docs)
max_length = 10
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)
# padded_docs
np.save('./cleaned_data/x_data',padded_docs)


# In[25]:


padded_docs.shape


# In[29]:


# prepare 3d trainig data

# def form_dictionary(samples):
#     token_index = {};
#     #FORM DICTIONARY WITH WORD INDICE MAPPINGS
#     for sample in samples:
#         for word in sample.split():
#             if word not in token_index:
#                 token_index[word] = len(token_index) + 1
# 
#     transformed_text=[]
#     for sample in samples:
#         tmp=[]
#         for word in sample.split():
#             tmp.append(token_index[word])
#         transformed_text.append(tmp)
#
# #     print("CONVERTED TEXT:", transformed_text)
# #     print("VOCABULARY-2 (SKLEARN): ",token_index)
#     return [token_index,transformed_text]
#
# [vocab,x]=form_dictionary(corpus)



# #CHOLLET:  LISTING 6.1 WORD-LEVEL ONE-HOT ENCODING (TOY EXAMPLE)
def one_hot_encode(samples):
    #ONE HOT ENCODE (CONVERT EACH SENTENCE INTO MATRIX)
    max_length = 10
    results = np.zeros(shape=(len(samples),max_length,max(vocab.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = vocab.get(word)
            results[i, j, index] = 1.
    results=results[:,:,1:]
#     print("ONE HOT")
#     print(results)
    return results

x=one_hot_encode(corpus)


# In[31]:


# np.save('./cleaned_data/x_data_3D',x)


# In[26]:


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(labels)


# In[27]:


np.save('./cleaned_data/y_data',transfomed_label)


# In[30]:


x.shape


# In[28]:


transfomed_label.shape


# In[ ]:
