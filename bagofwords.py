#Fit and Transform method for Bag of word creation

#Importing reruire libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix

# Amazon product review
#https://www.amazon.in/Girnar-Detox-Green-Desi-Kahwa/dp/B092W15FNN/ref=sr_1_11?crid=77VGUM2XZZ66&keywords=food%2Breview&qid=1663130729&sprefix=food%2Breview%2Caps%2C108&sr=8-11&th=1
#Converted reviews to lower case
review1 = "one can replace harmful tea with this healthy drink"
review2 = "it is good taste and each taste is worth for money"
review3 = "taste is great packaging was also good"
review4 = "true medicinal value"
#review5 ="this was recommended by a friend now I order the big box of 100 bags of other in my office to share"
#review6 ="no doubt my desk has become the favourite spot for green tea lovers"

"""<h3><strong>Fit method:</strong></h3>
With this function, find all unique words in the data and assign a dimension-number to each unique word.

Create a python dictionary to save all the unique words, such that the key of dictionary represents a unique word and the corresponding value represent it's dimension-number. Values are always sorted in ascending order.

For example, if you have a review, __'how are you'__ then you can represent each unique word with a dimension_number as,
dict = { 'are' : 0, 'how' : 1, 'you' : 2}
"""

#Accepts only list of sentances
def fit(dataset):    
    unique_words = set() # at first we will initialize an empty set
    # check if its list type or not
    if isinstance(dataset, (list,)):
        for row in dataset: # for each review in the dataset
            for word in row.split(" "): # for each word in the review. #split method converts a string into list of words
                if len(word) < 2: # word greater length should be greated then one
                    continue
                unique_words.add(word)
        unique_words = sorted(list(unique_words))
        vocab = {j:i for i,j in enumerate(unique_words)}
        
        return vocab
    else:
        print("you need to pass list of sentance")

whole_string=[review1,review2,review3,review4]
vocab = fit(whole_string)
print(vocab)

"""<h3><strong>Transform method:</strong></h3>
With this function, we will write a feature matrix using sprase matrix.
"""

# Return value in sparse matrix format
def transform(dataset,vocab):
    rows = []
    columns = []
    values = []
    if isinstance(dataset, (list,)):
        for idx, row in enumerate(tqdm(dataset)): # for each document in the dataset
            # it will return a dict type object where key is the word and values is its frequency, {word:frequency}
            word_freq = dict(Counter(row.split()))
            # for every unique word in the document
            for word, freq in word_freq.items():  # for each unique word in the review.                
                if len(word) < 2:
                    continue
                # we will check if its there in the vocabulary that we build in fit() function
                # dict.get() function will return the values, if the key doesn't exits it will return -1
                col_index = vocab.get(word, -1) # retreving the dimension number of a word
                # if the word exists
                if col_index !=-1:
                    # we are storing the index of the document
                    rows.append(idx)
                    # we are storing the dimensions of the word
                    columns.append(col_index)
                    # we are storing the frequency of the word
                    values.append(freq)
        return csr_matrix((values, (rows,columns)), shape=(len(dataset),len(vocab)))
    else:
        print("you need to pass list of strings")

print(list(vocab.keys()))
bow_0=transform(whole_string, vocab).toarray()
print(bow_0)

"""<h3>Results using countvectorizer</h3>"""

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(analyzer='word')

vec.fit(whole_string)
feature_matrix_2 = vec.transform(whole_string)
bow_1=feature_matrix_2.toarray()
print(bow_1)

"""<h3>Comparing both matrices</h3>"""

np.array_equal(bow_0,bow_1)
