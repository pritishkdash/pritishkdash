import pandas as pd
import sklearn as sk
import math

first_sentence = "Data Science is the demanding subject of the 21st century"
second_sentence = "machine learning is the key for data science"

#split so each word have their own string
first_sentence = first_sentence.split(" ")
second_sentence = second_sentence.split(" ")

#join them to remove common duplicate words
total= set(first_sentence).union(set(second_sentence))
#print(total)

wordDictA =dict.fromkeys(total, 0) 
wordDictB = dict.fromkeys(total, 0)

for word in first_sentence:
    wordDictA[word]+=1
    
for word in second_sentence:
    wordDictB[word]+=1
    

#print(wordDictA)
#print(wordDictB)

#TF(t) = (No of times term t appears in a doc) / (Total no of terms in the doc)
#TF Function
def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)
#running our sentences through the tf function:
tfFirst = computeTF(wordDictA, first_sentence)
tfSecond = computeTF(wordDictB, second_sentence)
#Converting to dataframe for visualization
tf = pd.DataFrame([tfFirst, tfSecond])


import nltk
from nltk.corpus import stopwords
set(stopwords.words('english'))
filtered_sentence = []
for word in wordDictA:
    if str(word) not in set(stopwords.words('english')):
        filtered_sentence.append(word)
#
def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val) + 1))
        
    return(idfDict)
#inputing our sentences in the log file
idfs = computeIDF([wordDictA, wordDictB])

#tf-idf(t, d) = tf(t, d) * idf(t)
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return(tfidf)
#running our two sentences through the IDF:
idfFirst = computeTFIDF(tfFirst, idfs)
idfSecond = computeTFIDF(tfSecond, idfs)
#putting it in a dataframe
idf= pd.DataFrame([idfFirst, idfSecond])

print(idf)

#refer..https://storytellingco.com/how-to-build-a-recommender-system-using-tf-idf-and-nmf-python/
'''
Specifically, we will:

Extract all of the links from a Wikipedia article.
Read text from Wikipedia articles.
Create a TF-IDF map.
Split queries into clusters.
Build a recommender system.

'''
