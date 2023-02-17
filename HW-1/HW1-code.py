#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup


# In[2]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# ## Read Data

# In[3]:


dataframe = pd.read_csv("amazon_reviews_us_Beauty_v1_00.tsv", sep="\t", on_bad_lines='skip')


# ## Keep Reviews and Ratings

# In[4]:


df = dataframe.loc[:, ['star_rating', 'review_body', 'review_headline']]
df['review_body'] = df['review_body'].astype(str)
df['review_headline'] = df['review_headline'].astype(str)


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[5]:


def assignClass(rating):
    if rating == 1 or rating == 2:
        return 1
    elif rating == 3:
        return 2
    elif rating == 4 or rating == 5:
        return 3
    return -1


# In[6]:


df['class'] = df['star_rating'].map(assignClass)


# In[7]:


df_1 = df[df["class"] == 1].sample(n=20000)
df_2 = df[df["class"] == 2].sample(n=20000)
df_3 = df[df["class"] == 3].sample(n=20000)
# print(len(df_1), len(df_2), len(df_3))


# In[8]:


df = pd.concat([df_1, df_2, df_3])


# In[9]:


# print("Dataframe that is being worked on:")
# df.head(10)


# In[10]:


# calculate length of raw reviews before any cleaning or preprocessing
len_before_cleaning = (df['review_headline'] + ". " + df['review_body']).apply(len).mean()


# # Data Cleaning
# 
# 

# In[11]:


import contractions


# In[12]:


def clean_data(text):
    text = str(text)

    # convert text to lowercase
    text = text.lower()

    # remove urls                                                                                                             
    text = re.sub(r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)',' ',text)
    # remove email ids    
    text = re.sub(r'([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',' ',text)       
    # html tag                                             
    text = re.sub('<[^<]+?>', '', text)                                                                                            
    
    # expand contractions
    text = contractions.fix(text)
    
    # replace non aplha numeric characters
    text = re.sub(r'[^a-zA-Z0-9. ]',' ',text)                                                                                      
    # remove isolated characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # remove consecutively repeating words                                                                                     
    text = re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    # replace every word following not/never/no as NEG_word until a fullstop is found
    text = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]',lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), text, flags=re.IGNORECASE)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)                                                                                                

    return text


# In[13]:


review_body = df["review_body"].apply(clean_data)
review_headline = df["review_headline"].apply(clean_data)


# #### Comparing length of raw uncleaned review with length of cleaned review 

# In[14]:


len_after_cleaning = (review_headline + ". " + review_body).apply(len).mean()
print(f"{len_before_cleaning}, {len_after_cleaning}\n")


# In[15]:


# store the cleaned data that will NOT be pre-processed
reviews_vanilla = review_headline.str.upper() + ". " + review_body


# # Pre-processing

# ## remove the stop words 

# In[16]:


# from nltk.corpus import stopwords
# nltk.download('stopwords')
# stop_words = stopwords.words('english')
# stop_words.remove('not')

# def remove_stop_words(text):
#     tokens = [w for w in text.split() if not w in stop_words]

    
#     text = " ".join(tokens)

#     return text


# In[17]:


# review_body = review_body.apply(remove_stop_words)
# review_headline = review_headline.apply(remove_stop_words)


# ## perform lemmatization  

# In[18]:


# from nltk.corpus import wordnet

# def get_wordnet_pos(treebank_tag):

#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return ''


# In[19]:


# from nltk.stem import WordNetLemmatizer
# from nltk.tag import pos_tag
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

# lemmatizer = WordNetLemmatizer()


# In[20]:


# def lemmatization(reviews):
#     lemmatized_reviews = []
#     for sent in reviews.values:
#         tagged = pos_tag(sent.split(" "))
#         temp_review = ""
#         for token in tagged:
#             tag = get_wordnet_pos(token[1])
#             if tag:
#                 temp_review += lemmatizer.lemmatize(token[0], tag)
#             else:
#                 temp_review += token[0]
#             temp_review += " "

#         lemmatized_reviews.append(temp_review)

#     return  pd.Series(lemmatized_reviews)


# In[21]:


# review_body = lemmatization(review_body)
# review_headline = lemmatization(review_headline)


# In[22]:


# # store the cleaned and pre-processed data
# reviews_processed = review_headline.str.upper() + ". " + review_body


# #### Comparing length of review that is NOT preprocessed with length of review before preprocessing

# In[23]:


len_without_preprocessing = reviews_vanilla.apply(len).mean()

print(f"{len_after_cleaning}, {len_without_preprocessing}\n")


# In[24]:


# print("Sample of cleaned Data:")
# reviews_vanilla.head(10)


# #### Comparing length of review that is preprocessed with length of review before preprocessing

# In[25]:


# len_after_preprocessing = reviews_processed.apply(len).mean()

# print(f"{len_after_cleaning}, {len_after_preprocessing}")


# In[26]:


# print("Sample of preprocessed Data:")
# reviews_processed.head(10)


# # TF-IDF Feature Extraction

# In[27]:


from sklearn.model_selection import train_test_split
X_train_vanilla, X_test_vanilla, Y_train_vanilla, Y_test_vanilla = train_test_split(reviews_vanilla, df['class'], test_size = 0.2, random_state = 0, stratify=df['class'])


# In[28]:


# X_train_processed, X_test_processed, Y_train_processed, Y_test_processed = train_test_split(reviews_processed, df['class'], test_size = 0.2, random_state = 0, stratify=df['class'])


# In[29]:


from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vect_vanilla = TfidfVectorizer(use_idf=True, max_features=5000)
X_train_tfidf_vanilla = tf_idf_vect_vanilla.fit_transform(X_train_vanilla)
X_test_tfidf_vanilla = tf_idf_vect_vanilla.transform(X_test_vanilla)


# In[30]:


# print("For cleaned/vanilla data:")
# print(f"\tX_train shape: {X_train_tfidf_vanilla.shape}")
# print(f"\tY_train shape: {Y_train_vanilla.shape}\n")

# print(f"\tX_test shape: {X_test_tfidf_vanilla.shape}")
# print(f"\tY_test shape: {Y_test_vanilla.shape}\n")


# In[31]:


# tf_idf_vect_processed = TfidfVectorizer(use_idf=True, max_features=5000)
# X_train_tfidf_processed = tf_idf_vect_processed.fit_transform(X_train_processed)
# X_test_tfidf_processed = tf_idf_vect_processed.transform(X_test_processed)


# In[32]:


# print("For pre-processed data:")
# print(f"\tX_train shape: {X_train_tfidf_processed.shape}")
# print(f"\tY_train shape: {Y_train_processed.shape}\n")

# print(f"\tX_test shape: {X_test_tfidf_processed.shape}")
# print(f"\tY_test shape: {Y_test_processed.shape}\n")


# # Perceptron

# ### Without Stop Word Removal and Without Lemmatization

# In[33]:


from sklearn.linear_model import Perceptron

prctrn_vanilla = Perceptron(tol=1e-3, random_state=0, validation_fraction=0.3)
prctrn_vanilla.fit(X_train_tfidf_vanilla, Y_train_vanilla)

# training accuracy
# print(prctrn_vanilla.score(X_train_tfidf_vanilla, Y_train_vanilla))

# testing accuracy
# print(prctrn_vanilla.score(X_test_tfidf_vanilla, Y_test_vanilla))


# In[34]:


# classification report
prctrn_pred_vanilla = prctrn_vanilla.predict(X_test_tfidf_vanilla)
# print(classification_report(Y_test_vanilla, prctrn_pred_vanilla))

# class-wise precision, recall and f1-score
prctrn_report_vanilla = classification_report(Y_test_vanilla, prctrn_pred_vanilla, output_dict=True)
print(f"{prctrn_report_vanilla['1']['precision']}, {prctrn_report_vanilla['1']['recall']}, {prctrn_report_vanilla['1']['f1-score']}")
print(f"{prctrn_report_vanilla['2']['precision']}, {prctrn_report_vanilla['2']['recall']}, {prctrn_report_vanilla['2']['f1-score']}")
print(f"{prctrn_report_vanilla['3']['precision']}, {prctrn_report_vanilla['3']['recall']}, {prctrn_report_vanilla['3']['f1-score']}")
print(f"{prctrn_report_vanilla['macro avg']['precision']}, {prctrn_report_vanilla['macro avg']['recall']}, {prctrn_report_vanilla['macro avg']['f1-score']}\n")


# ### With Stop Word Removal and Lemmatization

# In[35]:


# prctrn_processed = Perceptron(tol=1e-3, random_state=0, validation_fraction=0.3)
# prctrn_processed.fit(X_train_tfidf_processed, Y_train_processed)

# # training accuracy
# print(prctrn_processed.score(X_train_tfidf_processed, Y_train_processed))

# # testing accuracy
# print(prctrn_processed.score(X_test_tfidf_processed, Y_test_processed))


# In[36]:


# # classification report
# prctrn_pred_processed = prctrn_processed.predict(X_test_tfidf_processed)
# print(classification_report(Y_test_processed, prctrn_pred_processed))

# # class-wise precision, recall and f1-score
# prctrn_report_processed = classification_report(Y_test_processed, prctrn_pred_processed, output_dict=True)
# print(f"{prctrn_report_processed['1']['precision']}, {prctrn_report_processed['1']['recall']}, {prctrn_report_processed['1']['f1-score']}")
# print(f"{prctrn_report_processed['2']['precision']}, {prctrn_report_processed['2']['recall']}, {prctrn_report_processed['2']['f1-score']}")
# print(f"{prctrn_report_processed['3']['precision']}, {prctrn_report_processed['3']['recall']}, {prctrn_report_processed['3']['f1-score']}")
# print(f"{prctrn_report_processed['macro avg']['precision']}, {prctrn_report_processed['macro avg']['recall']}, {prctrn_report_processed['macro avg']['f1-score']}")


# # SVM

# ### Without Stop Word Removal and Without Lemmatization

# In[37]:


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SVM_vanilla = LinearSVC()
SVM_vanilla.fit(X_train_tfidf_vanilla, Y_train_vanilla)

# training accuracy
# print(SVM_vanilla.score(X_train_tfidf_vanilla, Y_train_vanilla))

# test accuracy
# print(SVM_vanilla.score(X_test_tfidf_vanilla, Y_test_vanilla))


# In[38]:


# classification report
SVM_pred_vanilla = SVM_vanilla.predict(X_test_tfidf_vanilla)
# print(classification_report(Y_test_vanilla, SVM_pred_vanilla))

# class-wise precision, recall and f1-score
SVM_report_vanilla = classification_report(Y_test_vanilla, SVM_pred_vanilla, output_dict=True)
print(f"{SVM_report_vanilla['1']['precision']}, {SVM_report_vanilla['1']['recall']}, {SVM_report_vanilla['1']['f1-score']}")
print(f"{SVM_report_vanilla['2']['precision']}, {SVM_report_vanilla['2']['recall']}, {SVM_report_vanilla['2']['f1-score']}")
print(f"{SVM_report_vanilla['3']['precision']}, {SVM_report_vanilla['3']['recall']}, {SVM_report_vanilla['3']['f1-score']}")
print(f"{SVM_report_vanilla['macro avg']['precision']}, {SVM_report_vanilla['macro avg']['recall']}, {SVM_report_vanilla['macro avg']['f1-score']}\n")


# ### With Stop Word Removal and Lemmatization

# In[39]:


# SVM_processed = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
# SVM_processed.fit(X_train_tfidf_processed, Y_train_processed)

# # training accuracy
# print(SVM_processed.score(X_train_tfidf_processed, Y_train_processed))

# # test accuracy
# print(SVM_processed.score(X_test_tfidf_processed, Y_test_processed))


# In[40]:


# # classification report
# SVM_pred_processed = SVM_processed.predict(X_test_tfidf_processed)
# print(classification_report(Y_test_processed, SVM_pred_processed))

# # class-wise precision, recall and f1-score
# SVM_report_processed = classification_report(Y_test_processed, SVM_pred_processed, output_dict=True)
# print(f"{SVM_report_processed['1']['precision']}, {SVM_report_processed['1']['recall']}, {SVM_report_processed['1']['f1-score']}")
# print(f"{SVM_report_processed['2']['precision']}, {SVM_report_processed['2']['recall']}, {SVM_report_processed['2']['f1-score']}")
# print(f"{SVM_report_processed['3']['precision']}, {SVM_report_processed['3']['recall']}, {SVM_report_processed['3']['f1-score']}")
# print(f"{SVM_report_processed['macro avg']['precision']}, {SVM_report_processed['macro avg']['recall']}, {SVM_report_processed['macro avg']['f1-score']}")


# # Logistic Regression

# ### Without Stop Word Removal and Without Lemmatization

# In[41]:


from sklearn.linear_model import LogisticRegression

lr_vanilla = LogisticRegression(multi_class='multinomial', max_iter=10000)
lr_vanilla.fit(X_train_tfidf_vanilla, Y_train_vanilla)

# training accuracy
# print(lr_vanilla.score(X_train_tfidf_vanilla, Y_train_vanilla))

# test accuracy
# print(lr_vanilla.score(X_test_tfidf_vanilla, Y_test_vanilla))


# In[42]:


# classification report
lr_pred_vanilla = lr_vanilla.predict(X_test_tfidf_vanilla)
# print(classification_report(Y_test_vanilla, lr_pred_vanilla))

# class-wise precision, recall and f1-score
lr_report_vanilla = classification_report(Y_test_vanilla, lr_pred_vanilla, output_dict=True)
print(f"{lr_report_vanilla['1']['precision']}, {lr_report_vanilla['1']['recall']}, {lr_report_vanilla['1']['f1-score']}")
print(f"{lr_report_vanilla['2']['precision']}, {lr_report_vanilla['2']['recall']}, {lr_report_vanilla['2']['f1-score']}")
print(f"{lr_report_vanilla['3']['precision']}, {lr_report_vanilla['3']['recall']}, {lr_report_vanilla['3']['f1-score']}")
print(f"{lr_report_vanilla['macro avg']['precision']}, {lr_report_vanilla['macro avg']['recall']}, {lr_report_vanilla['macro avg']['f1-score']}\n")


# ### With Stop Word Removal and Lemmatization

# In[43]:


# lr_processed = LogisticRegression(multi_class='multinomial', max_iter=10000)
# lr_processed.fit(X_train_tfidf_processed, Y_train_processed)

# # training accuracy
# print(lr_processed.score(X_train_tfidf_processed, Y_train_processed))

# # test accuracy
# print(lr_processed.score(X_test_tfidf_processed, Y_test_processed))


# In[44]:


# # classification report
# lr_pred_processed = lr_processed.predict(X_test_tfidf_processed)
# print(classification_report(Y_test_processed, lr_pred_processed))

# # class-wise precision, recall and f1-score
# lr_pred_processed = lr_processed.predict(X_test_tfidf_processed)

# lr_report_processed = classification_report(Y_test_processed, lr_pred_processed, output_dict=True)
# print(f"{lr_report_processed['1']['precision']}, {lr_report_processed['1']['recall']}, {lr_report_processed['1']['f1-score']}")
# print(f"{lr_report_processed['2']['precision']}, {lr_report_processed['2']['recall']}, {lr_report_processed['2']['f1-score']}")
# print(f"{lr_report_processed['3']['precision']}, {lr_report_processed['3']['recall']}, {lr_report_processed['3']['f1-score']}")
# print(f"{lr_report_processed['macro avg']['precision']}, {lr_report_processed['macro avg']['recall']}, {lr_report_processed['macro avg']['f1-score']}")


# # Naive Bayes

# ### Without Stop Word Removal and Without Lemmatization

# In[45]:


from sklearn.naive_bayes import MultinomialNB

MNB_vanilla = MultinomialNB()
MNB_vanilla.fit(X_train_tfidf_vanilla, Y_train_vanilla)

# training accuracy
# print(MNB_vanilla.score(X_train_tfidf_vanilla, Y_train_vanilla))

# test accuracy
# print(MNB_vanilla.score(X_test_tfidf_vanilla, Y_test_vanilla))


# In[46]:


# classification report
MNB_pred_vanilla = MNB_vanilla.predict(X_test_tfidf_vanilla)
# print(classification_report(Y_test_vanilla, MNB_pred_vanilla))

# class-wise precision, recall and f1-score
MNB_report_vanilla = classification_report(Y_test_vanilla, MNB_pred_vanilla, output_dict=True)
print(f"{MNB_report_vanilla['1']['precision']}, {MNB_report_vanilla['1']['recall']}, {MNB_report_vanilla['1']['f1-score']}")
print(f"{MNB_report_vanilla['2']['precision']}, {MNB_report_vanilla['2']['recall']}, {MNB_report_vanilla['2']['f1-score']}")
print(f"{MNB_report_vanilla['3']['precision']}, {MNB_report_vanilla['3']['recall']}, {MNB_report_vanilla['3']['f1-score']}")
print(f"{MNB_report_vanilla['macro avg']['precision']}, {MNB_report_vanilla['macro avg']['recall']}, {MNB_report_vanilla['macro avg']['f1-score']}\n")


# ### With Stop Word Removal and Lemmatization

# In[47]:


# from sklearn.naive_bayes import MultinomialNB

# MNB_processed = MultinomialNB()
# MNB_processed.fit(X_train_tfidf_processed, Y_train_processed)

# # train accuracy
# print(MNB_processed.score(X_train_tfidf_processed, Y_train_processed))

# # test accuracy
# print(MNB_processed.score(X_test_tfidf_processed, Y_test_processed))


# In[48]:


# # classification report
# MNB_pred_processed = MNB_processed.predict(X_test_tfidf_processed)
# print(classification_report(Y_test_processed, MNB_pred_processed))

# # class-wise precision, recall and f1-score
# MNB_report_processed = classification_report(Y_test_processed, MNB_pred_processed, output_dict=True)
# print(f"{MNB_report_processed['1']['precision']}, {MNB_report_processed['1']['recall']}, {MNB_report_processed['1']['f1-score']}")
# print(f"{MNB_report_processed['2']['precision']}, {MNB_report_processed['2']['recall']}, {MNB_report_processed['2']['f1-score']}")
# print(f"{MNB_report_processed['3']['precision']}, {MNB_report_processed['3']['recall']}, {MNB_report_processed['3']['f1-score']}")
# print(f"{MNB_report_processed['macro avg']['precision']}, {MNB_report_processed['macro avg']['recall']}, {MNB_report_processed['macro avg']['f1-score']}")