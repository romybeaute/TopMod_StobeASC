import os
import re
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spacy import load

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = load("en_core_web_sm")
nltk.download("stopwords")

def correct_typos(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)

def separate_attached_words(text):
    return " ".join(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", text))

def remove_single_word_rows(df, column_name, nwords=2):
    df = df[df[column_name].str.split().str.len() > nwords]
    return df

def handle_newlines(text):
    return text.replace("/n", " ")

def text_cleaning(text, stemming=False):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def full_cleaning_pipeline(df, column_name,
                           correct_typos_flag=True,
                           sep_words_flag=True,
                           rmv_single_flag=True,
                           new_lines_flag=True,
                           text_clean_flag=True):
    if correct_typos_flag:
        df[column_name] = df[column_name].apply(correct_typos)
    if sep_words_flag:
        df[column_name] = df[column_name].apply(separate_attached_words)
    if rmv_single_flag:
        df = remove_single_word_rows(df, column_name)
    if new_lines_flag:
        df[column_name] = df[column_name].apply(handle_newlines)
    if text_clean_flag:
        df[column_name] = df[column_name].apply(text_cleaning)
    return df
