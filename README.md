# CSPB NLP Final Project

## Overview

This repository contains code for the CSPB NLP Final Project focusing on text summarization, completed in Google Colab. The goal of the project is to explore multiple summarization approaches using the CNN/DailyMail dataset from HuggingFace (abisee/cnn_dailymail).
The project implements and compares:

TF-IDF Extractive Summarization
BART Abstractive Summarization
Hybrid Extractive Summarization (TF-IDF + DistilBERT)
HybridBART (Hybrid output passed through BART)

The report evaluates these models using ROUGE scoring. The last portion of this code will provide averaging scoring across 50 of the articles in the dataset. You can adjust how many articles you would like to run in the subset but will need to adjust code to allow for
GPU running. 

## Features
- TF-IDF summarization model
- BART summerization model
- Hybrid implementation of TF-IDF DistilBERT
- Previous Hybrid run in BART

## Installation

If running in Google Colab, install required packages with:
pip install sentence-transformers
pip install rouge-score
pip install datasets
pip install transformers

If you are running elsewhere, you might need to install these: 
import re
import os
import numpy as np
import pandas as pd
import nltk

from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

You'll need to download the NLTK resouces:
import nltk
nltk.download('punkt')
nltk.download('stopwords')

If using Google Colab, just open the .ipynb version of the file and run all cells or you can run based on what model you would like to implement. 
