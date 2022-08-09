import pandas as pd
import numpy as np

import re

from umap import UMAP
from umap import plot as umap_plot

import seaborn as sns
import matplotlib.pyplot as plt

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

config = {
    'data' : '../data/raw/posts_comms_20220712.xlsx',
    'sentence_model' : 'all-MiniLM-L6-v2',
    'cluster_params' : {
        'n_clusters' : 100,
        'random_state' : 42
    },
    'umap_params' : {
        'n_neighbors' : 100,
        'n_components' : 10,
        'min_dist' : 0.0,
        'metric' : 'cosine',
        'random_state' : 42
    },
    'ngram_representation' : {
        'ngram_range' : (1, 4),
        'stop_words' : 'english'
    },
    'linkage_params' : {
        'method' : 'ward', 
        'optimal_ordering' : True
    }
}

class BERedditTopics():
    """
    Class for handling BERTopic modeling as it pertains to Reddit. 
    """

    def __init__(self, config=config):
        self.config = config

        # Create our sentence model which will the level of our tokenization.
        self.sentence_model = SentenceTransformer(config['sentence_model'])

        # Create our vectorizer model to improve the keyword representation of our topics.
        self.vectorizer_model = CountVectorizer(**self.config['ngram_representation'])

        # Create our specific UMAP and clustering model. 
        self.umap_model = UMAP(**self.config['umap_params'])
        self.cluster_model = SpectralClustering(self.config['cluster_params'])


    def load_data_frame(self):
        """
        Load our Reddit post/comments dataframe.
        """
        if self.config['data'].split('.')[-1] != 'xlsx':
            raise TypeError('Expected an Excel file. Other input dataframe types not yet supported.')

        self.raw_df = pd.read_excel(self.config['data'])
        self.preprocess_dataframe()

        self.texts = self.raw_df['content'].to_list()

    def preprocess_dataframe(self):
        # Fill empty cells and remove some weird html tags
        self.raw_df['content'].fillna("", inplace=True)
        self.raw_df.content = self.raw_df.content.str.replace("http\S+", "")
        self.raw_df.content = self.raw_df.content.str.replace("\\n", " ")
        self.raw_df.content = self.raw_df.content.str.replace("&gt;", "") 

    def create_topic_model(self):
        """
        Creates the topic model as specified below.
        """
        
        # Create our embeddings
        self.embeddings = self.sentence_model.encode(self.texts, show_progress_bar=True)

        # Create our topic model.
        self.topic_model = BERTopic(umap_model=self.umap_model, hdbscan_model=self.cluster_model, vectorizer_model=self.vectorizer_model)
        self.topics, _ = self.topic_model.fit_transform(self.texts, self.embeddings)

        # Create the hierarchy of topics.
        self.hierarchy = self.topic_model.hierarchical_topics(self.texts, self.topics)

    def save_topic_model(self):
        raise NotImplementedError


def create_plot_style():
    sns.set_style('whitegrid')
    sns.set_context('talk')

if __name__ == '__main__':
    
    create_plot_style()

    topic_model = BERedditTopics(config)
    topic_model.load_data_frame()
    topic_model.create_topic_model()
    topic_model.save_topic_model()