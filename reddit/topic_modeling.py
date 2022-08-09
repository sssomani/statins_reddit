import pandas as pd
import numpy as np

import re

import seaborn as sns
import matplotlib.pyplot as plt

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

# from umap.umap_ import UMAP
from umap import UMAP

config = {
    'data' : '/Users/ssomani/research/heartlab/statins_reddit/data/raw/posts_comms_20220712.xlsx',

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
        self.sentence_model = SentenceTransformer(self.config['sentence_model'])

        # Create our vectorizer model to improve the keyword representation of our topics.
        self.vectorizer_model = CountVectorizer(**self.config['ngram_representation'])

        # Create our specific UMAP and clustering model. 
        self.umap_model = UMAP(**self.config['umap_params'])
        self.cluster_model = SpectralClustering(**self.config['cluster_params'])

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
        """
        Preprocess dataframe for topic modeling.
        """
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

    def create_groups(self):
        c_tf_idf_mms = mms().fit_transform(self.topic_model.c_tf_idf.toarray())
        c_tf_idf_embed = UMAP(n_neighbors=4, n_components=3, metric='hellinger', random_state=42).fit_transform(c_tf_idf_mms)
        ideal_n_clusters = self.find_silhouette_scores(c_tf_idf_embed)
        self.groups = SpectralClustering(n_clusters=ideal_n_clusters, random_state=42).fit_predict(c_tf_idf_embed)

    @staticmethod
    def find_silhouette_scores(c_tf_idf_embed, llim=3, ulim=25):
        """
        Find the optimal number of clusters based on the maximum silhouette score across a range of clusters.
        """
        ss = []
        cluster_arr = np.arange(llim, ulim)
        
        for n_clusters in cluster_arr:
            clusters = SpectralClustering(n_clusters=n_clusters, random_state=42).fit_predict(c_tf_idf_embed)
            ss.append(silhouette_score(c_tf_idf_embed, clusters))
            
        with sns.plotting_context('notebook'):
            sns.set_style('ticks')
            plt.figure(figsize=(5, 3))
            ax = sns.lineplot(x=cluster_arr, y=ss, palette='autumn')
            ax.set_title('Silhouette Score per Cluster', {'fontsize' : 16})
            plt.show(ax)

        ideal_n_clusters = cluster_arr[np.argmax(ss)]           

        print("top silhouette score: {0:0.3f} for at n_clusters {1}".format(np.max(ss), ideal_n_clusters))
        
        return ideal_n_clusters

    def save_topic_model(self):
        raise NotImplementedError

    def load_topic_model(self, path_to_topic_model):
        # Load a saved topic model.
        raise NotImplementedError

if __name__ == '__main__':
    
    create_plot_style()

    topic_model = BERedditTopics(config)
    topic_model.load_data_frame()
    topic_model.create_topic_model()
    topic_model.save_topic_model()