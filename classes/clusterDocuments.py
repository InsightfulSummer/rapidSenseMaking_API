import json
import re
import nltk
from nltk.corpus import stopwords
# nltk.download("stopwords")
# nltk.download('punkt')
from nltk.stem import PorterStemmer
# nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from classes.processText import ProcessText


class ClusterDocuments:
    def __init__(self, data) -> None:
        self.customStopWords = [
            'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
            'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
            'al.', 'Elsevier', 'PMC', 'CZI', 'www'
        ]
        self.vectorizer = SentenceTransformer('allenai-specter')
        self.data = data
        self.pca = PCA(n_components=0.99)
        self.processText = ProcessText()

    def customTextCleaning(self, text):
        text = text.replace('\r',' ').replace('\n',' ')
        text = re.sub('[^a-zA-z0-9\s]', '', text) # removes special characters
        text = text.lower() # transforms every character into lowercase
        stops = stopwords.words('english')
        #     stops = stops.append(custom_stop_words)
        text = " ".join([word for word in word_tokenize(text) if word not in stops]) # removes the stop words
        text = " ".join([word for word in word_tokenize(text) if len(word) > 1]) # removes seperated characters
        #     stemmer = PorterStemmer()
        #     text = " ".join([stemmer.stem(word) for word in word_tokenize(text)]) # stemming words
        lemmatizer = WordNetLemmatizer()
        text = " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(text)]) # lemmatizing words
        return text

    def get_top_n_words(self, corpus, n=None):
        print(corpus)
        if len(corpus) == 1:
            corpus.append(" ")
        vec = TfidfVectorizer(stop_words={'english'}, ngram_range=(1,3)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]

    def create_label(self, word_list):
        bigrams = reversed([term[0] for term in word_list if len(term[0].split(" ")) == 2 and term[1] > 1])
        trigrams = reversed([term[0] for term in word_list if len(term[0].split(" ")) == 3 and term[1] > 1])
        
        most_occurance = word_list[0][1]
        labels = [term[0] for term in word_list if term[1] > most_occurance*0.7]
        label_ = labels[0]
        for label in labels:
            if(len(label_.split()) < 3):
                for term in bigrams:
                    if label in term:
                        label_ = term
                for term in trigrams:
                    if label in term:
                        label_ = term
        return label_

    def clusterDocuments(self):
        # 1- preprocess data
        self.data["content"] = self.data["title"] + " " + self.data["abstract"]
        self.data["content"] = self.data["content"].apply(lambda con : self.customTextCleaning(str(con)))
        # 2- vectorize content
        vectorized_content = self.vectorizer.encode(list(self.data["content"]))
        # 3- perform dimensionality reduction
        reduced_vectors= self.pca.fit_transform(vectorized_content)
        number_of_features = reduced_vectors.shape[1]
        # 4- find optimal k
        silhouette_scores = []
        K = range(2,5)
        for k in K:
            km = KMeans(n_clusters=k,random_state=1234, n_init=number_of_features)
            km = km.fit_predict(reduced_vectors)
            silhouette_scores.append(silhouette_score(reduced_vectors, km))
        optimalK = silhouette_scores.index(max(silhouette_scores)) + K[0]
        kmeans_ = KMeans(n_clusters=optimalK,random_state=1234, n_init=number_of_features)
        kmeans_ = kmeans_.fit_predict(reduced_vectors)
        clusters = np.unique(kmeans_)
        for c in clusters:
            arr = [c_ for c_ in kmeans_ if c_ == c]
            if(len(arr) < len(self.data)*0.05):
                optimalK = optimalK - 1
        # 5- cluster documents with the optimal K
        kmeans_ = KMeans(n_clusters=optimalK,random_state=1234, n_init=number_of_features)
        kmeans_ = kmeans_.fit_predict(reduced_vectors)
        self.data["cluster_id"] = kmeans_
        # 6- create label for each cluster
        self.data["title_"] = self.data["title"].apply(lambda title : self.processText.stopWordsOnly(title))
        for i in range(optimalK):
            filtered_docs = self.data[self.data["cluster_id"] == i]
            sampleContent = list(filtered_docs["title_"])
            topWords = self.get_top_n_words(sampleContent,25)
            topWords_ = " || ".join([t[0] for t in topWords])
            label = self.create_label(topWords)
            self.data.loc[self.data.cluster_id == i, 'cluster_label'] = label
            self.data.loc[self.data.cluster_id == i, 'cluster_wordCloud'] = topWords_
        # 7- clean data and return it
        self.data = self.data.drop(columns=['content','title_'])
        return self.data.to_json(orient="records")
        # return optimalK