import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class ProcessText:
    def __init__(self) -> None:
        self.stopWords = stopwords.words('english')

    def clearText(self, text):
        text = text.replace('\r', ' ').replace('\n', ' ')
        text = re.sub('[^a-zA-z0-9\s]', '', text)
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = " ".join([word for word in word_tokenize(text) if len(word) > 1]) # removes seperated characters
        return text

    def getNormalizedWordFrequency(self, text):
        text = self.clearText(text)
        word_frequencies = {}
        for word in word_tokenize(text):
            if word not in self.stopWords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        maximum_frequncy = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        
        
        return word_frequencies

    def stopWordsOnly(self, text):
        text = " ".join([word for word in word_tokenize(text) if word not in self.stopWords]) # removes the stop words
        return text