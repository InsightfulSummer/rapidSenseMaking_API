from sentence_transformers import SentenceTransformer
from classes.processPDF import ProcessPDF
from sklearn.metrics.pairwise import cosine_similarity
import json

class HyperSimilarity:
    def __init__(self, reqID, docID, topN) -> None:
        self.vectorizer = SentenceTransformer('allenai-specter')
        self.processPDF = ProcessPDF("./processResult/"+reqID+"/"+docID+".tei.xml", docID)
        self.topN = int(topN)
        self.reqID = reqID

    def getSentences(self):
        parsedBody = self.processPDF.getBody()
        sentences = []
        for div in parsedBody:
            for content in div["content"]:
                if content["type"] == "paragraph":
                    for sentence in content["content"]:
                        if sentence["type"] == "sentenceInP":
                            sentences.append(sentence)
        return sentences

    def findSimilarSentences(self, sentence):
        topNSentences = []
        scores = []
        sent_vec = self.vectorizer.encode(sentence)
        sentences = self.getSentences()
        sents = [s["content"] for s in sentences]
        sents_vec = self.vectorizer.encode(sents)
        for i in range(len(sentences)):
            cosineRes = cosine_similarity([sent_vec],[sents_vec[i]])
            scores.append((cosineRes, i))
        scores.sort(key = lambda x : x[0], reverse=True)
        scores = scores[0:self.topN]
        for s in scores:
            sentences[s[1]]["position"] = s[1] / len(sentences)
            topNSentences.append(sentences[s[1]])
        return topNSentences

    def getAbstracts(self):
        abstract_ = []
        with open("./processResult/"+self.reqID+"/"+self.reqID+".json","r") as overall:
            data_ = json.load(overall)
            for d in data_:
                abstract_.append({
                    "abstract" : d["abstract"],
                    "id" : d["id"],
                    "title" : d["title"]
                })
        return abstract_

    def findSimilarDocs(self, sentence):
        topNDocs = []
        scores = []
        sent_vec = self.vectorizer.encode(sentence)
        abstracts_ = self.getAbstracts()
        abstracts = [a["abstract"] for a in abstracts_]
        sents_vec = self.vectorizer.encode(abstracts)
        for i in range(len(abstracts_)):
            cosineRes = cosine_similarity([sent_vec],[sents_vec[i]])
            scores.append((cosineRes, i))
        scores.sort(key = lambda x : x[0], reverse=True)
        scores = scores[0:self.topN]
        for s in scores:
            topNDocs.append(abstracts_[s[1]])
        return topNDocs