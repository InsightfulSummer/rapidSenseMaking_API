import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from classes.processPDF import ProcessPDF
from classes.processText import ProcessText
from classes.hyperSimilarity import HyperSimilarity

class Comparison:
    def __init__(self, reqID, doc1, doc2) -> None:
        self.reqID = reqID
        self.vectorizer = SentenceTransformer('allenai-specter')
        self.processPDF_1 = ProcessPDF("./processResult/"+reqID+"/"+doc1+".tei.xml", doc1)
        self.processPDF_2 = ProcessPDF("./processResult/"+reqID+"/"+doc2+".tei.xml", doc2)
        self.hyper1 = HyperSimilarity(reqID, doc1, 10)
        self.hyper2 = HyperSimilarity(reqID, doc2, 10)
        self.processTxt = ProcessText()
        with open("./processResult/"+reqID+"/"+reqID+".json","r") as overall:
            data = json.load(overall)
            self.doc1 = [x for x in data if x['id'] == int(doc1)]
            self.doc2 = [x for x in data if x['id'] == int(doc2)]

    def calculateSimilarity(self):
        abs1 = self.vectorizer.encode(self.doc1[0]['abstract'])
        abs2 = self.vectorizer.encode(self.doc2[0]['abstract'])
        cosineRes = cosine_similarity([abs1], [abs2])
        return str(cosineRes[0][0])

    def parseBody(self):
        body1txt = self.processPDF_1.getBodyText()
        body2txt = self.processPDF_2.getBodyText()
        fullBodyTxt = body1txt + body2txt
        wordFreq_ = self.processTxt.getNormalizedWordFrequency(fullBodyTxt)
        parsedBody1 = self.processPDF_1.getBody(wordFreq_)
        parsedBody2 = self.processPDF_2.getBody(wordFreq_)
        return {"body1": parsedBody1, "body2": parsedBody2}

    def basicComparison(self):
        cosineRes = self.calculateSimilarity()
        parsedBodies = self.parseBody()
        return {
            "cosineRes" : cosineRes,
            "parsedBodies" : parsedBodies
        }

    def searchAndCompare(self, searchTerm):
        searchRes1 = self.hyper1.findSimilarSentences(searchTerm)
        searchRes2 = self.hyper2.findSimilarSentences(searchTerm)
        return [searchRes1, searchRes2]