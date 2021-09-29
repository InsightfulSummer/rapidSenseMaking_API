import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Comparison:
    def __init__(self, reqID, doc1, doc2) -> None:
        self.reqID = reqID
        self.vectorizer = SentenceTransformer('allenai-specter')
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
        return ""