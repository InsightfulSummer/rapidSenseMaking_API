from nltk.tokenize import sent_tokenize, word_tokenize
from classes.processPDF import ProcessPDF
from classes.processText import ProcessText

class SummarizeDocument:
    def __init__(self, reqID, docID, size) -> None:
        # process it's pdf and gets its text
        pdf = ProcessPDF("./processResult/"+reqID+"/"+docID+".tei.xml", docID)
        self.text = pdf.getBodyText()
        self.size = size
        self.processText = ProcessText()

    def sentTokenization(self) :
        sentences = []
        sents = sent_tokenize(self.text)
        for i, sent in enumerate(sents):
            sentences.append((sent, int(i/len(sents)*100)))
        return sentences

    def rankSentences(self, sentences, wordFreq):
        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent[0].lower()):
                if word in wordFreq.keys():
                    if len(sent[0].split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent[1]] = wordFreq[word]
                        else:
                            sentence_scores[sent[1]] += wordFreq[word]
        return sentence_scores

    def summarizeText(self):
        sentences = self.sentTokenization()
        self.text = self.processText.clearText(self.text)
        wordFreq = self.processText.getNormalizedWordFrequency(self.text)
        sents = self.rankSentences(sentences, wordFreq)
        # sort the sentences based on their score
        rankedSents = sorted(sents.items() , key=lambda a:a[1], reverse=True)
        # choose the top n sentences
        self.size = int(len(sentences)*float(self.size))
        topN = rankedSents[0:self.size]
        # sort the filtered sentences based on their position 
        sortedTopN = sorted(topN, key=lambda a:a[0])
        summary = ""
        for k in sortedTopN:
            summary = summary + " " + sentences[k[0]][0]
        return summary.strip()