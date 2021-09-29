from flask import Flask, request, send_from_directory
from flask_restful import reqparse, abort, Api, Resource
import os
import pandas as pd
import requests
from flask_cors import CORS
from grobid_client.grobid_client import GrobidClient
from classes.processPDF import ProcessPDF
from classes.clusterDocuments import ClusterDocuments
from classes.summarizeDocument import SummarizeDocument
from classes.hyperSimilarity import HyperSimilarity
from classes.comparison import Comparison
import json

app = Flask(__name__)
CORS(app)
api = Api(app)
client = GrobidClient(config_path="./grobid/config.json")

@app.route("/pdfUploading",methods=['POST'])
def pdfHandling():
    # get the post data -> 1- pdf file 2- request_id 3- iterationNumber
    reqFile = request.files['pdfFile']
    reqID = request.form.get('reqID')
    iterationNum = request.form.get('iterationNum')
    # print(reqFile, reqID, iterationNum)
    # check if a specific folder exist for this request id or not (if not, create one)
    pathCheck = os.path.isdir("./processResult/"+reqID)
    if pathCheck == False :
        os.mkdir("./processResult/"+reqID)
    # 1- save pdf file, 2- process the pdf file, 3- save corresponding xml file, 4- extract meta data, create json
    fileCheck = os.path.exists("./processResult/"+reqID+"/"+iterationNum+"/"+iterationNum+".pdf")
    if fileCheck == False:
        os.mkdir("./processResult/"+reqID+"/"+iterationNum)
        reqFile.save("./processResult/"+reqID+"/"+iterationNum+"/"+iterationNum+".pdf")
    else : 
        return {"success":"false", "message":"file already exists."}

    client.process("processFulltextDocument", "./processResult/"+reqID+"/"+iterationNum+"/", output="./processResult/"+reqID+"/", n=2)
    
    processedPDFCheck = os.path.exists("./processResult/"+reqID+"/"+iterationNum+".tei.xml")
    if processedPDFCheck == False:
        return {"success":"false","messgae":"file processing failed"}
    pdf = ProcessPDF("./processResult/"+reqID+"/"+iterationNum+".tei.xml", iterationNum)

    # create a json file if not existing and save json object of uploaded pdf in it ...
    jsonFileCheck = os.path.exists("./processResult/"+reqID+"/"+reqID+".json")
    if jsonFileCheck == False:
        with open("./processResult/"+reqID+"/"+reqID+".json","w") as jsonFile:
            json.dump([],jsonFile)

    data_ = []
    with open("./processResult/"+reqID+"/"+reqID+".json","r") as overall:
        data_ = json.load(overall)
        data_.append(pdf.getJson())
    with open("./processResult/"+reqID+"/"+reqID+".json", "w") as file_ : 
        json.dump(data_,file_)
    
    return {"success":"True"}

@app.route("/clusterDocuments", methods=['POST'])
def clusterHandling():
    reqID = request.form.get('reqID')
    documents = pd.read_json("./processResult/"+reqID+"/"+reqID+".json")
    cluster_ = ClusterDocuments(documents)
    clusteredData = cluster_.clusterDocuments()
    with open("./processResult/"+reqID+"/"+reqID+".json", "w") as file_ : 
        file_.write(clusteredData)
    return clusteredData

@app.route("/PDF", methods=['GET'])
def fetchPDF():
    # return send_from_directory("./processResult/1632254245253__81206/1/","1.pdf")
    directory = request.args.get('directory')
    pdfID = request.args.get('pdfID')
    return send_from_directory("./processResult/"+directory+"/"+pdfID+"/",pdfID+".pdf")

@app.route("/skimmingDocument", methods=['POST'])
def skimmingDocument():
    reqID = request.form.get('reqID')
    docID = request.form.get('docID')
    document = ProcessPDF("./processResult/"+reqID+"/"+docID+".tei.xml", docID)
    parsedBody = document.getBody()
    return {"parsedBody":parsedBody}

@app.route("/summarizeDocument", methods=['POST'])
def summarizeDocument():
    reqID = request.form.get('reqID')
    docID = request.form.get('docID')
    size = request.form.get('size')
    summarizer = SummarizeDocument(reqID, docID, size)
    summary = summarizer.summarizeText()
    return {"summary":summary}

@app.route("/hyperSimilarity/findSents", methods=['POST'])
def hyperSimilarity_sent():
    reqID = request.form.get('reqID')
    docID = request.form.get('docID')
    topN = request.form.get('topN')
    sentence = request.form.get('sentence')
    hyperSimilarity = HyperSimilarity(reqID, docID, topN)
    similarSentences = hyperSimilarity.findSimilarSentences(sentence)
    return {"similarSentences":similarSentences}

@app.route("/hyperSimilarity/findDocs", methods=['POST'])
def hyperSimilarity_doc():
    reqID = reqID = request.form.get('reqID')
    docID = request.form.get('docID')
    topN = request.form.get('topN')
    sentence = request.form.get('sentence')
    hyperSimilarity = HyperSimilarity(reqID, docID, topN)
    similarDocuments = hyperSimilarity.findSimilarDocs(sentence)
    return {"similarDocuments":similarDocuments}

@app.route("/comparison/basicComparison", methods=['POST'])
def basicComparison():
    reqID = request.form.get('reqID')
    doc1 = request.form.get('doc1')
    doc2 = request.form.get('doc2')
    comparison = Comparison(reqID, doc1, doc2)
    cosineRes = comparison.calculateSimilarity()
    return {"comparisonResult":cosineRes}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')