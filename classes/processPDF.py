from bs4 import BeautifulSoup as bs
# pip install lxml
import json
import re
from classes.processText import ProcessText
from nltk.tokenize import sent_tokenize, word_tokenize

class ProcessPDF:
    def __init__(self, xmlAddress,  documentId) -> None:
        with open(xmlAddress , "r", encoding="utf8") as file :
            content = file.readlines()
            content = "".join(content)
            self.bs_content = bs(content, "xml")
        self.processText_ = ProcessText()
        self.documentId = documentId

    def getTitle(self):
        mainTitle = ""
        titles = self.bs_content.find_all("title")
        
        for t in titles :
            if t.has_attr("level") and t.has_attr("type"):
                if t['level'] == 'a' or t['type'] == 'main':
                    mainTitle = t.text
                    break;

        if mainTitle == "":
            mainTitle = titles[0].text
            
        if mainTitle == "":
            mainTitle = "Unable to recognize title!"
        return mainTitle

    def getPublishingDate(self):
        publicationDate = ""
        publicationstmt = self.bs_content.find_all("publicationstmt")
        if len(publicationstmt) > 0 :
            dates = publicationstmt[0].find_all("date")
            if len(dates) > 0 :
                publicationDate = dates[0]['when']
        if publicationDate == "":
            dates = self.bs_content.find_all("date")
            if len(dates) > 0:
                if dates[0].has_attr("when"):
                    publicationDate = dates[0]['when']
        return publicationDate

    def getAuthors(self):
        authors = []
        filedesc = self.bs_content.find_all('fileDesc')
        authors_ = filedesc[0].find_all("author")
        for a in authors_:
            author_name = ""
            if len(a.find_all('persName')) > 0 : 
                author_name = " ".join([c.text for c in a.persName.contents])
            author_email = ""
            if len(a.find_all('email')) > 0 :
                author_email = a.email.text
            # affiliation : connection to an organization
            author_affiliation = ""
            if len(a.find_all('affiliation')) > 0 :
                author_affiliation = a.affiliation.text.replace("\n"," ")
            author_object = {
                'name' : author_name,
                'email' : author_email,
                'affiliation' : author_affiliation
            } 
            authors.append(author_object)

        return authors

    def getBodyText(self) : 
        body_ = self.bs_content.find_all("body")
        body_p = body_[0].find_all("p")
        bodyText = ""
        for p in body_p:
            bodyText = bodyText + p.text
        return bodyText

    def getKeyWords(self):
        keywords__ = []
        keywords = self.bs_content.find_all('keywords')
        if len(keywords) > 0 :
            terms = keywords[0].find_all("term")
            if len(terms) > 0 :
                for t in terms:
                    keywords__.append(t.text.strip().lower())
            else :
                keywords_ = re.split('(?=[A-Z])', keywords[0].text)
                for k in keywords_ :
                    if k != '':
                        keywords__.append(k.strip().lower())
                            
            #     keywords are extracted from the pdf itself
            #     now it should be extracted from the textual content of the body of the article
        bodyText_ = self.getBodyText()
        words_ = self.processText_.getNormalizedWordFrequency(bodyText_)
        words_ = sorted(words_ , key=words_.get, reverse=True)
        i=0
        while(len(keywords__) < 20):
            if words_[i] not in keywords__:
                keywords__.append(words_[i])
            i += 1
        return keywords__

    def getAbstract(self):
        abstract = ""
        abstract_ = self.bs_content.find_all("abstract")
        if len(abstract_) > 0:
            abstract = abstract_[0].text.replace("\n"," ")
        return abstract

    def getOutlinks(self):
        refs = []
        listbibl_ = self.bs_content.find_all("listBibl")
        if len(listbibl_) > 0:
            refs_ = listbibl_[0].find_all("biblStruct")
            for ref in refs_:
                title_ = ref.find_all("title")[0].text.lower()
                refs.append(title_)
        return refs

    def getBody(self, wf=None):
        body = []
        body_ = self.bs_content.find_all("body")
        wordFreq = None
        if wf == None:
            bodyText_ = self.getBodyText()
            wordFreq = self.processText_.getNormalizedWordFrequency(bodyText_)
        else:
            wordFreq = wf
        sentenceId = 0
        if len(body_) > 0 :
            body_divs = body_[0].find_all('div')
            charNo = 0
            for i,d in enumerate(body_divs):
                charNo += len(d.text)
                divContent = []
                charNo_ = 0
                for c in d.contents:
                    if str(type(c)) == "<class 'bs4.element.Tag'>":
                        charNo_ += len(c.text)
                        if c.name == "head" : 
                            if c.has_attr('n'):
                                header = c['n'] + " " + c.text
                            else :
                                header = c.text
                            headerObject = {
                                "type" : "header",
                                "content" : header,
                                "levelInBody" : 1,
                                "divId" : i,
                                "charCount" : charNo_
                            }
                            divContent.append(headerObject)
                        elif c.name == "p" :
                            pContent = []
                            sentences = sent_tokenize(c.text)
                            for s in sentences:
                                sentScore_ = 0
                                for word in word_tokenize(s.lower()):
                                    if word in wordFreq.keys():
                                        sentScore_ += wordFreq[word]
                                sentObject = {
                                    "type" : "sentenceInP",
                                    "tag" : "span",
                                    "content" : s,
                                    "sentScore" : sentScore_,
                                    "sentenceId" : sentenceId,
                                    "divId" : i
                                }
                                sentenceId += 1
                                pContent.append(sentObject)
                            pObject = {
                                "type" : "paragraph",
                                "tag" : "p",
                                "content" : pContent,
                                "divId" : i,
                                "charCount" : charNo_
                            }
                            divContent.append(pObject)
                        elif c.name == 'formula' :
                            formulaObject = {
                                "tag" : 'p',
                                "content" : c.text,
                                "type" : 'formula',
                                "divId" : i,
                                "charCount" : charNo_
                            }
                            divContent.append(formulaObject)
                        elif c.name == "list":
                            items = []
                            items_ = c.find_all('item')
                            for item in items_:
                                itemObject = {
                                    "tag" : 'p',
                                    "type" : 'item',
                                    "content" : item.text,
                                    "divId" : i
                                }
                                items.append(itemObject)
                            listObject = {
                                "type" : "list",
                                "content" : items,
                                "charCount" : charNo_
                            }
                            divContent.append(listObject)

                    divObject = {
                        "divId" : i,
                        "content" : divContent,
                        "charCount" : charNo
                    }
                body.append(divObject)
        return body

    def getPublisher(self):
        publisher = ''
        publicationstmt = self.bs_content.find_all("publicationstmt")
        if len(publicationstmt) > 0 :
            publisher_ = publicationstmt[0].find_all("publisher")
            if len(publisher_) > 0 :
                publisher = publisher_[0].text
        if publisher == '':
            publishers = self.bs_content.find_all("publisher")
            if len(publishers) > 0:
                publisher = publishers[0].text
        if publisher == '':
            publisher = "Unable to recognize publisher!"
        return publisher
    
    def getJson(self):
        return {
            "id": self.documentId,
            "title": self.getTitle(),
            "authors" : self.getAuthors(),
            "publishingDate" : self.getPublishingDate(),
            "publisher" : self.getPublisher(),
            "keywords" : self.getKeyWords(),
            "abstract" : self.getAbstract(),
            "outlinks" : self.getOutlinks()
        }

    def validateDictAsJson(self, dict_):
        json_str = json.dumps(dict_)
        try:
            json.loads(json_str)
        except ValueError as e:
            return False
        return True