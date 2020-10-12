import re
import os
import glob
import pprint
from bs4 import BeautifulSoup
import csv
import pandas as pd
from xml.dom import minidom
import urllib.request
import zipfile
import tarfile
from collections import defaultdict
import shutil

documents = []
for file in glob.glob("*.sgml"):
    with open(file,'r') as f:
        sgml = f.read()
        soup = BeautifulSoup(sgml,'lxml')
        # want - mistake, type, correction
        # filter first by doc. Save each doc in a object
        # for each doc, filter by paras and mistakes
        docs = soup.find_all('doc')
        for d in docs:
            paras = d.find_all('p')
            mistakes = d.find_all('mistake')
            # doc = {"paras": paras,"mistakes":mistakes}
            # documents.append(doc)
            for mistake in mistakes:
                # print("hi")
                # Fetching Error types and Corrections
                types = mistake.type.text.split("\n")
                corrections = mistake.correction.text.split("\n")
                # Fetching offsets for mistaken-words
                start_para = int(mistake.attrs["start_par"])
                end_para = int(mistake.attrs["end_par"])
                start_off = int(mistake.attrs["start_off"])
                end_off = int(mistake.attrs["end_off"])

                if start_para != end_para:
                    #print("unhandled case")
                    break
                # get word using offset
                word_text = paras[start_para-1].text
                word = word_text[start_off:end_off+1]
                #print(word)
                documents.append({'Paragraphs':str(paras[start_para-1]),'Start Offset':start_off, 'End Offset':end_off, 'Mistake-Words':str(word),'Error Type':str(types),'Corrections':str(corrections)})
              
df = pd.DataFrame(documents)
csv_data = df.to_csv(index=False)
