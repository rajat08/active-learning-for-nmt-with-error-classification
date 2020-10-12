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
for file in glob.glob("Scripts/CONLL14_Parser/data/*.sgml"):
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

            temp = []
            original = [para.text for para in paras]
            texts = [para.text for para in paras]
            
            mistake_counter = 0
            for mistake in mistakes:
                
                # Fetching Error types and Corrections
                # types = mistake.type.text.split("\n")
                tag = mistake.find_all('type')[0].text
                # corrections = mistake.correction.text.split("\n")
                correction = mistake.find_all('correction')[0].text
                # Fetching offsets for mistaken-words
                start_para = int(mistake.attrs["start_par"])
                end_para = int(mistake.attrs["end_par"])
                start_off = int(mistake.attrs["start_off"]) 
                end_off = int(mistake.attrs["end_off"])

                if start_para != end_para:
                    #print("unhandled case")
                    break
                # get word using offset
                # special case
                word_text = original[start_para-1]
                word = word_text[start_off:end_off+1]
                
                # replace word in text 
                word_text = word_text[:start_off] + " " + correction + " " + word_text[start_off+ len(correction):]
                texts[start_para - 1] = word_text
                
                temp.append({"para":start_para - 1, "stuff": {"tag": tag, "start_off": start_off, "end_off": end_off, "correct": correction, "incorrect": word}})
            
            output = []
            for t in temp: 
                para_id = t["para"]
                output.append({"incorrect": original[para_id], "correct": texts[para_id], "error": t["stuff"]})

            print(output)

                #print(word)
                # documents.append({'incorrect':str(paras[start_para-1]),'start_off':start_off, 'end_off':end_off, 'Mistake-Words':str(word),'Error Type':str(types),'Corrections':str(corrections)})
              
# df = pd.DataFrame(documents)
# csv_data = df.to_csv(,index=False)
