
from xml.dom import minidom
import utils
import os
import urllib.request
import zipfile
import tarfile
from collections import defaultdict
import shutil
import glob
from bs4 import BeautifulSoup
import csv

__author__ = "Gwena Cunha"

"""
- Downloads dataset
- Extracts information from XML (original and corrected essays)
- Saves as txt files: target, source, vocab
"""


class FCECorpusHandler:
    def __init__(self, args, in_file_ext='xml', out_file_ext='txt'):
        self.args = args
        self.in_file_ext = in_file_ext
        self.out_file_ext = out_file_ext

        self.results_dir = args.results_dir
        utils.ensure_dir(self.results_dir)

        self.fce_xml_dir = self.args.fce_xml_dir
        print("Create data directory: {}".format(self.fce_xml_dir))
        utils.ensure_dir(self.fce_xml_dir)

        self.fce_error_detection_dir = self.args.fce_xml_dir + 'fce-error-detection'
        self.download_fce_error_detection_corpus()

        self.fce_dir = self.args.fce_xml_dir + 'fce-released-dataset'
        self.download_fce_corpus()

    def download_fce_error_detection_corpus(self):
        """ Check if FCE Error Detection Corpus exists and only download it if it doesn't

        :return: directory where FCE Error Detection Corpus is located
        """
        download_link = 'https://s3-eu-west-1.amazonaws.com/ilexir-website-media/fce-error-detection.tar.gz'
        if not os.path.exists(self.fce_error_detection_dir):
            print("Downloading FCE Error Detection Corpus")
            targz_fce_filename = self.fce_error_detection_dir + '.tar.gz'
            # Download file
            urllib.request.urlretrieve(download_link, targz_fce_filename)

            # Untar compressed file
            tar = tarfile.open(targz_fce_filename)
            tar.extractall(self.fce_xml_dir)
            tar.close()

            # Delete .tar.gz file
            os.remove(targz_fce_filename)
        else:
            print("FCE Error Detection Corpus has already been downloaded in: {}".format(
                self.fce_error_detection_dir))

    def download_fce_corpus(self):
        """ Check if FCE Corpus exists and only download it if it doesn't

        :return: directory where FCE Corpus is located
        """
        download_link = 'https://s3-eu-west-1.amazonaws.com/ilexir-website-media/fce-released-dataset.zip'
        if not os.path.exists(self.fce_dir):
            print("Downloading FCE Corpus")
            zip_fce_filename = self.fce_dir + '.zip'
            # Download file
            urllib.request.urlretrieve(download_link, zip_fce_filename)

            # Unzip compressed file
            zip_ref = zipfile.ZipFile(zip_fce_filename, 'r')
            zip_ref.extractall(self.fce_xml_dir)
            zip_ref.close()

            # Delete .tar.gz file
            os.remove(zip_fce_filename)
        else:
            print("FCE Corpus has already been downloaded in: {}".format(self.fce_dir))

    def get_train_dev_test_sets(self):
        print("\nGet train-dev-test sets")

        # Copy all files from "fce-released-dataset/dataset/*" to "fce-released-dataset/dataset_all/"
        fce_dir_dataset = self.fce_dir + '/dataset/'
        fce_dir_save = self.fce_dir + '/dataset_all/'
        utils.ensure_dir(fce_dir_save)

        subdirs = os.listdir(fce_dir_dataset)
        for s in subdirs:
            for txt_file in glob.glob(fce_dir_dataset + s + "/*.xml"):
                shutil.copy2(txt_file, fce_dir_save)

        # Separate train, dev, test
        train_dev_test_path = self.fce_error_detection_dir + '/filenames/'
        train_dev_test_files = [f for f in os.listdir(
            train_dev_test_path) if f.endswith('.txt')]
        print(train_dev_test_files)
        for file in train_dev_test_files:
            set_dir = self.fce_dir + '/' + file.split('.')[1] + '/'
            utils.ensure_dir(set_dir)

            f = open(train_dev_test_path + file, 'r')
            f_lines = f.read().split('\n')
            for l in f_lines:
                if len(l) > 0:
                    shutil.copy2(fce_dir_save + l, set_dir)
            f.close()

    def xml_to_txt(self, data_type='train', verbose=False):
        print("\nGet train-dev-test sets")
        fce_dir_dataset = '{}/{}/'.format(self.fce_dir, data_type)
        train_dev_test_files = [f for f in os.listdir(
            fce_dir_dataset) if f.endswith('.xml')]

        # Convert from xml to txt
        incorrect_sentences = []
        correct_sentences = []
        for f in train_dev_test_files:
            if verbose:
                print()
                print(f)
            mydoc = minidom.parse(fce_dir_dataset + f)
            items_essay = mydoc.getElementsByTagName('p')
            for item_essay in items_essay:
                incorrect_sent, correct_sent = self.strip_str(
                    item_essay, verbose=verbose)
                incorrect_sentences.append(incorrect_sent)
                correct_sentences.append(correct_sent)

        # Save in results_dir
        # Task: generate incorrect sentences from correct ones, thus source is incorrect and target is correct
        fce_txt_dir_dataset = '{}{}/'.format(self.results_dir, data_type)
        utils.ensure_dir(fce_txt_dir_dataset)
        print(fce_txt_dir_dataset)

        file_source_txt = open(fce_txt_dir_dataset + "source.txt", 'w')
        # file_source_txt.write('\n'.join(sent for sent in incorrect_sentences))
        file_source_txt.write('\n'.join(correct_sentences))
        file_target_txt = open(fce_txt_dir_dataset + "target.txt", 'w')
        file_target_txt.write('\n'.join(incorrect_sentences))
        file_source_txt.close()
        file_target_txt.close()

        print("Finished writing {} files".format(data_type))

    def strip_str(self, item_essay, verbose=False):
        incorrect_sent = ''
        correct_sent = ''
        error_tags = ''
        errors = []
        pos = 0
        for child in item_essay.childNodes:
            if child.localName is None:  # no child nodes
                segment = child.data
                incorrect_sent += segment
                correct_sent += segment

                # increment pos 
                pos += len(segment)
                # print(segment)
            else:  # 'NS', 'i', 'c'
                inc_sent, cor_sent = self.recursive_NS_tag_strip(child)
                incorrect_sent += inc_sent
                correct_sent += cor_sent
                if child._attrs is not None:
                    # to get incorrect word, incorrect_sent[errors[err_no]["start_off"]: errors[err_no]["end_off"]]
                    errors.append({"tag": child._attrs['type'].value, "incorrect": inc_sent, "correct": cor_sent, "start_off": pos, 
                    "end_off": pos + len(inc_sent)})
                else: 
                    print("No Error tag!; In-Correct:", inc_sent, "Correct:", cor_sent)    

                # update pos
                pos += len(inc_sent)

        if verbose:
            print("Incorrect sentence: " + incorrect_sent)
            print("Correct sentence: " + correct_sent)
        return incorrect_sent, correct_sent, errors

    def recursive_NS_tag_strip(self, item_ns):
        incorrect_sent = ''
        correct_sent = ''
        if item_ns.localName is None:  # base case
            segment = item_ns.data
            return segment, segment
        elif item_ns.localName == 'i':  # incorrect
            if item_ns.childNodes[0].localName is None:
                segment = item_ns.childNodes[0].data
                return segment, ''
            else:
                for child in item_ns.childNodes:
                    inc_sent, cor_sent = self.recursive_NS_tag_strip(child)
                    incorrect_sent += inc_sent
                    correct_sent += cor_sent
        elif item_ns.localName == 'c':  # correct
            if item_ns.childNodes[0].localName is None:
                segment = item_ns.childNodes[0].data
                return '', segment
            else:
                for child in item_ns.childNodes:
                    inc_sent, cor_sent = self.recursive_NS_tag_strip(child)
                    incorrect_sent += inc_sent
                    correct_sent += cor_sent
        else:  # NS
            for child in item_ns.childNodes:
                inc_sent, cor_sent = self.recursive_NS_tag_strip(child)
                incorrect_sent += inc_sent
                correct_sent += cor_sent
        return incorrect_sent, correct_sent

    def save_file(self, text, dir, filename):
        utils.ensure_dir(dir)
        file = open(dir + filename, 'w')
        file.write(text)
        file.close()

    def parse_data(self, data_type='train', verbose=False):
        print("\Get train-dev-test sets")
        fce_dir_dataset = '{}/{}/'.format(self.fce_dir, data_type)
        train_dev_test_files = [f for f in os.listdir(
            fce_dir_dataset) if f.endswith('.xml')]

        fields = ["incorrect", "correct", "error_tag", "inc_word", "cor_word", "start_off", "end_off"]

        all_output = []
        for f in train_dev_test_files:
            if verbose:
                print()
                print(f)
            mydoc = minidom.parse(fce_dir_dataset + f)
            # fetch each answer in doc
            answer = mydoc.getElementsByTagName('coded_answer')
            # items_essay = mydoc.getElementsByTagName('p')

            # loop through every answer/essay
            ans_count = 0
            for ans in answer: 
                # convert xml to txt
                incorrect_sentences = []
                correct_sentences = []
                error_tags = []

                items_essay = ans.getElementsByTagName('p')
                ans_count += 1
                output = []

                for item_essay in items_essay:
                    incorrect_sent, correct_sent, errors = self.strip_str(
                        item_essay, verbose=verbose)
                    # incorrect_sentences.append(incorrect_sent)
                    # correct_sentences.append(correct_sent)
                    # error_tags.append(errors)
                    for error in errors: 
                        output.append({"incorrect": incorrect_sent, "correct": correct_sent, "error_tag": error['tag'],
                         "inc_word": error['incorrect'], "cor_word": error['correct'], "start_off": error['start_off'], "end_off":['end_off']})
                        all_output.append({"incorrect": incorrect_sent, "correct": correct_sent, "error_tag": error['tag'],
                         "inc_word": error['incorrect'], "cor_word": error['correct'], "start_off": error['start_off'], "end_off":['end_off']})
                
                
                fce_txt_dir_dataset = '{}{}/{}/'.format(self.results_dir, data_type, os.path.splitext(f)[0])
                utils.ensure_dir(fce_txt_dir_dataset)
                print(fce_txt_dir_dataset)

                outfile_name = fce_txt_dir_dataset+"essay_"+str(ans_count)+".csv"
                with open(outfile_name, "w") as outfile: 
                    writer = csv.DictWriter(outfile, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(output)
                
                
                
        with open(self.results_dir + "all_converted.csv", "w") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_output)

        print("Finished writing {} files".format(data_type))


            # with open(fce_dir_dataset + f) as current_file: 
            #     text = current_file.read()
            #     soup = BeautifulSoup(text, "lxml")
            #     answers = soup.find_all('coded_answer')

            #     for ans in answers: 
            #         essay = ans.find_all('p')
            #         print()
        
        # print(error_tags)
