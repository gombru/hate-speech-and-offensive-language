from preprocess_tweets import tweet_preprocessing
import os
import json
import random

base_path = '../../../datasets/HateSPic/HateSPicLabeler/generated_json_v2mm/'
datasets = ['HateSPic/','SemiSupervised/','WZ-LS/']
out_file_train = open('../data/MMHS10K_data.csv', 'w')
out_file_test = open('../data/MMHS10K_test_data.csv', 'w')

# Load test indices
test_indices = []
with open('../../../datasets/HateSPic/HateSPic/tweet_embeddings/MMHS-v2mm-lstm_embeddings_test_hate.txt') as f:
    for line in f:
        data = line.split(',')
        test_indices.append(int(data[0]))

with open('../../../datasets/HateSPic/HateSPic/tweet_embeddings/MMHS-v2mm-lstm_embeddings_test_nothate.txt') as f:
    for line in f:
        data = line.split(',')
        test_indices.append(int(data[0]))


# count=0
for dataset in datasets:
    for file in os.listdir(base_path + dataset):
        info = json.load(open(base_path + dataset + file, 'r'))
        label = 0
        if info['hate_votes'] > info['not_hate_votes']:
            label = 1
        if 'mm_hate_votes' in info:
            if info['mm_hate_votes'] > info['not_hate_votes']:
                label = 1
        text = tweet_preprocessing(info['text'].encode('utf-8'))
        # Discard short tweets
        # if len(text) < 5: continue
        # if len(text.split(' ')) < 3: continue
        if label == 0: class_id = 2
        if label == 1: class_id = 0
        labels = str(info['id']) + ',3,' + str(label) + ',0,' + str(1-label) + ',' + str(class_id)
        if info['id'] in test_indices:
            out_file_test.write(labels + ',' + text + '\n')
        else:
            out_file_train.write(labels + ',' + text + '\n')


        # count+=1

print("DONE")