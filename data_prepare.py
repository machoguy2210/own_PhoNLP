import pickle
from transformers import AutoTokenizer
import torch

def split_data(data):
    return data.split()
 
with open('sample_data/ner_train.txt', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    nerdata = list(map(split_data, data))

with open('sample_data/pos_train.txt', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    posdata = list(map(split_data, data))
k = ""
q = []
ner_sentences = []
ner_labels = []
ner_set = set()
for i in nerdata:
    if i == []:
        ner_sentences.append(k.strip())
        ner_labels.append(q)
        k = ""
        q = []
    else:
        k += " " + i[0]
        q.append(i[1])
        ner_set.add(i[1])

k = ""
q = []
pos_sentences = []
pos_labels = []
pos_set = set()
for i in posdata:
    if i == []:
        pos_sentences.append(k.strip())
        pos_labels.append(q)
        k = ""
        q = []
    else:
        k += " " + i[0]
        q.append(i[1])
        pos_set.add(i[1])

labels = []

for i in range(len(ner_labels)):
    labels.append(list(zip(ner_labels[i], pos_labels[i])))

ner_dict_w2i = {}
ner_dict_i2w = {}

for i,j in enumerate(ner_set):
    ner_dict_w2i[j] = i
    ner_dict_i2w[i] = j

pos_dict_w2i = {}
pos_dict_i2w = {}

for i,j in enumerate(pos_set):
    pos_dict_w2i[j] = i
    pos_dict_i2w[i] = j

encode_label = []

for i in labels:
    encode_label.append(list(map(lambda x: (ner_dict_w2i[x[0]], pos_dict_w2i[x[1]]), i)))
                    

with open('sample_data/ner_dict_w2i.pkl', 'wb') as f:
    pickle.dump(ner_dict_w2i, f)

with open('sample_data/ner_dict_i2w.pkl', 'wb') as f:
    pickle.dump(ner_dict_i2w, f)

with open('sample_data/pos_dict_w2i.pkl', 'wb') as f:
    pickle.dump(pos_dict_w2i, f)

with open('sample_data/pos_dict_i2w.pkl', 'wb') as f:
    pickle.dump(pos_dict_i2w, f)

with open('sample_data/encode_label.pkl', 'wb') as f:
    pickle.dump(encode_label, f)

with open('sample_data/input_sentences.pkl', 'wb') as f:
    pickle.dump(ner_sentences, f)

with open('sample_data/input_labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

sentences = []

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

#Đảm bảo tokenize xong câu vẫn giữ nguyên độ dài
for i in ner_sentences:
    en_input = tokenizer.encode(i)
    if len(en_input) != len(i.split()) + 2:
        en_input = [0]
        for j in i.split():
            en_input.append(tokenizer.encode(j)[1])
        en_input.append(2)
    sentences.append(en_input)

with open('sample_data/input_encode.pkl', 'wb') as f:
    pickle.dump(sentences, f)

