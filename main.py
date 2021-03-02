import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import csv



def read_line(data, x, cnt):
    y = data.readline()
    if cnt == 0:
      x = y
    else:
      x.append(y)
    return y


f = open("data/train_samples.txt", "r", encoding='utf-8')
if f.mode == 'r':
    cnt = 0
    train_samples = []
    while read_line(f, train_samples, cnt) and cnt <= 5717:
        cnt += 1

f = open("data/train_labels.txt", "r", encoding='utf-8')
if f.mode == 'r':
    cnt = 0
    train_labels = []
    while read_line(f, train_labels, cnt) and cnt <= 5717:
        cnt += 1

f = open("data/validation_source_samples.txt", "r", encoding='utf-8')
if f.mode == 'r':
    cnt = 0
    validation_source_samples = []
    while read_line(f, validation_source_samples, cnt) and cnt <= 5919:
        cnt += 1

f = open("data/validation_source_labels.txt", "r", encoding='utf-8')
if f.mode == 'r':
    cnt = 0
    validation_source_labels = []
    while read_line(f, validation_source_labels, cnt) and cnt <= 5919:
        cnt += 1

f = open("data/validation_target_samples.txt", "r", encoding ='utf-8')
if f.mode == 'r':
    cnt = 0
    validation_target_samples = []
    while read_line(f, validation_target_samples, cnt) and cnt <= 213:
        cnt += 1

f = open("data/validation_target_labels.txt", "r", encoding ='utf-8')
if f.mode == 'r':
    cnt = 0
    validation_target_labels = []
    while read_line(f, validation_target_labels, cnt) and cnt <= 213:
        cnt += 1

f = open("data/test_samples.txt", "r", encoding='utf-8')
if f.mode == 'r':
    test_samples = []
    while read_line(f, test_samples, cnt):
        pass


id_list = []


for idx in range(len(train_samples)):
    train_samples[idx] = train_samples[idx].replace('$NE$', '')
    train_samples[idx] = re.sub('[0123456789!$@&+%#?/|,.:„”“";()\n\t…''"]', '', train_samples[idx])


# train_labels parsing
for idx in range(len(train_labels)):
    train_labels[idx] = train_labels[idx].split()
    train_labels[idx].pop(0)

aux = []
for idx in range(len(train_labels)):
    aux.append(train_labels[idx])

train_labels = aux

for idx in range(len(validation_source_samples)):
    validation_source_samples[idx] = validation_source_samples[idx].replace('$NE$', '')
    validation_source_samples[idx] = re.sub('[0123456789!$@&+%#?/|,.:„”“";()\n\t…''"]', '', validation_source_samples[idx])

for idx in range(len(validation_source_labels)):
    validation_source_labels[idx] = validation_source_labels[idx].split()
    validation_source_labels[idx].pop(0)

aux = []
for idx in range(len(validation_source_labels)):
    aux.append(validation_source_labels[idx])

validation_source_labels = aux

for idx in range(len(validation_target_samples)):
   validation_target_samples[idx]= validation_target_samples[idx].replace('$NE$', '')
   validation_target_samples[idx] = re.sub('[0123456789!$@&+%#?/|,.:„”“";()\n\t…''"]', '', validation_target_samples[idx])

for idx in range(len(validation_target_labels)):
    validation_target_labels[idx] = validation_target_labels[idx].split()
    validation_target_labels[idx].pop(0)

aux = []
for idx in range(len(validation_target_labels)):
    aux.append(validation_target_labels[idx])

validation_target_labels = aux


for idx in range(len(test_samples) - 1):
    test_samples[idx] = test_samples[idx].replace('$NE$', '')
    line = test_samples[idx].split()
    id_list.append(line[0])
    test_samples[idx] = re.sub('[0123456789!$@&+%#?/|,.:„”“";()\n\t…''"]', '', test_samples[idx])

file = open("data/result.txt", "w", encoding='utf-8')
file.write(str(test_samples))

for idx in range(len(validation_source_samples)):
    train_samples.append(validation_source_samples[idx])

for idx in range(len(validation_source_labels)):
    train_labels.append(validation_source_labels[idx])



tf_vectorizer = TfidfVectorizer(ngram_range=(1, 4))
train_samples_tfidf = tf_vectorizer.fit_transform(train_samples)
validation_source_samples_tfidf = tf_vectorizer.fit_transform(validation_source_samples)
validation_target_samples_tfidf = tf_vectorizer.transform(validation_target_samples)
test_samples_tfidf = tf_vectorizer.transform(test_samples)



# print(accuracy_score(predicted_labels, validation_source_labels))

# print(f1_score(predicted_labels, validation_target_labels))



C_param = 1
svm_model = svm.LinearSVC(C=C_param)  # kernel liniar
svm_model.fit(train_samples_tfidf, np.ravel(train_labels))  # train
svm_model.fit(validation_source_samples_tfidf, np.ravel(validation_source_labels))
svm_model.fit(validation_target_samples_tfidf, np.ravel(validation_target_labels))

predicted_labels = svm_model.predict(validation_target_samples_tfidf)
with open('sample_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["id", "label"])
    for idx in range(len(predicted_labels) - 1):
        writer.writerow([id_list[idx], predicted_labels[idx]])

# print(accuracy_score(predicted_labels, validation_target_labels))