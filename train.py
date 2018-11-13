from gensim.models.fasttext import FastText

import re
import glob

# Taken from: https://coderwall.com/p/rcmaea/flatten-a-list-of-lists-in-one-line-in-python
def flatten(lines):
  return [y for x in lines for y in x]

# Taken from: https://stackoverflow.com/questions/43358857/how-to-remove-special-characters-except-space-from-a-file-in-python/43358965#answer-43359001
def remove_special_char(line):
  return re.sub(r'\W+', ' ', line).strip()

# Taken from: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python#answer-50036508
def remove_link(line):
  return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", line).strip()

def get_sentence(line):
  return line.split('|')[-1].lower()

# Taken from: https://stackoverflow.com/questions/275018/how-can-i-remove-chomp-a-trailing-newline-in-python#answer-50870896
def strip_newline(line):
  return re.sub('(\\r|)\\n$', '', line)

def remove_empty_string(lines):
  return list(filter(str.strip, lines))

# Taken from: https://stackoverflow.com/questions/3277503/in-python-how-do-i-read-a-file-line-by-line-into-a-list#answer-43625375
def read_file(filename):
  with open(filename, encoding = 'utf-8', errors = 'ignore') as file:
    return remove_empty_string([remove_special_char(remove_link(get_sentence(strip_newline(line)))) for line in file])

def get_lines():
  return [read_file(file_name) for file_name in glob.glob('datasets/*.txt')]

datasets = [data.split(' ') for data in flatten(get_lines())]

# print(datasets[:100])
model = FastText(sentences = datasets, size = 100, window = 5, min_count = 5, workers = 4, sg = 0, hs = 1)
model.save('model.bin')

# print(model.wv.most_similar('ebola'))
