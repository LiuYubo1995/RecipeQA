import gensim
import os
import collections
import smart_open
import random
import json
from gensim.test.utils import get_tmpfile

def load_cleaned_data(file = 'train_cleaned.json'):
    file = open(file, 'r', encoding='utf8').read()
    recipe = json.loads(file) #json file contains data in str, convert str to dict
    recipe_context = recipe['context']
    recipe_answer = recipe['answer']
    recipe_choice = recipe['choice']
    recipe_question = recipe['question']
    recipe_images = recipe['images'] 
    return recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer 
recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = load_cleaned_data('train_cleaned.json')
recipe_context_valid, recipe_images_valid, recipe_question_valid, recipe_choice_valid, recipe_answer_valid = load_cleaned_data('val_cleaned.json')
new_train = []
for i in recipe_context:
    new_train.append(' '.join(i))
for i in recipe_question:
    for j in i:
        new_train.append(j)
for i in recipe_choice:
    for j in i:
        new_train.append(j)
for i in recipe_context:
    new_train.append(' '.join(i))
for i in recipe_question:
    for j in i:
        new_train.append(j)
for i in recipe_choice:
    for j in i:
        new_train.append(j)
new_val = []
for i in recipe_context_valid:
    new_val.append(' '.join(i))
for i in recipe_question_valid:
    for j in i:
        new_val.append(j)
for i in recipe_choice_valid:
    for j in i:
        new_val.append(j) 
def read_corpus(data_list, tokens_only=False):
    for i, line in enumerate(data_list):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags     
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(new_train))
test_corpus = list(read_corpus(new_val))
model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=100)
model.build_vocab(test_corpus)
print(len(model.wv.vocab)) 
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
fname = get_tmpfile("../my_doc2vec_model_val")
model.save(fname) 
