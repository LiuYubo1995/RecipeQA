import gensim
import os
import collections
import smart_open
import random
import json
import torch
import torch.nn.functional as F
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
def accuracy(preds, y):
    preds = F.softmax(preds, dim=1)
    correct = 0 
    pred = preds.max(1, keepdim=True)[1]
    correct += pred.eq(y.view_as(pred)).sum().item()
    acc = correct/len(y)
    return acc 

recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = load_cleaned_data('train_cleaned.json')
recipe_context_valid, recipe_images_valid, recipe_question_valid, recipe_choice_valid, recipe_answer_valid = load_cleaned_data('val_cleaned.json')
fname = get_tmpfile("my_doc2vec_model")
model = gensim.models.doc2vec.Doc2Vec.load(fname) 
question = []
for i in recipe_question_valid:
    question.append(gensim.utils.simple_preprocess(' '.join(i)))
for i in range(len(question)):
    question[i] = model.infer_vector(question[i])
for i in range(len(recipe_choice_valid)):
    for j in range(len(recipe_choice_valid[i])):
        recipe_choice_valid[i][j] = model.infer_vector(gensim.utils.simple_preprocess(recipe_choice_valid[i][j])) 
question = torch.FloatTensor(question)
choice = torch.FloatTensor(recipe_choice_valid)
answer = [] 
for i in range(len(question)):
    answer.append(torch.matmul(choice[i], question[0]).numpy())
answer = torch.FloatTensor(answer) 
answer_valid = torch.LongTensor(recipe_answer_valid)
acc_val = accuracy(answer, answer_valid)
print('validation accuracy', acc_val) 




