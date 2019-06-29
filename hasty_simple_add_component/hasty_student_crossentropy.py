import gensim
import os
import collections
import smart_open
import random
import json
import torch
import torch.nn.functional as F
from gensim.test.utils import get_tmpfile
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import json
import argparse 
from utils import shuffle_data, save_model, log_data, load_cleaned_data, accuracy, split_batch

def get_args():
    parser = argparse.ArgumentParser("Hasty_student_recipeQA")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--word_hidden_size", type=int, default=256)
    parser.add_argument("--sent_hidden_size", type=int, default=256)
    parser.add_argument("--log_path", type=str, default="result/log_data.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=str, default=None)
    args = parser.parse_args() 
    return args 

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
def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Model(nn.Module):
    
    def __init__(self, hidden_size=100):
        super(Model, self).__init__() 
        self.hidden_size = hidden_size
        self.linear = nn.Linear(100, 100) 
        self.fname = get_tmpfile("/Users/LYB/Desktop/coursework/Msc-project/recipe_baseline/hasty_simple_baseline/my_doc2vec_model") 
        self.linear_choice = nn.Linear(100, 100)
        self.model = gensim.models.doc2vec.Doc2Vec.load(self.fname)
    def forward(self, input, choice):
        #model = gensim.models.doc2vec.Doc2Vec.load(self.fname)
        question = []
        for i in input: 
            question.append(gensim.utils.simple_preprocess(' '.join(i)))
        for i in range(len(question)):
            question[i] = self.model.infer_vector(question[i])

        for i in range(len(choice)):
            for j in range(len(choice[i])):
                choice[i][j] = self.model.infer_vector(gensim.utils.simple_preprocess(choice[i][j]))
        question = self.linear(torch.FloatTensor(question))
        
        choice = self.linear_choice(torch.FloatTensor(choice))

        answer = []  
        for i in range(len(question)):
            answer.append(torch.nn.functional.cosine_similarity(choice[i], question[i].expand(4, -1), dim=1))

        #answer = torch.FloatTensor(answer) 
        return answer

def train_run(model, train_question, train_answer, train_choice, optimizer, criterion, batch_size):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch_question, batch_answer, batch_choice in tqdm(zip(train_question, train_answer, train_choice)):          
        optimizer.zero_grad() 
        output = model(batch_question, batch_choice)
        output = torch.cat(output, 0).view(-1, len(batch_question))
        output = output.permute(1, 0)
        loss = criterion(output, batch_answer)
        acc = accuracy(output, batch_answer)
        loss.backward() 
        optimizer.step()
        epoch_loss += loss.item() 
        epoch_acc += acc

    return epoch_loss / len(train_question), epoch_acc / len(train_question) 


def eval_run(model, val_question, val_answer, val_choice, criterion):
    if torch.cuda.is_available():
        val_answer = torch.LongTensor(val_answer).cuda()
    else:
        val_answer = torch.LongTensor(val_answer)
    epoch_loss = 0
    epoch_acc = 0
    model.eval() 
    batch_size = len(val_question) 

    with torch.no_grad():
        predictions = model(val_question, val_choice)
        predictions = torch.cat(predictions, 0).view(-1, len(val_question))
        predictions = predictions.permute(1, 0)
        loss = criterion(predictions, val_answer)
        acc = accuracy(predictions, val_answer)
    return loss, acc



def main(args):

    seed_everything()

    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs


    model = Model()
    if args.load_model: 
        model.load_state_dict(torch.load(args.load_model))
        model.eval()
        print("LOAD MODEL:",args.load_model)

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = criterion.to(device)

    max_val_acc = 0.0
    recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = load_cleaned_data('train_cleaned.json')
    recipe_context_valid, recipe_images_valid, recipe_question_valid, recipe_choice_valid, recipe_answer_valid = load_cleaned_data('val_cleaned.json')

    for epoch in tqdm(range(num_epochs)):

        print(epoch)

        train_context_new,train_question_new,train_choice_new,train_answer_new = shuffle_data(recipe_context,recipe_question,recipe_choice,recipe_answer)
        train_context, train_question, train_choice, train_answer = split_batch(batch_size, train_context_new,train_question_new,train_choice_new,train_answer_new)
        
        print(len(train_question)) 

        train_loss, train_acc = train_run(model, train_question, train_answer, train_choice, optimizer, criterion, batch_size)
        valid_loss, valid_acc = eval_run(model, recipe_question_valid, recipe_answer_valid, recipe_choice_valid, criterion)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        # if valid_acc > max_val_acc:
        #     max_val_acc = valid_acc
        #     save_model(model,epoch,valid_acc,args.saved_path)

if __name__ == "__main__":
    args = get_args()
    main(args)





