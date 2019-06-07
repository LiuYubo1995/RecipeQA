
from impatient_reader_model import Impatient_Reader_Model
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import json
from data_processing import load_cleaned_data
import argparse 

def get_args():
    parser = argparse.ArgumentParser("Hasty_student_recipeQA")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--word_hidden_size", type=int, default=256)
    parser.add_argument("--sent_hidden_size", type=int, default=256)
    parser.add_argument("--log_path", type=str, default="result/log_data.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=str, default=None)
    args = parser.parse_args() 
    return args 

def accuracy(preds, y):
    preds = F.softmax(preds, dim=1)
    correct = 0 
    pred = preds.max(1, keepdim=True)[1]
    correct += pred.eq(y.view_as(pred)).sum().item()
    acc = correct/len(y)

    return acc
def train_run(model, train_context, train_question, train_choice, train_answer, optimizer, criterion, batch_size):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    a = 0
    # max_acc = 0.0
    for batch_context, batch_question, batch_choice, batch_answer in tqdm(zip(train_context, train_question, train_choice, train_answer)):          
        optimizer.zero_grad() 
        output = model(batch_context, batch_question, batch_choice)
        output = torch.cat(output, 0).view(-1, len(batch_context))
        output = output.permute(1, 0)  
        loss = criterion(output, batch_answer)
        acc = accuracy(output, batch_answer)
        loss.backward() 
        optimizer.step()
        epoch_loss += loss.item() 
        epoch_acc += acc

    return epoch_loss / len(train_context), epoch_acc / len(train_context) 


def eval_run_batch(model, val_context, val_question, val_choice, val_answer, criterion, batch_size):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch_context, batch_question, batch_choice, batch_answer in tqdm(zip(val_context, val_question, val_choice, val_answer)):           
            output = model(batch_context, batch_question, batch_choice)
            output = torch.cat(output, 0).view(-1, len(batch_context))
            output = output.permute(1, 0)  
            loss = criterion(output, batch_answer)
            acc = accuracy(output, batch_answer) 
            epoch_loss += loss.item() 
            epoch_acc += acc

    return epoch_loss / len(val_context), epoch_acc / len(val_context)

def eval_run(model, val_context, val_question, val_choice, val_answer,criterion):
    if torch.cuda.is_available():
        val_answer = torch.LongTensor(val_answer).cuda()
    else:
        val_answer = torch.LongTensor(val_answer)
    epoch_loss = 0
    epoch_acc = 0
    model.eval() 
    batch_size = len(val_context) 

    with torch.no_grad():
        predictions = model(val_context, val_question, val_choice)
        predictions = torch.cat(predictions, 0).view(-1, len(val_context))
        predictions = predictions.permute(1, 0)
        loss = criterion(predictions, val_answer)
        acc = accuracy(predictions, val_answer)
    return loss, acc


def shuffle_data(recipe_context,recipe_question,recipe_choice,recipe_answer):
    #shuffle
    combine = list(zip(recipe_context,recipe_question,recipe_choice,recipe_answer))
    np.random.shuffle(combine)
    recipe_context_shuffled,recipe_question_shuffled, recipe_choice_shuffled, recipe_answer_shuffled = zip(*combine)
    recipe_context_shuffled = list(recipe_context_shuffled)
    recipe_question_shuffled = list(recipe_question_shuffled) 
    recipe_choice_shuffled = list(recipe_choice_shuffled)
    recipe_answer_shuffled = list(recipe_answer_shuffled)
    return recipe_context_shuffled,recipe_question_shuffled, recipe_choice_shuffled, recipe_answer_shuffled

def save_model(model, epoch,accuracy, saved_path):
    torch.save(model.state_dict(),
               '%s/hasty_student_epoch_%d_%f_acc.pth' % (saved_path, epoch,accuracy))
    print('Save model with accuracy:',accuracy)

def log_data(log_path,train_loss,train_accuracy,val_loss,val_accuracy):
    file = open(log_path,'a')
    if torch.cuda.is_available():
        data = str(train_loss) +' '+ str(f'{train_accuracy:.2f}') \
            +' '+ str(val_loss)+ ' ' + str(f'{val_accuracy:.2f}')    #####如果不是batch， 这要改
    else:
        data = str(train_loss) + ' '+ str(f'{train_accuracy:.2f}') \
                +' '+str(val_loss)+' '+str(f'{val_accuracy:.2f}')
    file.write(data)
    file.write('\n')
    file.close() 
        



def main(args):

    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    word_hidden_size = args.word_hidden_size
    sent_hidden_size = args.sent_hidden_size


    recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = load_cleaned_data('train_cleaned.json')
    recipe_context_val, recipe_images_val, recipe_question_val, recipe_choice_val, recipe_answer_val = load_cleaned_data('val_cleaned.json')

    model = Impatient_Reader_Model(word_hidden_size, sent_hidden_size, batch_size)
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
    for epoch in tqdm(range(num_epochs)):

        print(epoch)

        recipe_context_new,recipe_question_new,recipe_choice_new,recipe_answer_new = shuffle_data(recipe_context,recipe_question,recipe_choice,recipe_answer)
        val_context_new,val_question_new,val_choice_new,val_answer_new = shuffle_data(recipe_context_val,recipe_question_val,recipe_choice_val,recipe_answer_val)
        # recipe_context_new = recipe_context_new[0:15]
        # recipe_question_new = recipe_question_new[0:15]
        # recipe_choice_new = recipe_choice_new[0:15]
        # recipe_answer_new = recipe_answer_new[0:15]
        train_context = []
        train_question = [] 
        train_choice = []
        train_answer = []

        for i in tqdm(range(0, len(recipe_question_new), batch_size)):
            train_context.append(recipe_context_new[i : i + batch_size]) 
            train_question.append(recipe_question_new[i : i + batch_size])  
            train_choice.append(recipe_choice_new[i : i + batch_size])
            actual_scores = recipe_answer_new[i : i + batch_size]
            if torch.cuda.is_available():
                actual_scores = torch.LongTensor(actual_scores).cuda()
            else:
                actual_scores = torch.LongTensor(actual_scores)
            train_answer.append(actual_scores)


        val_context = []
        val_question = [] 
        val_choice = []
        val_answer = []

        for i in tqdm(range(0, len(val_question_new), batch_size)):
            val_context.append(val_context_new[i : i + batch_size]) 
            val_question.append(val_question_new[i : i + batch_size])  
            val_choice.append(val_choice_new[i : i + batch_size])
            actual_scores = val_answer_new[i : i + batch_size]
            if torch.cuda.is_available():
                actual_scores = torch.LongTensor(actual_scores).cuda()
            else:
                actual_scores = torch.LongTensor(actual_scores)
            val_answer.append(actual_scores)  

        train_loss, train_acc = train_run(model, train_context, train_question, train_choice, train_answer, optimizer, criterion, batch_size)
        valid_loss, valid_acc = eval_run_batch(model, val_context, val_question, val_choice, val_answer, criterion, batch_size)
        #valid_loss, valid_acc = eval_run(model, recipe_context_valid, recipe_question_valid, recipe_choice_valid, recipe_answer_valid, criterion)
        log_data(args.log_path, train_loss, train_acc, valid_loss, valid_acc)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        if valid_acc > max_val_acc:
            max_val_acc = valid_acc
            save_model(model,epoch,valid_acc,args.saved_path)
        

if __name__ == "__main__":
    args = get_args()
    main(args)
