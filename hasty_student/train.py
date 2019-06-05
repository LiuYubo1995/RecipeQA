
from hasty_student_model import HierNet
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import json
from data_processing import pre_process_data
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
def train_run(model, X_train, y_train, optimizer, criterion, X_choice, batch_size,recipe_question_valid, recipe_answer_valid,recipe_choice_valid):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    a = 0
    # max_acc = 0.0
    for batch_phrase, batch_label, batch_choice in tqdm(zip(X_train, y_train, X_choice)):          
        optimizer.zero_grad() 
        output = model(batch_phrase, batch_choice)
        output = torch.cat(output, 0).view(-1, len(batch_phrase))
        output = output.permute(1, 0)
        loss = criterion(output, batch_label)
        acc = accuracy(output, batch_label)
        loss.backward() 
        optimizer.step()
        epoch_loss += loss.item() 
        epoch_acc += acc

    return epoch_loss / len(X_train), epoch_acc / len(X_train) 


def eval_run(model, X_val, y_val, criterion, X_choice):
    if torch.cuda.is_available():
        y_val = torch.LongTensor(y_val).cuda()
    else:
        y_val = torch.LongTensor(y_val)
    epoch_loss = 0
    epoch_acc = 0
    model.eval() 
    batch_size = len(X_val) 

    with torch.no_grad():
        predictions = model(X_val, X_choice)
        predictions = torch.cat(predictions, 0).view(-1, len(X_val))
        predictions = predictions.permute(1, 0)
        loss = criterion(predictions, y_val)
        acc = accuracy(predictions, y_val)
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
            +' '+ str(val_loss.cpu().numpy())+ ' ' + str(f'{val_accuracy:.2f}')
    else:
        data = str(train_loss) + ' '+ str(f'{train_accuracy:.2f}') \
                +' '+str(val_loss.numpy())+' '+str(f'{val_accuracy:.2f}')
    file.write(data)
    file.write('\n')
    file.close() 
        



def main(args):

    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    word_hidden_size = args.word_hidden_size
    sent_hidden_size = args.sent_hidden_size


    recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = pre_process_data('train_text_cloze.json')
    recipe_context_valid, recipe_images_valid, recipe_question_valid, recipe_choice_valid, recipe_answer_valid = pre_process_data('val_text_cloze.json')

    model = HierNet(word_hidden_size, sent_hidden_size)
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
        b_train = []
        b_train_answer = []
        b_train_choice = []
        for i in tqdm(range(0, len(recipe_question_new), batch_size)):
            a = recipe_question_new[i : i + batch_size]
            b_train.append(a)  
            c = recipe_choice_new[i : i + batch_size]
            b_train_choice.append(c)
            actual_scores = recipe_answer_new[i : i + batch_size]
            if torch.cuda.is_available():
                actual_scores = torch.LongTensor(actual_scores).cuda()
            else:
                actual_scores = torch.LongTensor(actual_scores)
            b_train_answer.append(actual_scores) 

        train_loss, train_acc = train_run(model, b_train, b_train_answer, optimizer, criterion, b_train_choice, batch_size,recipe_question_valid, recipe_answer_valid, recipe_choice_valid)
        valid_loss, valid_acc = eval_run(model, recipe_question_valid, recipe_answer_valid, criterion, recipe_choice_valid)
        log_data(args.log_path, train_loss, train_acc, valid_loss, valid_acc)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        if valid_acc > max_val_acc:
            max_val_acc = valid_acc
            save_model(model,epoch,valid_acc,args.saved_path)
        

if __name__ == "__main__":
    args = get_args()
    main(args)
