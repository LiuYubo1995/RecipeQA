
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
import argparse 
from data_processing import load_cleaned_data
from data_processing import recipeDataset
import torch.utils.data as Data

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

def save_model(model, epoch,accuracy, saved_path):
    torch.save(model.state_dict(),
               '%s/hasty_student_epoch_%d_%f_acc.pth' % (saved_path, epoch,accuracy))
    print('Save model with accuracy:',accuracy)

def log_data(log_path,train_loss,train_accuracy,val_loss,val_accuracy):
    file = open(log_path,'a')
    if torch.cuda.is_available():
        data = str(train_loss) +' '+ str(f'{train_accuracy:.2f}') \
            +' '+ str(val_loss)+ ' ' + str(f'{val_accuracy:.2f}')
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

    train_dataset = recipeDataset('train_cleaned.json')
    val_dataset = recipeDataset('val_cleaned.json')

    loader_train = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        )
    loader_val = Data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,

        )
    
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


        epoch_loss_train = 0
        epoch_acc_train = 0
        epoch_loss_val = 0
        epoch_acc_val = 0

        model.train()
        for image, context, question, choice, answer in tqdm(loader_train):
    
            optimizer.zero_grad() 
            output = model(question, choice)
            output = torch.cat(output, 0).view(-1, len(answer)) 
            output = output.permute(1, 0)
            if torch.cuda.is_available():
                answer = answer.cuda()
            loss = criterion(output, answer)
            acc = accuracy(output, answer)
            loss.backward() 
            optimizer.step()
            epoch_loss_train += loss.item() 
            epoch_acc_train += acc

        model.eval()
        for image, context, question, choice, answer in tqdm(loader_val):
            
 
            with torch.no_grad(): 
                predictions = model(question, choice)
                predictions = torch.cat(predictions, 0).view(-1, len(answer))
                predictions = predictions.permute(1, 0)
                if torch.cuda.is_available():
                    answer = answer.cuda()
                loss = criterion(predictions, answer)
                acc = accuracy(predictions, answer)
                epoch_loss_val += loss.item() 
                epoch_acc_val += acc



        train_loss = epoch_loss_train / len(loader_train)
        train_acc = epoch_acc_train / len(loader_train)
        valid_loss = epoch_loss_val / len(loader_val)
        valid_acc = epoch_acc_val / len(loader_val)
        
        log_data(args.log_path, train_loss, train_acc, valid_loss, valid_acc)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        if valid_acc > max_val_acc:
            max_val_acc = valid_acc
            save_model(model,epoch,valid_acc,args.saved_path)
        

if __name__ == "__main__":
    args = get_args()
    main(args)
