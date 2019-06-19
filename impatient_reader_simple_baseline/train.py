
from impatient_reader_model import Impatient_Reader_Model
from utils import shuffle_data, save_model, log_data, load_cleaned_data, accuracy, split_batch,split_batch_val
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



def get_args():
    parser = argparse.ArgumentParser("Hasty_student_recipeQA")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--word_hidden_size", type=int, default=100)
    parser.add_argument("--sent_hidden_size", type=int, default=100)
    parser.add_argument("--log_path", type=str, default="result/log_data.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--Doc2vec_model", type=str, default="/Users/LYB/Desktop/coursework/Msc-project/recipe_baseline/impatient_reader_simple_baseline/train_doc2vec/my_doc2vec_model")
    parser.add_argument("--seed", type=str, default=4)
    args = parser.parse_args() 
    return args 

def train_run(model, train_context, train_question, train_choice, train_answer, optimizer, criterion, batch_size,train_question_true, train_question_false):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch_context, batch_question, batch_choice, batch_answer, batch_question_true, batch_question_false in tqdm(zip(train_context, train_question, train_choice, train_answer,train_question_true, train_question_false)):          
        optimizer.zero_grad() 
        output, answer_true, answer_false, g = model(batch_context, batch_question, batch_choice, batch_question_true, batch_question_false)
        output = torch.cat(output, 0).view(-1, len(batch_context))
        output = output.permute(1, 0)
        loss = criterion(g, answer_true, answer_false)
        acc = accuracy(output, batch_answer) 
        loss.backward() 
        optimizer.step()
        #print(loss.item())
        #print(acc)
        epoch_loss += loss.item() 
        epoch_acc += acc 

    return epoch_loss / len(train_context), epoch_acc / len(train_context) 


def eval_run_batch(model, val_context, val_question, val_choice, val_answer, criterion, batch_size, a, b):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch_context, batch_question, batch_choice, batch_answer, batch_a, batch_b in tqdm(zip(val_context, val_question, val_choice, val_answer,a,b)):           
            output, _, _, _ = model(batch_context, batch_question, batch_choice, batch_a, batch_b)
            output = torch.cat(output, 0).view(-1, len(batch_context))
            output = output.permute(1, 0)  
            # loss = criterion(output, batch_answer)
            acc = accuracy(output, batch_answer)  
            # epoch_loss += loss.item() 
            epoch_acc += acc

    return epoch_acc / len(val_context)




def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)#为CPU设置随机种子 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)#为当前GPU设置随机种子 
        torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子

    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    word_hidden_size = args.word_hidden_size
    sent_hidden_size = args.sent_hidden_size
    Doc2vec_model = args.Doc2vec_model


    recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = load_cleaned_data('train_cleaned.json')
    val_context_new, recipe_images_val, val_question_new, val_choice_new, val_answer_new = load_cleaned_data('val_cleaned.json')
    

    true_answer = []
    for i in range(len(recipe_answer)):
        true_answer.append(recipe_choice[i][recipe_answer[i]]) 
    false_answer = []
    for i in range(len(recipe_answer)):
        false_answer.append(recipe_choice[i][:recipe_answer[i]]+ recipe_choice[i][(recipe_answer[i]+1):]) 
    random_choice_false_answer = []
    for i in range(len(false_answer)):
        random_choice_false_answer.append(np.random.choice(false_answer[i]))
    recipe_question_true = []
    for i in range(len(recipe_question)):
        temp = []
        for j in recipe_question[i]:
            if j == '@placeholder':
                temp.append(true_answer[i])
            else:
                temp.append(j)
        recipe_question_true.append(temp)
    recipe_question_false = []
    for i in range(len(recipe_question)):
        temp = []
        for j in recipe_question[i]:
            if j == '@placeholder':
                temp.append(random_choice_false_answer[i])
            else:
                temp.append(j)
        recipe_question_false.append(temp)

    
    model = Impatient_Reader_Model(word_hidden_size, sent_hidden_size, batch_size, Doc2vec_model)
    if args.load_model: 
        model.load_state_dict(torch.load(args.load_model))
        model.eval()
        print("LOAD MODEL:",args.load_model)

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=0.001)
    
    criterion = nn.TripletMarginLoss(margin=1.5, p=2.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = criterion.to(device)
    max_val_acc = 0.0
    for epoch in tqdm(range(num_epochs)):

        print(epoch)

        train_context_new,train_question_new,train_choice_new,train_answer_new, train_question_true_new, train_question_false_new = shuffle_data(recipe_context,recipe_question,recipe_choice,recipe_answer, recipe_question_true, recipe_question_false)
        # train_context_new = train_context_new[0:15]
        # train_question_new = train_question_new[0:15]
        # train_choice_new = train_choice_new[0:15]
        # train_answer_new = train_answer_new[0:15]
        # train_question_true_new = train_question_true_new[:15]
        # train_question_false_new =  train_question_false_new[:15]
        train_context, train_question, train_choice, train_answer, train_question_true, train_question_false = split_batch(batch_size, train_context_new,train_question_new,train_choice_new,train_answer_new,train_question_true_new, train_question_false_new)
        val_context, val_question, val_choice, val_answer, a,b = split_batch(batch_size, val_context_new, val_question_new,val_choice_new,val_answer_new,val_question_new,val_question_new) 

        # print(len(train_context)) 
        # print(len(val_context))

        train_loss, train_acc = train_run(model, train_context, train_question, train_choice, train_answer, optimizer, criterion, batch_size, train_question_true, train_question_false)
        valid_acc = eval_run_batch(model, val_context, val_question, val_choice, val_answer, criterion, batch_size, a, b)
        valid_loss = 0
        log_data(args.log_path, train_loss, train_acc, valid_loss, valid_acc)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        if valid_acc > max_val_acc:
            max_val_acc = valid_acc
            save_model(model,epoch,valid_acc,args.saved_path)
        

if __name__ == "__main__":
    args = get_args()
    main(args)
