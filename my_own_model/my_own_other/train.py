
from impatient_reader_model_visual import Impatient_Reader_Model
from utils import shuffle_data, save_model, log_data, load_cleaned_data, accuracy, split_batch
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
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--word_hidden_size", type=int, default=256)
    parser.add_argument("--sent_hidden_size", type=int, default=256)
    parser.add_argument("--log_path", type=str, default="result/log_data.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--num_attention", type=int, default=2)
    parser.add_argument("--use_lexical", type=str, default=True) 
    parser.add_argument("--use_image", type=str, default=True)  
    args = parser.parse_args() 
    return args 

def train_run(model, train_context, train_question, train_choice, train_answer, train_images, image_path, optimizer, criterion, batch_size):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch_context, batch_question, batch_choice, batch_answer, batch_images in tqdm(zip(train_context, train_question, train_choice, train_answer, train_images)):          
        optimizer.zero_grad() 
        output = model(batch_context, batch_question, batch_choice, batch_images, image_path) 
        output = torch.cat(output, 0).view(-1, len(batch_context))
        output = output.permute(1, 0)  
        loss = criterion(output, batch_answer)
        acc = accuracy(output, batch_answer)
        loss.backward() 
        optimizer.step()
        epoch_loss += loss.item() 
        epoch_acc += acc

    return epoch_loss / len(train_context), epoch_acc / len(train_context) 


def eval_run_batch(model, val_context, val_question, val_choice, val_answer, val_images, image_path, criterion, batch_size):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch_context, batch_question, batch_choice, batch_answer, batch_images in tqdm(zip(val_context, val_question, val_choice, val_answer, val_images)):           
            output = model(batch_context, batch_question, batch_choice, batch_images, image_path)
            output = torch.cat(output, 0).view(-1, len(batch_context))
            output = output.permute(1, 0)  
            loss = criterion(output, batch_answer)
            acc = accuracy(output, batch_answer) 
            epoch_loss += loss.item() 
            epoch_acc += acc

    return epoch_loss / len(val_context), epoch_acc / len(val_context)


def main(args):

    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    word_hidden_size = args.word_hidden_size
    sent_hidden_size = args.sent_hidden_size
    num_attention = args.num_attention
    use_lexical = args.use_lexical
    use_image = args.use_image

    recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = load_cleaned_data('train_cleaned.json')
    recipe_context_val, recipe_images_val, recipe_question_val, recipe_choice_val, recipe_answer_val = load_cleaned_data('val_cleaned.json')

    model = Impatient_Reader_Model(word_hidden_size, sent_hidden_size, batch_size, use_lexical, use_image, num_attention) 
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

        train_context_new,train_question_new,train_choice_new,train_answer_new, train_images_new = shuffle_data(recipe_context,recipe_question,recipe_choice,recipe_answer, recipe_images)
        val_context_new,val_question_new,val_choice_new,val_answer_new, val_images_new = shuffle_data(recipe_context_val,recipe_question_val,recipe_choice_val,recipe_answer_val, recipe_images_val)
        
        # train_context_new = train_context_new[0:5]
        # train_question_new = train_question_new[0:5]
        # train_choice_new = train_choice_new[0:5]
        # train_answer_new = train_answer_new[0:5]
        # train_images_new = train_images_new[0:5]
        # val_context_new = val_context_new[0:5]
        # val_question_new = val_question_new[0:5]
        # val_choice_new = val_choice_new[0:5]
        # val_answer_new = val_answer_new[0:5]
        # val_images_new = val_images_new[0:5] 
        train_context, train_question, train_choice, train_answer, train_images = split_batch(batch_size, train_context_new,train_question_new,train_choice_new,train_answer_new, train_images_new)
        val_context, val_question, val_choice, val_answer, val_images = split_batch(batch_size, val_context_new,val_question_new,val_choice_new,val_answer_new, val_images_new)

        print(len(train_context)) 
        print(len(val_context)) 

        train_feature_path = open('training_features.json', 'r', encoding='utf8').read()
        train_images_feature = json.loads(train_feature_path)

        val_feature_path = open('validation_features.json', 'r', encoding='utf8').read()
        val_images_feature = json.loads(val_feature_path)

        train_loss, train_acc = train_run(model, train_context, train_question, train_choice, train_answer, train_images, train_images_feature, optimizer, criterion, batch_size)
        valid_loss, valid_acc = eval_run_batch(model, val_context, val_question, val_choice, val_answer, val_images, val_images_feature, criterion, batch_size)
        log_data(args.log_path, train_loss, train_acc, valid_loss, valid_acc)

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        # if valid_acc > max_val_acc:
        #     max_val_acc = valid_acc
        #     save_model(model,epoch,valid_acc,args.saved_path)
        
        

if __name__ == "__main__":
    args = get_args()
    main(args)
