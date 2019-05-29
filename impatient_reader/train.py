# from src.model import Model
from hierarchical_att_model import HierAttNet
#from src.model import Model
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import json
import re



def extract_dataset(path, store_path, task):
    file = open(path, 'r', encoding='utf8').read()
    task_dict = json.loads(file)#json file contains data in str, convert str to dict
    data = task_dict['data']#train file format {'data':[{},{},{}]} 
    text_cloze = [] 
    for recipe in data:
        if recipe['task'] == task:
            text_cloze.append(recipe) 
    new_data={}
    new_data['data'] = text_cloze 
    with open(store_path, 'w', encoding='utf8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False) # convert dict to str and write, indent means change row


extract_dataset('train.json', 'train_text_cloze.json', 'textual_cloze')
extract_dataset('val.json', 'val_text_cloze.json', 'textual_cloze')

text_cloze_train = json.loads(open('train_text_cloze.json', 'r', encoding='utf8').read())
print(len(text_cloze_train['data']))




def clean_text(text):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
'·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
'“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
'▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
'∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    text = str(text)
    for punct in puncts:
        text = text.replace(punct, '')
    return text

def clean_numbers(text):
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    return text

mispell_dict = {"aren't" : "are not", 
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"I'd" : "I would",
"I'd" : "I had",
"I'll" : "I will",
"I'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying",
"Aren't" : "are not",
"Can't" : "cannot",
"Couldn't" : "could not",
"Didn't" : "did not",
"Doesn't" : "does not",
"Don't" : "do not",
"Hadn't" : "had not",
"Hasn't" : "has not",
"Haven't" : "have not",
"He'd" : "he would",
"He'll" : "he will",
"He's" : "he is",
"I'd" : "I would",
"I'd" : "I had",
"I'll" : "I will",
"I'm" : "I am",
"I'd" : "I would",
"I'd" : "I had",
"I'll" : "I will",
"I'm" : "I am",
"Isn't" : "is not",
"It's" : "it is",
"It'll":"it will",
"I've" : "I have",
"Let's" : "let us",
"Mightn't" : "might not",
"Mustn't" : "must not",
"Shan't" : "shall not",
"She'd" : "she would",
"She'll" : "she will",
"She's" : "she is",
"Shouldn't" : "should not",
"That's" : "that is",
"There's" : "there is",
"They'd" : "they would",
"They'll" : "they will",
"They're" : "they are",
"They've" : "they have",
"We'd" : "we would",
"We're" : "we are",
"Weren't" : "were not",
"We've" : "we have",
"What'll" : "what will",
"What're" : "what are",
"What's" : "what is",
"What've" : "what have",
"Where's" : "where is",
"Who'd" : "who would",
"Who'll" : "who will",
"Who're" : "who are",
"Who's" : "who is",
"Who've" : "who have",
"Won't" : "will not",
"Wouldn't" : "would not",
"You'd" : "you would",
"You'll" : "you will",
"You're" : "you are",
"You've" : "you have",
"Wasn't": "was not",
"We'll":" will",
"Didn't": "did not",
"'s":"is"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def process(string):
    
    string = replace_typical_misspell(string)
    string = clean_text(string)
    string = string.lower()
    string = string.split()
    return string


train_file = open('train_text_cloze.json', 'r', encoding='utf8').read()
train_dict = json.loads(train_file)
train_data = train_dict['data'] 


recipe_context = []
recipe_answer = []
recipe_choice = []
recipe_question = [] 
recipe_images = []
for recipe in train_data: 
    new_recipe = [] 
    new_question = []
    new_choice = []
    new_images = []
    for step in recipe['context']:
        step_recipe = process(step['body'])
        new_images.append(step['images'])   
        new_recipe.append(step_recipe)
    recipe_context.append(new_recipe)
    recipe_images.append(new_images)
    for step in recipe['question']:
        new_question.append(process(step))
    recipe_question.append(new_question)
    for step in recipe['choice_list']: 
        new_choice.append(process(step))
    recipe_choice.append(new_choice)
    recipe_answer.append(recipe['answer'])  




def main():
    def train_run(model, X_train, y_train, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        a = 0
        for batch_phrase, batch_label in tqdm(zip(X_train, y_train)):
            
            
            optimizer.zero_grad() 
            output = model(batch_phrase).squeeze(1)
            # print('!!!!!')
            # print(output.shape)
            # print('@@@@@@@@@@')
            # print(batch_label.shape)
            loss = criterion(output, batch_label)
            acc = binary_accuracy(output, batch_label)
            loss.backward() 
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc
        
        return epoch_loss / len(X_train), epoch_acc / len(X_train) 
    batch_size = 3
    LR = 0.001
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 128
    OUTPUT_DIM = 5
    NUM_LAYERS = 3
    DROPOUT = 0.2
    N_EPOCHS = 20
    PATH = './weight/weight_w_attention.pth'
    word_hidden_size = 256
    sent_hidden_size = 256

    model = HierAttNet(word_hidden_size, sent_hidden_size)

    optimizer = optim.Adam(model.parameters(), lr = LR, weight_decay=0.0001)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    # for a gpu environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in tqdm(range(N_EPOCHS)):
        print(epoch)
        b_train = []
        b_train_answer = []
        for i in tqdm(range(0, len(recipe_context), batch_size)):
            a = recipe_context[i : i + batch_size]
            b_train.append(a)
            actual_scores = recipe_answer[i : i + batch_size]
            if torch.cuda.is_available():
                actual_scores = torch.LongTensor(actual_scores).cuda()
            else:
                actual_scores = torch.LongTensor(actual_scores)
            b_train_answer.append(actual_scores)  

    
        train_loss, train_acc = train_run(model, b_train, b_train_answer, optimizer, criterion)
    

    # c = list(zip(X_train,y_train))
    # d = list(zip(X_valid,y_valid))
    # e = list(zip(X_test,y_test))
    # for epoch in tqdm(range(N_EPOCHS)):
    #     print(epoch)
    #     np.random.shuffle(c)
    #     X_train,y_train = zip(*c)
    #     X_train = list(X_train)
    #     y_train = list(y_train)
    #     b_train = [] 
    #     b_train_y = []
    #     for i in tqdm(range(0,len(X_train), batch_size)):   
    #         a = X_train[i : i + batch_size] # Batch Size x Sequence Length     
    #         b_train.append(a)
    #         actual_scores = y_train[i : i + batch_size]
    #         if torch.cuda.is_available():
    #             actual_scores = torch.LongTensor(actual_scores).cuda()
    #         else:
    #             actual_scores = torch.LongTensor(actual_scores)
    #         b_train_y.append(actual_scores)
        

    #     np.random.shuffle(d)
    #     X_valid,y_valid = zip(*d)
    #     X_train = list(X_valid)
    #     y_train = list(y_valid)
    #     b_valid = [] 
    #     b_valid_y = []
    #     for i in tqdm(range(0,len(X_valid), batch_size)):   
    #         a = X_valid[i : i + batch_size] # Batch Size x Sequence Length     
    #         b_valid.append(a)
    #         actual_scores = y_valid[i : i + batch_size]
    #         if torch.cuda.is_available():
    #             actual_scores = torch.LongTensor(actual_scores).cuda()
    #         else:
    #             actual_scores = torch.LongTensor(actual_scores)
    #         b_valid_y.append(actual_scores)

    #     np.random.shuffle(e)
    #     X_test,y_test = zip(*e)
    #     X_test = list(X_test)
    #     y_test = list(y_test)
    #     b_test = [] 
    #     b_test_y = []
    #     for i in tqdm(range(0,len(X_test), batch_size)):   
    #         a = X_test[i : i + batch_size] # Batch Size x Sequence Length     
    #         b_test.append(a)
    #         actual_scores = y_test[i : i + batch_size]
    #         if torch.cuda.is_available():
    #             actual_scores = torch.LongTensor(actual_scores).cuda()
    #         else:  
    #             actual_scores = torch.LongTensor(actual_scores)
    #         b_test_y.append(actual_scores)
        
    #     train_loss, train_acc = train_run(model, b_train, b_train_y, optimizer, criterion)
    #     valid_loss, valid_acc, valid_f1 = eval_run(model, b_valid, b_valid_y, criterion)
    #     test_loss, test_acc, test_f1 = eval_run(model, b_test, b_test_y, criterion)
    #     print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |Val. F1: {valid_f1*100:.3f}%')
    #     print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test.F1: {test_f1*100:.3f}%')
    

    # torch.save(model.state_dict(), PATH)

# def train_run(model, X_train, y_train, optimizer, criterion):
#     epoch_loss = 0
#     epoch_acc = 0
#     model.train()

#     a = 0
#     for batch_phrase, batch_label in tqdm(zip(X_train, y_train)):
        
        
#         optimizer.zero_grad() 
#         output = model(batch_phrase).squeeze(1)
#         # print('!!!!!')
#         # print(output.shape)
#         # print('@@@@@@@@@@')
#         # print(batch_label.shape)
#         loss = criterion(output, batch_label)
#         acc = binary_accuracy(output, batch_label)
#         loss.backward() 
#         optimizer.step()
#         epoch_loss += loss.item()
#         epoch_acc += acc
    
#     return epoch_loss / len(X_train), epoch_acc / len(X_train) 


# def eval_run(model, X_test, y_test, criterion):
#     epoch_loss = 0
#     epoch_acc = 0
#     epoch_f1 = 0
#     model.eval()

#     with torch.no_grad():
#         for batch_phrase, batch_label in tqdm(zip(X_test, y_test)):
#             predictions = model(batch_phrase).squeeze(1)
#             loss = criterion(predictions, batch_label)
#             acc = binary_accuracy(predictions, batch_label)
#             f1 = f1_score(predictions, batch_label)
#             epoch_loss += loss.item()
#             epoch_acc += acc
#             epoch_f1 += f1
    
#     return epoch_loss / len(X_test), epoch_acc / len(X_test), epoch_f1 / len(X_test)

# def f1_score(preds, y):
#     from sklearn.metrics import f1_score
#     pred = F.softmax(preds, dim=1)
#     pred = pred.max(1, keepdim=True)[1]
#     return f1_score(y.data.cpu().numpy(), pred.data.cpu().numpy(), average='weighted')
# def binary_accuracy(preds, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """

#     #round predictions to the closest integer
#     pred = F.softmax(preds)
#     correct = 0
#     pred = pred.max(1, keepdim=True)[1]
#     correct += pred.eq(y.view_as(pred)).sum().item()
#      #convert into float for division 
#     acc = correct/len(y)

#     return acc

if __name__ == "__main__":
    main()
