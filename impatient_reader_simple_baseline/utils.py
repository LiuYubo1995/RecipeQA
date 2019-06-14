import numpy as np
import torch
import json
import torch.nn.functional as F

def accuracy(preds, y):
    preds = F.softmax(preds, dim=1)
    correct = 0 
    pred = preds.max(1, keepdim=True)[1]
    correct += pred.eq(y.view_as(pred)).sum().item()
    acc = correct/len(y)

    return acc

def shuffle_data(recipe_context,recipe_question,recipe_choice,recipe_answer,recipe_question_true, recipe_question_false):
    #shuffle
    combine = list(zip(recipe_context,recipe_question,recipe_choice,recipe_answer,recipe_question_true, recipe_question_false))
    np.random.shuffle(combine)
    recipe_context_shuffled,recipe_question_shuffled, recipe_choice_shuffled, recipe_answer_shuffled, recipe_question_true_shuffled, recipe_question_false_shuffled = zip(*combine)
    recipe_context_shuffled = list(recipe_context_shuffled)
    recipe_question_shuffled = list(recipe_question_shuffled) 
    recipe_choice_shuffled = list(recipe_choice_shuffled)
    recipe_answer_shuffled = list(recipe_answer_shuffled)
    recipe_question_true_shuffled = list(recipe_question_true_shuffled)
    recipe_question_false_shuffled = list(recipe_question_true_shuffled)
    return recipe_context_shuffled,recipe_question_shuffled, recipe_choice_shuffled, recipe_answer_shuffled,recipe_question_true_shuffled,recipe_question_false_shuffled

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

def load_cleaned_data(file = 'train_cleaned.json'):
    file = open(file, 'r', encoding='utf8').read()
    recipe = json.loads(file) #json file contains data in str, convert str to dict
    recipe_context = recipe['context']
    recipe_answer = recipe['answer']
    recipe_choice = recipe['choice']
    recipe_question = recipe['question']
    recipe_images = recipe['images']
    return recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer 

def split_batch(batch_size, recipe_context_new, recipe_question_new, recipe_choice_new, recipe_answer_new, recipe_question_true_new, recipe_question_false_new):
    train_context = []
    train_question = []
    train_answer = []
    train_choice = []
    train_question_true = []
    train_question_false = []
    for i in range(0, len(recipe_question_new), batch_size):
        train_context.append(recipe_context_new[i : i + batch_size])
        train_question.append(recipe_question_new[i : i + batch_size]) 
        train_question_true.append(recipe_question_true_new[i : i + batch_size])
        train_question_false.append(recipe_question_false_new[i : i + batch_size])  
        train_choice.append(recipe_choice_new[i : i + batch_size])
        actual_scores = recipe_answer_new[i : i + batch_size]

        if torch.cuda.is_available():
            actual_scores = torch.LongTensor(actual_scores).cuda()
        else: 
            actual_scores = torch.LongTensor(actual_scores)
        train_answer.append(actual_scores) 
    return train_context, train_question, train_choice, train_answer, train_question_true, train_question_false

def split_batch_val(batch_size, recipe_context_new, recipe_question_new, recipe_choice_new, recipe_answer_new):
    train_context = []
    train_question = []
    train_answer = []
    train_choice = []
    for i in range(0, len(recipe_question_new), batch_size):
        train_context.append(recipe_context_new[i : i + batch_size])
        train_question.append(recipe_question_new[i : i + batch_size])  
        train_choice.append(recipe_choice_new[i : i + batch_size])
        actual_scores = recipe_answer_new[i : i + batch_size]

        if torch.cuda.is_available():
            actual_scores = torch.LongTensor(actual_scores).cuda()
        else: 
            actual_scores = torch.LongTensor(actual_scores)
        train_answer.append(actual_scores) 
    return train_context, train_question, train_choice, train_answer