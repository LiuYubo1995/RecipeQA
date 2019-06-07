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

def load_cleaned_data(file = 'train_cleaned.json'):
    file = open(file, 'r', encoding='utf8').read()
    recipe = json.loads(file) #json file contains data in str, convert str to dict
    recipe_context = recipe['context']
    recipe_answer = recipe['answer']
    recipe_choice = recipe['choice']
    recipe_question = recipe['question']
    recipe_images = recipe['images']
    return recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer 
def transport_1_0_2(a):
        max_step = 0
        for i in a:
            if max_step < len(i):
                max_step = len(i)
        new = []
        for i in range(max_step):
            step = []
            for j in a:
                if len(j) <= i:
                    step.append(['0','0'])
                else:
                    step.append(j[i])      
            new.append(step)
        return new