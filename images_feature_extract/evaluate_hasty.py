
#! /Users/cuijie/anaconda3/envs/allennlp1/bin/python



import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import data_processing
import extract_images
from hasty_student import recipeDataset, hasty_student

import sys
import json


def Loading_val_data(val_json,task):
    DATA_DIR = '../images/val/images-qa'
    val_file = data_processing.extract_dataset(val_json,'val_visual_cloze.json',task)
    _, val_question_name, val_choice_name, val_answer = data_processing.extract_data(val_file)
    val_answer = np.array(val_answer).reshape(-1,1)

    val_questions,val_choices = extract_images.extract_test_festures(DATA_DIR, val_question_name, val_choice_name)  # 842x4x1000


    val_dataset = recipeDataset(val_questions,val_choices,val_answer)
    if torch.cuda.is_available():
        val_loader = Data.DataLoader(
            dataset=val_dataset,
            batch_size=len(val_answer),
            shuffle=False,
            num_workers=1,
        )
    else:
        val_loader = Data.DataLoader(
            dataset=val_dataset,
            batch_size=len(val_answer),
            shuffle=False,
        )
    return val_loader

def eval_run(model, dataloader):
    model.eval()
    pred_list = []
    for val_question,val_choice,val_answer in tqdm(dataloader):
        val_question = val_question.permute(1,0,2)
        val_choice = val_choice.permute(1,0,2)
        val_answer = val_answer.view(-1).long()
        if torch.cuda.is_available():
            val_answer = val_answer.cuda()
        with torch.no_grad():
            predictions = model(val_question, val_choice)
            predictions = F.softmax(predictions, dim=1)
            predictions = predictions.max(1, keepdim=True)[1].view(-1)
            if torch.cuda.is_available():
                pred_list.extend(predictions.cpu().numpy().tolist())
            else:
                pred_list.extend(predictions.numpy().tolist())
    return pred_list


def save_preds(pred_list, file):
    Task = 'visual_cloze'
    pred_dict = {}
    pred_dict[Task] = pred_list
    with open(file, 'w') as json_file:
        json.dump(pred_dict, json_file)
    print("Done!")




def main():

    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    load_model = 'trained_models/hasty_student_epoch_6_0.600230_acc.pth'
    model = hasty_student()
    model.load_state_dict(torch.load(load_model))
    model.eval()
    print("LOAD MODEL:", load_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    val_loader = Loading_val_data(input_file,'visual_cloze')

    pred_list = eval_run(model, val_loader)
    print("len(pred_list): ",len(pred_list))
    save_preds(pred_list, output_file)



if __name__ == "__main__":
    main()
