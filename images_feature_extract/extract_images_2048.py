'''
Function of this file:
    Extract the image features by inception-resnet-v2 pretrained model
    or resnet50
'''
import pretrainedmodels
import torch
from pretrainedmodels import utils
import numpy as np
from tqdm import tqdm
import os
import json
from data_processing import extract_data
from resnet50_2048 import ResNet50_2048


TRAINDIR = '../images/train/images-qa'
VALDIR = '../images/val/images-qa'
def extract_all_images():

    model_name = 'resnet50_2048'

    # model = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
    model = ResNet50_2048()
    # print(model.settings)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_dict = {}
    val_dict = {}

    print('Start processing training data...')
    train_files = os.listdir(TRAINDIR)
    for file in tqdm(train_files):
        load_img = utils.LoadImage()

        # transformations depending on the model
        # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
        tf_img = utils.TransformImage(model.settings)
        path_img = TRAINDIR + '/' + file
        input_img = load_img(path_img)
        image = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
        # images = torch.cat((images, image), 0)
        # print("images_size", images.size())
        input_tensor = image.unsqueeze(0)     # 1x3x299x299
        input = torch.autograd.Variable(input_tensor,
                                    requires_grad=False)

        output_features = model(input)# 1x1x1000
        print(output_features.size())
        if torch.cuda.is_available():
            output_features = output_features.cpu().detach().numpy().tolist()
        else:
            output_features = output_features.detach().numpy().tolist()

        if file in train_dict.keys():
            raise KeyError('This file has been added twice in training set!')
        train_dict[file] = output_features
    print(len(train_dict.keys()))
    print('Start logging training data.')
    output_file = 'training_features_'+model_name+'.json'
    with open(output_file, 'w') as json_file:
        json.dump(train_dict, json_file)
    print("Done!")



    print('Start processing validation data.')
    val_files = os.listdir(VALDIR)
    for file in tqdm(val_files):
        load_img = utils.LoadImage()

        # transformations depending on the model
        # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
        tf_img = utils.TransformImage(model)
        path_img = VALDIR + '/' + file
        input_img = load_img(path_img)
        image = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
        # images = torch.cat((images, image), 0)
        # print("images_size", images.size())
        input_tensor = image.unsqueeze(0)     # 1x3x299x299
        input = torch.autograd.Variable(input_tensor,
                                    requires_grad=False)

        output_features = model(input)  # 1x1x1000
        if torch.cuda.is_available():
            output_features = output_features.cpu().detach().numpy().tolist()
        else:
            output_features = output_features.detach().numpy().tolist()
        if file in val_dict.keys():
            raise KeyError('This file has been added twice in validation set!')
        val_dict[file] = output_features
    print(len(val_dict.keys()))
    print('Start logging validation data.')
    output_file = 'validation_features_'+ model_name + '.json'
    with open(output_file, 'w') as json_file:
        json.dump(val_dict, json_file)
    print("Done!")


def extract_vc_features(model_name):
    place_holder = np.zeros((1, 1, 1000))
    train_img_file = 'training_features_'+model_name+'.json'
    val_img_file = 'validation_features_'+ model_name + '.json'

    print('Start loading training features.')
    file = open(train_img_file, 'r', encoding='utf8').read()
    train_dict_img = json.loads(file)

    train_file = '../train_visual_cloze.json'
    print('Start loading training data.')
    _, recipe_question, recipe_choice, recipe_answer = extract_data(train_file)
    # len_recipe = len(recipe_answer)
    image_array = np.array([])
    images_file = np.concatenate((np.array(recipe_question),np.array(recipe_choice)),1).reshape(-1)  # 7144x4 -> 7144 x 8 -> 57152,
    print('Start processing training data.')
    for image in tqdm(images_file):

        if image == "@placeholder":
            image_array = np.append(image_array,place_holder)
        else:
            image_array = np.append(image_array,train_dict_img[image])

    train_array = image_array.reshape(-1,8000)   #    57152, -> 7144 x 8000
    print(train_array.shape)
    output_file = 'train_vc_features_'+ model_name+'.npz'
    np.savez_compressed(output_file, train_array = train_array)
    print("Done!")


    # Validation data

    print('Start processing validation features.')

    file = open(val_img_file, 'r', encoding='utf8').read()
    val_dict_img = json.loads(file)
    print('Start logging validation data.')
    val_file = '../val_visual_cloze.json'
    _, recipe_question, recipe_choice, recipe_answer = extract_data(val_file)
    # len_recipe = len(recipe_answer)  # 842

    image_array = np.array([])
    images_file = np.concatenate((np.array(recipe_question),np.array(recipe_choice)),1).reshape(-1)  # 842x4 -> 842 x 8 -> 6736,
    print('Start processing validation data.')
    for image in tqdm(images_file):

        if image == "@placeholder":
            image_array = np.append(image_array,place_holder)
        else:
            image_array = np.append(image_array,val_dict_img[image])
    print(image_array.shape)
    val_array = image_array.reshape(-1,8000)   #    6736, -> 842 x 8000
    print(val_array.shape)
    output_file = 'val_vc_features_'+model_name+'.npz'
    np.savez_compressed(output_file, val_array = val_array)
    print("Done!")





















if __name__ == "__main__":

    # extract_vc_features()
    # args = get_args()
    # main(args)
    extract_all_images()
    # extract_vc_features('resnet50_2048')





