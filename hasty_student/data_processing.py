# import pandas as pd
import numpy as np
import json
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize



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
    string = re.sub(r"[^a-zA-Z0-9]"," ",string)
    string = string.lower() 
    string = word_tokenize(string) 
    string = [w for w in string if w not in stopwords.words('english')]
    string = [PorterStemmer().stem(w) for w in string] 
    return string

def process_data(from_file='train_text_cloze.json' , tofile='train_cleaned'):
    from_file = open(from_file, 'r', encoding='utf8').read()
    from_dict = json.loads(from_file)
    from_data = from_dict['data'] 

    recipe_context = []
    recipe_answer = []
    recipe_choice = []
    recipe_question = [] 
    recipe_images = []
    for recipe in from_data: 
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
    recipe = {}
    recipe['context'] = recipe_context
    recipe['answer'] = recipe_answer
    recipe['choice'] = recipe_choice
    recipe['question'] = recipe_question
    recipe['images'] = recipe_images
    with open(tofile, 'w', encoding='utf8') as f:
        json.dump(recipe, f, indent=4, ensure_ascii=False) # convert dict to str and write, indent means change row
    return recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer



#load cleaned data


extract_dataset('train.json', 'train_text_cloze.json', 'textual_cloze')
extract_dataset('val.json', 'val_text_cloze.json', 'textual_cloze')
process_data(from_file='train_text_cloze.json' , tofile='train_cleaned.json')
print('process training data finish')
process_data(from_file='val_text_cloze.json' , tofile='val_cleaned.json')
print('process validation data finish')


    
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