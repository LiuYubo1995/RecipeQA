import torch
import torch.nn as nn
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from utils import transport_1_0_2, transport_1_0_2_image, extract_image_features
import torch.nn.functional as F
class WordLevel(nn.Module): 
    
    def __init__(self, word_hidden_size):
        super(WordLevel, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad = False)
        self.lstm = nn.LSTM(1024, word_hidden_size, num_layers=1, 
                        bidirectional=True, dropout=0.2)

    def forward(self, input):
        sentences = input
        if torch.cuda.is_available():
            character_ids = batch_to_ids(sentences).cuda()
        else:
            character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)['elmo_representations'][0]
        embedded = embeddings.permute(1, 0 , 2)
        output, (hidden,_) = self.lstm(embedded)
        hidden_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        return output, hidden_output 


class SentLevel(nn.Module):
    def __init__(self, sent_hidden_size=256, word_hidden_size=256):
        super(SentLevel, self).__init__()
        self.lstm = nn.LSTM(2 * word_hidden_size, sent_hidden_size, bidirectional=True)


    def forward(self, input):
        output, (hidden, _) = self.lstm(input)
        hidden_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return output, hidden_output 

class ChoiceNet(nn.Module):
    
    def __init__(self, word_hidden_size):
        super(ChoiceNet, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad = False)
        self.lstm = nn.LSTM(1024, word_hidden_size, num_layers=1, 
                        bidirectional=True, dropout=0.2)

    def forward(self, input):
        sentences = input
        if torch.cuda.is_available():
            character_ids = batch_to_ids(sentences).cuda()
        else:
            character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)['elmo_representations'][0]
        embedded = embeddings.permute(1, 0, 2) 
        output, (hidden,_) = self.lstm(embedded)
        hidden_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        return output, hidden_output


class Text_Net(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size):
        super(Text_Net, self).__init__()
        self.step_net = WordLevel(word_hidden_size)
        self.text_net = SentLevel(sent_hidden_size, word_hidden_size)

    def forward(self, input_text): 
        output_list = []
        for i in input_text:  
            output, step_hidden_state = self.step_net(i)
            if torch.cuda.is_available():
                output_list.append(step_hidden_state.cpu().detach().numpy())
            else:
                output_list.append(step_hidden_state.detach().numpy())
        if torch.cuda.is_available():
            output_step = torch.FloatTensor(output_list).cuda()
        else:
            output_step = torch.FloatTensor(output_list)

        output, hidden_output = self.text_net(output_step)

        return output, hidden_output 


class Question_Net(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size):
        super(Question_Net, self).__init__()
        self.word_net = WordLevel(word_hidden_size)
        self.sen_net = SentLevel(sent_hidden_size, word_hidden_size)

    def forward(self, input_question):
        output_list = []
        for i in input_question:       
            output, word_hidden_state = self.word_net(i)
            if torch.cuda.is_available():
                output_list.append(word_hidden_state.cpu().detach().numpy())
            else:
                output_list.append(word_hidden_state.detach().numpy())
        if torch.cuda.is_available():
            output_word = torch.FloatTensor(output_list).cuda()
        else:
            output_word = torch.FloatTensor(output_list)

        output, hidden_output = self.sen_net(output_word)

        return output, hidden_output




class Attention(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size):
        super(Attention, self).__init__() 
        self.dim = word_hidden_size*2
        self.batch_size = batch_size
        self.fc1 = nn.Linear(self.dim, self.dim)
        self.fc2 = nn.Linear(self.dim, self.dim)
        self.linear_dm = nn.Linear(self.dim,self.dim)
        self.linear_ms = nn.Linear(self.dim, 1)
        self.linear_rm = nn.Linear(self.dim,self.dim)
        self.linear_qm = nn.Linear(self.dim,self.dim)
        self.linear_rr = nn.Linear(self.dim,self.dim)
        self.linear_rg = nn.Linear(self.dim,self.dim)
        self.linear_qg = nn.Linear(self.dim,self.dim)
    def forward(self, context_output, context_hidden_state, question_output, question_hidden_state, image_output, image_hidden_state): 
        
        if torch.cuda.is_available(): 
            r = torch.zeros(context_output.size()[1], 1, self.dim).cuda() 
        else:
            r = torch.zeros(context_output.size()[1], 1, self.dim)
        print(context_output.size())
        print(context_hidden_state.size())
        print(question_output.size())
        print(question_hidden_state.size())
        print(image_output.size())
        print(image_hidden_state.size())

        

        context_output = self.fc1(context_output.permute(1,0,2)) + self.fc2(image_output.permute(1,0,2)) 

        for i in question_output: 
            output1 = self.linear_dm(context_output) #(seq_leng, batch, dim) -> (batch, seq, dim)
            output2 = self.linear_rm(r) # (batch, 1, dim)
            output3 = self.linear_qm(i.unsqueeze(1)) # (batch, 1, dim)
            m = torch.tanh(output1 + output2 + output3) 
            s = F.softmax(self.linear_ms(m), dim=1).permute(0,2,1)
            r = torch.matmul(s, context_output) + torch.tanh(self.linear_rr(r))
        g = self.linear_rg(r).squeeze(1) + self.linear_qg(question_hidden_state) # g (batch, 1, 512)

        return g 


class Impatient_Reader_Model(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size):
        super(Impatient_Reader_Model, self).__init__()
        self.text = Text_Net(word_hidden_size, sent_hidden_size)
        self.question = Question_Net(word_hidden_size, sent_hidden_size) 
        self.attention = Attention(word_hidden_size, sent_hidden_size, batch_size)
        self.choice = ChoiceNet(word_hidden_size)
        self.fc3 = nn.Linear(word_hidden_size*8, word_hidden_size*2)
        self.dropout = nn.Dropout(p = 0.2)
        self.fc4 = nn.Linear(word_hidden_size*2, 1) 
        self.images = nn.LSTM(1000, word_hidden_size, bidirectional=True) 

    

    def exponent_neg_manhattan_distance(self, x1, x2):
        return torch.sum(torch.abs(x1 - x2), dim=1)
    def cosine_dot_distance(self, x1, x2):
        return torch.sum(torch.mul(x1, x2), dim=1)
    def Infersent(self, x1, x2):
        return torch.cat((x1, x2, torch.abs(x1 - x2), x1 * x2), 1)

    def forward(self, input_context,  input_question, input_choice, input_images, image_path):
        input_context = transport_1_0_2(input_context)
        input_question = transport_1_0_2(input_question)
        input_choice = transport_1_0_2(input_choice) 
        input_images = transport_1_0_2_image(input_images) 

        input_images = extract_image_features(input_images, image_path)

        context_output, context_hidden_state = self.text(input_context)
        question_output, question_hidden_state = self.question(input_question)
        image_output, (image_hidden_state, _)  = self.images(input_images) 
        image_hidden_state = torch.cat((image_hidden_state[-2,:,:], image_hidden_state[-1,:,:]), dim=1)

        output_list = []
        g = self.attention(context_output, context_hidden_state, question_output, question_hidden_state, image_output, image_hidden_state) 

        output_choice_list = [] 
        for i in input_choice:  
            output_choice, hidden_output_choice = self.choice(i)
            similarity_scores = self.Infersent(g, hidden_output_choice)
            similarity_scores = self.dropout(torch.tanh(self.fc3(similarity_scores)))
            similarity_scores = self.fc4(similarity_scores)
            #similarity_scores = self.cosine_dot_distance(g, hidden_output_choice)
            #similarity_scores = self.exponent_neg_manhattan_distance(hidden_output_question,hidden_output_choice)
            output_choice_list.append(similarity_scores)
            
        return output_choice_list




