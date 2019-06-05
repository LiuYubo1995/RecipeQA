import torch
import torch.nn as nn
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from data_processing import transport_1_0_2
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
            output, word_hidden_state = self.word_att_net(i)
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
        self.text = Text_Net(word_hidden_size, sent_hidden_size)
        self.question = Question_Net(word_hidden_size, sent_hidden_size)
        self.dim = word_hidden_size*2
        self.batch_size = batch_size
        self.linear_dm = nn.Linear(self.dim,self.dim)
        self.linear_rm = nn.Linear(self.dim,self.dim)
        self.linear_qm = nn.Linear(self.dim,self.dim)
        self.linear_ms = nn.Linear(self.dim, 1) 
        self.linear_rr = nn.Linear(self.dim,self.dim)
        self.linear_rg = nn.Linear(self.dim,self.dim)
        self.linear_qg = nn.Linear(self.dim,self.dim)
    def forward(self, input_context, input_question): 
        context_output, _ = self.text(input_context)
        
        question_output, u = self.question(input_question)
        
        if torch.cuda.is_available():
            r = torch.zeros(self.batch_size, 1, self.dim).cuda() 
        else:
            r = torch.zeros(self.batch_size, 1, self.dim)

        for i in question_output: 
            output1 = self.linear_dm(context_output.permute(1,0,2)) #(seq_leng, batch, dim) -> (batch, seq, dim)
            output2 = self.linear_rm(r) # (batch, 1, dim)

            output3 = self.linear_qm(i.unsqueeze(1)) # (batch, 1, dim
            m = F.tanh(output1 + output2 + output3) 
            s = F.softmax(self.linear_ms(m), dim=1).permute(0,2,1)
            output4 = F.tanh(self.linear_rr(r))
            output5 = torch.matmul(s, context_output.permute(1, 0, 2))
            r = output5 + output4
        # print('r', r.size())
        # print('u', u.size())6
        g = self.linear_rg(r).squeeze(1) + self.linear_qg(u) # g (batch, 1, 512)

        return g 


class Impatient_Reader_Model(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size):
        super(Impatient_Reader_Model, self).__init__()
        self.step_net = WordLevel(word_hidden_size)
        self.text_net = SentLevel(sent_hidden_size, word_hidden_size)
        self.attention = Attention(word_hidden_size, sent_hidden_size, batch_size)
        self.choice = ChoiceNet(word_hidden_size)
    

    def exponent_neg_manhattan_distance(self, x1, x2):
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))
    def cosine_neg_distance(self, x1, x2):
        return torch.mm(x1, x2.transpose(0, 1))/(torch.norm(x1, dim=1)*torch.norm(x2,dim=1))

    def forward(self, input_context,  input_question, input_choice):
        input_context = transport_1_0_2(input_context)
        input_question = transport_1_0_2(input_question)
        input_choice = transport_1_0_2(input_choice)
        output_list = []
        g = self.attention(input_context, input_question)

        output_choice_list = []
        for i in input_choice:  
            output_choice, hidden_output_choice = self.choice(i)
            #hidden_output_choice = self.fc2(hidden_output_choice)
            similarity_scores = torch.sum(torch.mul(g, hidden_output_choice), dim=1)
            #similarity_scores = self.exponent_neg_manhattan_distance(hidden_output_question,hidden_output_choice)
            output_choice_list.append(similarity_scores)
            
        return output_choice_list





