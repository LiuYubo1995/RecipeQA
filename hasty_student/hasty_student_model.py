import torch
import torch.nn as nn
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

class WordLevel(nn.Module):
    
    def __init__(self, hidden_size=256):
        super(WordLevel, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad = False)
        self.lstm = nn.LSTM(1024, hidden_size, num_layers=1, 
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
    
    def __init__(self, hidden_size=256):
        super(ChoiceNet, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad = False)
        self.lstm = nn.LSTM(1024, hidden_size, num_layers=1, 
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





class HierNet(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size):
        super(HierNet, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordLevel(word_hidden_size)
        self.sent_att_net = SentLevel(sent_hidden_size, word_hidden_size)
        self.choice = ChoiceNet(word_hidden_size)
        self.fc1 = nn.Linear(512, 50, bias=True)
        self.fc2 = nn.Linear(512, 50, bias = True)

    def transport_1_0_2(self, a):
        max_step = 0
        for i in a:
            if max_step < len(i):
                max_step = len(i)
        new = []
        for i in range(max_step):
            step = []
            for j in a:
                if len(j) <= i:
                    step.append([])
                else:
                    step.append(j[i])      
            new.append(step)
        return new

    def exponent_neg_manhattan_distance(self, x1, x2):
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))
    def cosine_neg_distance(self, x1, x2):
        return torch.mm(x1, x2.transpose(0, 1))/(torch.norm(x1, dim=1)*torch.norm(x2,dim=1))

    def forward(self, input_question, input_choice):
        input_question = self.transport_1_0_2(input_question) 
        input_choice = self.transport_1_0_2(input_choice)
        output_list = []
        for i in input_question:       
            output, word_hidden_state = self.word_att_net(i)
            output_list.append(step_hidden_state.detach().numpy()) 
        output_word = torch.FloatTensor(output_list)
        output, hidden_output_question = self.sent_att_net(output_word)
        #hidden_output_question = self.fc1(hidden_output_question)
        output_choice_list = []
        for i in input_choice:  
            output_choice, hidden_output_choice = self.choice(i)
            #hidden_output_choice = self.fc2(hidden_output_choice)
            similarity_scores = torch.sum(torch.mul(hidden_output_question, hidden_output_choice), dim=1)
            #similarity_scores = self.exponent_neg_manhattan_distance(hidden_output_question,hidden_output_choice)
            output_choice_list.append(similarity_scores)
            
        return output_choice_list




