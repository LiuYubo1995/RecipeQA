import torch
import torch.nn as nn
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

class WordAttNet(nn.Module):
    
    def __init__(self, hidden_size=256):
        super(WordAttNet, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad = False)
        self.lstm = nn.LSTM(1024, 256, num_layers=1, 
                        bidirectional=True, dropout=0.2)
        self.dropout1 = nn.Dropout(0.5)
        #self.dropout = nn.Dropout(d_rate)

    def forward(self, input):
        # print('wordlasdfsljfaskdf;as;fsl')
        # print(input)
        sentences = input
        if torch.cuda.is_available():
            character_ids = batch_to_ids(sentences).cuda()
        else:
            character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)['elmo_representations'][0]
        #embedded = self.dropout(embeddings) 
        embedded = embeddings.permute(1, 0 , 2)
        # print('@@@@@@@@@@@@@@@@@@@@')
        # print(embedded.size()) 
        output, (hidden,_) = self.lstm(embedded)
        hidden_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        return output, hidden_output


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=256, word_hidden_size=256):
        super(SentAttNet, self).__init__()

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
        self.lstm = nn.LSTM(1024, 256, num_layers=1, 
                        bidirectional=True, dropout=0.2)
        #self.dropout = nn.Dropout(d_rate)

    def forward(self, input):
        # print('wordlasdfsljfaskdf;as;fsl')
        # print(input)
        sentences = input
        if torch.cuda.is_available():
            character_ids = batch_to_ids(sentences).cuda()
        else:
            character_ids = batch_to_ids(sentences)
        embeddings = self.elmo(character_ids)['elmo_representations'][0]
        #embedded = self.dropout(embeddings) 
        embedded = embeddings.permute(1, 0, 2) 
        # print('@@@@@@@@@@@@@@@@@@@@')
        # print(embedded.size()) 
        output, (hidden,_) = self.lstm(embedded)
        hidden_output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        return output, hidden_output





class HierAttNet(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size):
        super(HierAttNet, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttNet(word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)
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
        ''' Helper function for the similarity estimate of the LSTMs outputs '''
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))
    def cosine_neg_distance(self, x1, x2):
        return torch.mm(x1, x2.transpose(0, 1))/(torch.norm(x1, dim=1)*torch.norm(x2,dim=1))

    def forward(self, input, input_choice):
        # print('hierlasjdlfalsflaslfaslfjlsdf')
        # print(input)    
        # print(len(input))
        input = self.transport_1_0_2(input) 
        input_choice = self.transport_1_0_2(input_choice)
        output_list = []
        for i in input:     
            # print('iiiiiiiiiiiiiiiiiiiiiiiiiiii')
            # print(i)    
            output, self.word_hidden_state = self.word_att_net(i)
            output_list.append(output) 
        output = torch.cat(output_list, 0)
        output, hidden_output_question = self.sent_att_net(output)
        hidden_output_question = self.fc1(hidden_output_question)
        #print(hidden_output.size()) 
        output_choice_list = []
        for i in input_choice:  
            output_choice, hidden_output_choice = self.choice(i)
            hidden_output_choice = self.fc2(hidden_output_choice)
            similarity_scores = self.exponent_neg_manhattan_distance(hidden_output_question,hidden_output_choice)
            output_choice_list.append(similarity_scores)
            
        #print('successfulllllllllllllllllllllllllllllll')
        return output_choice_list
