
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


class HierAttNet(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size):
        super(HierAttNet, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.word_att_net = WordAttNet(word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)

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

    def forward(self, input):
        # print('hierlasjdlfalsflaslfaslfjlsdf')
        # print(input)    
        # print(len(input))
        input = self.transport_1_0_2(input) 
        output_list = []
        for i in input:     
            # print('iiiiiiiiiiiiiiiiiiiiiiiiiiii')
            # print(i)    
            output, self.word_hidden_state = self.word_att_net(i)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, hidden_output = self.sent_att_net(output)
        print(hidden_output.size())  
    
        #print('successfulllllllllllllllllllllllllllllll')
        return output
