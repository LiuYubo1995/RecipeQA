import torch
import torch.nn as nn
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch.nn.functional as F
from gensim.test.utils import get_tmpfile
import numpy as np
import gensim

class contextNet(nn.Module):
    def __init__(self, word_hidden_size, Doc2vec_model):
        super(contextNet, self).__init__()
        self.lstm = nn.LSTM(100, 100, num_layers = 3,
                            bidirectional = False, dropout = 0.2)
        self.Doc2vec_model = Doc2vec_model
    def forward(self, input):
        fname = get_tmpfile(self.Doc2vec_model)
        model = gensim.models.doc2vec.Doc2Vec.load(fname)
        input_new = []
        for i in range(len(input)):
            temp = []
            for j in range(len(input[i])):
                temp.append(model.infer_vector(input[i][j]))
            input_new.append(torch.FloatTensor(temp))
        input_new = torch.nn.utils.rnn.pad_sequence(input_new, batch_first=False, padding_value=0) 
        output, (hidden,_) = self.lstm(input_new) 

        return output 

class questionNet(nn.Module): 
    def __init__(self, word_hidden_size, Doc2vec_model):
        super(questionNet, self).__init__()
        self.Doc2vec_model = Doc2vec_model
        self.lstm = nn.LSTM(100, 100, num_layers = 3,
                            bidirectional = False, dropout = 0.2)
    def forward(self, input):
        fname = get_tmpfile(self.Doc2vec_model) 
        model = gensim.models.doc2vec.Doc2Vec.load(fname)

        input_new = []
        for i in range(len(input)):
            temp = []
            for j in range(len(input[i])):
                temp.append(model.infer_vector(input[i][j].lower().split())) 
            input_new.append(temp) 
        input_new = torch.FloatTensor(input_new).permute(1, 0, 2)
        output, (hidden, _) = self.lstm(input_new)
        return output, hidden[-1] 


class Attention(nn.Module): 
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, Doc2vec_model):
        super(Attention, self).__init__() 
        self.text = contextNet(word_hidden_size, Doc2vec_model)
        self.question = questionNet(word_hidden_size, Doc2vec_model)
        self.dim = word_hidden_size
        self.batch_size = batch_size 
        self.linear_dm = nn.Linear(self.dim,self.dim)
        self.linear_ms = nn.Linear(self.dim, 1)
        self.linear_rm = nn.Linear(self.dim,self.dim)
        self.linear_qm = nn.Linear(self.dim,self.dim)
        self.linear_rr = nn.Linear(self.dim,self.dim)
        self.linear_rg = nn.Linear(self.dim,self.dim)
        self.linear_qg = nn.Linear(self.dim,self.dim)
    def forward(self, input_context, input_question): 
        context_output = self.text(input_context)
        
        question_output, u = self.question(input_question)
        
        if torch.cuda.is_available(): 
            r = torch.zeros(context_output.size()[1], 1, self.dim).cuda() 
        else:
            r = torch.zeros(context_output.size()[1], 1, self.dim)

        for i in question_output: 
            output1 = self.linear_dm(context_output.permute(1,0,2)) #(seq_leng, batch, dim) -> (batch, seq, dim)
            output2 = self.linear_rm(r) # (batch, 1, dim)
            output3 = self.linear_qm(i.unsqueeze(1)) # (batch, 1, dim)
            m = torch.tanh(output1 + output2 + output3) 
            s = F.softmax(self.linear_ms(m), dim=1).permute(0,2,1)
            r = torch.matmul(s, context_output.permute(1, 0, 2)) + torch.tanh(self.linear_rr(r))

        g = self.linear_rg(r).squeeze(1) + self.linear_qg(u) # g (batch, 1, 512)
        return g 


class Impatient_Reader_Model(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, Doc2vec_model):
        super(Impatient_Reader_Model, self).__init__() 
        self.attention = Attention(word_hidden_size, sent_hidden_size, batch_size, Doc2vec_model)
        self.answer_true = questionNet(word_hidden_size, Doc2vec_model)
        self.answer_false = questionNet(word_hidden_size, Doc2vec_model)
        self.Doc2vec_model = Doc2vec_model 
        self.linear = nn.Linear(100, 64)

    def forward(self, input_context,  input_question, input_choice, input_answer_true, input_answer_false):
        _, answer_true = self.answer_true(input_answer_true)
        _, answer_false = self.answer_false(input_answer_false)
        fname = get_tmpfile(self.Doc2vec_model)
        model = gensim.models.doc2vec.Doc2Vec.load(fname)
        output_list = []
        g = self.attention(input_context, input_question)

        choice = []
        for i in range(len(input_choice)):
            temp = []
            for j in range(len(input_choice[i])): 
                temp.append(model.infer_vector(input_choice[i][j].lower().split())) 
            choice.append(temp) 
        choice = torch.FloatTensor(choice).permute(1, 0, 2) 

        output_choice_list = [] 
        for i in choice: 
            # i = self.linear(i)
            similarity_scores = F.cosine_similarity(g, i) 
            output_choice_list.append(similarity_scores)

        return output_choice_list, answer_true, answer_false, g





