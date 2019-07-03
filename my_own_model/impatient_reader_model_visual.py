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


class linear_attention(nn.Module):
    def __init__(self, question_dim, vector_dim, attention_dim):
        super(linear_attention, self).__init__()
        self.fc1 = nn.Linear(question_dim, attention_dim)
        self.fc2 = nn.Linear(vector_dim, attention_dim)
        self.fc3 = nn.Linear(attention_dim, 1)
    def forward(self, question, vector): 
        H = torch.tanh(self.fc1(question)+self.fc2(vector))
        attention_score = F.softmax(self.fc3(H), dim=0)
        x = torch.matmul(attention_score.permute(1,2,0), question.permute(1,0,2))
        return x.squeeze(1)

class alternating_co_attention(nn.Module):
    def __init__(self, batch_size, question_dim, vector_dim, attention_dim):
        super(alternating_co_attention, self).__init__()
        self.question_g_attention = linear_attention(question_dim, vector_dim, attention_dim)
        self.context_q_attention = linear_attention(question_dim, vector_dim, attention_dim)
        self.question_c_attention = linear_attention(question_dim, vector_dim, attention_dim)
    def forward(self, question, context):
        if torch.cuda.is_available():  
            g = torch.zeros(question.shape[1], vector_dim).cuda()
        else: 
            g = torch.zeros(question.shape[1], vector_dim)  
        temp_vector = self.question_g_attention(question, self.g)
        c_vector = self.context_q_attention(context, temp_vector)
        q_vector = self.question_c_attention(question, c_vector)
        return q_vector, c_vector


class MultiAttention(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size):
        super(MultiAttention, self).__init__() 
        self.dim = word_hidden_size*2
        self.batch_size = batch_size
        self.fc1 = nn.Linear(self.dim, self.dim) # W_u 
        self.fc2 = nn.Linear(self.dim, self.dim) # W_a_k
        self.fc3 = nn.Linear(self.dim, self.dim) # W_q 
        if torch.cuda.is_available():  
            self.u = torch.zeros(self.dim, requires_grad = True).cuda()
        else:
            self.u = torch.zeros(self.dim, requires_grad = True)
        self.attention = alternating_co_attention(batch_size, self.dim, self.dim, self.dim) 
        self.attention1 = linear_attention(self.dim, self.dim, self.dim)
        self.linear = nn.Linear(self.dim*2, self.dim)
    def forward(self, answer_hidden, question_output, context_output, num_attention=2):
        '''
        input
        answer_hidden (batch_size, dim)
        question_output (4, batch_size, dim)
        '''
        for i in range(num_attention):
            q_new = torch.tanh(self.fc1(self.u) + self.fc2(answer_hidden) + self.fc3(question_output))
            q_attention_vector, context_attention_vector = self.attention(question_output, context_output)
            u = torch.tanh(self.linear(torch.cat((q_attention_vector, context_attention_vector), dim=1)))
            self.u = u 
        return u 


class Impatient_Reader_Model(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_attention):
        super(Impatient_Reader_Model, self).__init__()
        self.text = Text_Net(word_hidden_size, sent_hidden_size)
        self.question = Question_Net(word_hidden_size, sent_hidden_size) 
        self.attention = MultiAttention(word_hidden_size, sent_hidden_size, batch_size)
        self.dim  = 2*word_hidden_size
        self.choice = ChoiceNet(word_hidden_size)
        #self.images = nn.LSTM(1000, word_hidden_size, bidirectional=True) 
        self.num_attention = num_attention
        self.q = nn.Linear(self.dim, self.dim)
        self.a = nn.Linear(self.dim, self.dim)
    def Infersent(self, x1, x2):
        return torch.cat((x1, x2, torch.abs(x1 - x2), x1 * x2), 1)

    def forward(self, input_context,  input_question, input_choice, input_images, image_path):
        input_context = transport_1_0_2(input_context)
        input_question = transport_1_0_2(input_question)
        input_choice = transport_1_0_2(input_choice) 
        input_images = transport_1_0_2_image(input_images) 

        #input_images = extract_image_features(input_images, image_path)

        context_output, _ = self.text(input_context)
        question_output, question_final_hidden = self.question(input_question)
        #image_output, _  = self.images(input_images) 
        output_choice_list = [] 

        for i in input_choice:   
            output_choice, hidden_output_choice = self.choice(i)
            u = self.attention(hidden_output_choice, question_output, context_output, self.num_attention)
            q_u = torch.tanh(self.q(question_final_hidden)) + u
            similarity_scores = torch.sum(torch.mul(q_u, torch.tanh(self.a(hidden_output_choice))), dim=1)
            # similarity_scores = self.Infersent(g, hidden_output_choice)
            # similarity_scores = self.dropout(torch.tanh(self.fc3(similarity_scores)))
            # similarity_scores = self.fc4(similarity_scores)
            output_choice_list.append(similarity_scores) 
            
        return output_choice_list 





