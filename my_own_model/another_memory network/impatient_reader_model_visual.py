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

        return output, hidden_output, output_step 


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

        return output, hidden_output, output_word


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
        return x.squeeze(1), attention_score.permute(1, 2, 0) 

class alternating_co_attention(nn.Module):
    def __init__(self, batch_size, question_dim, vector_dim, attention_dim):
        super(alternating_co_attention, self).__init__()
        self.question_g_attention = linear_attention(question_dim, vector_dim, attention_dim)
        self.context_q_attention = linear_attention(question_dim, vector_dim, attention_dim)
        self.question_c_attention = linear_attention(question_dim, vector_dim, attention_dim)
        self.dim = vector_dim
    def forward(self, question, context):
        if torch.cuda.is_available():  
            g = torch.zeros(question.size(1), self.dim).cuda()
        else: 
            g = torch.zeros(question.size(1), self.dim)  
        temp_vector, _ = self.question_g_attention(question, g)
        c_vector, score_q2c = self.context_q_attention(context, temp_vector)
        q_vector, score_c2q = self.question_c_attention(question, c_vector)
        return q_vector, c_vector, score_c2q, score_q2c


class MultiAttention(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, use_lexical):
        super(MultiAttention, self).__init__() 
        self.dim = word_hidden_size*2
        self.batch_size = batch_size
        self.fc1 = nn.Linear(self.dim, self.dim) # W_u 
        self.fc2 = nn.Linear(self.dim, self.dim) # W_a_k
        self.fc3 = nn.Linear(self.dim, self.dim) # W_q 
        self.attention = alternating_co_attention(batch_size, self.dim, self.dim, self.dim) 
        self.lexical_question = linear_attention(self.dim, self.dim, self.dim)
        self.lexical_context = linear_attention(self.dim, self.dim, self.dim)
        self.linear = nn.Linear(self.dim*2, self.dim) 
        self.linear_u = nn.Linear(self.dim*2, self.dim)
        self.use_lexical = use_lexical
        if self.use_lexical:
            self.lexical_projection_q = nn.Linear(self.dim, self.dim) 
            self.lexical_projection_c= nn.Linear(self.dim, self.dim)
    def forward(self, question_output, question_hidden, context_output, context_input, question_input, num_attention=2):

        for i in range(num_attention): 
            q_attention_vector, context_attention_vector, score_c2q, score_q2c = self.attention(question_output, context_output)
            

            if self.use_lexical: 
                q_lexical = torch.tanh(torch.matmul(score_c2q, question_input.permute(1,0,2))).squeeze(1)
                c_lexical = torch.tanh(torch.matmul(score_q2c, context_input.permute(1,0,2))).squeeze(1)
                q_attention_vector = torch.tanh(self.lexical_projection_q(q_lexical)) + q_attention_vector
                context_attention_vector = torch.tanh(self.lexical_projection_c(c_lexical)) + context_attention_vector

            u = torch.tanh(self.linear(torch.cat((q_attention_vector, context_attention_vector), dim=1))) 
            question_hidden = torch.tanh(self.linear_u(torch.cat((u, question_hidden), dim=1)))
            
        return u 

class MultiAttention_image(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, use_lexical):
        super(MultiAttention_image, self).__init__() 
        self.dim = word_hidden_size*2
        self.batch_size = batch_size
        self.fc1 = nn.Linear(self.dim, self.dim) # W_u 
        self.fc2 = nn.Linear(self.dim, self.dim) # W_a_k
        self.fc3 = nn.Linear(self.dim, self.dim) # W_q 

        self.attention_q_c = alternating_co_attention(batch_size, self.dim, self.dim, self.dim) 
        self.attention_q_v = alternating_co_attention(batch_size, self.dim, self.dim, self.dim) 
        self.attention_c_v = alternating_co_attention(batch_size, self.dim, self.dim, self.dim) 

        self.lexical_question = linear_attention(self.dim, self.dim, self.dim)
        self.lexical_context = linear_attention(self.dim, self.dim, self.dim)
        self.linear = nn.Linear(self.dim*6, self.dim) 
        self.use_lexical = use_lexical 

        if self.use_lexical:
            self.lexical_projection_q_c = nn.Linear(self.dim, self.dim) 
            self.lexical_projection_c_q = nn.Linear(self.dim, self.dim)
            self.lexical_projection_q_v = nn.Linear(self.dim, self.dim)
            self.lexical_projection_v_q = nn.Linear(1000, self.dim)
            self.lexical_projection_c_v = nn.Linear(self.dim, self.dim)
            self.lexical_projection_v_c = nn.Linear(1000, self.dim)


    def forward(self, answer_hidden, question_output, context_output, image_output, context_input, question_input, image_input, num_attention=2):

        for i in range(num_attention):
            q_attention_vector_c, c_attention_vector_q, score_c2q, score_q2c = self.attention_q_c(question_output, context_output)
            q_attention_vector_v, v_attention_vector_q, score_v2q, score_q2v = self.attention_q_v(question_output, image_output)
            c_attention_vector_v, v_attention_vector_c, score_v2c, score_c2v = self.attention_q_v(context_output, image_output)

            if self.use_lexical: 
                q_lexical_c = torch.tanh(torch.matmul(score_c2q, question_input.permute(1,0,2))).squeeze(1)
                c_lexical_q = torch.tanh(torch.matmul(score_q2c, context_input.permute(1,0,2))).squeeze(1)
                q_attention_vector_c = torch.tanh(self.lexical_projection_q_c(q_lexical_c)) + q_attention_vector_c
                c_attention_vector_q = torch.tanh(self.lexical_projection_c_q(c_lexical_q)) + c_attention_vector_q

                q_lexical_v = torch.tanh(torch.matmul(score_v2q, question_input.permute(1,0,2))).squeeze(1)
                v_lexical_q = torch.tanh(torch.matmul(score_q2v, image_input.permute(1,0,2))).squeeze(1)
                q_attention_vector_v = torch.tanh(self.lexical_projection_q_v(q_lexical_c)) + q_attention_vector_v
                v_attention_vector_q = torch.tanh(self.lexical_projection_v_q(v_lexical_q)) + v_attention_vector_q

                v_lexical_c = torch.tanh(torch.matmul(score_c2v, image_input.permute(1,0,2))).squeeze(1)
                c_lexical_v = torch.tanh(torch.matmul(score_v2c, context_input.permute(1,0,2))).squeeze(1)
                v_attention_vector_c = torch.tanh(self.lexical_projection_v_c(v_lexical_c)) + v_attention_vector_c
                c_attention_vector_v = torch.tanh(self.lexical_projection_c_v(c_lexical_v)) + c_attention_vector_v

            u = torch.tanh(self.linear(torch.cat((q_attention_vector_c, c_attention_vector_q, q_attention_vector_v, v_attention_vector_q, c_attention_vector_v, v_attention_vector_c), dim=1))) 
        return u


class Impatient_Reader_Model(nn.Module):       
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, use_lexical, use_image, num_attention):
        super(Impatient_Reader_Model, self).__init__()
        self.text = Text_Net(word_hidden_size, sent_hidden_size)
        self.question = Question_Net(word_hidden_size, sent_hidden_size) 
        self.attention = MultiAttention(word_hidden_size, sent_hidden_size, batch_size, use_lexical)
        self.dim  = 2*word_hidden_size
        self.choice = ChoiceNet(word_hidden_size)
        self.use_image  = use_image
        self.num_attention = num_attention
        self.q = nn.Linear(self.dim, self.dim)
        self.a = nn.Linear(self.dim, self.dim)
        if self.use_image: 
            self.images = nn.LSTM(1000, word_hidden_size, bidirectional=True) 
            self.attention_image = MultiAttention_image(word_hidden_size, sent_hidden_size, batch_size, use_lexical)
    
    def Infersent(self, x1, x2):
        return torch.cat((x1, x2, torch.abs(x1 - x2), x1 * x2), 1)

    def forward(self, input_context,  input_question, input_choice, input_images, image_path):
        input_context = transport_1_0_2(input_context)
        input_question = transport_1_0_2(input_question)
        input_choice = transport_1_0_2(input_choice) 
        input_images = transport_1_0_2_image(input_images) 
        if self.use_image:
            input_images = extract_image_features(input_images, image_path)
            image_output, _  = self.images(input_images) 

        context_output, _, context_input = self.text(input_context)
        question_output, question_final_hidden, question_input = self.question(input_question)

        if self.use_image:
                u = self.attention_image(question_output, context_output, image_output, context_input, question_input, input_images, self.num_attention)
            else:
                u = self.attention(question_output, context_output, context_input, question_input, self.num_attention)
         
        output_choice_list = []
        for i in input_choice:   
            output_choice, hidden_output_choice = self.choice(i) 

            similarity_scores = self.Infersent(u, hidden_output_choice) 
            similarity_scores = self.dropout(torch.tanh(self.fc3(similarity_scores))) 
            similarity_scores = self.fc4(similarity_scores)

            output_choice_list.append(similarity_scores) 
            
        return output_choice_list 





