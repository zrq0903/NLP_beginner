import torch
import torch.nn as nn
from numpy.ma.core import squeeze


def get_input_label(sentences):
    input_batch= []
    label_batch = []
    for sen in sentences:
        words = sen.split()
        input = [word_to_label[i] for i in words[:-1]]
        label = word_to_label[words[-1]]
        input_batch.append(input)
        label_batch.append(label)
    return input_batch, label_batch
class NNLMs(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden,n_pred):
        super(NNLMs,self).__init__()
        self.C = nn.Embedding(vocab_size,embed_size)
        self.H = nn.Linear(n_pred*embed_size,hidden)
        self.d = nn.Parameter(torch.ones(hidden))
        self.U = nn.Linear(hidden,vocab_size)
        self.W = nn.Linear(n_pred*embed_size,vocab_size)
        self.b = nn.Parameter(torch.ones(vocab_size))
    def forward(self,x):
        x = self.C(x) #(batch_size,n_pred,embed_size)
        x = x.view(x.shape[0],-1) #(batch_size, n_pred*emned_size)
        hidden_layer = torch.tanh(self.H(x)+ self.d) #(batch_size,hidden)
        output = self.U(hidden_layer) + self.W(x) + self.b #(batch_size,vocab_size)
        return output

if __name__ == '__main__':
    sentences = ['mamba is out','mamba comes back','mamba loves you','what can say']
    word_list = []
    for sen in sentences:
        word_list += sen.split()
    word_list = list(set(word_list))
    word_to_label = {w: l for l, w in enumerate(word_list)}
    label_to_word = {l: w for l, w in enumerate(word_list)}
    dict_num = len(word_list)

    model = NNLMs(dict_num,2,2,2)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

    input_batch, label_batch = get_input_label(sentences)
    input_batch = torch.LongTensor(input_batch)
    label_batch = torch.LongTensor(label_batch)

    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(input_batch)
        l = loss(output ,label_batch)

        l.backward()
        optimizer.step()

        __,pred = torch.max(output.data, 1)
        acc = (pred==label_batch).sum().item()/label_batch.shape[0]
        if(epoch%100 ==0):
            print("epoch {}, loss = {}, acc {}".format(epoch,l.item(),acc))
    predict =  model(input_batch).data.max(1,keepdim = True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [label_to_word[n.item()] for n in predict.squeeze()])
