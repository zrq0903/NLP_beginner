#skip_gram
import torch
import torch.nn as nn
import numpy as np
window_size = 1
class skipgram(nn.Module):
    def __init__(self,hidden,vocab_size):
        super(skipgram,self).__init__()
        self.w1 = nn.Linear(vocab_size,hidden,bias=False)
        self.w2 = nn.Linear(hidden,vocab_size,bias=False)
    def forward(self,x):    #x -> (batch, vocab_size)
        x = self.w1(x)       #  (batch_hidden)
        output = self.w2(x)    #(batch_vocab_size)
        return output

def get_pair(sentences):
    pairs = []
    for sen in sentences:
        words = sen.split()
        for i in range(1,len(words)-1):
            center = word_to_label[words[i]]
            context = word_to_label[words[i-1]], word_to_label[words[i+1]]
            for c in context:
                pairs.append([center, c])
    return pairs
def get_input_label(pairs,vocab_size):
    input_batch = []
    label_batch = []
    for pair in pairs:
        input_batch.append(np.eye(vocab_size)[pair[0]])
        label_batch.append(pair[1])
    return input_batch,label_batch



if __name__ == '__main__':
    sentences = ['mamba is coming back','mamba will love you','man what can say']
    word_list = []
    for sen in sentences:
        word_list += sen.split()
    word_list = list(set(word_list))
    word_to_label = {w: l for l, w in enumerate(word_list)}
    label_to_word = {l: w for l, w in enumerate(word_list)}
    dict_num = len(word_list)
    pairs = get_pair(sentences)
    input_batch, label_batch = get_input_label(pairs, dict_num)
    model = skipgram(hidden=10, vocab_size=dict_num)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    input_batch = torch.Tensor(input_batch)
    label_batch = torch.LongTensor(label_batch)
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(input_batch)
        l = loss(output ,label_batch)

        l.backward()
        optimizer.step()

        __,pred = torch.max(output.data, 1)
        acc = (pred==label_batch).sum().item()/label_batch.shape[0]
        if((epoch+1)%100 ==0):
            print("epoch {}, loss = {}, acc {}".format(epoch+1,l.item(),acc))
    predict =  model(input_batch).data.max(1,keepdim = True)[1]
    print([label_to_word[pair[0]] for pair in pairs], '->', [label_to_word[n.item()] for n in predict.squeeze()])
    print([label_to_word[pair[0]] for pair in pairs], '->', [label_to_word[pair[1]] for pair in pairs])

