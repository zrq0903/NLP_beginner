import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import random
from docutils.nodes import label
#nltk.download('punkt')
#nltk.download('gutenberg')
from nltk.corpus import gutenberg
from collections import Counter

from torch.nn.functional import embedding

#print(gutenberg.sents('melville-moby_dick.txt')[:10])
'''
preprocessing
'''
window_size = 5
corpus = list(gutenberg.sents('melville-moby_dick.txt'))[:500]
corpus = [[word.lower() for word in sent] for sent in corpus]
flatten_corpus = [word for sent in corpus for word in sent]

included = []
word_count = Counter(flatten_corpus)
for w,n in word_count.items():
    if n>=3:
        included.append(w)
#print(len(excluded))
window_size = 5
word_to_label = {w:l for l, w in enumerate(included)}
label_to_word = {l:w for l, w in enumerate(included)}
#def get_pairs(corpus):
windows = [list(nltk.ngrams(['<DUMMY>']*window_size + c + ['<DUMMY>']*window_size, 2*window_size+1)) for c in corpus]
windows = [list(w) for ww in windows for w in ww]
#print(windows)
pairs = []
for window in windows:
    for i in range(2*window_size+1):
        if (window[i] not in included) or (window[window_size] not in included):
            continue
        if  i == window_size or window[i] == '<DUMMY>':
            continue
        pairs.append((window[window_size],window[i]))
#print(pairs)

def unigram_distribution(word_count,included):
    distribution = {w:n**0.75 for w,n in word_count.items() if w in included}
    total_count = sum(distribution.values())
    prob = {w:n/total_count for w,n in distribution.items()}
    vocab, weight = zip(*prob.items())
    return vocab, weight
vocab, uni_weights = unigram_distribution(word_count, included)

k = 10
def negative_sampling(pos,uni_weights,k):
    batch_size = pos.shape[0]
    neg_samples = []
    for i in range(batch_size):
        #pos_index = pos[i].data.tolist()[0]
        pos_index = pos[i].data.tolist()
        neg_index = []
        while len(neg_samples)<k:
            neg = random.choices(vocab,weights=uni_weights,k=1)[0]
            if word_to_label[neg] != pos_index:
                neg_index.append(word_to_label[neg])
        neg_samples.append(torch.LongTensor(neg_index).view(1,-1)) #(1,k)
    return torch.cat(neg_samples) #(batch,k)

class skipgram_negsampling(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super(skipgram_negsampling,self).__init__()
        self.embedding_i = nn.Embedding(vocab_size,embed_size)
        self.embedding_o = nn.Embedding(vocab_size,embed_size)

    def forward(self,target,center,neg):
        # nn.Embedding added in the last dim
        center_embedding = self.embedding_i(target) #(batch,emb)
        target_embedding = self.embedding_o(center) #(batch,emb)
        neg_embedding = self.embedding_o(neg)  #(batch,k,emb)
        positive_score = torch.sum(center_embedding*target_embedding,dim=1)
        negative_score = torch.bmm(neg_embedding,center_embedding.unsqueeze(2))
        #center -> (batch, emb, 1) -> neg(batch, k, emb) bmm (batch, emb, 1)-> (batch, k, 1)
        loss = - F.logsigmoid(positive_score) - F.logsigmoid(-negative_score)
        return torch.mean(loss)

# to be continue
def get_input_batch(pairs):
    center_batch = torch.LongTensor([word_to_label[pair[0]] for pair in pairs]).view(1,-1)
    context_batch = torch.LongTensor([word_to_label[pair[1]] for pair in pairs]).view(1,-1)
    return list(zip(center_batch,context_batch))

input_batch = get_input_batch(pairs)
epoch = 100
embedding_size = 30
model = skipgram_negsampling(len(vocab),embedding_size)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
for i in range(epoch):
    optimizer.zero_grad()
    center, target = zip(*input_batch)
    center = torch.cat(center)
    target = torch.cat(target)
    negative = negative_sampling(target,uni_weights,k)
    loss = model(center,target,negative)
    loss.backward()
    optimizer.step()

