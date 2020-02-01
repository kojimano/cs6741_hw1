"""
Name: Noriyuki Kojima
Date: January 31th, 2020
Assignment 1: Classification
"""
import os, sys
import torch
import torch.nn as nn
import numpy as np
from IPython import embed

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


class NaiveBayes(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.f_dim = config["f_dim"]
        self.binarize = config["nb_binarize"]
        self.alpha = config["nb_alpha"]
        self.linear = nn.Linear(self.f_dim, 1)
        self.embeddings = nn.Embedding(self.f_dim, self.f_dim)
        self.embeddings.weight = nn.Parameter(torch.eye(self.f_dim).detach())
        self.embeddings.weight[0,0] = 0
        self.embeddings.weight.requires_grad = False

        if torch.cuda.is_available():
            self.linear.cuda()
            self.embeddings.cuda()

    def get_features(self, x):
        return torch.sum(self.embeddings(x),0)

    def set_weight(self, train_iter):
        p_ct, q_ct = 0., 0.
        p_weight, q_weight = torch.zeros((1, self.f_dim)), torch.zeros((1, self.f_dim))

        if torch.cuda.is_available():
            p_weight = p_weight.cuda()
            q_weight = q_weight.cuda()

        for batch in train_iter:
            x = self.get_features(batch.text)
            if self.binarize:
                local_p = (torch.sum(x[batch.label==1, :], 0) > 0).float()
                local_q = (torch.sum(x[batch.label==0, :], 0) > 0).float()
            else:
                local_p = torch.sum(x[batch.label==1, :], 0)
                local_q = torch.sum(x[batch.label==0, :], 0)

            p_weight += local_p
            q_weight += local_q
            p_ct += torch.sum(batch.label==1)
            q_ct += torch.sum(batch.label==0)

        p_weight = p_weight+ self.alpha
        p_weight = p_weight / torch.sum(p_weight)
        q_weight = q_weight+ self.alpha
        q_weight = q_weight / torch.sum(q_weight)

        if torch.cuda.is_available():
            p_ct = p_ct.cuda()
            q_ct = q_ct.cuda()

        r = torch.log(p_weight / q_weight)
        b = torch.log(p_ct.float() / q_ct.float())
        self.linear.weight = nn.Parameter(r)
        self.linear.bias = nn.Parameter(b)

    def forward(self, x):
        x = self.get_features(x)
        if self.binarize:
            x = (x > 0).float()

        pos_score = torch.sign(self.linear(x))
        neg_score = torch.sign(-self.linear(x))

        final_scores =  torch.cat([neg_score, pos_score], 1)
        return final_scores

class LogisticRegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.f_dim = config["f_dim"]
        self.linear = nn.Linear(self.f_dim, 1)

        self.embeddings = nn.Embedding(self.f_dim, self.f_dim)
        weight = torch.eye(self.f_dim)
        weight[0,0] = 0
        self.embeddings.weight = nn.Parameter(weight.detach())
        self.embeddings.weight.requires_grad = config["finetune_w2v"]

        if torch.cuda.is_available():
            self.linear.cuda()
            self.embeddings.cuda()

    def get_features(self, x):
        return torch.sum(self.embeddings(x),0)

    def forward(self, x):
        x = self.get_features(x)
        x = self.linear(x)
        pos_score = torch.sigmoid(x)
        neg_score = 1.-pos_score
        final_scores =  torch.cat([neg_score, pos_score], 1)
        return final_scores


class feedforwardNN(nn.Module):

    def __init__(self, config, w2v):
        super().__init__()
        self.f_dim = config["f_dim"]
        self.embeddings = nn.Embedding(self.f_dim, w2v.shape[1])
        self.embeddings.weight = nn.Parameter(w2v)
        self.embeddings.weight.requires_grad = config["finetune_w2v"]
        self.ff_layer =  nn.Sequential(
                nn.Linear(w2v.shape[1], 50),
                nn.ReLU(),
                nn.Dropout(config["ff_dropout_p"]),
                nn.Linear(50, 2),
                )

        if torch.cuda.is_available():
            self.ff_layer.cuda()
            self.embeddings.cuda()

    def get_features(self, x):
        return torch.sum(self.embeddings(x).transpose(0,1),1)

    def forward(self, x):
        x = self.get_features(x)
        x = self.ff_layer(x)
        final_scores = torch.softmax(x, 1)
        return final_scores


class convolutionalNN(nn.Module):

    def __init__(self, config, w2v):
        super().__init__()
        self.f_dim = config["f_dim"]
        self.embeddings = nn.Embedding(self.f_dim, w2v.shape[1])
        self.embeddings.weight = nn.Parameter(w2v)
        self.embeddings.weight.requires_grad = config["finetune_w2v"]
        self.convs =  nn.ModuleList([
                nn.Conv2d(1, 100, (3, w2v.shape[1]), padding=(1,0)),
                nn.Conv2d(1, 100, (4, w2v.shape[1]), padding=(1,0)),
                nn.Conv2d(1, 100, (5, w2v.shape[1]), padding=(1,0)),])
        self.dropout = nn.Dropout(config["ff_dropout_p"])
        """
        self.ff_layer = nn.Sequential(
            nn.Linear(300, 50),
            nn.ReLU(),
            nn.Dropout(config["ff_dropout_p"]),
            nn.Linear(50, 2),
        )
        """
        self.linear =  nn.Linear(300, 2)


        if torch.cuda.is_available():
            self.convs.cuda()
            self.linear.cuda()
            self.embeddings.cuda()


    def get_features(self, x):
        return self.embeddings(x).transpose(0,1)

    def forward(self, x):
        feat_x = self.get_features(x)
        feat_x = feat_x.unsqueeze(1)
        all_feats =[]
        for layer in self.convs:
            feat = layer(feat_x)
            feat = feat.squeeze(3)
            feat = feat.max(2)[0]
            all_feats.append(feat)
        cat_feats = torch.cat(all_feats, 1)
        cat_feats = self.dropout(cat_feats)
        logits = self.linear(cat_feats)
        final_scores = torch.softmax(logits, 1)
        return final_scores

class BERTfeedforward(nn.Module):

    def __init__(self, config, vocab_list):
        super().__init__()
        self.f_dim = config["f_dim"]
        self.vocab_list = np.array(vocab_list)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.ff_layer =  nn.Sequential(
                nn.Linear(768, 50),
                nn.ReLU(),
                nn.Dropout(config["ff_dropout_p"]),
                nn.Linear(50, 2),
                )
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        if torch.cuda.is_available():
            self.ff_layer.cuda()
            self.bert_model.cuda()

    def get_features(self, x):
        sents = self.vocab_list[x.cpu()]

        all_sents = []
        for i in range(len(sents[1])):
            tokens = list(sents[:,i])
            sent = " ".join(tokens)
            sent = sent.replace(" <pad>", "")
            sent = "[CLS] " + sent + " [SEP]"
            tokenized_sent = self.tokenizer.tokenize(sent)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
            all_sents.append(indexed_tokens)

        max_len = np.max([len(sent) for sent in all_sents])
        for i in range(len(all_sents)):
            all_sents[i] += [0] * (max_len - len(all_sents[i]))

        all_sents = np.stack(all_sents, 0)
        all_sents = torch.Tensor(all_sents).long()

        if torch.cuda.is_available():
            all_sents = all_sents.cuda()


        encoded_layers, _ = self.bert_model(all_sents)
        bert_rep = encoded_layers[11]
        sent_rep = torch.mean(bert_rep, dim=1)

        return sent_rep

    def forward(self, x):
        feat_x = self.get_features(x)
        logits = self.ff_layer(feat_x)
        final_scores = torch.softmax(logits, 1)
        return final_scores
