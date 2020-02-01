"""
Name: Noriyuki Kojima
Date: January 31th, 2020
Assignment 1: Classification
"""
import os, sys
import torch
import argparse
import operator
import numpy as np
import sklearn.metrics as eval
from typing import Dict, List

# Text text processing library and methods for pretrained word embeddings
import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors, GloVe
from IPython import embed

from models import NaiveBayes, LogisticRegression, feedforwardNN, convolutionalNN, BERTfeedforward

# parser
parser = argparse.ArgumentParser(description='CS 6741 HW 1')
parser.add_argument('--mode', required=True, type=str, help='train | test')
parser.add_argument('--model_type', required=True, type=str, help='nb | lr | ff | cnn')
parser.add_argument('--load_model_name', default='', type=str, help='model path')
parser.add_argument('--save_model_name', default='', type=str, help='model path')


def load_data():
    # Our input $x$
    TEXT = torchtext.data.Field()

    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False, unk_token=None)
    train, val, test = torchtext.datasets.SST.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral')

    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('len(LABEL.vocab)', len(LABEL.vocab))

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=10, device=torch.device("cuda"))

    batch = next(iter(train_iter))
    print("Size of text batch:", batch.text.shape)
    example = batch.text[1,:]
    print("Second in batch", example)
    print("Converted back to string:", " ".join([TEXT.vocab.itos[i] for i in example.tolist()]))

    print("Size of label batch:", batch.label.shape)
    example = batch.label[1]
    print("Second in batch", example.item())
    print("Converted back to string:", LABEL.vocab.itos[example.item()])


    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
    # TEXT.vocab.load_vectors(vectors=Vectors('glove.twitter.27B.100d.txt', "./embeddings"))
    #print("Word embeddings size ", TEXT.vocab.vectors.size())
    print("Word embedding of 'follows', first 10 dim ", TEXT.vocab.vectors[TEXT.vocab.stoi['follows']][:10])
    # vec1 = TEXT.vocab.vectors[TEXT.vocab.stoi['man']] / (torch.sum(TEXT.vocab.vectors[TEXT.vocab.stoi['man']] ** 2) )**0.5
    # vec2 = TEXT.vocab.vectors[TEXT.vocab.stoi['guy']] / (torch.sum(TEXT.vocab.vectors[TEXT.vocab.stoi['guy']] ** 2) )**0.5
    # vec3 = TEXT.vocab.vectors[TEXT.vocab.stoi['woman']] / (torch.sum(TEXT.vocab.vectors[TEXT.vocab.stoi['woman']] ** 2) )**0.5
    #  vec4 = TEXT.vocab.vectors[TEXT.vocab.stoi['ball']] / (torch.sum(TEXT.vocab.vectors[TEXT.vocab.stoi['ball']] ** 2) )**0.5
    # print(np.dot(vec1, vec2))
    return train, val, test, TEXT.vocab.itos, LABEL.vocab.itos, TEXT.vocab.vectors

def get_metrics(prediction: List[int], ground_truth: List[int]):
    f_score = eval.f1_score(prediction, ground_truth)
    accuracy = eval.accuracy_score(prediction, ground_truth)
    print("f-score: {}".format(f_score))
    print("accuracy: {}".format(accuracy))

    return f_score, accuracy

def test_code(model: nn.Module, test_data: torchtext.datasets):
    "All models should be able to be run with following command."
    upload = []
    ground_truth = []

    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test_data, train=False, batch_size=10, device=torch.device("cuda"))
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        # here we assume that the name for dimension classes is `classes`
        #_, argmax = probs.max('classes') # Noriyuki: Named tensor is not supported for pytorch 1.1.0 (I am using CUDA 9.0)
        _, argmax = probs.max(1)
        upload += argmax.tolist()
        ground_truth += batch.label.cpu().tolist()

    f_score, accuracy = get_metrics(upload, ground_truth)

    with open("predictions.txt", "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")

    return f_score


def get_model(model_type: str, model_config: Dict, w2v: torch.Tensor, vocab_list: List, model_name: str) -> nn.Module:
    # Instantiate model and configuration
    train_config = {
                    "num_epochs": 30,
                    "lr_rate": 2e-5,
                    "log_step": 100,
                    "l2norm": False,
                    "l2factor": 3.,
                    "lambda": 0.01,
                   }

    if model_type == "nb":
        model = NaiveBayes(model_config)
    elif model_type == "lr":
        model = LogisticRegression(model_config)
        train_config["lr_rate"] = 2e-3
    elif model_type == "ff":
        model = feedforwardNN(model_config, w2v)
        train_config["num_epochs"] = 50
        train_config["lr_rate"] = 2e-4
    elif model_type ==  "cnn":
        model = convolutionalNN(model_config, w2v)
        train_config["num_epochs"] = 30
        train_config["lr_rate"] = 2e-4
        train_config["l2norm"] = False
    elif model_type ==  "bertff":
        model = BERTfeedforward(model_config, vocab_list)
        train_config["num_epochs"] = 30
        train_config["lr_rate"] = 1e-5
    else:
        raise ValueError("Model type is not supported.")

    # Load model
    if model_name is not "":
        model = torch.load("./models/"+model_name)

    return model, train_config

def train_model(model: nn.Module, train_config: Dict, train_data: torchtext.datasets, val_data: torchtext.datasets, model_name: str, debug=False):
    # Create train / val iterators
    train_iter, val_iter = torchtext.data.BucketIterator.splits((train_data, val_data), batch_size=10, device=torch.device("cuda"))
    if debug:
        val_iter = torchtext.data.BucketIterator(val_data, train=False, batch_size=10, device=torch.device("cuda"))

    if type(model) == NaiveBayes:
        # Train non-gradient-based method
        with torch.no_grad():
            model.set_weight(train_iter)

        # Validation
        model.eval()
        print("Validation")
        f_score = test_code(model, val_data)

        # Save model
        val_score = np.round(f_score,3)
        model_name = model_name + "-score-{}-epoch-{}.pkl".format(val_score, 0)
        save_model(model, model_name)

        return model_name
    else:
        # Gradient-based method
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr_rate"])
        criterion = torch.nn.CrossEntropyLoss()
        log_step = train_config["log_step"]
        iterator = train_iter
        if debug:
            iterator = val_iter
        model_val_scores = {}

        # Train
        for epoch in range(train_config["num_epochs"]):
            loss_history = []
            model.train()
            for i, batch in enumerate(iterator):
                outputs = model(batch.text)
                loss = criterion(outputs, batch.label)
                optimizer.zero_grad()

                if train_config["l2norm"]:
                    reg_loss = 0
                    for param in model.parameters():
                        reg = param.pow(2) > train_config["l2factor"]
                        reg_loss += torch.sum(param.pow(2)[reg] - train_config["l2factor"])
                    loss += train_config["lambda"] / 2. * reg_loss

                loss.backward()
                optimizer.step()
                loss_history.append(loss.cpu().item())
                if i % log_step == 0:
                    avg_loss = np.round(np.mean(loss_history), 3)
                    print("Epoch {}, step {}, average loss: {}".format(epoch, i, avg_loss))

            avg_loss = np.round(np.mean(loss_history), 3)
            print("Epoch {}, average loss: {}".format(epoch,avg_loss))
            print("")
            print("")

            # Validation
            model.eval()
            print("Epoch {} ... Validation".format(epoch))
            f_score = test_code(model, val_data)

            # Save model
            val_score = np.round(f_score,3)
            pkl_name = model_name + "-score-{}-epoch-{}.pkl".format(val_score, epoch)
            if not debug:
                save_model(model, pkl_name)
            model_val_scores[pkl_name] = val_score

        sorted_models = sorted(model_val_scores.items(), key=operator.itemgetter(1), reverse=True)
        for pkl_name, _ in sorted_models[1:]:
                os.remove("./models/{}".format(pkl_name))
        final_model_name = "best-{}".format(sorted_models[0][0])
        os.rename("./models/{}".format(sorted_models[0][0]), "./models/{}".format(final_model_name))

    return final_model_name

def save_model(model: nn.Module, model_name: str):
    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(model, "./models/"+model_name)

if __name__ == '__main__':
    args = parser.parse_args()

    train_data, val_data, test_data, vocab_list, label_list, word_vec = load_data()

    model_config = {
                    "f_dim": len(vocab_list),
                    "nb_alpha": 1.,
                    "nb_binarize": True,
                    "ff_dropout_p": 0.5,
                    "finetune_w2v": True,
                    }


    model, train_config = get_model(args.model_type, model_config, word_vec, vocab_list, args.load_model_name)

    if args.mode == "train":
        print("Start training ...")
        best_model_name = train_model(model, train_config, train_data, val_data, args.save_model_name)
        model, train_config = get_model(args.model_type, model_config, word_vec, vocab_list, best_model_name)
        print("Start testing ...")
        test_code(model, test_data)
    elif args.mode == "test":
        model.eval()
        print("Testing")
        test_code(model, test_data)
    else:
        raise ValueError("Mode not supported.")
