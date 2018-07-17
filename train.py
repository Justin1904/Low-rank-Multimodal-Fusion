from __future__ import print_function
from model import TFN
from utils import MultimodalDataset, total
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import csv

def preprocess(options):
    # parse the input args
    dataset = options['dataset']
    epochs = options['epochs']
    model_path = options['model_path']
    max_len = options['max_len']

    # prepare the paths for storing models
    model_path = os.path.join(
        model_path, "tfn.pt")
    print("Temp location for saving model: {}".format(model_path))

    # prepare the datasets
    print("Currently using {} dataset.".format(dataset))
    mosi = MultimodalDataset(dataset, max_len=max_len)
    train_set, valid_set, test_set = mosi.train_set, mosi.valid_set, mosi.test_set

    audio_dim = train_set[0][0].shape[1]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[1]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # normalize the visual features
    visual_max = np.max(np.max(np.abs(train_set.visual), axis=0), axis=0)
    visual_max[visual_max==0] = 1
    train_set.visual = train_set.visual / visual_max
    valid_set.visual = valid_set.visual / visual_max
    test_set.visual = test_set.visual / visual_max

    # for visual and audio modality, we average across time
    # here the original data has shape (max_len, num_examples, feature_dim)
    # after averaging they become (1, num_examples, feature_dim)
    train_set.visual = np.mean(train_set.visual, axis=0, keepdims=True)
    train_set.audio = np.mean(train_set.audio, axis=0, keepdims=True)
    valid_set.visual = np.mean(valid_set.visual, axis=0, keepdims=True)
    valid_set.audio = np.mean(valid_set.audio, axis=0, keepdims=True)
    test_set.visual = np.mean(test_set.visual, axis=0, keepdims=True)
    test_set.audio = np.mean(test_set.audio, axis=0, keepdims=True)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims

def display(mae, corr, multi_acc, bi_acc, f1):
    print("MAE on test set is {}".format(mae))
    print("Correlation w.r.t human evaluation on test set is {}".format(corr))
    print("Multiclass accuracy on test set is {}".format(multi_acc))
    print("Binary accuracy on test set is {}".format(bi_acc))
    print("F1-score on test set is {}".format(f1))

def main(options):
    DTYPE = torch.FloatTensor

    # parse the input args
    run_id = options['run_id']
    epochs = options['epochs']
    model_path = options['model_path']
    output_path = options['output_path']
    signiture = options['signiture']
    patience = options['patience']
    output_dim = 1

    print("Training initializing... Setup ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(
        model_path, "model_{}_{}.pt".format(signiture, run_id))
    output_path = os.path.join(
        output_path, "results_{}_{}.csv".format(signiture, run_id))
    print("Temp location for models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))

    train_set, valid_set, test_set, input_dims = preprocess(options)


    batch_sz = 16
    model = TFN(input_dims, (8, 32, 256), 128, (0.3, 0.2, 0.2, 0.5), output_dim, 4)
    params_ = list(model.parameters())


    if options['cuda']:
        model = model.cuda()
        DTYPE = torch.cuda.FloatTensor
    print("Model initialized")
    criterion = nn.L1Loss(size_average=False)
    factors = list(model.parameters())[:3]
    other = list(model.parameters())[5:]
    optimizer = optim.Adam([{"params": factors, "lr": 0.0003}, {"params": other, "lr": 0.0003}], weight_decay=0.001) # don't optimize the first 2 params, they should be fixed (output_range and shift)

    # setup training
    complete = True
    min_valid_loss = float('Inf')
    train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True)
    valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
    curr_patience = patience
    train_time = []
    train_loss_list = []
    test_loss_list = []
    for e in range(epochs):
        model.train()
        model.zero_grad()
        avg_train_loss = 0.0
        start = time.clock()
        for batch in train_iterator:
            model.zero_grad()

            # the provided data has format [batch_size, seq_len, feature_dim] or [batch_size, 1, feature_dim]
            x = batch[:-1]
            x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
            x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
            x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
            y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
            output = model(x_a, x_v, x_t)
            loss = criterion(output, y)
            loss.backward()
            avg_loss = loss.data[0]
            avg_train_loss += avg_loss / len(train_set)
            optimizer.step()
        elapsed = (time.clock() - start)
        print(elapsed)
        train_time.append(elapsed)
        train_loss_list.append(avg_train_loss)
        print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

        # Terminate the training process if run into NaN
        if np.isnan(avg_train_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        # On validation set we don't have to compute metrics other than MAE and accuracy
        model.eval()
        for batch in valid_iterator:
            x = batch[:-1]
            x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
            x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
            x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
            y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
            output = model(x_a, x_v, x_t)
            valid_loss = criterion(output, y)
            avg_valid_loss = valid_loss.data[0] 
        output_valid = output.cpu().data.numpy().reshape(-1, output_dim)
        y = y.cpu().data.numpy().reshape(-1, output_dim)

        if np.isnan(avg_valid_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        # valid_binacc = accuracy_score(output_valid>=0, y>=0)

        print("Validation loss is: {}".format(avg_valid_loss / len(valid_set)))
        # print("Validation binary accuracy is: {}".format(valid_binacc))

        if (avg_valid_loss < min_valid_loss):
            curr_patience = patience
            min_valid_loss = avg_valid_loss
            torch.save(model, model_path)
            print("Found new best model, saving to disk...")
        else:
            curr_patience -= 1
        
        if curr_patience <= 0:
            break
        print("\n\n")

        model.eval()
        for batch in test_iterator:
            x = batch[:-1]
            x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
            x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
            x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
            y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
            output_test = model(x_a, x_v, x_t)
            loss_test = criterion(output_test, y)
            avg_test_loss = loss_test.data[0] / len(test_set)

        output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
        y = y.cpu().data.numpy().reshape(-1, output_dim)

        # these are the needed metrics
        output_test = output_test.reshape((len(output_test),))
        y = y.reshape((len(y),))
        mae = np.mean(np.absolute(output_test-y))
        test_loss_list.append(avg_test_loss)
    return train_loss_list, test_loss_list
    train_avg = np.mean(train_time)

    test_loss = []
    if complete:
        
        best_model = torch.load(model_path)
        best_model.eval()
        start = time.clock()
        for batch in test_iterator:
            x = batch[:-1]
            x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
            x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
            x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
            y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
            output_test = model(x_a, x_v, x_t)
            loss_test = criterion(output_test, y)
            avg_test_loss = loss_test.data[0]
        test_time = (time.clock() - start)
        print("train_time:", len(train_set) / train_avg, "IPS")
        print("test_time:", len(test_set)/ test_time, "IPS")

        output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
        y = y.cpu().data.numpy().reshape(-1, output_dim)

        # these are the needed metrics
        output_test = output_test.reshape((len(output_test),))
        y = y.reshape((len(y),))
        mae = np.mean(np.absolute(output_test-y))
        corr = round(np.corrcoef(output_test,y)[0][1],5)
        multi_acc = round(sum(np.round(output_test)==np.round(y))/float(len(y)),5)
        true_label = (y >= 0)
        predicted_label = (output_test >= 0)
        bi_acc = accuracy_score(true_label, predicted_label)
        f1 = f1_score(true_label, predicted_label, average='weighted')
        print(multi_acc)
        print(bi_acc)
     
    return

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--dataset', dest='dataset',
                         type=str, default='MOSI')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=500)
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='mosi')     
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results')
    OPTIONS.add_argument('--max_len', dest='max_len', type=int, default=20)
    PARAMS = vars(OPTIONS.parse_args())

    train_loss_list, test_loss_list = main(PARAMS)
    with open("overfit_lrf.csv", 'w+') as out:
        writer = csv.writer(out)
        writer.writerow(train_loss_list)
        writer.writerow(test_loss_list)