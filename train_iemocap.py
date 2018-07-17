from __future__ import print_function
from model import LMF
from utils import MultimodalDataset, total
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report, confusion_matrix
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import cPickle as pickle



def load_iemocap(emotion, verbose = False):

    timestamps = "absolute"  # absolute or relative, relative will output features relative to segment time

    root = "/media/bighdd4/Paul/mosi2/experiments/iemocap/"
    labels = pickle.load(open(root + "labels.p", "rb"))
    e_labels = pickle.load(open(root + "e_labels.p", "rb"))
    happy_labels = pickle.load(open(root + "happy_labels.p", "rb"))
    angry_labels = pickle.load(open(root + "angry_labels.p", "rb"))
    sad_labels = pickle.load(open(root + "sad_labels.p", "rb"))
    neutral_labels = pickle.load(open(root + "neutral_labels.p", "rb"))
    fru_labels = pickle.load(open(root + "fru_labels.p", "rb"))
    exc_labels = pickle.load(open(root + "exc_labels.p", "rb"))
    text_dict = pickle.load(open(root + "text_dict.p", "rb"))
    audio_dict = pickle.load(open(root + "audio_dict.p", "rb"))
    video_dict = pickle.load(open(root + "video_dict.p", "rb"))

    for video_id in e_labels:
        if video_id not in text_dict:
            print(video_id)

    train_vids = [video_id for video_id in text_dict if
                  'Ses03' in video_id or 'Ses04' in video_id or 'Ses05' in video_id]
    valid_vids = [video_id for video_id in text_dict if 'Ses02' in video_id]
    test_vids = [video_id for video_id in text_dict if 'Ses01' in video_id]

    all_labels = happy_labels
    train_i = []
    for video_id in train_vids:
        for segment_id in text_dict[video_id]:
            if video_id in all_labels:
                if segment_id in all_labels[video_id]:
                    train_i.append((video_id, segment_id))
    valid_i = []
    for video_id in valid_vids:
        for segment_id in text_dict[video_id]:
            if video_id in all_labels:
                if segment_id in all_labels[video_id]:
                    valid_i.append((video_id, segment_id))
    test_i = []
    for video_id in test_vids:
        for segment_id in text_dict[video_id]:
            if video_id in all_labels:
                if segment_id in all_labels[video_id]:
                    test_i.append((video_id, segment_id))

    # assert False
    def pad(data, max_segment_len, t):
        curr = []
        try:
            dim = data.shape[1]
        except:
            if t == 1:
                return np.zeros((max_segment_len, 300))
            if t == 2:
                return np.zeros((max_segment_len, 74))
            if t == 3:
                return np.zeros((max_segment_len, 35))
        if max_segment_len >= len(data):
            for vec in data:
                curr.append(vec)
            for i in xrange(max_segment_len - len(data)):
                curr.append([0 for i in xrange(dim)])
        else:  # max_segment_len < len(text), take last max_segment_len of text
            for vec in data[len(data) - max_segment_len:]:
                curr.append(vec)
        curr = np.array(curr)
        return curr

    max_segment_len = 20

    text_train_emb = np.array(
        [pad(text_dict[video_id][segment_id], max_segment_len, 1) for (video_id, segment_id) in train_i])
    covarep_train = np.array(
        [pad(audio_dict[video_id][segment_id], max_segment_len, 2) for (video_id, segment_id) in train_i])
    facet_train = np.array(
        [pad(video_dict[video_id][segment_id], max_segment_len, 3) for (video_id, segment_id) in train_i])
    y_train = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id, segment_id) in train_i]))
    ey_train = np.nan_to_num(np.array([e_labels[video_id][segment_id] for (video_id, segment_id) in train_i]))

    text_valid_emb = np.array(
        [pad(text_dict[video_id][segment_id], max_segment_len, 1) for (video_id, segment_id) in valid_i])
    covarep_valid = np.array(
        [pad(audio_dict[video_id][segment_id], max_segment_len, 2) for (video_id, segment_id) in valid_i])
    facet_valid = np.array(
        [pad(video_dict[video_id][segment_id], max_segment_len, 3) for (video_id, segment_id) in valid_i])
    y_valid = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id, segment_id) in valid_i]))
    ey_valid = np.nan_to_num(np.array([e_labels[video_id][segment_id] for (video_id, segment_id) in valid_i]))

    text_test_emb = np.array(
        [pad(text_dict[video_id][segment_id], max_segment_len, 1) for (video_id, segment_id) in test_i])
    covarep_test = np.array(
        [pad(audio_dict[video_id][segment_id], max_segment_len, 2) for (video_id, segment_id) in test_i])
    facet_test = np.array(
        [pad(video_dict[video_id][segment_id], max_segment_len, 3) for (video_id, segment_id) in test_i])
    y_test = np.nan_to_num(np.array([labels[video_id][segment_id] for (video_id, segment_id) in test_i])) # 2717 x 3
    ey_test = np.nan_to_num(np.array([e_labels[video_id][segment_id] for (video_id, segment_id) in test_i])) # 2717 x 4


    happy_train = np.nan_to_num(np.array([happy_labels[video_id][segment_id] for (video_id, segment_id) in train_i]))
    happy_valid = np.nan_to_num(np.array([happy_labels[video_id][segment_id] for (video_id, segment_id) in valid_i]))
    happy_test = np.nan_to_num(np.array([happy_labels[video_id][segment_id] for (video_id, segment_id) in test_i]))

    angry_train = np.nan_to_num(np.array([angry_labels[video_id][segment_id] for (video_id, segment_id) in train_i]))
    angry_valid = np.nan_to_num(np.array([angry_labels[video_id][segment_id] for (video_id, segment_id) in valid_i]))
    angry_test = np.nan_to_num(np.array([angry_labels[video_id][segment_id] for (video_id, segment_id) in test_i]))

    sad_train = np.nan_to_num(np.array([sad_labels[video_id][segment_id] for (video_id, segment_id) in train_i]))
    sad_valid = np.nan_to_num(np.array([sad_labels[video_id][segment_id] for (video_id, segment_id) in valid_i]))
    sad_test = np.nan_to_num(np.array([sad_labels[video_id][segment_id] for (video_id, segment_id) in test_i]))

    neutral_train = np.nan_to_num(
        np.array([neutral_labels[video_id][segment_id] for (video_id, segment_id) in train_i]))
    neutral_valid = np.nan_to_num(
        np.array([neutral_labels[video_id][segment_id] for (video_id, segment_id) in valid_i]))
    neutral_test = np.nan_to_num(np.array([neutral_labels[video_id][segment_id] for (video_id, segment_id) in test_i]))

    facet_train_max = np.max(np.max(np.abs(facet_train), axis=0), axis=0)
    facet_train_max[facet_train_max == 0] = 1
    covarep_train_max = np.max(np.max(np.abs(covarep_train), axis=0), axis=0)
    covarep_train_max[covarep_train_max == 0] = 1

    facet_train = facet_train / facet_train_max
    facet_valid = facet_valid / facet_train_max
    facet_test = facet_test / facet_train_max
    covarep_train = covarep_train / covarep_train_max
    covarep_valid = covarep_valid / covarep_train_max
    covarep_test = covarep_test / covarep_train_max

    # average audio & video features
    covarep_train = np.mean(covarep_train, axis=1)
    covarep_valid = np.mean(covarep_valid, axis=1)
    covarep_test = np.mean(covarep_test, axis=1)

    facet_train = np.mean(facet_train, axis=1)
    facet_valid = np.mean(facet_valid, axis=1)
    facet_test = np.mean(facet_test, axis=1)

    if verbose:
        print (text_train_emb.shape)  # 2717 x 20 x 300
        print (covarep_train.shape ) # n x seq x 74
        print (facet_train.shape  )# n x seq x 35

        print (text_valid_emb.shape)  # 798 x seq x 300
        print (covarep_valid.shape ) # n x seq x 74
        print (facet_valid.shape ) # n x seq x 35

        print (text_test_emb.shape)  # 938 x seq x 300
        print (covarep_test.shape)  # n x seq x 74
        print (facet_test.shape)  # n x seq x 35

        print(happy_train.shape)
        print(happy_valid.shape)
        print(happy_test.shape)


    if emotion == 'happy':
        return facet_train, text_train_emb, covarep_train, happy_train, \
               facet_valid, text_valid_emb, covarep_valid, happy_valid, \
               facet_test, text_test_emb, covarep_test, happy_test
    if emotion == 'angry':
        return facet_train, text_train_emb, covarep_train, angry_train, \
               facet_valid, text_valid_emb, covarep_valid, angry_valid, \
               facet_test, text_test_emb, covarep_test, angry_test
    if emotion == 'sad':
        return facet_train, text_train_emb, covarep_train, sad_train, \
               facet_valid, text_valid_emb, covarep_valid, sad_valid, \
               facet_test, text_test_emb, covarep_test, sad_test
    if emotion == 'neutral':
        return facet_train, text_train_emb, covarep_train, neutral_train, \
               facet_valid, text_valid_emb, covarep_valid, neutral_valid, \
               facet_test, text_test_emb, covarep_test, neutral_test


def preprocess(emotion):
    # parse the input args
    class IEMOCAP(Dataset):
        '''
        PyTorch Dataset for IEMOCAP, don't need to change this
        '''
        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels
        
        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]


    # TODO: put the data loading code here, so you can populate xxx_audio, xxx_visual, xxx_text and xxx_labels with numpy
    # arrays. xxx_audio and xxx_visual should be of shape (num_examples, feature_dim), and xxx_text will be (num_examples, maxlen, feature_dim)
    # xxx_labels will be of shape (num_examples, output_dim)
    
    train_visual, train_text, train_audio, train_labels, \
    valid_visual, valid_text, valid_audio, valid_labels, \
    test_visual, test_text, test_audio, test_labels = load_iemocap(emotion)

    # code that instantiates the Dataset objects
    train_set = IEMOCAP(train_audio, train_visual, train_text, train_labels)
    valid_set = IEMOCAP(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = IEMOCAP(test_audio, test_visual, test_text, test_labels)


    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims

def display(f1_score, accuracy_score):
    print("F1-score on test set is {}".format(f1_score))
    print("Accuracy score on test set is {}".format(accuracy_score))

def main(options):
    DTYPE = torch.FloatTensor
    LONG = torch.LongTensor

    # parse the input args
    epochs = options['epochs']
    model_path = options['model_path']
    output_path = options['output_path']
    signiture = options['signiture']
    patience = options['patience']
    emotion = options['emotion']
    output_dim = 2

    # print("Training initializing... Setup ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(
        model_path, "model_{}_{}.pt".format(signiture, emotion))
    output_path = os.path.join(
        output_path, "results_{}_{}.csv".format(signiture, emotion))
    print("Temp location for models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))

    train_set, valid_set, test_set, input_dims = preprocess(emotion)


    params = dict()
    params['audio_hidden'] = [4, 8, 16, 32]
    params['video_hidden'] = [4, 8, 16]
    params['text_hidden'] = [64, 128, 256]
    params['audio_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['video_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['text_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['rank'] = [1, 4, 8, 16]
    params['batch_size'] = [8, 16, 32, 64, 128]
    params['weight_decay'] = [0, 0.001, 0.002, 0.01]

    total_settings = total(params)

    print("There are {} different hyper-parameter settings in total.".format(total_settings))

    seen_settings = set()

    if not os.path.isfile(output_path):
        with open(output_path, 'w+') as out:
            writer = csv.writer(out)
            writer.writerow(["audio_hidden", "video_hidden", 'text_hidden', 'audio_dropout', 'video_dropout', 'text_dropout',
                            'factor_learning_rate', 'learning_rate', 'rank', 'batch_size', 'weight_decay', 
                            'Best Validation CrossEntropyLoss', 'Test CrossEntropyLoss', 'Test F1-score', 'Test Accuracy Score'])

    for i in range(total_settings):

        ahid = random.choice(params['audio_hidden'])
        vhid = random.choice(params['video_hidden'])
        thid = random.choice(params['text_hidden'])
        thid_2 = thid / 2
        adr = random.choice(params['audio_dropout'])
        vdr = random.choice(params['video_dropout'])
        tdr = random.choice(params['text_dropout'])
        factor_lr = random.choice(params['factor_learning_rate'])
        lr = random.choice(params['learning_rate'])
        r = random.choice(params['rank'])
        batch_sz = random.choice(params['batch_size'])
        decay = random.choice(params['weight_decay'])

        # reject the setting if it has been tried
        current_setting = (ahid, vhid, thid, adr, vdr, tdr, factor_lr, lr, r, batch_sz, decay)
        if current_setting in seen_settings:
            continue
        else:
            seen_settings.add(current_setting)

        model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
        if options['cuda']:
            model = model.cuda()
            DTYPE = torch.cuda.FloatTensor
            LONG = torch.cuda.LongTensor
        print("Model initialized")
        criterion = nn.CrossEntropyLoss(size_average=False)
        factors = list(model.parameters())[:3]
        other = list(model.parameters())[5:]
        optimizer = optim.Adam([{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}], weight_decay=decay) # don't optimize the first 2 params, they should be fixed (output_range and shift)

        # setup training
        complete = True
        min_valid_loss = float('Inf')
        train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True)
        valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
        test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
        curr_patience = patience
        for e in range(epochs):
            model.train()
            model.zero_grad()
            avg_train_loss = 0.0
            for batch in train_iterator:
                model.zero_grad()

                # the provided data has format [batch_size, seq_len, feature_dim] or [batch_size, 1, feature_dim]
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                try:
                    output = model(x_a, x_v, x_t)
                except ValueError as e:
                    print(x_a.data.shape)
                    print(x_v.data.shape)
                    print(x_t.data.shape)
                    raise e
                loss = criterion(output, torch.max(y, 1)[1])
                loss.backward()
                avg_loss = loss.data[0]
                avg_train_loss += avg_loss / len(train_set)
                optimizer.step()


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
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                output = model(x_a, x_v, x_t)
                valid_loss = criterion(output, torch.max(y, 1)[1])
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

        if complete:
            
            best_model = torch.load(model_path)
            best_model.eval()
            for batch in test_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False)
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False)
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                output_test = model(x_a, x_v, x_t)
                loss_test = criterion(output_test, torch.max(y, 1)[1])
                test_loss = loss_test.data[0]
            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)

            # these are the needed metrics
            all_true_label = np.argmax(y,axis=1)
            all_predicted_label = np.argmax(output_test,axis=1)

            f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
            acc_score = accuracy_score(all_true_label, all_predicted_label)

            display(f1, acc_score)

            with open(output_path, 'a+') as out:
                writer = csv.writer(out)
                writer.writerow([ahid, vhid, thid, adr, vdr, tdr, factor_lr, lr, r, batch_sz, decay, 
                                min_valid_loss / len(valid_set), test_loss / len(test_set), f1, acc_score])


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    # OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--emotion', dest='emotion', type=str, default='happy')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=100)
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='')
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=False)
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results')
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
