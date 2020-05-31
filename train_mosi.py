from __future__ import print_function
from model import LMF
from utils import total, load_mosi
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv


def display(mae, corr, multi_acc, bi_acc, f1):
    print("MAE on test set is {}".format(mae))
    print("Correlation w.r.t human evaluation on test set is {}".format(corr))
    print("Multiclass accuracy on test set is {}".format(multi_acc))
    print("Binary accuracy on test set is {}".format(bi_acc))
    print("F1-score on test set is {}".format(f1))


def main(options):
    DTYPE = torch.FloatTensor

    # parse the input args
    run_id = options["run_id"]
    epochs = options["epochs"]
    data_path = options["data_path"]
    model_path = options["model_path"]
    output_path = options["output_path"]
    signiture = options["signiture"]
    patience = options["patience"]
    output_dim = options["output_dim"]

    print("Training initializing... Setup ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(model_path, "model_{}_{}.pt".format(signiture, run_id))
    output_path = os.path.join(
        output_path, "results_{}_{}.csv".format(signiture, run_id)
    )
    print("Temp location for models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_set, valid_set, test_set, input_dims = load_mosi(data_path)

    params = dict()
    params["audio_hidden"] = [4, 8, 16]
    params["video_hidden"] = [4, 8, 16]
    params["text_hidden"] = [64, 128, 256]
    params["audio_dropout"] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params["video_dropout"] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params["text_dropout"] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params["factor_learning_rate"] = [0.0003, 0.0005, 0.001, 0.003]
    params["learning_rate"] = [0.0003, 0.0005, 0.001, 0.003]
    params["rank"] = [1, 4, 8, 16]
    params["batch_size"] = [4, 8, 16, 32, 64, 128]
    params["weight_decay"] = [0, 0.001, 0.002, 0.01]

    total_settings = total(params)

    print(
        "There are {} different hyper-parameter settings in total.".format(
            total_settings
        )
    )

    seen_settings = set()

    with open(output_path, "w+") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "audio_hidden",
                "video_hidden",
                "text_hidden",
                "audio_dropout",
                "video_dropout",
                "text_dropout",
                "factor_learning_rate",
                "learning_rate",
                "rank",
                "batch_size",
                "weight_decay",
                "Best Validation MAE",
                "Test MAE",
                "Test Corr",
                "Test multiclass accuracy",
                "Test binary accuracy",
                "Test f1_score",
            ]
        )

    for i in range(total_settings):

        ahid = random.choice(params["audio_hidden"])
        vhid = random.choice(params["video_hidden"])
        thid = random.choice(params["text_hidden"])
        thid_2 = thid // 2
        adr = random.choice(params["audio_dropout"])
        vdr = random.choice(params["video_dropout"])
        tdr = random.choice(params["text_dropout"])
        factor_lr = random.choice(params["factor_learning_rate"])
        lr = random.choice(params["learning_rate"])
        r = random.choice(params["rank"])
        batch_sz = random.choice(params["batch_size"])
        decay = random.choice(params["weight_decay"])

        # reject the setting if it has been tried
        current_setting = (
            ahid,
            vhid,
            thid,
            adr,
            vdr,
            tdr,
            factor_lr,
            lr,
            r,
            batch_sz,
            decay,
        )
        if current_setting in seen_settings:
            continue
        else:
            seen_settings.add(current_setting)

        model = LMF(
            input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r
        )
        if options["cuda"]:
            model = model.cuda()
            DTYPE = torch.cuda.FloatTensor
        print("Model initialized")
        criterion = nn.L1Loss(size_average=False)
        factors = list(model.parameters())[:3]
        other = list(model.parameters())[3:]
        optimizer = optim.Adam(
            [{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}],
            weight_decay=decay,
        )

        # setup training
        complete = True
        min_valid_loss = float("Inf")
        train_iterator = DataLoader(
            train_set, batch_size=batch_sz, num_workers=4, shuffle=True
        )
        valid_iterator = DataLoader(
            valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True
        )
        test_iterator = DataLoader(
            test_set, batch_size=len(test_set), num_workers=4, shuffle=True
        )
        curr_patience = patience

        for e in range(epochs):
            model.train()
            model.zero_grad()
            avg_train_loss = 0.0
            for batch in train_iterator:
                model.zero_grad()

                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(
                    batch[-1].view(-1, output_dim).float().type(DTYPE),
                    requires_grad=False,
                )
                output = model(x_a, x_v, x_t)
                loss = criterion(output, y)
                loss.backward()
                avg_loss = loss.data[0]
                avg_train_loss += avg_loss / len(train_set)
                optimizer.step()

            print(
                "Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss)
            )

            # Terminate the training process if run into NaN
            if np.isnan(avg_train_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            model.eval()
            for batch in valid_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(
                    batch[-1].view(-1, output_dim).float().type(DTYPE),
                    requires_grad=False,
                )
                output = model(x_a, x_v, x_t)
                valid_loss = criterion(output, y)
                avg_valid_loss = valid_loss.data[0]
            y = y.cpu().data.numpy().reshape(-1, output_dim)

            if np.isnan(avg_valid_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            avg_valid_loss = avg_valid_loss / len(valid_set)
            print("Validation loss is: {}".format(avg_valid_loss))

            if avg_valid_loss < min_valid_loss:
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
                y = Variable(
                    batch[-1].view(-1, output_dim).float().type(DTYPE),
                    requires_grad=False,
                )
                output_test = model(x_a, x_v, x_t)
                loss_test = criterion(output_test, y)
                avg_test_loss = loss_test.data[0] / len(test_set)

            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)

            # these are the needed metrics
            output_test = output_test.reshape((len(output_test),))
            y = y.reshape((len(y),))
            mae = np.mean(np.absolute(output_test - y))

        if complete:

            best_model = torch.load(model_path)
            best_model.eval()
            for batch in test_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(
                    batch[-1].view(-1, output_dim).float().type(DTYPE),
                    requires_grad=False,
                )
                output_test = best_model(x_a, x_v, x_t)
                loss_test = criterion(output_test, y)

            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)

            # these are the needed metrics
            output_test = output_test.reshape((len(output_test),))
            y = y.reshape((len(y),))
            mae = np.mean(np.absolute(output_test - y))
            corr = round(np.corrcoef(output_test, y)[0][1], 5)
            multi_acc = round(
                sum(np.round(output_test) == np.round(y)) / float(len(y)), 5
            )
            true_label = y >= 0
            predicted_label = output_test >= 0
            bi_acc = accuracy_score(true_label, predicted_label)
            f1 = f1_score(true_label, predicted_label, average="weighted")
            display(mae, corr, multi_acc, bi_acc, f1)

            with open(output_path, "a+") as out:
                writer = csv.writer(out)
                writer.writerow(
                    [
                        ahid,
                        vhid,
                        thid,
                        adr,
                        vdr,
                        tdr,
                        factor_lr,
                        lr,
                        r,
                        batch_sz,
                        decay,
                        min_valid_loss.cpu().data.numpy(),
                        mae,
                        corr,
                        multi_acc,
                        bi_acc,
                        f1,
                    ]
                )


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument("--run_id", dest="run_id", type=int, default=1)
    OPTIONS.add_argument("--epochs", dest="epochs", type=int, default=500)
    OPTIONS.add_argument("--patience", dest="patience", type=int, default=20)
    OPTIONS.add_argument("--output_dim", dest="output_dim", type=int, default=1)
    OPTIONS.add_argument("--signiture", dest="signiture", type=str, default="mosi")
    OPTIONS.add_argument("--cuda", dest="cuda", type=bool, default=False)
    OPTIONS.add_argument("--data_path", dest="data_path", type=str, default="./data/")
    OPTIONS.add_argument("--model_path", dest="model_path", type=str, default="models")
    OPTIONS.add_argument(
        "--output_path", dest="output_path", type=str, default="results"
    )
    OPTIONS.add_argument("--max_len", dest="max_len", type=int, default=20)
    PARAMS = vars(OPTIONS.parse_args())

    main(PARAMS)
