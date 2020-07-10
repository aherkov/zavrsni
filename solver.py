import numpy as np
import torch
from torch.autograd import Variable
from relaynet_pytorch.losses import CombinedLoss
from torch.optim import lr_scheduler
import os


def per_class_dice(y_pred, y_true, num_class, list):
    y_pred = y_pred.data.cpu().numpy()  # [batch_size, 1024, 512]
    y_true = y_true.data.cpu().numpy()  # [batch_size, 1024, 512]

    pom = []
    avgdice = 0
    for i in range(num_class):
        inter = np.sum(y_pred[y_true == i] == i)
        union = np.sum(y_pred[y_pred == i] == i) + np.sum(y_true[y_true == i] == i)
        dice = 2 * inter / union
        pom.append(dice)
        avgdice += dice
    list.append(pom)
    return avgdice / num_class


class Solver(object):
    default_optim_args = {"lr": 1e-2,
                          "betas": (0.9, 0.999),
                          "eps": 1e-8,
                          "weight_decay": 0.0001}
    gamma = 0.5
    step_size = 10
    NumClass = 4

    def __init__(self, optim=torch.optim.Adam, optim_args=None, loss_func=CombinedLoss()):
        if optim_args is None:
            optim_args = {}
        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self.train_loss_batches_history = []
        self.train_loss_history = []

        self.val_loss_batches_history = []
        self.val_loss_history = []

        self.dice_class_history = []
        self.dice_score_history = []

    def train(self, experiment, model, train_loader, val_loader, test_loader, exp_dir_name='exp_default'):
        optim = self.optim(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

        # decay LR by a factor of 0.1 every 30 epochs
        scheduler = lr_scheduler.StepLR(optim, step_size=self.step_size, gamma=self.gamma)

        self.train_loss_history = []
        self.val_loss_history = []

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAINING\n')
        if not os.path.exists('models/' + exp_dir_name):
            os.makedirs('models/' + exp_dir_name)
        epoch_count = 0
        while True:
            model.train()
            self.train_loss_batches_history = []
            epoch_count += 1

            for i_batch, sample_batched in enumerate(train_loader):
                X = Variable(sample_batched[0].float())  # [batch_size, 1, 1024, 512]
                y = Variable(sample_batched[1].float())  # [batch_size, 1024, 512]
                w = Variable(sample_batched[2].float())  # [batch_size, 1024, 512]

                if model.is_cuda:
                    X, y, w = X.cuda(), y.cuda(), w.cuda()

                optim.zero_grad()
                output = model(X)  # [batch_size, 4, 1024, 512]
                loss = self.loss_func(output, y, w)
                loss.backward()
                optim.step()
                loss_data = loss.data.item()
                self.train_loss_batches_history.append(loss_data)
                print('Batch ' + str(i_batch + 1) + '/' + str(len(train_loader)) + ' train loss: ' + str(loss_data))

            scheduler.step()
            print("START EPOCH VALIDATION")
            avg_dice = 0
            model.eval()
            self.val_loss_batches_history = []
            self.dice_class_history = []
            with torch.no_grad():
                for i_batch, sample_batched in enumerate(val_loader):
                    X = Variable(sample_batched[0].float())
                    y = Variable(sample_batched[1].float())

                    if model.is_cuda:
                        X, y = X.cuda(), y.cuda()

                    output = model(X)  # [batch_size, 4, 1024, 512]
                    loss = self.loss_func(output, y, None)
                    loss_data = loss.data.item()
                    self.val_loss_batches_history.append(loss_data)
                    print('Batch ' + str(i_batch + 1) + '/' + str(len(val_loader)) + ' val loss: ' + str(loss_data))
                    experiment.log_metric("val_loss", loss_data)
                    _, batch_output = torch.max(output, dim=1)  # [batch_size, 1024, 512], [batch_size, 1024, 512]
                    # _ vjerojatnosti po klasama, batch_output pripadnosti klasama
                    avg_dice += per_class_dice(batch_output, y, self.NumClass, self.dice_class_history)
            print("END EPOCH VALIDATION")

            avg_dice_val = avg_dice / len(val_loader)
            self.dice_score_history.append(avg_dice_val)

            avg_loss_train = sum(self.train_loss_batches_history) / len(self.train_loss_batches_history)
            self.train_loss_history.append(avg_loss_train)

            avg_loss_val = sum(self.val_loss_batches_history) / len(self.val_loss_batches_history)
            self.val_loss_history.append(avg_loss_val)

            avg_dice_per_class = [0, 0, 0, 0]
            for i in range(len(val_loader)):
                for j in range(self.NumClass):
                    avg_dice_per_class[j] += self.dice_class_history[i][j]
            avg_dice_bkg = avg_dice_per_class[0] / len(val_loader)
            avg_dice_ipl = avg_dice_per_class[1] / len(val_loader)
            avg_dice_inl = avg_dice_per_class[2] / len(val_loader)
            avg_dice_ped = avg_dice_per_class[3] / len(val_loader)

            print('Epoch [' + str(epoch_count) + '] train loss: ' + str(avg_loss_train))
            print('Epoch [' + str(epoch_count) + '] val loss: ' + str(avg_loss_val))
            print('Epoch [' + str(epoch_count) + '] average dice: ' + str(avg_dice_val))

            print("Vitreous Dice Score: " + str(avg_dice_bkg))
            print("Inner Plexiform Layer Dice Score: " + str(avg_dice_ipl))
            print("Inner Nuclear Layer Dice Score: " + str(avg_dice_inl))
            print("Pigment Epithelial Detachment Dice Score: " + str(avg_dice_ped))

            experiment.log_metric("Training Loss", avg_loss_train, epoch=epoch_count)
            experiment.log_metric("Validation Loss", avg_loss_val, epoch=epoch_count)
            experiment.log_metric("Average Dice Score", avg_dice_val, epoch=epoch_count)
            experiment.log_metric("Vitreous Dice Score", avg_dice_bkg, epoch=epoch_count)
            experiment.log_metric("Inner Plexiform Layer Dice Score", avg_dice_ipl, epoch=epoch_count)
            experiment.log_metric("Inner Nuclear Layer Dice Score", avg_dice_inl, epoch=epoch_count)
            experiment.log_metric("Pigment Epithelial Detachment Dice Score", avg_dice_ped, epoch=epoch_count)

            last_five_epochs_val = self.dice_score_history[-5:]
            if epoch_count > 20:
                worse = True
                for dice in last_five_epochs_val:
                    if avg_dice_val > dice:
                        worse = False
                if len(last_five_epochs_val) < 5:
                    worse = False
                if worse:
                    print("Dice Score not getting better. Stopping training.")
                    break
            model.save('models/' + exp_dir_name + '/relaynet_epoch' + str(epoch_count) + '.model')

        print('FINISH')
        print("Calculating Dice on test data...")
        self.dice_class_history = []
        avg_dice = 0
        for i_batch, sample_batched in enumerate(test_loader):
            X = Variable(sample_batched[0].float())
            y = Variable(sample_batched[1].float())
            if model.is_cuda:
                X, y = X.cuda(), y.cuda()
            output = model(X)
            _, val_preds = torch.max(output, dim=1)
            avg_dice += per_class_dice(val_preds, y, self.NumClass, self.dice_class_history)
        avg_dice_test = avg_dice / len(test_loader)
        avg_dice_per_class = [0, 0, 0, 0]
        for i in range(len(test_loader)):
            for j in range(self.NumClass):
                avg_dice_per_class[j] += self.dice_class_history[i][j]
        avg_dice_bkg = avg_dice_per_class[0] / len(test_loader)
        avg_dice_ipl = avg_dice_per_class[1] / len(test_loader)
        avg_dice_inl = avg_dice_per_class[2] / len(test_loader)
        avg_dice_ped = avg_dice_per_class[3] / len(test_loader)

        print("Testing Dataset average dice: " + str(avg_dice_test))
        print("Vitreous Dice Score: " + str(avg_dice_bkg))
        print("Inner Plexiform Layer Dice Score: " + str(avg_dice_ipl))
        print("Inner Nuclear Layer Dice Score: " + str(avg_dice_inl))
        print("Pigment Epithelial Detachment Dice Score: " + str(avg_dice_ped))

        experiment.log_metric("Test Vitreous", avg_dice_bkg)
        experiment.log_metric("Test Inner Plexiform Layer", avg_dice_ipl)
        experiment.log_metric("Test Inner Nuclear Layer", avg_dice_inl)
        experiment.log_metric("Test Pigment Epithelial Detachment", avg_dice_ped)
        experiment.end()
