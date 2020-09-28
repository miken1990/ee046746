import torch
import numpy as np
import time
import os


class Trainer:
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    # training loop
    def train(self, num_epochs, dl_train, dl_val, early_stopping=None):
        epochs_without_improvement = 0
        train_acc_list = []
        val_acc_list = []
        best_val_acc = 0.
        for epoch in range(1, num_epochs + 1):
            self.model.train()  # put in training mode
            running_epoch_loss = 0.0
            epoch_time = time.time()
            for data in dl_train:
                # get the inputs
                inputs, labels = data
                # send them to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs) # forward pass
                loss = self.loss_fn(outputs, labels) # calculate the loss
                # always the same 3 steps
                self.optimizer.zero_grad()  # zero the parameter gradients
                loss.backward()  # backpropagation
                self.optimizer.step()  # update parameters
            # print statistics
            running_epoch_loss += loss.data.item()
            # Normalizing the loss by the total number of train batches
            running_epoch_loss /= len(dl_train)
            # Calculate training/test set accuracy of the existing model
            train_accuracy, _ = self.calculate_accuracy(self.model, dl_train, self.device)
            train_acc_list.append(train_accuracy)
            cur_epoch_val_accuracy, _ = self.calculate_accuracy(self.model, dl_val, self.device)
            val_acc_list.append(cur_epoch_val_accuracy)
            log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch
            , running_epoch_loss, train_accuracy, cur_epoch_val_accuracy)
            epoch_time = time.time() - epoch_time
            log += "Epoch Time: {:.2f} secs".format(epoch_time)
            print(log)

            if len(val_acc_list) > 1 and cur_epoch_val_accuracy < val_acc_list[-2]:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

            if early_stopping is not None and epochs_without_improvement == early_stopping:
                break

            if cur_epoch_val_accuracy > best_val_acc:
                best_val_acc = cur_epoch_val_accuracy
                if not os.path.isdir('part1/checkpoints'):
                    os.mkdir('part1/checkpoints')
                # save model
                state = {
                    'net': self.model.state_dict(),
                    'epoch': epoch,
                }
                print('saving model')
                torch.save(state, 'part1/checkpoints/svhn_cnn.pth')

        print('==> Finished Training ...')

    def test(self, dl_test):
        train_acc_list = []
        val_acc_list = []
        running_epoch_loss = 0.0
        epoch_time = time.time()
        with torch.no_grad():
            self.model.eval()
            for data in dl_test:
                # get the inputs
                inputs, labels = data
                # send them to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = self.model(inputs)  # forward pass
                loss = self.loss_fn(outputs, labels)  # calculate the loss
            # print statistics
            running_epoch_loss += loss.data.item()
            # Normalizing the loss by the total number of train batches
            running_epoch_loss /= len(dl_test)
            # Calculate training/test set accuracy of the existing model
            test_accuracy, _ = self.calculate_accuracy(self.model, dl_test, self.device)
            train_acc_list.append(test_accuracy)
            cur_epoch_val_accuracy, _ = self.calculate_accuracy(self.model, dl_test, self.device)
            val_acc_list.append(cur_epoch_val_accuracy)
            log = "Test: | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(
                                                                                                         running_epoch_loss,
                                                                                                         test_accuracy,
                                                                                                         cur_epoch_val_accuracy)
            epoch_time = time.time() - epoch_time
            log += "Test Time: {:.2f} secs".format(epoch_time)
            print(log)

        print('==> Finished Test ...')

    # function to calcualte accuracy of the model
    @staticmethod
    def calculate_accuracy(model, dataloader, device):
        model.eval() # put in evaluation mode
        total_correct = 0
        total_images = 0
        confusion_matrix = np.zeros([10,10], int)
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_images += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                for i, l in enumerate(labels):
                    confusion_matrix[l.item(), predicted[i].item()] += 1
        model_accuracy = (total_correct / total_images) * 100
        return model_accuracy, confusion_matrix