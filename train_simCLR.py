import torch
import torch.nn.functional as F
import os
import numpy as np
import time
import copy

class SimCLR(object):
    def __init__(self, model, optimizer, scheduler, log_dir, stratified, device, temperature, epochs_pretrain, max_tol_pretrain):
        self.device = device
        self.temperature = temperature
        self.epochs_pretrain = epochs_pretrain
        self.max_tol_pretrain = max_tol_pretrain
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.stratified = stratified
        self.log_dir = log_dir

    def info_nce_loss(self, features, stratified):
        # print(features.shape)
        bs = int(features.shape[0] // 2)
        labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # Normlize the features according to subject
        if stratified == 'stratified':
            features_str = features.clone()
            features_str[:bs, :] = (features[:bs, :] -  features[:bs, :].mean(
                dim=0)) / (features[:bs, :].std(dim=0) + 1e-3)
            features_str[bs:, :] = (features[bs:, :] -  features[bs:, :].mean(
                dim=0)) / (features[bs:, :].std(dim=0) + 1e-3)
            features = F.normalize(features_str, dim=1)
        elif stratified == 'bn':
            features_str = features.clone()
            features_str = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-3)
            features = F.normalize(features_str, dim=1)
        elif stratified == 'no':
            features = F.normalize(features, dim=1)
        elif stratified == 'minmax':
            f_max = features.max(dim=1, keepdim=True)[0]
            f_min = features.min(dim=1, keepdim=True)[0]
            features = (features - f_min) / (f_max - f_min)
            features = features * (f_max - f_min) + f_min
            features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Put the positive column at the end (when all the entries are the same, the top1 acc will be 0; while if the
        # positive column is at the start, the top1 acc might be exaggerated)
        logits = torch.cat([negatives, positives], dim=1)
        # The label means the last column contain the positive pairs
        labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
        labels = labels.to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, val_loader, print_paramStats):
        n_iter = 0

        bad_count = 0
        best_acc, best_loss = -1, 1000
        model_epochs, optimizer_epochs = {}, {}
        train_top1_history, val_top1_history = np.zeros(self.epochs_pretrain), np.zeros(self.epochs_pretrain)
        train_top5_history, val_top5_history = np.zeros(self.epochs_pretrain), np.zeros(self.epochs_pretrain)
        train_loss_history, val_loss_history = np.zeros(self.epochs_pretrain), np.zeros(self.epochs_pretrain)
        for epoch_counter in range(self.epochs_pretrain):

            if print_paramStats:
                timeParams = np.squeeze(self.model.timeConv.weight.detach().cpu().numpy())
                print('timeParams max:', np.max(abs(timeParams), axis=1))
                spatialParams = np.squeeze(self.model.spatialConv.weight.detach().cpu().numpy())
                print('spatialParams max:', np.max(abs(spatialParams), axis=1))

            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_acc5 = 0
            self.model.train()
            for count, data in enumerate(train_loader):
                data = data.to(self.device)
                features = self.model(data)
                logits, labels = self.info_nce_loss(features, self.stratified)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                top1, top5 = accuracy(logits, labels, topk=(1,5))

                n_iter += 1

                train_loss = train_loss + loss.data.cpu().numpy()
                train_acc = train_acc + top1[0].cpu().numpy()
                train_acc5 = train_acc5 + top5[0].cpu().numpy()

            train_loss = train_loss / (count + 1)
            train_acc = train_acc / (count + 1)
            train_acc5 = train_acc5 / (count + 1)

            val_loss = 0
            val_acc = 0
            val_acc5 = 0
            self.model.eval()
            for count, data in enumerate(val_loader):
                data = data.to(self.device)

                features = self.model(data)
                logits, labels = self.info_nce_loss(features, self.stratified)
                loss = self.criterion(logits, labels)

                top1, top5 = accuracy(logits, labels, topk=(1,5))

                val_loss = val_loss + loss.data.cpu().numpy()
                val_acc = val_acc + top1[0].cpu().numpy()
                val_acc5 = val_acc5 + top5[0].cpu().numpy()

            val_loss = val_loss / (count + 1)
            val_acc = val_acc / (count + 1)
            val_acc5 = val_acc5 / (count + 1)

            train_top1_history[epoch_counter] = train_acc
            val_top1_history[epoch_counter] = val_acc
            train_top5_history[epoch_counter] = train_acc5
            val_top5_history[epoch_counter] = val_acc5
            train_loss_history[epoch_counter] = train_loss
            val_loss_history[epoch_counter] = val_loss

            # Revise the bug
            model_epochs[epoch_counter] = copy.deepcopy(self.model)
            optimizer_epochs[epoch_counter] = copy.deepcopy(self.optimizer)
            
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            
            

            # warmup for the first 10 epochs
            # if epoch_counter >= 10:
            # No warmup
            self.scheduler.step()
            print(f"Epoch: {epoch_counter}   Train loss: {train_loss}   Top1 accuracy: {train_acc}   Top5 accuracy: {train_acc5}")
            print(
                f"\tVal loss: {val_loss}   Top1 accuracy: {val_acc}   Top5 accuracy: {val_acc5}")
            # print('learning rate', self.scheduler.get_lr()[0])
            # print('logits:', logits[0])

            if val_acc > best_acc:
                bad_count = 0
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch_counter
            else:
                bad_count += 1

            if bad_count > self.max_tol_pretrain:
                break

            end_time = time.time()
            print('time consumed:', end_time - start_time)

        self.best_model = model_epochs[best_epoch]
        self.best_optimizer = optimizer_epochs[best_epoch]

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
        torch.save({
            'epoch': epoch_counter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.log_dir, checkpoint_name))
            
        # save model checkpoints
        checkpoint_name = 'checkpoint_best_{:04d}.pth.tar'.format(best_epoch)
        # torch.save(self.best_model, os.path.join(self.log_dir, checkpoint_name))
        torch.save({
            'epoch': best_epoch,
            'state_dict': self.best_model.state_dict(),
            'optimizer': self.best_optimizer.state_dict(),
        }, os.path.join(self.log_dir, checkpoint_name))

        # checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
        # # torch.save(model_epochs[epoch_counter], os.path.join(self.log_dir, checkpoint_name))
        # torch.save({
        #     'epoch': epoch_counter,
        #     'state_dict': model_epochs[epoch_counter].state_dict(),
        #     'optimizer': optimizer_epochs[epoch_counter].state_dict(),
        # }, os.path.join(self.log_dir, checkpoint_name))

        print('best epoch: %d, train top1 acc:%.3f, top5 acc:%.3f; val top1 acc:%.3f, top5 acc:%.3f, train loss:%.4f, val loss: %.4f' % (
            best_epoch, train_top1_history[best_epoch], train_top5_history[best_epoch],
            val_top1_history[best_epoch], val_top5_history[best_epoch], 
            train_loss_history[best_epoch], val_loss_history[best_epoch]))
        return self.best_model, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history
    
    def extract_feature(self, data_tensor, stratified):
        self.model.train()
        data = data_tensor.to(self.device)
        features = self.model(data)

        # Normlize the features according to subject
        if stratified == 'minmax':
            f_max = features.max(dim=1, keepdim=True)[0]
            f_min = features.min(dim=1, keepdim=True)[0]
            features = (features - f_min) / (f_max - f_min)
            features = features * (f_max - f_min) + f_min
            features = F.normalize(features, dim=1)
        return features


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # eq_out = (torch.abs(output - output[:, 0].repeat(output.shape[1], 1).t()) < 1e-5).sum(axis=1)
        # print(eq_out)
        # eq_num = (eq_out > 1).sum()
        # if eq_out.sum() > len(eq_out):
        #     print('Equal logits for different entries!')
        #     print(output[eq_out>1, :].shape, output[eq_out>1, :])

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # if k == 1:
            #     correct_k = correct_k - eq_num
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
