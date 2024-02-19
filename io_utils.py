import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import bisect



class Dataset_raw(Dataset):
    def __init__(self, data, timeLen, timeStep, n_samples, n_points_remain, fs, n_subs):
        self.data = data.transpose()  # change to (n_channs, n_samples)
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.fs = fs

        n_samples_all = np.tile(n_samples, n_subs)
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples_all)))
        n_points_remain_all = np.tile(n_points_remain, n_subs)
        self.n_points_remain_cum = np.concatenate((np.array([0]), np.cumsum(n_points_remain_all)))

        self.sample_num = int(np.sum(n_samples)) * n_subs

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        # The sample should not be across different videos
        # for i in range(len(self.n_samples_cum)-1):
        #     if (idx >= self.n_samples_cum[i]) & (idx < self.n_samples_cum[i+1]):
        #         pass_videos = i
        #         break
        # print('1:', pass_videos)
        pass_videos = bisect.bisect_right(self.n_samples_cum, idx)
        pass_videos = pass_videos - 1
        # print('2:', pass_videos)

        one_seq = self.data[:, int(float(idx) * self.timeStep * float(self.fs) + float(self.n_points_remain_cum[pass_videos])):
                               int((float(idx) * self.timeStep + float(self.timeLen)) * float(self.fs) + float(self.n_points_remain_cum[pass_videos]))]
        # print(idx, self.n_points_remain_cum[pass_videos], one_seq.shape)

        one_seq = torch.FloatTensor(one_seq).unsqueeze(0)
        return one_seq


class TrainSampler():
    def __init__(self, n_subs, n_times, batch_size, n_samples, phase):
        self.n_per = int(np.sum(n_samples))
        self.n_subs = n_subs
        # Number of data points per session
        self.batch_size = batch_size
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
        self.n_samples_per_trial = int(batch_size / len(n_samples))

        self.sub_pairs = []
        if phase == 'train':
            for i in range(self.n_subs):
                for j in range(i+1, self.n_subs):
                    self.sub_pairs.append([i, j])
        elif phase == 'val':
            j = self.n_subs - 1
            for i in range(self.n_subs-1):
                self.sub_pairs.append([i, j])
        self.n_times = n_times

    def __len__(self):
        return self.n_times * len(self.sub_pairs)

    def __iter__(self):
        for s in range(len(self.sub_pairs)):
            for t in range(self.n_times):
                [sub1, sub2] = self.sub_pairs[s]
                # print(sub1, sub2)

                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum)-1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum)-1), self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]), 1, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum)-2):
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))

                    i = len(self.n_samples_cum) - 2
                    ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))

                assert len(ind_abs) == self.batch_size
                ind_this1 = ind_abs + self.n_per*sub1
                ind_this2 = ind_abs + self.n_per*sub2

                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                yield batch


class ValSampler():
    def __init__(self, n_subs, n_samples, train_sub, val_sub):
        self.n_per = int(np.sum(n_samples))
        self.n_subs = n_subs
        # Number of data points per session
        # self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))

        self.sub_pairs = []
        for j in val_sub:
            for i in train_sub:
                self.sub_pairs.append([i, j])

    def __len__(self):
        return len(self.sub_pairs) * min(self.n_samples) # Loop over all samples in order

    def __iter__(self):
        for s in range(len(self.sub_pairs)):
            for t in range(min(self.n_samples)):
                [sub1, sub2] = self.sub_pairs[s]
                # print(sub1, sub2)

                ind_abs = np.array([self.n_samples_cum[i]+t for i in range(len(self.n_samples))])
                # print(ind_abs)
                ind_this1 = ind_abs + self.n_per*sub1
                ind_this2 = ind_abs + self.n_per*sub2

                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                yield batch