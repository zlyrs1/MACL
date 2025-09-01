import torch
import numpy as np
from torch.utils.data import Dataset


def Train_Test_Split(args, gt):
    indices = np.nonzero(gt)
    X = list(zip(*indices))
    y = gt[indices]

    train_gt, test_gt = np.zeros_like(gt), np.zeros_like(gt)
    train_samples, train_indexes = [], []

    class_statistic = [[] for _ in range(args.class_num)]
    for i in range(y.shape[0]):
        class_statistic[y[i] - 1].append(i)

    for i in range(args.class_num):
        index = np.random.permutation(np.arange(len(class_statistic[i])))[0:args.train_per_class_Num]
        train_indexes.extend(np.array(class_statistic[i])[index])

    for i in range(args.train_samples):
        train_samples.append(X[train_indexes[i]])

    test_samples = list(np.delete(X, train_indexes, axis=0))

    train_samples = tuple(zip(*train_samples))
    test_samples = tuple(zip(*test_samples))

    train_gt[train_samples] = gt[train_samples]
    test_gt[test_samples] = gt[test_samples]

    return train_gt, test_gt


# -------  Stage1: DataSet for Contrast learning pre-training  -------
class DataSetS1(torch.utils.data.Dataset):
    def __init__(self, HSI, LiDAR, HSI_SPA, gt, args):
        super().__init__()
        imgH, imgW, imgC = HSI.shape[0], HSI.shape[1], HSI.shape[2]

        self.p_lidar, self.p_hsi = args.patch_lidar, args.patch_hsi
        self.half_lidar = int((self.p_lidar - 1) / 2)
        self.half_hsi = int((self.p_hsi - 1) / 2)

        self.hsi_offset = np.ones((imgH + self.p_lidar - 1, imgW + self.p_lidar - 1, imgC))
        self.hsi_offset[:, :, 0:-1:2] = 0
        self.hsi_offset[self.half_lidar:imgH + self.half_lidar,
                        self.half_lidar:imgW + self.half_lidar, :] = HSI

        self.lidar_offset = np.ones((imgH + self.p_lidar - 1, imgW + self.p_lidar - 1)) * 5
        self.lidar_offset[self.half_lidar:imgH + self.half_lidar,
                          self.half_lidar:imgW + self.half_lidar] = LiDAR

        self.hsi_spa_offset = np.ones((imgH + self.p_lidar - 1, imgW + self.p_lidar - 1)) * 5
        self.hsi_spa_offset[self.half_lidar:imgH + self.half_lidar,
                            self.half_lidar:imgW + self.half_lidar] = HSI_SPA

        trainMap = np.zeros((imgH, imgW))
        trainMap[0:-1:2, 0:-1:2] = 1
        trainMap[1:-1:2, 1:-1:2] = 1
        if args.name == 'Houston':
            trainMap = gt
        x_pos, y_pos = np.nonzero(trainMap)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.s_lambda = args.s_lambda

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]

        hsi = self.hsi_offset[x:x + self.p_lidar, y:y + self.p_lidar]
        hsi = np.asarray(hsi, dtype='float32')
        hsi_c = hsi[self.half_lidar - self.half_hsi:self.half_lidar + self.half_hsi + 1,
                    self.half_lidar - self.half_hsi:self.half_lidar + self.half_hsi + 1]
        hsi_c = torch.from_numpy(hsi_c)
        hsi_c = hsi_c.permute(2, 0, 1)

        lidar = self.lidar_offset[x:x + self.p_lidar, y:y + self.p_lidar]
        lidar = np.asarray(lidar, dtype='float32')

        lidar_central = lidar[self.half_lidar, self.half_lidar]
        height_sim = abs(lidar - lidar_central).reshape(-1)

        hsi_central = hsi[self.half_lidar, self.half_lidar]
        hsi_central = torch.from_numpy(hsi_central).unsqueeze(0)
        A = hsi_central.softmax(dim=1)
        A_log = hsi_central.log_softmax(dim=1)

        b = hsi.reshape(-1, hsi.shape[2])
        b = torch.from_numpy(b)
        B = b.softmax(dim=1)
        B_log = b.log_softmax(dim=1)

        batch_klB = (B * B.log()).sum(dim=1).unsqueeze(0) - torch.einsum('ik, jk -> ij', [A_log, B])
        batch_klA = (A * A.log()).sum(dim=1) - torch.einsum('ik, jk -> ij', [A, B_log])
        spec_sim = (batch_klA + batch_klB).t().numpy().squeeze(1)

        hs_sim = height_sim + self.s_lambda * spec_sim
        random_pick = np.random.permutation(np.arange(1, 11))[0]
        [x_s, y_s] = np.unravel_index(np.argsort(hs_sim)[random_pick], lidar.shape)
        [x_r, y_r] = x + x_s - self.half_lidar, y + y_s - self.half_lidar

        hsi_r = self.hsi_offset[x_r:x_r + self.p_lidar, y_r:y_r + self.p_lidar]
        lidar_spec = hsi_r[self.half_lidar - self.half_hsi:self.half_lidar + self.half_hsi + 1,
                           self.half_lidar - self.half_hsi:self.half_lidar + self.half_hsi + 1]
        lidar_spec = np.asarray(lidar_spec, dtype='float32')
        lidar_spec = torch.from_numpy(lidar_spec)
        lidar_spec = lidar_spec.permute(2, 0, 1)

        lidar = torch.from_numpy(lidar)
        lidar = lidar.unsqueeze(0)

        # The representation of HSI image in spatial domain
        hsi_spa = self.hsi_spa_offset[x:x + self.p_lidar, y:y + self.p_lidar]
        hsi_spa = np.asarray(hsi_spa, dtype='float32')
        hsi_spa = torch.from_numpy(hsi_spa)
        hsi_spa = hsi_spa.unsqueeze(0)

        return hsi_c, lidar_spec, lidar, hsi_spa


# -------  Stage 2: DataSet for HSI+Lidar classification task -------
class DataSetS2(torch.utils.data.Dataset):
    def __init__(self, HSI, LiDAR, gt, args):
        super().__init__()
        imgH, imgW, imgC = HSI.shape[0], HSI.shape[1], HSI.shape[2]

        self.p_lidar, self.p_hsi = args.patch_lidar, args.patch_hsi
        self.half_lidar = int((self.p_lidar - 1) / 2)
        self.half_hsi = int((self.p_hsi - 1) / 2)

        self.hsi_offset = np.zeros((imgH + self.p_lidar - 1, imgW + self.p_lidar - 1, imgC))
        self.hsi_offset[self.half_lidar:imgH + self.half_lidar,
                        self.half_lidar:imgW + self.half_lidar, :] = HSI

        self.lidar_offset = np.zeros((imgH + self.p_lidar - 1, imgW + self.p_lidar - 1))
        self.lidar_offset[self.half_lidar:imgH + self.half_lidar,
                          self.half_lidar:imgW + self.half_lidar] = LiDAR

        x_pos, y_pos = np.nonzero(gt)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.label = gt - 1
        self.s_lambda = args.s_lambda

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]

        label = self.label[x, y]
        label = np.asarray(label, dtype='int64')
        label = torch.from_numpy(label)

        hsi = self.hsi_offset[x:x + self.p_lidar, y:y + self.p_lidar]
        hsi = np.asarray(hsi, dtype='float32')
        hsi_c = hsi[self.half_lidar - self.half_hsi:self.half_lidar + self.half_hsi + 1,
                    self.half_lidar - self.half_hsi:self.half_lidar + self.half_hsi + 1]
        hsi_c = torch.from_numpy(hsi_c)
        hsi_c = hsi_c.permute(2, 0, 1)

        lidar = self.lidar_offset[x:x + self.p_lidar, y:y + self.p_lidar]
        lidar = np.asarray(lidar, dtype='float32')
        lidar = np.expand_dims(lidar, axis=0)
        lidar = torch.from_numpy(lidar)

        return hsi_c, lidar, label, x, y


# -------  DataSet for HSI+Lidar classification visual -------
class DataSetVisual(torch.utils.data.Dataset):
    def __init__(self, HSI, LiDAR, args):
        super().__init__()
        imgH, imgW, imgC = HSI.shape[0], HSI.shape[1], HSI.shape[2]

        self.p_lidar, self.p_hsi = args.patch_lidar, args.patch_hsi
        self.half_lidar = int((self.p_lidar - 1) / 2)
        self.half_hsi = int((self.p_hsi - 1) / 2)

        self.hsi_offset = np.zeros((imgH + self.p_lidar - 1, imgW + self.p_lidar - 1, imgC))
        self.hsi_offset[self.half_lidar:imgH + self.half_lidar,
                        self.half_lidar:imgW + self.half_lidar, :] = HSI

        self.lidar_offset = np.zeros((imgH + self.p_lidar - 1, imgW + self.p_lidar - 1))
        self.lidar_offset[self.half_lidar:imgH + self.half_lidar,
                          self.half_lidar:imgW + self.half_lidar] = LiDAR

        fullMap = np.ones((imgH, imgW))
        x_pos, y_pos = np.nonzero(fullMap)
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]

        hsi = self.hsi_offset[x:x + self.p_lidar, y:y + self.p_lidar]
        hsi = np.asarray(hsi, dtype='float32')
        hsi_c = hsi[self.half_lidar - self.half_hsi:self.half_lidar + self.half_hsi + 1,
                    self.half_lidar - self.half_hsi:self.half_lidar + self.half_hsi + 1]
        hsi_c = torch.from_numpy(hsi_c)
        hsi_c = hsi_c.permute(2, 0, 1)

        lidar = self.lidar_offset[x:x + self.p_lidar, y:y + self.p_lidar]
        lidar = np.asarray(lidar, dtype='float32')
        lidar = np.expand_dims(lidar, axis=0)
        lidar = torch.from_numpy(lidar)

        return hsi_c, lidar, x, y
