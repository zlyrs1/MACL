import os
from scipy import io
import numpy as np
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from utils.initial import initialize
from utils.data import DataSetVisual
from utils.visualize import visual_attr
from model.extractor import E_SPA, E_SPEC
from model_02_classification import Classifier


def main():
    # ------------------------  Dataset  ------------------------
    # -----------------------------------------------------------
    dataset_name = "Houston"  # "Houston" / "Houston" / "Houston"
    args, device, labels_text, ignored_label = initialize(dataset_name)

    HSI = io.loadmat(args.hsi_dataset_dir)['HSI']
    HSI = HSI.reshape(-1, args.hsi_bands_num)
    HSI = StandardScaler().fit_transform(HSI)
    HSI = HSI.reshape(args.H, args.W, -1)

    LiDAR = io.loadmat(args.lidar_dataset_dir)['LiDAR']
    if LiDAR.ndim == 3 and args.lidar_bands_num == 1:
        LiDAR = LiDAR[:, :, 0]
    LiDAR = LiDAR.reshape(-1, args.lidar_bands_num)
    LiDAR = StandardScaler().fit_transform(LiDAR)
    LiDAR = LiDAR.reshape(args.H, args.W)

    gt = io.loadmat(args.gt_dataset_dir)['gt']

    classify_visualData = DataSetVisual(HSI, LiDAR, args)
    classify_visualLoader = torch.utils.data.DataLoader(classify_visualData, pin_memory=True, shuffle=False,
                                                        batch_size=args.c_test_batch_size)

    model_path = os.path.join('checkpoints', 'Houston')

    model_name = os.path.join(model_path, 'encoder_spa', 'wo.pth')
    encoder_spa = E_SPA(args).to(device)
    encoder_spa.load_state_dict(torch.load(model_name))

    model_name = os.path.join(model_path, 'encoder_spec', 'wo.pth')
    encoder_spec = E_SPEC(args).to(device)
    encoder_spec.load_state_dict(torch.load(model_name))

    model_name = os.path.join(model_path, 'baseNet', 'wo.pth')
    baseNet = Classifier(fea_dims=args.fea_dims, class_num=args.class_num).to(device)
    baseNet.load_state_dict(torch.load(model_name))

    # ---------------------------  visual  ---------------------------
    encoder_spa.eval()
    encoder_spec.eval()
    baseNet.eval()

    predicts, targets = np.zeros(0), np.zeros(0)
    indicesX, indicesY = [], []

    for iter_, (hsi, lidar, x, y) in enumerate(classify_visualLoader):
        hsi, lidar = hsi.to(device), lidar.to(device)

        features_spa = encoder_spa(lidar)
        features_spec = encoder_spec(hsi)
        features = torch.cat((features_spa, features_spec), dim=1).unsqueeze(2)
        hsi_p = baseNet(features)

        _, predict = torch.max(hsi_p.data, 1)
        predicts = np.append(predicts, predict.cpu().numpy())
        indicesX.extend(x.cpu().numpy().tolist())
        indicesY.extend(y.cpu().numpy().tolist())

    visual_attr(gt, predicts, indicesX, indicesY, args.palette, 'Houston_wo')


if __name__ == '__main__':
    main()
