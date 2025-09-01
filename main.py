import time
from scipy import io
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler

from utils.initial import initialize
from utils.data import Train_Test_Split, DataSetS1, DataSetS2
from utils.network import save_model, collect_params, regression_loss
from utils.report import compute_metrics
from utils.visualize import visual_label
from modelTool.LARSSGD import LARS
from modelTool.extractor import E_SPA, E_SPEC
from modelTool.mlp_head import MLPHead
from model import ContrastNet, Classifier


def main():
    # ------------------------  Dataset  ------------------------
    # -----------------------------------------------------------
    dataset_name = "Trento"
    args, device, labels_text, ignored_label = initialize(dataset_name)
    start_time = time.time()

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

    HSI_PCA = io.loadmat(args.hsi_pca_dataset_dir)['HSI_PCA']
    HSI_PCA = HSI_PCA.reshape(-1, 1)
    HSI_PCA = StandardScaler().fit_transform(HSI_PCA)
    HSI_PCA = HSI_PCA.reshape(args.H, args.W)

    gt = io.loadmat(args.gt_dataset_dir)['gt']

    # DataSet
    train_gt, test_gt = Train_Test_Split(args, gt)
    num_train = np.count_nonzero(train_gt)
    num_test = np.count_nonzero(test_gt)

    print("Dataset_name:", dataset_name)
    print("The number of train/test samples is %d/%d" % (num_train, num_test))

    # ------------------  stage one: pretraining  ------------------
    # --------------------------------------------------------------
    DataPretext = DataSetS1(HSI, LiDAR, HSI_PCA, gt, args)
    Pretext_loader = torch.utils.data.DataLoader(DataPretext, pin_memory=True, shuffle=True,
                                                 batch_size=args.cl_batch_size,
                                                 num_workers=args.cl_num_workers)
    Len_Pretext_loader = len(Pretext_loader)
    print('Pretext BatchNum:', Len_Pretext_loader)

    online_network_spa = ContrastNet(args, 0).to(device)
    target_network_spa = ContrastNet(args, 0).to(device)
    predictor_spa = MLPHead(args.projection_size, args.mlp_hidden_size, args.projection_size).to(device)
    optimizer_spa = LARS(params=collect_params([online_network_spa, predictor_spa]),
                         lr=args.lr_base, momentum=0.9, weight_decay=1.0e-6)

    online_network_spec = ContrastNet(args, 1).to(device)
    target_network_spec = ContrastNet(args, 1).to(device)
    predictor_spec = MLPHead(args.projection_size, args.mlp_hidden_size, args.projection_size).to(device)
    optimizer_spec = LARS(params=collect_params([online_network_spec, predictor_spec]),
                          lr=args.lr_base, momentum=0.9, weight_decay=1.0e-6)

    for param_q, param_k in zip(online_network_spa.parameters(), target_network_spa.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False

    for param_q, param_k in zip(online_network_spec.parameters(), target_network_spec.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False

    online_network_spa.train()
    predictor_spa.train()
    online_network_spec.train()
    predictor_spec.train()

    scaler = GradScaler()

    for e in range(1, args.cl_epochs + 1):
        Loss_spa, Loss_spec = 0, 0

        for iter_, (hsi_c, lidar_spec, lidar, hsi_spa) in enumerate(Pretext_loader):
            lidar, hsi_spa = lidar.to(device), hsi_spa.to(device)
            hsi_c, lidar_spec = hsi_c.to(device), lidar_spec.to(device)

            with autocast():
                predictions_spa_hsi = predictor_spa(online_network_spa(hsi_spa)[-1])
                predictions_spa_lidar = predictor_spa(online_network_spa(lidar)[-1])
                with torch.no_grad():
                    targets_spa_hsi = target_network_spa(hsi_spa)[-1]
                    targets_spa_lidar = target_network_spa(lidar)[-1]

                loss_spa = regression_loss(predictions_spa_hsi, targets_spa_lidar)
                loss_spa += regression_loss(predictions_spa_lidar, targets_spa_hsi)
                loss_spa = loss_spa.mean()

                predictions_spec_hsi = predictor_spec(online_network_spec(hsi_c)[-1])
                predictions_spec_lidar = predictor_spec(online_network_spec(lidar_spec)[-1])
                with torch.no_grad():
                    targets_spec_hsi = target_network_spec(hsi_c)[-1]
                    targets_spec_lidar = target_network_spec(lidar_spec)[-1]

                loss_spec = regression_loss(predictions_spec_hsi, targets_spec_lidar)
                loss_spec += regression_loss(predictions_spec_lidar, targets_spec_hsi)
                loss_spec = loss_spec.mean()

            optimizer_spa.zero_grad()
            scaler.scale(loss_spa).backward()
            scaler.step(optimizer_spa)
            scaler.update()

            optimizer_spec.zero_grad()
            scaler.scale(loss_spec).backward()
            scaler.step(optimizer_spec)
            scaler.update()

            for param_q, param_k in zip(online_network_spa.parameters(), target_network_spa.parameters()):
                param_k.data = args.m_rate * param_k.data + (1. - args.m_rate) * param_q.data

            for param_q, param_k in zip(online_network_spec.parameters(), target_network_spec.parameters()):
                param_k.data = args.m_rate * param_k.data + (1. - args.m_rate) * param_q.data

            Loss_spa += loss_spa.item() / Len_Pretext_loader
            Loss_spec += loss_spec.item() / Len_Pretext_loader

        elapse_time = round((time.time() - start_time) / 60, 2)
        print('Epoch %2d, eps_time %2.2f min, Pretext E_spa loss %.5f, Pretext E_spec loss %.5f'
              % (e, elapse_time, Loss_spa, Loss_spec))

    # ------------------------ stage two: HSI + Lidar Classification task--------------------
    # -------------------------------------------------------------
    classify_trainData = DataSetS2(HSI, LiDAR, train_gt, args)
    classify_trainLoader = torch.utils.data.DataLoader(classify_trainData, pin_memory=True, shuffle=True,
                                                       batch_size=args.c_train_batch_size)
    Len_classify_trainLoader = len(classify_trainLoader)
    print('Classification Train BatchNum:', Len_classify_trainLoader)

    classify_testData = DataSetS2(HSI, LiDAR, test_gt, args)
    classify_testLoader = torch.utils.data.DataLoader(classify_testData, pin_memory=True, shuffle=False,
                                                      batch_size=args.c_test_batch_size)
    Len_classify_testLoader = len(classify_testLoader)
    print('Classification Test BatchNum:', Len_classify_testLoader)

    # Classification Model: encoder + baseNet(MAFF + FC)
    encoder_spa = E_SPA(args).to(device)
    encoder_spa.load_state_dict(online_network_spa.encoder.state_dict())

    encoder_spec = E_SPEC(args).to(device)
    encoder_spec.load_state_dict(online_network_spec.encoder.state_dict())

    baseNet = Classifier(fea_dims=args.fea_dims, class_num=args.class_num).to(device)

    optimizer = torch.optim.SGD([{'params': encoder_spa.parameters(), 'lr': args.c_lr_encoder},
                                 {'params': encoder_spec.parameters(), 'lr': args.c_lr_encoder},
                                 {'params': baseNet.parameters()}], lr=args.c_lr, weight_decay=0.0001)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.95)

    criterion = nn.CrossEntropyLoss().cuda()

    for e in range(1, args.c_epochs + 1):
        # ---------------------------  training ---------------------------
        encoder_spa.train()
        encoder_spec.train()
        baseNet.train()
        avg_loss_classify, correct = 0, 0

        for iter_, (hsi, lidar, target, x, y) in enumerate(classify_trainLoader):
            hsi, lidar, target = hsi.to(device), lidar.to(device), target.to(device)

            features_spa = encoder_spa(lidar)
            features_spec = encoder_spec(hsi)
            features = torch.cat((features_spa, features_spec), dim=1).unsqueeze(2)
            hsi_p = baseNet(features)

            loss_classify = criterion(hsi_p, target)

            optimizer.zero_grad()
            loss_classify.backward(retain_graph=True)
            optimizer.step()

            _, predicts = torch.max(hsi_p.data, 1)
            correct += (predicts == target).sum()
            avg_loss_classify += loss_classify.item() / Len_classify_trainLoader

        acc = correct / num_train
        print('Epoch %2d, train classifyLoss %.5f, train acc %.5f' % (e, avg_loss_classify, acc))

        # ---------------------------  testing  ---------------------------
        encoder_spa.eval()
        encoder_spec.eval()
        baseNet.eval()

        predicts, targets = np.zeros(0), np.zeros(0)
        indicesX, indicesY = [], []
        for iter_, (hsi, lidar, target, x, y) in enumerate(classify_testLoader):
            hsi, lidar, target = hsi.to(device), lidar.to(device), target.to(device)

            features_spa = encoder_spa(lidar)
            features_spec = encoder_spec(hsi)
            features = torch.cat((features_spa, features_spec), dim=1).unsqueeze(2)
            hsi_p = baseNet(features)

            _, predict = torch.max(hsi_p.data, 1)
            predicts = np.append(predicts, predict.cpu().numpy())
            targets = np.append(targets, target.cpu().numpy())
            indicesX.extend(x.cpu().numpy().tolist())
            indicesY.extend(y.cpu().numpy().tolist())

        acc = accuracy_score(predicts, targets)
        elapse_time = round((time.time() - start_time) / 60, 2)
        print('Epoch %2d, test acc %.5f, eps_time %2.2f min' % (e, acc, elapse_time))

        oa, aa, kappa = compute_metrics(predicts, targets, args, [], labels_text)
        if acc > args.best_acc:
            args.best_acc = acc
            print('Epoch %2d, Update Best Acc %.4f, Best OA %.4f, Best AA %.4f, Best KAPPA %.4f'
                  '' % (e, args.best_acc, oa, aa, kappa))
            visual_label(gt, predicts, indicesX, indicesY, args.palette, args.best_acc)

            save_model(encoder_spa, 'Trento', 'encoder_spa', args.best_acc)
            save_model(encoder_spec, 'Trento', 'encoder_spec', args.best_acc)
            save_model(baseNet, 'Trento', 'baseNet', args.best_acc)

        scheduler.step()


if __name__ == '__main__':
    main()
