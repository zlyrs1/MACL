seed = 20
name = 'Trento'
hsi_dataset_dir = "./datasets/Trento/HSI.mat"
lidar_dataset_dir = "./datasets/Trento/LiDAR.mat"
gt_dataset_dir = "./datasets/Trento/gt.mat"
hsi_pca_dataset_dir = "./datasets/Trento/HSI_PCA.mat"

# --------------------   Image information   --------------------
H = 166
W = 600
lidar_bands_num = 1
hsi_bands_num = 63
class_num = 6
patch_lidar = 21   # Lidar patchSize
patch_hsi = 7      # HSI patchSize

# --------------------   stage 1：Pre-training   --------------------
cl_epochs = 50
cl_batch_size = 1024
cl_num_workers = 3
lr_base = 1.2
m_rate = 0.99
s_lambda = 1  # 0.005

# -------- E_SPA network --------
en_spa_input_channel = lidar_bands_num
en_spa_channels = [16, 32, 64, 64]

# -------- E_SPEC network --------
en_spec_input_channel = hsi_bands_num
en_spec_channels = [128, 256, 512, 512]

# -------- Predictor network --------
mlp_hidden_size = 4096
projection_size = 512


# --------------------    stage 2： Classification   --------------------
train_per_class_Num = 10
train_samples = class_num * train_per_class_Num

c_epochs = 500
c_train_batch_size = 32
c_test_batch_size = 2048

c_lr_encoder = 0.005
c_lr = 0.05

fea_dims = en_spa_channels[-1] + en_spec_channels[-1]


# --------------------    visualization   --------------------
palette = {
            0: (0, 0, 0),        # background
            1: (223, 190, 167),  # Apple trees
            2: (136, 102, 204),  # Buildings
            3: (177, 211, 236),  # Ground
            4: (170, 102, 57),   # Woods
            5: (45, 136, 45),    # Vineyard
            6: (255, 255, 0),    # Roads
            7: (0, 255, 255)     # selected train samples
        }

best_acc = 0.95

print_oa = 0.98
