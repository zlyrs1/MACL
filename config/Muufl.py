seed = 20
name = 'Muufl'
hsi_dataset_dir = "./datasets/MUUFL/HSI.mat"
lidar_dataset_dir = "./datasets/MUUFL/LiDAR.mat"
gt_dataset_dir = "./datasets/MUUFL/gt.mat"
hsi_pca_dataset_dir = "./datasets/MUUFL/HSI_PCA.mat"

# --------------------   Image information   --------------------
H = 325
W = 220
lidar_bands_num = 1
hsi_bands_num = 64
class_num = 11
patch_lidar = 21  # Lidar patchSize
patch_hsi = 7     # HSI patchSize

# --------------------   stage 1：Pre-training   --------------------
cl_epochs = 50
cl_batch_size = 1024
cl_num_workers = 3
lr_base = 0.8
m_rate = 0.99
s_lambda = 2

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
            1: (45, 135, 45),    # Trees
            2: (116, 230, 116),  # Mostly grass
            3: (182, 242, 182),  # Mixed ground surface
            4: (154, 102, 0),    # Dirt and sand
            5: (222, 190, 167),  # Road
            6: (177, 211, 236),  # Water
            7: (74, 134, 232),   # Building shadow
            8: (38, 97, 197),    # Buildings
            9: (255, 255, 0),    # Sidewalk
            10: (217, 25, 24),   # Yellow curb
            11: (0, 51, 204),    # Cloth panels
            12: (255, 0, 0)      # selected train samples
          }

best_acc = 0.84

print_oa = 0.89
