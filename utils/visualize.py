import cv2
import numpy as np
from scipy import io
import matplotlib.pyplot as plt


# 图像上采样
def imageUp(image, ratio=1):
    # 图像分辨率提升2的ratio次方
    for i in range(np.uint8(ratio)):
        image = cv2.pyrUp(image, dstsize=(2 * image.shape[1], 2 * image.shape[0]))
    return image


# 中断打点时可视化显示图像
def images_save(image, dpi, savePath):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(image.shape[1] * 5.0 * 16 / dpi, image.shape[2] * 5.0 * 16 / dpi)

    for i in range(1, image.shape[0] + 1):
        ax = fig.add_subplot(4, 4, i)
        ax.set_title('# %d' % i, fontsize=500)
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # 获取图像cube当前对应地物类别的丰度图
        curr_Img = image[i - 1]
        # curr_Img = np.uint8(np.rint(255 * curr_Img))
        # colorImg = cv2.applyColorMap(curr_Img, cv2.COLORMAP_PARULA)
        # RGB = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
        ax.imshow(imageUp(curr_Img, 5))

    plt.tight_layout()
    fig.savefig('./ComponentMap/' + savePath, dpi=dpi)
    plt.close('all')


def visual_label(template, predicts, indicesX, indicesY, palette, best_acc):
    X = np.array(indicesX)
    Y = np.array(indicesY)

    v_rgb = template
    v_rgb[X[:], Y[:]] = np.array(predicts) + 1

    x = np.ravel(v_rgb)
    y = np.zeros((x.shape[0], 3))
    for index, item in enumerate(x):
        y[index] = palette[item]
    y_re = np.reshape(y, (template.shape[0], template.shape[1], 3))
    y_re = cv2.cvtColor(np.float32(y_re), cv2.COLOR_BGR2RGB)

    filename = "./visualize/Classification_Result_" + str(round(best_acc * 100, 2)) + '.png'
    cv2.imwrite(filename, y_re)


def visual_attr(template, predicts, indicesX, indicesY, palette, attr):
    X = np.array(indicesX)
    Y = np.array(indicesY)

    v_rgb = template
    v_rgb[X[:], Y[:]] = np.array(predicts) + 1

    x = np.ravel(v_rgb)
    y = np.zeros((x.shape[0], 3))
    for index, item in enumerate(x):
        y[index] = palette[item]
    y_re = np.reshape(y, (template.shape[0], template.shape[1], 3))
    y_re = cv2.cvtColor(np.float32(y_re), cv2.COLOR_BGR2RGB)

    filename = "./visualize/Result_" + attr + '.png'
    cv2.imwrite(filename, y_re)


if __name__ == "__main__":
    gt = io.loadmat("../dataset/Trento/processed/gt.mat")['gt']
    TRLabel = io.loadmat("../dataset/Trento/processed/TRLabel.mat")['TRLabel']
    TSLabel = io.loadmat("../dataset/Trento/processed/TSLabel.mat")['TSLabel']

    # # 记录HSI图像块的可视化RGB图
    # hsi_rgb = hsi[:, :, [6, 28, 45]]
    # Caption = 'HSI_rgb_' + str(x) + '_' + str(y)
    # Img = wandb.Image(hsi_rgb)
    # wandb.log({Caption: Img})
    # # 记录Lidar图像块的可视化图
    # Caption = 'LiDAR_' + str(x) + '_' + str(y)
    # Img = wandb.Image(lidar)
    # wandb.log({Caption: Img})
    #
    # # 可视化显示hs_sim的热力图
    # SimilarImage = hs_sim.reshape(self.patch_size, self.patch_size)
    # minV = np.min(SimilarImage)
    # maxV = np.max(SimilarImage)
    # Img = np.uint8(np.rint(255 * ((SimilarImage - minV) / (maxV - minV))))
    # colorImg = cv2.applyColorMap(Img, cv2.COLORMAP_JET)
    # Caption = 'hs_sim_rgb_' + str(x) + '_' + str(y)
    # Img = wandb.Image(colorImg)
    # wandb.log({Caption: Img})
    #
    # # 可视化显示中心像元以及挑选出来的对应正样本
    # pick = hsi_rgb
    # pick[self.start, self.start, :] = [1, 0, 0]
    # pick[x_s, y_s, :] = [0, 1, 0]
    # Caption = 'pick_' + str(x) + '_' + str(y)
    # Img = wandb.Image(pick)
    # wandb.log({Caption: Img})


