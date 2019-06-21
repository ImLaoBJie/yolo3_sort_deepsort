import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

PATH = 'cifar-100-python/'
REMOVE = list(range(0, 100))
REMAIN = list(np.concatenate([[11, 35, 46, 98], [8, 13, 48, 58], [81, 85]]))
for i in REMAIN:
    REMOVE.remove(i)


def filter(image, label):
    # filter
    remove_index = []
    for index, element in enumerate(label):
        if int(element) in REMOVE:
            remove_index.append(index)

    label = np.delete(label, remove_index)
    image = np.delete(image, remove_index, 0)

    if not REMAIN == []:
        value = 0
        for index in REMAIN:
            label[label == np.int32(index)] = np.int32(value)
            value = value + 1

    return image, label


def load_CIFAR_batch(filename, N, data_filter: bool):
    # 单个batch
    # load single batch of cifar
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')  # dict类型
        image = datadict['data']  # X, ndarray, 像素值
        label = datadict['fine_labels']  # Y, list, 标签, 分类

        # check the id of fine_labels relevant to the coarse_labels
        # label = np.array(label)
        # coarse = np.array(datadict['coarse_labels'])
        # print(np.unique(label[np.array(np.where(coarse == 19))[0]]))

        # reshape, 一维数组转为矩阵10000行3列。每个entries是32x32
        # transpose，转置
        # astype，复制，同时指定类型
        image = image.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        label = np.array(label)

        if data_filter:
            image, label = filter(image, label)

        return image, label


def load_CIFAR100(path, data_filter: bool):
    # 所有batch
    # load all of cifar
    images = []  # list
    labels = []

    # 训练集
    f = os.path.join(path, 'train')
    image, label = load_CIFAR_batch(f, 50000, data_filter)
    images.append(image)
    labels.append(label)

    images = np.concatenate(images)  # [ndarray, ndarray] 合并为一个ndarray
    labels = np.concatenate(labels)

    # 测试集
    img_val, lab_val = load_CIFAR_batch(os.path.join(path, 'test'), 10000, data_filter)
    return images, labels, img_val, lab_val


# 警告：使用该函数可能会导致内存溢出，可以适当修改减少扩充量
# WARNING：Using this function may cause out of memory and OS breakdown
def creat_more_data(images):
    # 通过旋转、翻转扩充数据 expand dataset through rotation and mirroring
    images_rot90 = []
    images_rot180 = []
    images_rot270 = []
    img_lr = []
    img_ud = []

    for index in range(0, images.shape[0]):
        band_1 = images[index, :, :, 0]
        band_2 = images[index, :, :, 1]
        band_3 = images[index, :, :, 2]

        # 旋转90, rotating 90 degrees
        band_1_rot90 = np.rot90(band_1)
        band_2_rot90 = np.rot90(band_2)
        band_3_rot90 = np.rot90(band_3)
        images_rot90.append(np.dstack((band_1_rot90, band_2_rot90, band_3_rot90)))

        # 180
        band_1_rot180 = np.rot90(band_1_rot90)
        band_2_rot180 = np.rot90(band_2_rot90)
        band_3_rot180 = np.rot90(band_3_rot90)
        images_rot180.append(np.dstack((band_1_rot180, band_2_rot180, band_3_rot180)))

        # 270
        band_1_rot270 = np.rot90(band_1_rot180)
        band_2_rot270 = np.rot90(band_2_rot180)
        band_3_rot270 = np.rot90(band_3_rot180)
        images_rot270.append(np.dstack((band_1_rot270, band_2_rot270, band_3_rot270)))

        # 左右翻转 flip horizontally
        lr1 = np.flip(band_1, 0)
        lr2 = np.flip(band_2, 0)
        lr3 = np.flip(band_3, 0)
        img_lr.append(np.dstack((lr1, lr2, lr3)))

        # 上下反转 flip vertical
        ud1 = np.flip(band_1, 1)
        ud2 = np.flip(band_2, 1)
        ud3 = np.flip(band_3, 1)
        img_ud.append(np.dstack((ud1, ud2, ud3)))

    rot90 = np.array(images_rot90)
    rot180 = np.array(images_rot180)
    rot270 = np.array(images_rot270)
    lr = np.array(img_lr)
    ud = np.array(img_ud)

    images = np.concatenate((rot90, rot180, rot270, lr, ud))

    return images


def shuffle(images, labels):
    permutation = np.random.permutation(images.shape[0])
    shuffled_dataset = images[permutation, :, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def data(path, more_data: bool, shuffle_data: bool, data_filter: bool):
    images, labels, img_val, lab_val = load_CIFAR100(path, data_filter)

    if more_data:
        # 扩充数据 expand dataset
        images = creat_more_data(np.array(images))
        # 扩充标签 expend labels
        labels = np.concatenate((labels, labels, labels, labels, labels, labels))

    if shuffle_data:
        images, labels = shuffle(images, labels)
        img_val, lab_val = shuffle(img_val, lab_val)

    return images, labels, img_val, lab_val


def main():
    images, labels, img_val, lab_val = data(PATH, False, True, True)
    # test
    print(len(images))
    print(len(labels))
    plt.imshow(images[0] / 255)
    print(images[0])
    print(labels[0])
    plt.show()


if __name__ == '__main__':
    main()
