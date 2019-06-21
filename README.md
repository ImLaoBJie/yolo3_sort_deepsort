# YOLOv3+SORT

# 介绍 Introduction

YOLOV3及其训练的实现借鉴：[qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

SORT的实现借鉴：[abewley/sort](https://github.com/abewley/sort)

参考文献：[SIMPLE ONLINE AND REALTIME TRACKING](https://arxiv.org/pdf/1602.00763.pdf)

---

# 搞快点 Quick Start

1. 打开`yolo_video.py`

2. 修改`DEFAULTS`（个人原因不太喜欢用`argparse`）

```
DEFAULTS = {
        "model_path": './model_h5/yolo.h5',
        "anchors_path": './model_data/yolo_anchors.txt',
        "classes_path": './model_data/coco_classes.txt',
        "gpu_num": 1,
        "image": False,
        "tracker": True,
        "write_to_file": True,
        "input": './input/Demo2_tiny.mp4',
        "output": './output/Demo2_tiny.mp4',
        "output_path": './output/',
        "score": 0.4,  # threshold
        "iou": 0.4,  # threshold
        "repeat_iou": 0.95,  # threshold
    }
```

3. 运行`yolo_video.py`，结果可在`"output_path"`中指定的文件夹查看

```
python yolo_video.py
```

4. 如果想适用轻量级的YOLOv3模型，修改'"model_path"'和'"anchors_path"'即可

*关于YOLOV3的内容，可以查看[YOLO WEBSITE](https://pjreddie.com/darknet/yolo/)

*tiny-YOLOv3下载：[tiny-YOLOv3](https://pjreddie.com/media/files/yolov3-tiny.weights)

*YOLOv3下载：[YOLOv3](https://pjreddie.com/media/files/yolov3.weights)

---

# 参数含义 Parameter

```
model_path  # h5文件路径
anchors_path  # anchor的路径
classes_path  # 存放识别对象类别的路径
gpu_num  # gpu数
image  # 处理video(False)或处理图片(True)
tracker  # 是否使用追踪
write_to_file  # 是否写入到文件
input  # video的路径
output  # 输出video的路径
output_path  # 其他文件output的路径
score  # 分数低于该阈值的物体会被忽略
iou  # iou低于该阈值的物体会被忽略
repeat_iou  # 去除重复bounding box
```

*写入到文件的格式为：

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

---

# 训练自己的模型 Training

选取的图片从CIFAR-100 dataset中提取，由于主要研究对象是交通方面的，因此选取的物体种类主要围绕车辆和
人，详细分类见`model_data/cifar_classes.txt`

CIFAR数据集可在此网站查看：[The CIFAR-10 and CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

CIFAR-100 dataset下载：[CIFAR-100 python version](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

1. 可通过修改`read_data_cifar100.py`确定训练数据集的物体类别

```
REMAIN = list(np.concatenate([[11, 35, 46, 98], [8, 13, 48, 58], [81, 85]]))
```

2. 运行train.py

```
python train.py
```

可自行修改`epochs`，`batch_size`

3. 可先使用训练好的YOLOv3模型`yolo.h5`获取bounding box数据，再使用`kmeans_anchors.py`
计算获得anchors

---

# TIPS

1. 环境 Environment

 * 软

    * python 3.6.5
    * Keras 2.1.5
    * tensorflow-gpu 1.11.0

 * 硬

    * NVIDIA GTX 850M DDR5
    * CUDA 9.0.176
    * Driver 390.65
    
2. 显卡和内存（8GB）太拉闸了，显存4GB训练YOLO甚至是tiny_YOLO时，epoch超过10直接就ResourceExhaustError，
处理视频还是没问题的，性能在3fp左右，CPU没使用过，估计会更加龟速吧。
所以你问我有没有针对vehicle和pedestrian训练好的YOLO或者tiny_YOLO模型，我也很难受:worried:

3. 缺少`openh264-1.8.0-win64.dll`可能会发生未知错误，因此需要将此文件和`python yolo_video.py`放置在
同一目录下（貌似少了也没啥事）

4. 暂时不会搞DeepSort里的图像特征抽取，只用SORT的话会导致ID Switch过于频繁，对于有遮挡物、双向重交通流等
情况，ID会非常不准确

5. **DEMO**上传至[百度云](https://pan.baidu.com/s/1VLKI8OGDbzsfqtzMe1amxg) PW: pb34

6. 有点懒，以后弄个英语README  I am exhausted, En version to be done.


