# YOLOv3+SORT+DeepSort

* Update 2020.7.16 增加deepsort，并作了大量调整

# 介绍 Introduction

YOLOV3及其训练的实现借鉴：[qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

SORT的实现借鉴：[abewley/sort](https://github.com/abewley/sort)

DeepSort的实现借鉴：[theAIGuysCode/yolov3_deepsort](https://github.com/theAIGuysCode/yolov3_deepsort)

参考文献：

1. [SIMPLE ONLINE AND REALTIME TRACKING](https://arxiv.org/pdf/1602.00763.pdf)

2. [SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC](https://arxiv.org/pdf/1703.07402.pdf)

演示视频：[Demo](https://www.bilibili.com/video/av56450343/)

---

# 搞快点 Quick Start

1. 打开`yolo_video.py`

2. 修改`DEFAULTS`（个人原因不太喜欢用`argparse`）

```
DEFAULTS = {
        "model_path": './model_h5/yolo.h5',
        "anchors_path": './model_data/yolo_anchors.txt',
        "classes_path": './model_data/coco_classes.txt',
        "deepsort_model": './model_data/mars-small128.pb',
        "gpu_num": 1,
        "image": False,  # 如果此处设置了True，"tracker"则被忽略
        "tracker": 'deepsort',  # 此处根据需要为'sort'或'deepsort'
        "write_to_file": True,
        "input": './input/your_video.format',
        "output": './output/your_video.format',
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

*预训练的DeepSort网络：Google Drive: [DeepSort](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp), BaiduDisk: [DeepSort](https://pan.baidu.com/s/1B4xKXYWckM4TLIg6WGW6uw)  pw:9i6p

---

# 参数含义 Parameter

```
model_path  # h5文件路径
anchors_path  # anchor的路径
classes_path  # 存放识别对象类别的路径
deepsort_model  # DeepSort预训练权重存放路径
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

 * 主要依赖

    * python 3.6
    * Keras 2.3.1
    * tensorflow-gpu 1.13.0
    * numpy 1.17.0
    
    (较低版本貌似也支持)

3. 缺少`openh264-1.8.0-win64.dll`可能会发生未知错误，因此需要将此文件和`python yolo_video.py`放置在
同一目录下（貌似少了也没啥事）

4. DeepSort能解决短时遮挡问题，解决不了长时间object消失或被遮挡问题

5. **DEMO**上传至[百度云](https://pan.baidu.com/s/1VLKI8OGDbzsfqtzMe1amxg) PW: pb34

6. **MOT_DEMO** [Multiple Object Tracking Benchmark](https://motchallenge.net/data/MOT16/)


