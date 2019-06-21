# YOLOv3+SORT

# ���� Introduction

YOLOV3����ѵ����ʵ�ֽ����[qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

SORT��ʵ�ֽ����[abewley/sort](https://github.com/abewley/sort)

�ο����ף�[SIMPLE ONLINE AND REALTIME TRACKING](https://arxiv.org/pdf/1602.00763.pdf)

---

# ���� Quick Start

1. ��`yolo_video.py`

2. �޸�`DEFAULTS`������ԭ��̫ϲ����`argparse`��

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

3. ����`yolo_video.py`���������`"output_path"`��ָ�����ļ��в鿴

```
python yolo_video.py
```

4. �����������������YOLOv3ģ�ͣ��޸�'"model_path"'��'"anchors_path"'����

*����YOLOV3�����ݣ����Բ鿴[YOLO WEBSITE](https://pjreddie.com/darknet/yolo/)

*tiny-YOLOv3���أ�[tiny-YOLOv3](https://pjreddie.com/media/files/yolov3-tiny.weights)

*YOLOv3���أ�[YOLOv3](https://pjreddie.com/media/files/yolov3.weights)

---

# �������� Parameter

```
model_path  # h5�ļ�·��
anchors_path  # anchor��·��
classes_path  # ���ʶ���������·��
gpu_num  # gpu��
image  # ����video(False)����ͼƬ(True)
tracker  # �Ƿ�ʹ��׷��
write_to_file  # �Ƿ�д�뵽�ļ�
input  # video��·��
output  # ���video��·��
output_path  # �����ļ�output��·��
score  # �������ڸ���ֵ������ᱻ����
iou  # iou���ڸ���ֵ������ᱻ����
repeat_iou  # ȥ���ظ�bounding box
```

*д�뵽�ļ��ĸ�ʽΪ��

```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

---

# ѵ���Լ���ģ�� Training

ѡȡ��ͼƬ��CIFAR-100 dataset����ȡ��������Ҫ�о������ǽ�ͨ����ģ����ѡȡ������������ҪΧ�Ƴ�����
�ˣ���ϸ�����`model_data/cifar_classes.txt`

CIFAR���ݼ����ڴ���վ�鿴��[The CIFAR-10 and CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

CIFAR-100 dataset���أ�[CIFAR-100 python version](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

1. ��ͨ���޸�`read_data_cifar100.py`ȷ��ѵ�����ݼ����������

```
REMAIN = list(np.concatenate([[11, 35, 46, 98], [8, 13, 48, 58], [81, 85]]))
```

2. ����train.py

```
python train.py
```

�������޸�`epochs`��`batch_size`

3. ����ʹ��ѵ���õ�YOLOv3ģ��`yolo.h5`��ȡbounding box���ݣ���ʹ��`kmeans_anchors.py`
������anchors

---

# TIPS

1. ���� Environment

 * ��

    * python 3.6.5
    * Keras 2.1.5
    * tensorflow-gpu 1.11.0

 * Ӳ

    * NVIDIA GTX 850M DDR5
    * CUDA 9.0.176
    * Driver 390.65
    
2. �Կ����ڴ棨8GB��̫��բ�ˣ��Դ�4GBѵ��YOLO������tiny_YOLOʱ��epoch����10ֱ�Ӿ�ResourceExhaustError��
������Ƶ����û����ģ�������3fp���ң�CPUûʹ�ù������ƻ���ӹ��ٰɡ�
������������û�����vehicle��pedestrianѵ���õ�YOLO����tiny_YOLOģ�ͣ���Ҳ������:worried:

3. ȱ��`openh264-1.8.0-win64.dll`���ܻᷢ��δ֪���������Ҫ�����ļ���`python yolo_video.py`������
ͬһĿ¼�£�ò������Ҳûɶ�£�

4. ��ʱ�����DeepSort���ͼ��������ȡ��ֻ��SORT�Ļ��ᵼ��ID Switch����Ƶ�����������ڵ��˫���ؽ�ͨ����
�����ID��ǳ���׼ȷ

5. **DEMO**�ϴ���[�ٶ���](https://pan.baidu.com/s/1VLKI8OGDbzsfqtzMe1amxg) PW: pb34

6. �е������Ժ�Ū��Ӣ��README  I am exhausted, En version to be done.


