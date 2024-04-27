# Self-Supervised Representation Learning with Spatial-Temporal Consistency for Sign Language Recognition
[Weichao Zhao](https://scholar.google.com/citations?user=v-ASmMIAAAAJ&hl=zh-CN), [Hezhen Hu](https://scholar.google.com/citations?user=Fff-9WAAAAAJ&hl=zh-CN&oi=ao), [Wengang Zhou](https://scholar.google.com/citations?user=8s1JF8YAAAAJ&hl=zh-CN&oi=ao), [Min Wang](https://scholar.google.com/citations?user=FFDionEAAAAJ&hl=zh-CN&oi=ao) and [Houqiang Li](https://scholar.google.com/citations?user=7sFMIKoAAAAJ&hl=zh-CN&oi=ao)

This repository includes Python (PyTorch) implementation of this paper.

Under review in TIP2024

![](./images/framework.png)


## Requirements

```bash
python==3.8.13
torch==1.8.1+cu111
torchvision==0.9.1+cu111
tensorboard==2.9.0
scikit-learn==1.1.1
tqdm==4.64.0
numpy==1.22.4
```

## Training and Testing
Please refer to the bash scripts

## Datasets
* Download the original datasets, including [SLR500](https://ustc-slr.github.io/datasets/), [NMFs_CSL](https://ustc-slr.github.io/datasets/), [WLASL](https://dxli94.github.io/WLASL/) and [MSASL](https://www.microsoft.com/en-us/research/project/ms-asl/)

* Utilize the off-the-shelf pose estimator [MMPose](https://mmpose.readthedocs.io/en/latest/model_zoo/wholebody_2d_keypoint.html) with the setting of [Topdown Heatmap + Hrnet + Dark on Coco-Wholebody](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth) to extract the 2D keypoints for sign language videos.
* The final data is formatted as follows:
  
```
    Data
    ├── NMFs_CSL
    ├── SLR500
    ├── WLASL
    └── MSASL
        ├── Video
        ├── Pose
        └── Annotations
```

## Pretrained Model
You can download the pretrained model from this link: [pretrained model on four ISLR datasets](https://rec.ustc.edu.cn/share/f3af4d10-0454-11ef-b462-57ac733a8ba6)


## Acknowledgment
The framework of our code is based on [skeleton-contrast](https://github.com/fmthoker/skeleton-contrast).
