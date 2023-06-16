# Hierarchical Semantic Contrast

## Overview

The Pytorch implementation of *Hierarchical Semantic Contrast* for weakly supervised semantic segmentation, IJCAI 2023.

The paper can be downloaded in `paper` folder.

![image](https://github.com/Wu0409/HSC_WSSS/blob/main/paper/overview.png)



## Abstract

>Weakly supervised semantic segmentation (WSSS) with image-level annotations has achieved great processes through class activation map (CAM). Since vanilla CAMs are hardly served as guidance to bridge the gap between full and weak supervision, recent studies explore semantic representations to make CAM ﬁt for WSSS better and demonstrate encouraging results. However, they generally exploit single-level semantics, which may hamper the model to learn a comprehensive semantic structure. Motivated by the prior that each image has multiple levels of semantics, we propose hierarchical semantic contrast (HSC) to ameliorate the above problem. It conducts semantic contrast from coarse-grained to ﬁne-grained perspective, including ROI level, class level, and pixel level, making the model learn a better object pattern understanding. To further improve CAM quality, building upon HSC, we explore consistency regularization of cross supervision and develop momentum prototype learning to utilize abundant semantics across different images. Extensive studies manifest that our plug-and-play learning paradigm, HSC, can signiﬁcantly boost CAM quality on both nonsaliency-guided and saliency-guided baselines, and establish new state-of-the-art WSSS performance on PASCAL VOC 2012 dataset.



## Environment

- Python 3.6
- pytorch>=1.13.0
- torchvision
- CUDA>=11.1
- pydensecrf from https://github.com/lucasb-eyer/pydensecrf 
- others (opencv-python etc.)

**NOTES:** please do not use `pip install pydensecrf` to install the incompatiable version. Use `pip install git+https://github.com/lucasb-eyer/pydensecrf.git` instead.



## Preparation

1. **Clone this repository.**
2. **Data preparation.**
   * Download PASCAL VOC 2012 devkit following instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit. 
   * Download saliency maps ([PFAN](https://arxiv.org/abs/1903.00179)) of VOC 2012 trainaug set in https://drive.google.com/file/d/19AjSmgdMlIZH4FXVZ5zjlUZcoZZCkwrI/view.
   * Then download the annotation of VOC 2012 trainaug set (containing 10582 images) from  [here](https://1drv.ms/u/s!As-yzQ0hGhUXiL0wnsQ6Cd65GiV6kw?e=wXW1UH) and place them all as ```VOC2012/SegmentationClassAug/xxxxxx.png```. 
3. **Download ImageNet pretrained backbones.**
   We use ResNet-38 for initial seeds generation and ResNet-101 (DeepLab v2) for segmentation training. 
   Download pretrained ResNet-38 weight from [here](https://1drv.ms/u/s!As-yzQ0hGhUXiL0x_fRjWGgcMn9IBA?e=qRzIpF) and put it to `network` folder.
   Download pretrained ResNet-101 from https://download.pytorch.org/models/resnet101-5d3b4d8f.pth.



## Experiment Guide

### Step1: Initial Seed Generation with HSC.

#### 1. Classification with HSC.

```
python train_voc.py  \
  --data_root $your_voc_image_data_root(VOC2012/images/JPEGImages) \
  --session_name $your_session_name
```

**Optional Parameters:**

* `--work_dir`: inference saving directory (default: `./exps`)

**NOTE:** Our tain model can be downloaded [here (HSC_eps_v2.pth)]([HSC_eps_v2.pth](https://1drv.ms/u/s!As-yzQ0hGhUXiL026T2ZqmbId1w7vg?e=PMxV89)).

#### 2. Infer CAM with CRF.

Download the our trained model (HSC_eps_v2.pth) or train from scratch, set ```--weights``` and then run:

```
python inference.py \
  --weights $your_trained_model_weight \ 
  --infer_list $[voc12/val.txt | voc12/train.txt | voc12/train_aug.txt] \
  --n_gpus 1 \
  --n_processes_per_gpu 1 \
  --crf_t 8 \
  --crf_alpha 7
```

**Optional Parameters:**

* `--thr`: the background CAM threshold (default: 0.22) *****
* `--work_dir`: inference saving directory (default: exps)
* `--cam_png`: output cam (.png) directory  (default: work_dir/cam_png)
* `--cam_npy`: output cam (.npy) directory (default: work_dir/cam_npy)

**NOTE:** 

* *****The inference script will also generate the pseudo-label (.png) with the fixed background threshold (default: 0.22). Your can directly evaluate them in the next step of CASE #1. 
* You can set `n_gpus=N` if you have multiple gpus for inference. 
* You can adjust the `n_processes_per_gpu` to boost the inference time (<u>if GPU memory is sufficient</u>)
* The CAM (.png) and CAM (.npy) of <u>the train set (evaluate)</u> and <u>trainaug set (train the segmentation model)</u> can be downloaded [here (cam_inference)](https://1drv.ms/f/s!As-yzQ0hGhUXiL0zgK4X7vILCcfhjA?e=wYuvOI).

#### 3. Evaluation.

**CASE #1:** For the psuedo-labels (.png) with the fixed background threshold:

```
python eval.py \
	--predict_dir $your_cam_png_dir \
	--logfile $your_log_file_name
	--comment $your_comment \
	--type png \
	--curve False
```

**CASE #2:** Following SEAM, we recommend you to use ```--curve``` to readjust an optimial background threshold to generate the optimal pseudo-labels if you modify our framework:

```
python eval.py \
  --list VOC2012/ImageSets/Segmentation/$[val.txt | train.txt] \
  --predict_dir $your_result_dir \
  --gt_dir VOC2012/SegmentationClass \
  --comment $your_comments \
  --type $[npy | png] \
  --curve True
```

**NOTE:** 

* The evaluation results will saved in “logfile” with your “comment”
* Following previous works, we just use train set (not the trainaug set) to evaluate the CAM (pseudo-label) performance.



### Step2: Training refinment network to refine pseudo-labels.

***NO additional post refinement stage is employed in HSC with EPS baseline.***



### Step3: Segmentation training with DeepLab

We borrow the segmentation repo from https://github.com/YudeWang/semantic-segmentation-codebase. 

The modified segmentation scripts are in `segmentation` folder. You can follow the following guide to finish Step 3.

#### 1. Preparation.

The training code is at `segmentation/experiment` directory. Please set the following parameters  in `config.py` :

* The experiment name: `EXP_NAME` (Line 12)

* The directory of the generated pseudo-labels of the **train_aug** set: `DATA_PSEUDO_GT` (Line 27) 

**Optional Parameters:**

* GPU num: `GPUS` (Line 13) (the multiple gpu setting is not tested for work)

**NOTE:** you can directly use our generated pseudo-labels of HSC ([here](https://1drv.ms/u/s!As-yzQ0hGhUXiL07SirUd1Z_P7GEeA?e=VUyMH9)) to train the segmentation model.

#### 2. Training.

After setting `config.py`, then run:

```
python train.py
```

**NOTE:** the trained model weight can be downloaded [here](https://1drv.ms/u/s!As-yzQ0hGhUXiL1Crl4j4WCp8EqAZQ?e=PmhOd5).

#### 3. Evaluation.

Check the path of the trained model `config_dict['TEST_CKPT']` in `config.py` (Line 69) and val/test set selection (Line 45) in ```test.py```.  Then run:

```
python test.py
```

For test set evaluation, you need to download test set images and submit the segmentation results (with color mapping) to the official eval server.

Here is the official evaluation result of HSC on the VOC test set: http://host.robots.ox.ac.uk/anonymous/11DHLZ.html

**NOTE:** we provide the predictions of the *val* set and *test* set [here](https://1drv.ms/f/s!As-yzQ0hGhUXiL08wy7TQlEvndVhww?e=s1yzRk).



## Download List

You can download the trained models and evaluate the performance below. We also provides the predictions for visualization :)

| content                                       | download link                                                | note                                                         |
| :-------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **Step #1:**  HSC (ResNet38)                  | [download](https://1drv.ms/u/s!As-yzQ0hGhUXiL026T2ZqmbId1w7vg?e=PMxV89) | HSC_eps_v2.pth                                               |
| **Step #3:**  segmentation model (DeepLab V2) | [download](https://1drv.ms/u/s!As-yzQ0hGhUXiL1Crl4j4WCp8EqAZQ?e=PmhOd5) | deeplabv2…itr20000_all.pth                                   |
| CAM (.npy) from HSC (train set)               | [download](https://1drv.ms/u/s!As-yzQ0hGhUXiL06M7tIucmFBCwoQg?e=nv9GMq) | You can adjust background threhold to evaluate the CAM performance (Step #1 - Eval - CASE #2) |
| CAM (.png) from HSC (train set)               | [download](https://1drv.ms/u/s!As-yzQ0hGhUXiL03RHkv8O9RByfTMw?e=Ap9H0b) | Pseudo-labels actually.                                      |
| Pseudo-labels from HSC (train_aug set)        | [download](https://1drv.ms/u/s!As-yzQ0hGhUXiL07SirUd1Z_P7GEeA?e=BX9QBl) | This is not used for evaluation in Step #1.                  |
| Segmentation results                          | [download](https://onedrive.live.com/?cid=17151a210dcdb2cf&id=17151A210DCDB2CF%21138940&ithint=folder&authkey=%21AMMu00JRL53VYcM) | We also provides color mapped results in `colormap` folder.  |



## Acknowledgements

We sincerely thank [Ye Du et al.](https://github.com/usr922/wseg) and [Yude Wang](https://scholar.google.com/citations?user=5aGpONMAAAAJ&hl=en) for their great work PPC and SEAM in CVPR. We borrow codes heavly from their repositories. We also thank [Seungho Lee](https://scholar.google.com/citations?hl=zh-CN&user=vUM0nAgAAAAJ) for his nice baseline work [EPS](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.pdf). Many thanks to their brilliant works!



## Citation

```
@inproceedings{wu2023hierarchical,
  title={Hierarchical Semantic Contrast for Weakly Supervised Semantic Segmentation},
  author={Wu, Yuanchen and Li, Xiaoqiang and Dai, Songmin and Li, Jide and Liu, Tong and Xie, Shaorong},
  booktitle={IJCAI},
  year={2023}
}
```
