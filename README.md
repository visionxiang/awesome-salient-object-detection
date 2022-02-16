# Salient Object Detection

A curated list of awesome resources for salient object detection (SOD),  including RGB-D SOD, CoSOD, and part image SOD. We will keep updating it.

:heavy_exclamation_mark:Updated 2022-01.



--------------------------------------------------------------------------------------

<!--TOC-->

## Content:
- [Overview](#Overview)
- [RGBD SOD](#RGBD-SOD)
- [CoSOD](#CoSOD)
- [Image SOD](#Image-SOD)
- [Appendix](#Appendix)

--------------------------------------------------------------------------------------



## Overview

| **Year** | **Pub.** | **Title**                                | **Author**       | **Links**                                                    |
| :------: | :------: | :--------------------------------------- | :--------------- | :----------------------------------------------------------- |
|   2020   |   CVM    | RGB-D Salient Object Detection: A Survey | Tao Zhou, et al. | [Paper](https://arxiv.org/abs/2008.00230)/[Proj](https://github.com/taozh2017/RGBD-SODsurvey) |



## RGBD SOD

### Preprint

| **Year** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :------- | :----------------------------------------------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
|   2021   | arXiv12  | MutualFormer: Multi-Modality Representation Learning via Mutual Transformer | Xixi Wang, Bo Jiang, et al. | [Paper](https://arxiv.org/abs/2112.01177)/Code
|   2021   | arXiv12  | Transformer-based Network for RGB-D Saliency Detection | Yue Wang, Huchuan Lu, et al. | [Paper](https://arxiv.org/abs/2112.00582)/Code
|   2021   | arXiv09  | ACFNet: Adaptively-Cooperative Fusion Network for RGB-D Salient Object Detection | Jinchao Zhu, et al.                                          | [arXiv](https://arxiv.org/pdf/2109.04627.pdf)/Code           |
|   2021   | arXiv06  | Dynamic Knowledge Distillation with A Single Stream Structure for RGB-D Salient Object Detection | Guangyu Ren, Tania Stathaki                                  | [Paper](https://arxiv.org/pdf/2106.09517.pdf)/Code           |
|   2021   | arXiv06  | Progressive Multi-scale Fusion Network for RGB-D Salient Object Detection | Guangyu Ren, et al.                                          | [Paper](https://arxiv.org/pdf/2106.03941.pdf)/Code           |
|   2021   | arXiv04  | Middle-level Fusion for Lightweight RGB-D Salient Object Detection | N. Huang, Qiang Jiao, Qiang Zhang, Jungong Han               | [Paper](https://arxiv.org/pdf/2104.11543.pdf)/Code           |
|   2020   | arXiv12  | A Unified Structure for Efficient RGB and RGB-D Salient Object Detection | Peng Peng and Yong-Jie Li                                    | [arXiv](https://arxiv.org/pdf/2012.00437.pdf)/Code           |
|   2020   | arXiv08  | <span style="white-space:nowrap;">Knowing Depth Quality In Advance: A Depth Quality Assessment Method For RGB-D Salient Object Detection&emsp; </span> | Xuehao Wang, et al.                                          | [Paper](https://arxiv.org/abs/2008.04157)/[Code](https://github.com/XueHaoWang-Beijing/DQSF?utm_source=catalyzex.com) |




### 2022 

| **No.** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   01   | AAAI  | Self-Supervised Representation Learning for RGB-D Salient Object Detection | Xiaoqi Zhao, Huchuan Lu, et al.                              | [Paper](https://arxiv.org/pdf/2101.12482.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD) |




### 2021

| **No.** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   17    |   PAMI   | Uncertainty Inspired RGB-D Saliency Detection                | Jing Zhang, Deng-Ping Fan, et al.                            | Paper/Code                                                   |
|   16    |   PAMI   | Siamese Network for RGB-D Salient Object Detection and Beyond | Keren Fu, Deng-Ping Fan, et al.                              | Paper/Code                                                   |
|   15    |   IJCV   | CNN-Based RGB-D Salient Object Detection: Learn, Select, and Fuse | Hao Chen, Youfu Li, et al.                                   | [Paper](https://arxiv.org/pdf/1909.09309.pdf)/Code           |
|   14    |   TIP    | Hierarchical Alternate Interaction Network for RGB-D Salient Object Detection | Gongyang Li, Haibin Ling, et al.                             | Paper/Code                                                   |
|   13    |   TIP    | Bilateral Attention Network for RGB-D Salient Object Detection | Zhao Zhang, Deng-Ping Fan, et al.                            | [arXiv](https://arxiv.org/abs/2004.14582)/[Code](https://github.com/zzhanghub/bianet) |
|   12    |   TIP    | CDNet: Complementary Depth Network for RGB-D Salient Object Detection | Wen-Da Jin, Ming-Ming Cheng, et al.                          | Paper/Code                                                   |
|   11    |   TIP    | Data-Level Recombination and Lightweight Fusion Scheme for RGB-D Salient Object Detection | Xuehao Wang, et al.                                          | [Paper](https://arxiv.org/abs/2009.05102)/[Code](https://github.com/XueHaoWang-Beijing/DRLF?utm_source=catalyzex.com) |
|   10    |   TIP    | RGB-D Salient Object Detection With Ubiquitous Target Awareness | Yifan Zhao, et al.                                           | [arXiv](https://arxiv.org/abs/2109.03425)/Code               |
|   09    |   TIP    | DPANet: Depth Potentiality-Aware Gated Attention Network for RGB-D Salient Object Detection | Zuyao Chen, Runmin Cong, et al.                              | [Paper](https://arxiv.org/abs/2003.08608)/[Code](https://github.com/JosephChenHub/DPANet) |
|   08    |   TIP    | Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection | Zhengzheng Tu, et al.                                        | Paper/Code                                                   |
|   07    |   TCYB   | ASIF-Net: Attention Steered Interweave Fusion Network for RGB-D Salient Object Detection | Chongyi Li, Runmin Cong, et al.                              | [Paper](https://ieeexplore.ieee.org/document/8998588)/[Code](https://github.com/Li-Chongyi/ASIF-Net) |
|   06    |   TNN    | IRFR-Net: Interactive Recursive Feature-Reshaping Network for Detecting Salient Objects in RGB-D Images | Wujie Zhou, Qinling Guo, et al.                              | [Paper](https://ieeexplore.ieee.org/abstract/document/9519891)/Code |
|   05    |   TNN    | Rethinking RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks | Deng-Ping Fan, et al.                                        | [Paper](https://arxiv.org/abs/1907.06781)/[Code](https://github.com/DengPingFan/D3NetBenchmark) |
|   04    |   TMM    | Employing Bilinear Fusion and Saliency Prior Information for RGB-D Salient Object Detection | Nianchang Huang, Dingwen Zhang, et al.                       | Paper/Code                                                   |
|   03    |   TMM    | Attentive Cross-Modal Fusion Network for RGB-D Saliency Detection | <span style="white-space:nowrap;">Di Liu, Kao Zhang, Zhenzhong Chen&emsp;</span> | Paper/Cod                                                    |
|   02    |   TMM    | Deep RGB-D Saliency Detection without Depth                  | Yuan-Fang Zhang, et al.                                      | Paper/Code                                                   |
|   01    |   TMM    | <span style="white-space:nowrap;">CCAFNet: Crossflow and Cross-scale Adaptive Fusion Network for Detecting Salient Objects in RGB-D Images&emsp; </span> | Wujie Zhou, et al.                                           | Paper/Code                                                   |

| **No.** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   10    |   ICCV   | Visual Saliency Transformer                                  | Nian Liu, Junwei Han, et al.                                 | [Paper](https://arxiv.org/pdf/2104.12099.pdf)/[Code](https://github.com/nnizhang/VST#visual-saliency-transformer-vst) |
|   09    |   ICCV   | Specificity-preserving RGB-D Saliency Detection              | Tao Zhou, et al.                                             | [Paper](https://arxiv.org/abs/2108.08162)/[Code](https://github.com/taozh2017/SPNet) |
|   08    |   ICCV   | RGB-D Saliency Detection via Cascaded Mutual Information Minimization | Jing Zhang, Deng-Ping Fan, et al.                            | [Paper](https://arxiv-download.xixiaoyao.cn/pdf/2109.07246.pdf)/[Code](https://github.com/JingZhang617/cascaded_rgbd_sod) |
|   07    |    MM    | Depth Quality-Inspired Feature Manipulation for Efficient RGB-D Salient Object Detection | Wenbo Zhang, Keren Fu, et al.                                | [Paper](https://arxiv.org/abs/2107.01779)/[Code](https://github.com/zwbx/DFM-Net) |
|   06    |    MM    | Cross-modality Discrepant Interaction Network for RGB-D Salient Object Detection | Chen Zhang, Runmin Cong, et al.                              | [Paper](https://arxiv.org/abs/2108.01971)/[Proj](https://rmcong.github.io/proj_CDINet.html?utm_source=catalyzex.com) |
|   05    |    MM    | TriTransNet: RGB-D salient object detection with a triplet transformer embedding network | Zhengyi Liu, et al.                                          | [Paper](https://arxiv.org/abs/2108.03990)/Code               |
|   04    |   ICME   | BTS-Net: Bi-directional Transfer-and-Selection Network For RGB-D Salient Object Detection | Wenbo Zhang, Keren Fu, et al.                                | [arXiv](https://arxiv.org/pdf/2104.01784.pdf)/[Code](https://github.com/zwbx/BTS-Net) |
|   03    |   CVPR   | Deep RGB-D Saliency Detection With Depth-Sensitive Attention and Automatic Multi-Modal Fusion | Peng Sun, Xi Li, et al.                                      | [Paper](https://arxiv.org/abs/2103.11832)/[Code](https://github.com/sunpeng1996/DSA2F) |
|   02    |   CVPR   | Calibrated RGB-D Salient Object Detection                    | Wei Ji, Shuang Yu, et al.                                    | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.pdf)/[Code](https://github.com/jiwei0921/DCF) |
|   01    |   AAAI   | <span style="white-space:nowrap;">RGB-D Salient Object Detection via 3D Convolutional Neural Networks &emsp; </span> | <span style="white-space:nowrap;">Qian Chen, Keren Fu, et al. &emsp;</span> | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16191)/[Code](https://github.com/PPOLYpubki/RD3D) |



### 2020

| **No.** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   09    |   TIP    | RGBD Salient Object Detection via Disentangled Cross-Modal Fusion | Hao Chen, et al.                                             | Paper/Code                                                   |
|   08    |   TIP    | RGBD Salient Object Detection via Deep Fusion                | Liangqiong Qu, et al.                                        | [Paper](http://www.shengfenghe.com/qfy-content/uploads/2019/12/4853ab33ac9a11e019f165388e57acf1.pdf)/Code |
|   07    |   TIP    | Boundary-Aware RGBD Salient Object Detection With Cross-Modal Feature Sampling | Yuzhen Niu, et al.                                           | Paper/Code                                                   |
|   06    |   TIP    | ICNet: Information Conversion Network for RGB-D Based Salient Object Detection | Gongyang Li, Zhi Liu, Haibin Ling                            | Paper/Code                                                   |
|   05    |   TIP    | Improved Saliency Detection in RGB-D Images Using Two-Phase Depth Estimation and Selective Deep Fusion | Chenglizhao Chen, et al.                                     | Paper/Code                                                   |
|   04    |   TCYB   | Discriminative Cross-Modal Transfer Learning and Densely Cross-Level Feedback Fusion for RGB-D Salient Object Detection | Hao Chen, Youfu Li, Dan Su                                   | Paper/Code                                                   |
|   03    |   TCYB   | Going From RGB to RGBD Saliency: A Depth-Guided Transformation Model | Runmin Cong, et al.                                          | Paper/Code                                                   |
|   02    |   TMM    | cmSalGAN: RGB-D Salient Object Detection With Cross-View Generative Adversarial Networks | Bo Jiang, et al.                                             | [Paper](https://arxiv.org/pdf/1912.10280.pdf)/[Code](https://github.com/wangxiao5791509/cmSalGAN_PyTorch)<br>[Proj](https://sites.google.com/view/cmsalgan/) |
|   01    |   TMM    | <span style="white-space:nowrap;">Joint Cross-Modal and Unimodal Features for RGB-D Salient Object Detection &emsp;</span> | <span style="white-space:nowrap;">N. Huang, Yi Liu, Qiang Zhang, Jungong Han&emsp;</span> | Paper/Code                                                   |


| **No.** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   20    |   ECCV   | RGB-D Salient Object Detection with Cross-Modality Modulation and Selection | Chongyi Li, Runmin Cong, et al.                              | [Paper](https://arxiv.org/abs/2007.07051)/[Code](https://github.com/Li-Chongyi/cmMS-ECCV20)<br>[Proj](https://li-chongyi.github.io/Proj_ECCV20) |
|   19    |   ECCV   | Progressively Guided Alternate Refinement Network for RGB-D Salient Object Detection | Shuhan Chen, Yun Fu                                          | [Paper](https://arxiv.org/abs/2008.07064)/[Code](https://github.com/ShuhanChen/PGAR_ECCV20?utm_source=catalyzex.com) |
|   18    |   ECCV   | BBS-Net: RGB-D Salient Object Detection with a Bifurcated Backbone Strategy Network | Yingjie Zhai, Deng-Ping Fan, et al.                          | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570273.pdf)/[Code](https://github.com/DengPingFan/BBS-Net)<br>[Extension](https://arxiv.org/pdf/2007.02713.pdf) |
|   17    |   ECCV   | Cascade Graph Neural Networks for RGB-D Salient Object Detection | Ao Luo, et al.                                               | [Paper](https://arxiv.org/abs/2008.03087)/[Code](https://github.com/LA30/Cas-Gnn?utm_source=catalyzex.com) |
|   16    |   ECCV   | Cross-Modal Weighting Network for RGB-D Salient Object Detection | Gongyang Li, Haibin Ling, et al.                             | [Paper](https://arxiv.org/pdf/2007.04901.pdf)/[arXiv](https://arxiv.org/abs/2007.04901)<br>[Code](https://github.com/MathLee/CMWNet) |
|   15    |   ECCV   | Accurate RGB-D Salient Object Detection via Collaborative Learning | Wei Ji, Huchuan Lu, et al.                                   | [Paper](https://arxiv.org/abs/2007.11782)/[Code](https://github.com/jiwei0921/CoNet) |
|   14    |   ECCV   | A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection | Xiaoqi Zhao, Lihe Zhang, et al.                              | [Paper](https://arxiv.org/pdf/2007.06811.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/DANet-RGBD-Saliency) |
|   13    |   ECCV   | Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection | Youwei Pang, Lihe Zhang, Xiaoqi Zhao, Huchuan Lu             | [arXiv](https://arxiv.org/abs/2007.06227)/[Code](https://github.com/lartpang/HDFNet) |
|   12    |   ECCV   | Asymmetric Two-Stream Architecture for Accurate RGB-D Saliency Detection | Miao Zhang, Huchuan Lu, et al.                               | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730375.pdf)/[Code](https://github.com/OIPLab-DUT/ATSA) |
|   11    |    MM    | Is Depth Really Necessary for Salient Object Detection?      | Jiawei Zhao, Yifan Zhao, Jia Li, Xiaowu Chen                 | [Paper](https://arxiv.org/pdf/2006.00269.pdf)/[Code](https://github.com/iCVTEAM/DASNet)<br>[Proj](http://cvteam.net/projects/2020/DASNet/) |
|   10    |   CVPR   | Select, Supplement and Focus for RGB-D Saliency Detection    | Miao Zhang, Yongri Piao, et al.                              | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Select_Supplement_and_Focus_for_RGB-D_Saliency_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/OIPLab-DUT/CVPR_SSF-RGBD) |
|   09    |   CVPR   | JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection | Keren Fu, Deng-Ping Fan, Ge-Peng Ji, Qijun Zhao              | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf)/[arXiv](https://arxiv.org/abs/2004.08515)/<br>[PAMI21](https://arxiv.org/pdf/2008.12134.pdf)/[Code](https://github.com/kerenfu/JLDCF/) |
|   08    |   CVPR   | A2dele: Adaptive and Attentive Depth Distiller for Efficient RGB-D Salient Object Detection | Yongri Piao, Huchuan Lu, et al.                              | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Piao_A2dele_Adaptive_and_Attentive_Depth_Distiller_for_Efficient_RGB-D_Salient_CVPR_2020_paper.html)/Code |
|   07    |   CVPR   | UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders | Jing Zhang, Deng-Ping Fan, et al.                            | [Paper](https://arxiv.org/abs/2009.03075)/[Code](https://github.com/JingZhang617/UCNet?utm_source=catalyzex.com) |
|   06    |   CVPR   | Multi-Scale Interactive Network for Salient Object Detection | Youwei Pang, Huchuan Lu, et al.                              | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/lartpang/MINet) |
|   05    |   CVPR   | Interactive Two-Stream Decoder for Accurate and Fast Saliency Detection | H. Zhou, Xiaohua Xie, Jian-Huang Lai, et al.                 | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/moothes/ITSD-pytorch) |
|   04    |   CVPR   | Label Decoupling Framework for Salient Object Detection      | Jun Wei, et al.                                              | Paper/[Code](https://github.com/weijun88/LDF)                |
|   03    |   CVPR   | <span style="white-space:nowrap;">Learning Selective Self-Mutual Attention for RGB-D Saliency Detection&emsp;</span> | <span style="white-space:nowrap;">Nian Liu, Ni Zhang, et al.&emsp;</span> | Paper/[Code](https://github.com/nnizhang/SMAC?utm_source=catalyzex.com)<br>[Extension](https://arxiv.org/abs/2010.05537) |




### 2019 

| **No.** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   07    |   TIP    | Three-Stream Attention-Aware Network for RGB-D Salient Object Detection | <span style="white-space:nowrap;">Hao Chen, Youfu Li   &emsp;</span> | [Paper](https://ieeexplore.ieee.org/document/8603756)/Code   |
|   06    |   TIP    | RGB-‘D’ Saliency Detection With Pseudo Depth                 | Xiaolin Xiao, Yicong Zhou, Yue-Jiao Gong                     |                                                              |
|   05    |   ICCV   | EGNet: Edge Guidance Network for Salient Object Detection    | Jia-Xing Zhao, Ming-Ming Cheng, et al.                       | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/JXingZhao/EGNet) |
|   04    |   ICCV   | Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection | Yongri Piao, Wei Ji Miao Zhang, et al.                       | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Piao_Depth-Induced_Multi-Scale_Recurrent_Attention_Network_for_Saliency_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/jiwei0921/DMRA) |
|   03    |   CVPR   | <span style="white-space:nowrap;">Contrast Prior and Fluid Pyramid Integration for RGBD Salient Object Detection&emsp;</span> | Jia-Xing Zhao, Ming-Ming Cheng, et al.                       | [Paper](http://mftp.mmcheng.net/Papers/19cvprRrbdSOD.pdf)/[Code](https://github.com/JXingZhao/ContrastPrior)<br>[Proj](https://mmcheng.net/rgbdsalpyr/) |
|   02    |   CVPR   | BASNet: Boundary-Aware Salient Object Detection              |                                                              |                                                              |
|   01    |   CVPR   | S4Net: Single Stage Salient-Instance Segmentation            |                                                              |                                                              |


### 2018

| **No.** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links** |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :-------- |
|   03    |   TCYB   | CNNs-Based RGB-D Saliency Detection via Cross-View Transfer and Multiview Fusion | Junwei Han, et al.                                           |           |
|   02    |   CVPR   | Progressively Complementarity-Aware Fusion Network for RGB-D Salient Object Detection | <span style="white-space:nowrap;">Hao Chen, Youfu Li &emsp;</span> |           |
|   01    |   CVPR   | <span style="white-space:nowrap;">PiCANet: Learning Pixel-wise Contextual Attention for Saliency Detection &emsp;</span> |                                                              |           |


### 2017

| **No.** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links** |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :-------- |
|   03    |   TIP    | Depth-Aware Salient Object Detection and Segmentation via Multiscale Discriminative Saliency Fusion and Bootstrap Learning | Hangke Song, et al.                                          |           |
|   02    |   TIP    | <span style="white-space:nowrap;">Edge Preserving and Multi-Scale Contextual Neural Network for Salient Object Detection &emsp;</span> | <span style="white-space:nowrap;">Xiang Wang, Huimin Ma, Xiaozhi Chen, Shaodi You&emsp;</span> |           |
|   01    |   CVPR   | Instance-Level Salient Object Segmentation                   |                                                              |           |



### Beyond

| **Year** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 2021 | arXiv02 | Active Boundary Loss for Semantic Segmentation | Chi Wang, Yunke Zhang, et al. | [Paper](https://arxiv.org/abs/2102.02696)/Code |
| 2021 | arXiv02 | CPP-Net: Context-aware Polygon Proposal Network for Nucleus Segmentation | Shengcong Chen, Dacheng Tao, et al. | [Paper](https://arxiv.org/abs/2102.06867)/Code |
| 2021 | CVPR | Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation | Chufeng Tang, Xiaolin Hu, et al. | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html)/[Code](https://github.com/tinyalpha/BPR) |
| 2021 | CVPR | Boundary IoU: Improving Object-Centric Image Segmentation Evaluation | Bowen Cheng, Ross Girshick, Piotr Dollár, et al. | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)/[Code](https://bowenc0221.github.io/boundary-iou/) | 
| 2021 | TIP | Progressive Self-Guided Loss for Salient Object Detection | Sheng Yang, Weisi Lin, et al. | [Paper](https://arxiv.org/abs/2101.02412)/Code |
|   2021   |   ICCV   | **Contrastive Multimodal Fusion with TupleInfoNCE**          | Yunze Liu, Li Yi, et al.                                     | [Paper](https://arxiv.org/pdf/2107.02575.pdf)/Code           |
|   2021   |   TCYB   | PANet: Patch-Aware Network for Light Field Salient Object Detection | Yongri Piao, Huchuan Lu, et al.                              | Paper/Code                                                   |
|   2021   |    MM    | Occlusion-aware Bi-directional Guided Network for Light Field Salient Object Detection | Dong Jing, Runmin Cong, et al.                               | Paper/Code                                                   |
|   2020   |   TIP    | RGB-T Salient Object Detection via Fusing Multi-Level CNN Features | Qiang Zhang, Dingwen Zhang, et al.                           | Paper/Code                                                   |
|   2020   |   TIP    | LFNet: Light Field Fusion Network for Salient Object Detection | Miao Zhang, Huchuan Lu, et al.                               | Paper/Code                                                   |
|   2020   | NeurIPS  | Deep Multimodal Fusion by Channel Exchanging                 | Yikai Wang, et al.                                           | [Paper](https://papers.nips.cc/paper/2020/file/339a18def9898dd60a634b2ad8fbbd58-Paper.pdf)/[Code](https://github.com/yikaiw/CEN) |
|   2019   |   TIP    | Cross-Modal Attentional Context Learning for RGB-D Object Detection | <span style="white-space:nowrap;">G. Li, Liang Lin, et al.&emsp;</span> | Paper/Code                                                   |
|   2019   |   TMM    | RGB-T Image Saliency Detection via Collaborative Graph Learning | Zhengzheng Tu, et al.                                        | Paper/Code                                                   |
|   2019   | NeurIPS  | <span style="white-space:nowrap;">One-Shot Object Detection with Co-Attention and Co-Excitation &emsp;</span> | Ting-I Hsieh, et al.                                         | [Paper](https://papers.nips.cc/paper/2019/file/92af93f73faf3cefc129b6bc55a748a9-Paper.pdf)/[Code](https://github.com/timy90022/One-Shot-Object-Detection) |
|   2019   |   ICCV   | RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment | Guan’an Wang, et al.                                         | Paper/[Code](https://github.com/wangguanan/AlignGAN)         |



## CoSOD

- More related works can be found in: [CoSOD paper list](http://dpfan.net/CoSOD3k/) 

| **Year** | **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   2021   |  arXiv08 | Free Lunch for Co-Saliency Detection: Context Adjustment | Lingdong Kong, Prakhar Ganesh, et al. | [Paper](https://arxiv.org/abs/2108.02093v4)/[Data](http://ldkong.com/data/sets/cat/home)
|   2021   |  arXiv04 | CoSformer: Detecting Co-Salient Object with Transformers | Lv Tang | [Paper](https://arxiv.org/pdf/2104.14729.pdf)/Code
|   2021   |   PAMI   | Re-thinking Co-Salient Object Detection                      | Deng-Ping Fan, et al.                                        | Paper/[Proj](http://dpfan.net/CoSOD3k/)                      |
|   2021   |   TMM    | Image Co-saliency Detection and Instance Co-segmentation using Attention Graph Clustering based Graph Convolutional Network | Tengpeng Li, Qingshan Liu, et al.                            |                                                              |
|   2021   |   ICCV   | Summarize and Search: Learning Consensus-Aware Dynamic Convolution for Co-Saliency Detection | Ni Zhang, Junwei Han, Nian Liu, Ling Shao                    | Paper/Code                                                   |
|   2021   |   CVPR   | DeepACG: Co-Saliency Detection via Semantic-Aware Contrast Gromov-Wasserstein Distance |                                                              |                                                              |
|   2021   |   CVPR   | Group Collaborative Learning for Co-Salient Object Detection |                                                              |                                                              |
|   2021   |   AAAI   | Multi-scale Graph Fusion for Co-saliency Detection           | Rongyao Hu, Zhenyun Deng, Xiaofeng Zhu                       | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16951)/Code |
|          |          |                                                              |                                                              |                                                              |
|          |          |                                                              |                                                              |                                                              |
|   2020   |   PAMI   | Zero-Shot Video Object Segmentation with Co-Attention Siamese Networks | Xiankai Lu, Wenguan Wang, et al.                             | Paper/[Code](https://github.com/carrierlxk/COSNet)           |
|   2020   |   TNN    | Robust Deep Co-Saliency Detection With Group Semantic and Pyramid Attention | Zheng-Jun Zhang, et al.                                      | Paper/Code                                                   |
|   2020   |   TMM    | Deep Co-Saliency Detection via Stacked Autoencoder-Enabled Fusion and Self-Trained CNNs | Chung-Chi Tsai, et al.                                       | Paper/Code                                                   |
|   2020   |   TMM    | A New Method and Benchmark for Detecting Co-Saliency Within a Single Image | Hongkai You, et al.                                          | Paper/Code                                                   |
|   2020   | NeurIPS  | ICNet: Intra-saliency Correlation Network for Co-Saliency Detection | Wen-Da Jin, Ming-Ming Cheng, et al.                          | [Paper](https://proceedings.neurips.cc/paper/2020/file/d961e9f236177d65d21100592edb0769-Paper.pdf)/Code |
|   2020   | NeurIPS  | CoADNet: Collaborative Aggregation-and-Distribution Networks for Co-Salient Object Detection | Qijian Zhang, Runmin Cong                                    | [Paper](https://arxiv.org/pdf/2011.04887.pdf)/[Code](https://github.com/rmcong/CoADNet_NeurIPS20) |
|   2020   |   CVPR   | Taking a Deeper Look at Co-Salient Object Detection          | Deng-Ping Fan, et al                                         | Paper/[Proj](http://dpfan.net/CoSOD3k/)                      |
|   2020   |   CVPR   | Adaptive Graph Convolutional Network With Attention Graph Clustering for Co-Saliency Detection |                                                              |                                                              |
|   2020   |   ECCV   | Gradient-Induced Co-Saliency Detection                       |                                                              |                                                              |
|          |          |                                                              |                                                              |                                                              |
|          |          |                                                              |                                                              |                                                              |
|   2019   |   TIP    | Class Agnostic Image Common Object Detection                 | Shuqiang Jiang, et al.                                       |                                                              |
|   2019   |   TIP    | Deep Group-Wise Fully Convolutional Network for Co-Saliency Detection With Graph Propagation | Lina Wei, et al.                                             |                                                              |
|   2019   |   TIP    | Image Co-Saliency Detection and Co-Segmentation via Progressive Joint Optimization | Chung-Chi Tsai, et al.                                       |                                                              |
|   2019   |   TIP    | Salient Object Detection With Lossless Feature Reflection and Weighted Structural Loss | P. Zhang, Wei Liu, Huchuan Lu, et al.                        |                                                              |
|   2019   |   TCYB   | An Iterative Co-Saliency Framework for RGBD Images           | Runmin Cong, et al.                                          |                                                              |
|   2019   |   ICCV   | Group-Wise Deep Object Co-Segmentation With Co-Attention Recurrent Neural Network | Bo Li, et al.                                                | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Group-Wise_Deep_Object_Co-Segmentation_With_Co-Attention_Recurrent_Neural_Network_ICCV_2019_paper.pdf)/[Code](https://github.com/francesco-p/group-wise-iccv19) |
|   2019   |   CVPR   | Co-Saliency Detection via Mask-Guided Fully Convolutional Networks With Multi-Scale Label Smoothing |                                                              |                                                              |
|   2019   |   AAAI   | Robust Deep Co-Saliency Detection with Group Semantic        | Chong Wang, Zheng-Jun Zha, et al.                            | [Paper](https://ojs.aaai.org//index.php/AAAI/article/view/4919)/Code |
|   2019   |    MM    | A Unified Multiple Graph Learning and Convolutional Network Model for Co-saliency Estimation | Bo Jiang, et al.                                             | [Paper](https://doi.org/10.1145/3343031.3350860)/Code        |
|   2019   |    MM    | Co-saliency Detection Based on Hierarchical Consistency      | Bo Li, Zhengxing Sun, Quan Wang, Qian Li                     | [Paper](https://dl.acm.org/doi/10.1145/3343031.3351016)/Code |
|          |          |                                                              |                                                              |                                                              |
|          |          |                                                              |                                                              |                                                              |
|   2018   |   TIP    | Co-Saliency Detection for RGBD Images Based on Multi-Constraint Feature Matching and Cross Label Propagation | Ruiming Cong, et al.                                         |                                                              |
|   2018   |   TIP    | Co-Salient Object Detection Based on Deep Saliency Networks and Seed Propagation Over an Integrated Graph | Dong-ju Jeong, Insung Hwang, Nam Ik Cho                      |                                                              |
|   2018   |   TMM    | HSCS: Hierarchical Sparsity Based Co-saliency Detection for RGBD Images | Runmin Song, Huazhu Fu, et al.                               |                                                              |
|   2018   |   ECCV   | Unsupervised CNN-based Co-Saliency Detection with Graphical Optimization |                                                              |                                                              |
|   2018   |   AAAI   | Co-Saliency Detection Within a Single Image                  | Hongkai Yu, et al.                                           | [Paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16886)/Code |
|   2018   |    MM    | A Feature-Adaptive Semi-Supervised Framework for Co-saliency Detection | X. Zheng, Zheng-Jun Zha, Liansheng Zhuang                    | [Paper](https://dl.acm.org/doi/10.1145/3240508.3240648)/Code |
|          |          |                                                              |                                                              |                                                              |
|          |          |                                                              |                                                              |                                                              |
|   2017   |   PAMI   | Co-Saliency Detection via a Self-Paced Multiple-Instance Learning Framework | Dingwen Zhang, et al.                                        |                                                              |
|   2017   |   AAAI   | Image Cosegmentation via Saliency-Guided Constrained Clustering with Cosine Similarity | Zhiqiang Tao, Huazhu Fu, et al.                              | [Paper](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14331)/Code |
|   2016   |   IJCV   | Detection of Co-salient Objects by Looking Deep and Wide     | Dingwen Zhang, Junwei Han, et al.                            |                                                              |
|   2016   |   ECCV   | <span style="white-space:nowrap;">Image Co-segmentation Using Maximum Common Subgraph Matching and Region Co-growing &emsp;</span> | <span style="white-space:nowrap;">Avik Hati, S. Chaudhuri, Rajbabu Velmurugan&emsp;</span> |                                                              |




## Image SOD

#### 2021 

| **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|  arXiv08   | Unifying Global-Local Representations in Salient Object Detection with Transformer  | Sucheng Ren, Qiang Wen, et al. | [Paper](https://arxiv.org/abs/2108.02759)/Code
|  arXiv01   | Boundary-Aware Segmentation Network for Mobile and Web Applications | Xuebin Qin, Deng-Ping Fan, et al.                            | [Paper](https://arxiv.org/abs/2101.04704)/[Code](https://github.com/xuebinqin/BASNet) |
|   PAMI   | Part-Object Relational Visual Saliency                       | Yi Liu, Dingwen Zhang, et al.                                |                                                              |
|   PAMI   | Saliency Prediction in the Deep Learning Era: Successes and Limitations | Ali Borji                                                    |                                                              |
|   PAMI   | Salient Object Detection in the Deep Learning Era: An In-depth Survey | Wenguan Wang, et al.                                         |                                                              |
|   PAMI   | Revisiting Video Saliency Prediction in the Deep Learning Era | Wenguan Wang, et al.                                         |                                                              |
|   PAMI   | Relative Saliency and Ranking: Models, Metrics, Data and Benchmarks | Mahmoud Kalash, Md Amirul Islam, et al.                      |                                                              |
|   PAMI   | Instance-Level Relative Saliency Ranking with Graph Reasoning | Nian Liu, et al.                                             |                                                              |
|   PAMI   | Learning Saliency From Single Noisy Labelling: A Robust Model Fitting Perspective | Jing Zhang, Yuchao Dai, et al.                               |                                                              |
|   PAMI   | Deep Cognitive Gate: Resembling Human Cognition for Saliency Detection | Ke Yan, et al.                                               |                                                              |
|   PAMI   | A Highly Efficient Model to Study the Semantics of Salient Object Detection | Ming-Ming Cheng, et al.                                      |                                                              |
|   PAMI   | Learning to Detect Salient Object with Multi-source Weak Supervision | Hongshuang Zhang, Huchuan Lu, et al.                         |                                                              |
|   IJCV   | Saliency Detection Inspired by Topological Perception Theory | Peng Peng, Yong-Jie Li, etc                                  | [Paper](https://link.springer.com/article/10.1007/s11263-021-01478-4)/Code |
|   TIP    | Rethinking Image Salient Object Detection: Object-Level Semantic Saliency Reranking First, Pixelwise Saliency Refinement Later | Guangxiao Ma, et al.                                         |                                                              |
|   TIP    | Contour-Aware Loss: Boundary-Aware Learning for Salient Object Segmentation | Zixuan Chen, Jianhuang Lai, et al.                           |                                                              |
|   TIP    | Depth-Quality-Aware Salient Object Detection                 | Chenglizhao Chen, et al.                                     |                                                              |
|   TIP    | Dense Attention Fluid Network for Salient Object Detection in Optical Remote Sensing Images | Qijian Zhang, Runmin Cong, et al.                            |                                                              |
|   TIP    | SCG: Saliency and Contour Guided Salient Instance Segmentation | Nian Liu, et al.                                             |                                                              |
|   TIP    | Hierarchical and Interactive Refinement Network for Edge-Preserving Salient Object Detection | Sanping Zhou, et al.                                         |                                                              |
|   TIP    | Salient Object Detection With Purificatory Mechanism and Structural Similarity Loss | Jia Li, et al.                                               |                                                              |
|   TIP    | Hierarchical Edge Refinement Network for Saliency Detection  | D. Song, Yongsheng Dong, Xuelong Li                          |                                                              |
|   ICCV   | Scene Context-Aware Salient Object Detection                 | Avishek Siris, Jianbo Jiao, et al.                           | Paper/Code                                                   |
|   ICML   | DANCE: Enhancing saliency maps using decoys                  | Yang Lu, Willian Stafford Noble, et al.                      | Paper/[Code](https://bitbucket.org/noblelab/dance/src/master/) |
|   AAAI   | Pyramidal Feature Shrinking for Salient Object Detection     | Mingcan Ma, Changqun Xia, Jia Li                             |                                                              |
|   AAAI   | <span style="white-space:nowrap;">Locate Globally, Segment Locally: A Progressive Architecture With Knowledge Review Network for Salient Object Detection&emsp;</span> | <span style="white-space:nowrap;">Binwei Xu, et al.&emsp;&emsp;&emsp;&emsp;</span> | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16408)/Code |

#### 2020 

| **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| PAMI     | Inferring Salient Objects from Human Fixations               | Wenguan Wang, Jianbing Shen, et al.                          | Paper/[Code](https://github.com/wenguanwang/ASNet)           |
| PAMI     | Synthesizing Supervision for Learning Deep Saliency Network without Human Annotation | Dingwen Zhang, et al.                                        |                                                              |
| TIP      | PiCANet: Pixel-Wise Contextual Attention Learning for Accurate Saliency Detection | Nian Liu, Junwei Han, Ming-Hsuan Yang                        |                                                              |
| TIP      | Reverse Attention-Based Residual Network for Salient Object Detection | Shuhan Chen, Huchuan Lu, et al.                              |                                                              |
| TIP      | Boundary-Aware RGBD Salient Object Detection With Cross-Modal Feature Sampling | Yuzhen Liu, et al.                                           |                                                              |
| TCYB     | Complementarity-Aware Attention Network for Salient Object Detection | Junxia Li, Qingshan Liu, et al.                              |                                                              |
| NeurIPS  | Few-Cost Salient Object Detection with Adversarial-Paced Learning | Dingwen Zhang, Haibin Tian, et al.                           | Paper/[Code](https://github.com/hb-stone/FC-SOD)             |
| ECCV     | Suppress and Balance: A Simple Gated Network for Salient Object | Xiaoqi Zhao, Huchuan Lu, et al.                              | [Paper](https://arxiv.org/pdf/2007.08074.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency) |
| CVPR     | Inferring Attention Shift Ranks of Objects for Image Saliency | Avishek Siris, Jianbo Jiao, et al.                           | [Paper](https://jianbojiao.com/pdfs/CVPR20.pdf)/[Code](https://github.com/SirisAvishek/Attention_Shift_Ranks) |
| CVPR     | Weakly-Supervised Salient Object Detection via Scribble Annotations | Jing Zhang, Yuchao Dai, et al.                               | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Weakly-Supervised_Salient_Object_Detection_via_Scribble_Annotations_CVPR_2020_paper.html)/[Code](https://github.com/JingZhang617/Scribble_Saliency) |
| AAAI     | Learning Deep Relations to Promote Saliency Detection        | Changrui Chen, et al.                                        |                                                              |
| AAAI     | Global Context-Aware Progressive Aggregation Network for Salient Object Detection | <span style="white-space:nowrap;">Z. Chen, Runmin Cong, et al.&emsp;</span> | [Paper](https://ojs.aaai.org//index.php/AAAI/article/view/6633)/Code |
| AAAI     | <span style="white-space:nowrap;">F³Net: Fusion, Feedback and Focus for Salient Object Detection &emsp;</span> | Jun Wei, et al.                                              | [Paper](https://ojs.aaai.org//index.php/AAAI/article/view/6916)/[Code](https://github.com/weijun88/F3Net) |

#### 2019 

| **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   PAMI   | Salient Object Detection with Recurrent Fully Convolutional Networks | L. Wang, Huchuan Lu, et al.                                  |                                                              |
|   PAMI   | Deeply Supervised Salient Object Detection with Short Connections | Qibin Hou, et al.                                            |                                                              |
|   PAMI   | What Do Different Evaluation Metrics Tell Us About Saliency Models? | Zoya Bylinskii, et al.                                       |                                                              |
|   PAMI   | Personalized Saliency and Its Prediction                     | Y. Xu, Jingyi Yu, et al.                                     |                                                              |
|   IJCV   | Unsupervised Learning of Foreground Object Segmentation      | Ioana Croitoru, Simion-Vlad Bogolin, et al.                  |                                                              |
|   TIP    | Focal Boundary Guided Salient Object Detection               | Yupei Wang, et al.                                           |                                                              |
|   ICCV   | Employing Deep Part-Object Relationships for Salient Object Detection |                                                              |                                                              |
|   ICCV   | Selectivity or Invariance: Boundary-Aware Salient Object Detection | Jinming Su, Jia Li, et al.                                   |                                                              |
|   ICCV   | Stacked Cross Refinement Network for Edge-Aware Salient Object Detection |                                                              |                                                              |
|   CVPR   | Salient Object Detection With Pyramid Attention and Salient Edges | Wenguan Wang, Jianbing Shen, et al.                          | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Salient_Object_Detection_With_Pyramid_Attention_and_Salient_Edges_CVPR_2019_paper.pdf)/Code |
|   CVPR   | Attentive Feedback Network for Boundary-Aware Salient Object Detection |                                                              |                                                              |
|   CVPR   | Pyramid Feature Attention Network for Saliency Detection     |                                                              |                                                              |
|   AAAI   | Image Saliency Prediction in Transformed Domain: A Deep Complex Neural Network Method | Lai Jiang, et al.                                            |                                                              |
|   AAAI   | <span style="white-space:nowrap;">SuperVAE: Superpixelwise Variational Autoencoder for Salient Object Detection&emsp;</span> | <span style="white-space:nowrap;">Bo Li, Z. Sun, Yuqi Guo&emsp;</span> | [Paper](https://ojs.aaai.org//index.php/AAAI/article/view/4876)/Code |

#### 2018 

| **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   ECCV   | Reverse Attention for Salient Object Detection               |                                                              |                                                              |
|   ECCV   | Contour Knowledge Transfer for Salient Object Detection      |                                                              |                                                              |
|   ECCV   | Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks |                                                              |                                                              |
|   AAAI   | Lateral Inhibition-Inspired Convolutional Neural Network for Visual Attention and Saliency Detection | Chunshui Cao, Liang Wang, et al.                             |                                                              |
|   AAAI   | <span style="white-space:nowrap;">Recurrently Aggregating Deep Features for Salient Object Detection&emsp;</span> | <span style="white-space:nowrap;">Xiaowei Hu, et al.&emsp;</span> | [Paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16775)/Code |

#### 2017 

| **Pub.** | **Title**                                                    | **Author**                                                   | **Links** |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :-------- |
|   PAMI   | Ranking Saliency                                             | Lihe Zhang, Ming-Hsuan Yang, et al.                          |           |
|   IJCV   | Attentive Systems: A Survey                                  | Tam V. Nguyen, Qi Zhao, Shuicheng Yan                        |           |
|   IJCV   | <span style="white-space:nowrap;">Salient Object Detection: A Discriminative Regional Feature Integration Approach&emsp;</span> | <span style="white-space:nowrap;">Jingdong Wang, et al.&emsp;</span> |           |




## Appendix

- [SOD CNNs-based Read List](https://github.com/jiwei0921/SOD-CNNs-based-code-summary-)
- [RGB-D SOD Survey](https://github.com/taozh2017/RGBD-SODsurvey)

