# Salient Object Detection

A curated list of awesome resources for salient object detection (SOD), focusing more on multimodal SODs (e.g., RGB-D SOD). We will keep updating it.

:heavy_exclamation_mark:Updated 2022-06-27.



--------------------------------------------------------------------------------------

<!--TOC-->

## Content:
- [Overview](#Overview)
- [RGB-D SOD](#RGBD-SOD)
- [Image SOD](#Image-SOD)
- [Appendix](#Appendix)

--------------------------------------------------------------------------------------



## Overview

| **Year** | **Pub.** | **Title**                                      | **Links**                                              |
| :------: | :------: | :--------------------------------------- | :----------------------------------------------------------- |
|   2020   |   CVM    | RGB-D Salient Object Detection: A Survey <br> <sub><sup>*Tao Zhou, Deng-Ping Fan, Ming-Ming Cheng, Jianbing Shen, Ling Shao*</sup></sub> | [Paper](https://arxiv.org/abs/2008.00230)/[Proj](https://github.com/taozh2017/RGBD-SODsurvey) |



## RGBD SOD

### Preprint

| **Year** | **Pub.** | **Title**              | **Links**                                                    |
| :------: | :------- | :----------------------------------------------------------- | :------------------------------------------------------------ |
|   2022  |  arXiv | TANet: Transformer-based Asymmetric Network for RGB-D Salient Object Detection <br> <sub><sup>*Chang Liu, Gang Yang, et al.*</sup></sub>  | [Paper](https://arxiv.org/abs/2207.01172)/[Code](https://github.com/lc012463/TANet)
|   2022  | arXiv | GroupTransNet: Group Transformer Network for RGB-D Salient Object Detection <br> <sub><sup>*Xian Feng, Jinshao Zhu, et al.*</sup></sub>   | [Paper](https://arxiv.org/abs/2203.10785)/Cpde
|   2022  | arXiv | Dual Swin-Transformer based Mutual Interactive Network for RGB-D Salient Object Detection <br> <sub><sup>*Chao Zeng, Sam Kwong*</sup></sub> | [Paper](https://arxiv.org/abs/2206.03105)/Code
|   2021  | arXiv12 | TransCMD: Cross-Modal Decoder Equipped with Transformer for RGB-D Salient Object Detection <br> <sub><sup>*Youwei Pang, Lihe Zhang, et al.*</sup></sub>  | [Paper](https://arxiv.org/abs/2112.02363)/Code
|   2021   | arXiv12  | MutualFormer: Multi-Modality Representation Learning via Mutual Transformer <br> <sub><sup>*Xixi Wang, Bo Jiang, et al.*</sup></sub>   | [Paper](https://arxiv.org/abs/2112.01177)/Code
|   2021   | arXiv12  | Transformer-based Network for RGB-D Saliency Detection <br> <sub><sup>*Yue Wang, Huchuan Lu, et al.*</sup></sub>  | [Paper](https://arxiv.org/abs/2112.00582)/Code
|   2021   | arXiv09  | ACFNet: Adaptively-Cooperative Fusion Network for RGB-D Salient Object Detection <br> <sub><sup>*Jinchao Zhu, et al.*</sup></sub>    | [arXiv](https://arxiv.org/pdf/2109.04627.pdf)/Code           |
|   2021   | arXiv06  | Dynamic Knowledge Distillation with A Single Stream Structure for RGB-D Salient Object Detection <br> <sub><sup>*Guangyu Ren, Tania Stathaki*</sup></sub>      | [Paper](https://arxiv.org/pdf/2106.09517.pdf)/Code           |
|   2021   | arXiv06  | Progressive Multi-scale Fusion Network for RGB-D Salient Object Detection <br> <sub><sup>*Guangyu Ren, et al.*</sup></sub>      | [Paper](https://arxiv.org/pdf/2106.03941.pdf)/Code           |
|   2021   | arXiv04  | Middle-level Fusion for Lightweight RGB-D Salient Object Detection <br> <sub><sup>*N. Huang, Qiang Jiao, Qiang Zhang, Jungong Han*</sup></sub>          | [Paper](https://arxiv.org/pdf/2104.11543.pdf)/Code           |
|   2020   | arXiv12  | A Unified Structure for Efficient RGB and RGB-D Salient Object Detection <br> <sub><sup>*Peng Peng and Yong-Jie Li*</sup></sub>                       | [arXiv](https://arxiv.org/pdf/2012.00437.pdf)/Code           |
|   2020   | arXiv08  | <span style="white-space:nowrap;">Knowing Depth Quality In Advance: A Depth Quality Assessment Method For RGB-D Salient Object Detection&emsp; </span>   <br> <sub><sup>*Xuehao Wang, et al.*</sup></sub>    | [Paper](https://arxiv.org/abs/2008.04157)/[Code](https://github.com/XueHaoWang-Beijing/DQSF?utm_source=catalyzex.com) |




### 2022 

| **Year** | **Pub.** | **Title** |  **Links**                                       |
| :-----: | :------: | :----------------------------------------------------------- |   :----------------------------------------------------------- |
|   2022   |  AAAI | Self-Supervised Representation Learning for RGB-D Salient Object Detection  <br> <sub><sup>*Xiaoqi Zhao, Huchuan Lu, et al.*</sup></sub>       | [Paper](https://arxiv.org/pdf/2101.12482.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD) |
|   2022   |  TMM  | C2DFNet: Criss-Cross Dynamic Filter Network for RGB-D Salient Object Detection <br> <sub><sup>*Miao Zhang, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/abstract/document/9813422)/[Code](https://github.com/OIPLab-DUT/C2DFNet)
|   2022   |  TIP  | Boosting RGB-D Saliency Detection by Leveraging Unlabeled RGB Images  <br> <sub><sup>*Xiaoqiang Wang, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2201.00100)/Code




### 2021

| **Year** | **Pub.** | **Title**   | **Links**                                      |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   2021    |   ICCV   | Visual Saliency Transformer       <br> <sub><sup>*Nian Liu, Junwei Han, et al.*</sup></sub>  | [Paper](https://arxiv.org/pdf/2104.12099.pdf)/[Code](https://github.com/nnizhang/VST#visual-saliency-transformer-vst) |
|   2021    |   ICCV   | Specificity-preserving RGB-D Saliency Detection   <br> <sub><sup>*Tao Zhou, et al.*</sup></sub>  | [Paper](https://arxiv.org/abs/2108.08162)/[Code](https://github.com/taozh2017/SPNet) |
|   2021    |   ICCV   | RGB-D Saliency Detection via Cascaded Mutual Information Minimization <br> <sub><sup>*Jing Zhang, Deng-Ping Fan, et al.*</sup></sub>    | [Paper](https://arxiv-download.xixiaoyao.cn/pdf/2109.07246.pdf)/[Code](https://github.com/JingZhang617/cascaded_rgbd_sod) |
|   2021    |    MM    | Depth Quality-Inspired Feature Manipulation for Efficient RGB-D Salient Object Detection <br> <sub><sup>*Wenbo Zhang, Keren Fu, et al.*</sup></sub>    | [Paper](https://arxiv.org/abs/2107.01779)/[Code](https://github.com/zwbx/DFM-Net) |
|   2021    |    MM    | Cross-modality Discrepant Interaction Network for RGB-D Salient Object Detection <br> <sub><sup>*Chen Zhang, Runmin Cong, et al.*</sup></sub>   | [Paper](https://arxiv.org/abs/2108.01971)/[Proj](https://rmcong.github.io/proj_CDINet.html?utm_source=catalyzex.com) |
|   2021    |    MM    | TriTransNet: RGB-D salient object detection with a triplet transformer embedding network <br> <sub><sup>*Zhengyi Liu, et al.*</sup></sub>   | [Paper](https://arxiv.org/abs/2108.03990)/Code    |
|   2021    |   CVPR   | Deep RGB-D Saliency Detection With Depth-Sensitive Attention and Automatic Multi-Modal Fusion <br> <sub><sup>*Peng Sun, Xi Li, et al.*</sup></sub>   | [Paper](https://arxiv.org/abs/2103.11832)/[Code](https://github.com/sunpeng1996/DSA2F) |
|   2021    |   CVPR   | Calibrated RGB-D Salient Object Detection     <br> <sub><sup>*Wei Ji, Shuang Yu, et al.*</sup></sub>   | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.pdf)/[Code](https://github.com/jiwei0921/DCF) |
|   2021    |   ICME   | BTS-Net: Bi-directional Transfer-and-Selection Network For RGB-D Salient Object Detection <br> <sub><sup>*Wenbo Zhang, Keren Fu, et al.*</sup></sub>     | [arXiv](https://arxiv.org/pdf/2104.01784.pdf)/[Code](https://github.com/zwbx/BTS-Net) |
|   2021    |   AAAI   | <span style="white-space:nowrap;">RGB-D Salient Object Detection via 3D Convolutional Neural Networks &emsp; </span>   <br> <sub><sup>*Qian Chen, Keren Fu, et al.*</sup></sub> | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16191)/[Code](https://github.com/PPOLYpubki/RD3D) |
|      ---     |    ---      |    ---       |     ---      |
|   2021    |   PAMI   | Uncertainty Inspired RGB-D Saliency Detection <br> <sub><sup>*Jing Zhang, Deng-Ping Fan, Yuchao Dai, Saeed Anwaret, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/document/9405467)/Code                                                   |
|   2021    |   PAMI   | Siamese Network for RGB-D Salient Object Detection and Beyond <br> <sub><sup>*Keren Fu, Deng-Ping Fan, et al.*</sup></sub>       | Paper/Code                                                   |
|   2021    |   IJCV   | CNN-Based RGB-D Salient Object Detection: Learn, Select, and Fuse <br> <sub><sup>*Hao Chen, Youfu Li, et al.*</sup></sub>       | [Paper](https://arxiv.org/pdf/1909.09309.pdf)/Code           |
|   2021    |   TIP    | Hierarchical Alternate Interaction Network for RGB-D Salient Object Detection <br> <sub><sup>*Gongyang Li, Haibin Ling, et al.*</sup></sub>         | Paper/Code                                                   |
|   2021    |   TIP    | Bilateral Attention Network for RGB-D Salient Object Detection <br> <sub><sup>*Zhao Zhang, Deng-Ping Fan, et al.*</sup></sub>    | [arXiv](https://arxiv.org/abs/2004.14582)/[Code](https://github.com/zzhanghub/bianet) |
|   2021    |   TIP    | CDNet: Complementary Depth Network for RGB-D Salient Object Detection <br> <sub><sup>*Wen-Da Jin, Ming-Ming Cheng, et al.*</sup></sub>           | Paper/Code                                                   |
|   2021    |   TIP    | Data-Level Recombination and Lightweight Fusion Scheme for RGB-D Salient Object Detection <br> <sub><sup>*Xuehao Wang, et al.*</sup></sub>    | [Paper](https://arxiv.org/abs/2009.05102)/[Code](https://github.com/XueHaoWang-Beijing/DRLF?utm_source=catalyzex.com) |
|   2021    |   TIP    | RGB-D Salient Object Detection With Ubiquitous Target Awareness <br> <sub><sup>*Yifan Zhao, et al.*</sup></sub>                   | [arXiv](https://arxiv.org/abs/2109.03425)/Code               |
|   2021    |   TIP    | DPANet: Depth Potentiality-Aware Gated Attention Network for RGB-D Salient Object Detection <br> <sub><sup>*Zuyao Chen, Runmin Cong, et al.*</sup></sub>           | [Paper](https://arxiv.org/abs/2003.08608)/[Code](https://github.com/JosephChenHub/DPANet) |
|   2021    |   TIP    | Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection <br> <sub><sup>*Zhengzheng Tu, et al.*</sup></sub>       | Paper/Code                                                   |
|   2021    |   TCYB   | ASIF-Net: Attention Steered Interweave Fusion Network for RGB-D Salient Object Detection <br> <sub><sup>*Chongyi Li, Runmin Cong, et al.*</sup></sub>      | [Paper](https://ieeexplore.ieee.org/document/8998588)/[Code](https://github.com/Li-Chongyi/ASIF-Net) |
|   2021    |   TNNLS    | IRFR-Net: Interactive Recursive Feature-Reshaping Network for Detecting Salient Objects in RGB-D Images <br> <sub><sup>*Wujie Zhou, Qinling Guo, et al.*</sup></sub>     | [Paper](https://ieeexplore.ieee.org/abstract/document/9519891)/Code |
|   2021    |   TNNLS    | Rethinking RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks <br> <sub><sup>*Deng-Ping Fan, et al.*</sup></sub>                    | [Paper](https://arxiv.org/abs/1907.06781)/[Code](https://github.com/DengPingFan/D3NetBenchmark) |
|   2021    |   TMM    | Employing Bilinear Fusion and Saliency Prior Information for RGB-D Salient Object Detection <br> <sub><sup>*Nianchang Huang, Dingwen Zhang, et al.*</sup></sub>    | Paper/Code                                                   |
|   2021    |   TMM    | Attentive Cross-Modal Fusion Network for RGB-D Saliency Detection  <br> <sub><sup>*Di Liu, Kao Zhang, Zhenzhong Chen*</sup></sub>  | Paper/Cod                                                    |
|   2021    |   TMM    | Deep RGB-D Saliency Detection without Depth   <br> <sub><sup>*Yuan-Fang Zhang, et al.*</sup></sub>              | Paper/Code                                                   |
|   2021    |   TMM    | <span style="white-space:nowrap;">CCAFNet: Crossflow and Cross-scale Adaptive Fusion Network for Detecting Salient Objects in RGB-D Images&emsp; </span>  <br> <sub><sup>*Wujie Zhou, et al.*</sup></sub>     | Paper/Code        |





### 2020

| **Year** | **Pub.** | **Title**    | **Links**                                    |
| :-----: | :------: | :----------------------------------------------------------- |  :----------------------------------------------------------- |
|   2020    |   ECCV   | RGB-D Salient Object Detection with Cross-Modality Modulation and Selection <br> <sub><sup>*Chongyi Li, Runmin Cong, et al.*</sup></sub>             | [Paper](https://arxiv.org/abs/2007.07051)/[Code](https://github.com/Li-Chongyi/cmMS-ECCV20)<br>[Proj](https://li-chongyi.github.io/Proj_ECCV20) |
|   2020    |   ECCV   | Progressively Guided Alternate Refinement Network for RGB-D Salient Object Detection <br> <sub><sup>*Shuhan Chen, Yun Fu*</sup></sub>                   | [Paper](https://arxiv.org/abs/2008.07064)/[Code](https://github.com/ShuhanChen/PGAR_ECCV20?utm_source=catalyzex.com) |
|   2020    |   ECCV   | BBS-Net: RGB-D Salient Object Detection with a Bifurcated Backbone Strategy Network <br> <sub><sup>*Yingjie Zhai, Deng-Ping Fan, et al.*</sup></sub>            | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570273.pdf)/[Code](https://github.com/DengPingFan/BBS-Net)<br>[Extension](https://arxiv.org/pdf/2007.02713.pdf) |
|   2020    |   ECCV   | Cascade Graph Neural Networks for RGB-D Salient Object Detection  <br> <sub><sup>*Ao Luo, et al.*</sup></sub>   | [Paper](https://arxiv.org/abs/2008.03087)/[Code](https://github.com/LA30/Cas-Gnn?utm_source=catalyzex.com) |
|   2020    |   ECCV   | Cross-Modal Weighting Network for RGB-D Salient Object Detection <br> <sub><sup>*Gongyang Li, Haibin Ling, et al.*</sup></sub>            | [Paper](https://arxiv.org/pdf/2007.04901.pdf)/[arXiv](https://arxiv.org/abs/2007.04901)<br>[Code](https://github.com/MathLee/CMWNet) |
|   2020    |   ECCV   | Accurate RGB-D Salient Object Detection via Collaborative Learning <br> <sub><sup>*Wei Ji, Huchuan Lu, et al.*</sup></sub>     | [Paper](https://arxiv.org/abs/2007.11782)/[Code](https://github.com/jiwei0921/CoNet) |
|   2020    |   ECCV   | A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection <br> <sub><sup>*Xiaoqi Zhao, Lihe Zhang, et al.*</sup></sub>              | [Paper](https://arxiv.org/pdf/2007.06811.pdf)/[Code](https://github.com/Xiaoqi-Zhao-DLUT/DANet-RGBD-Saliency) |
|   2020    |   ECCV   | Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection <br> <sub><sup>*Youwei Pang, Lihe Zhang, Xiaoqi Zhao, Huchuan Lu*</sup></sub>  | [arXiv](https://arxiv.org/abs/2007.06227)/[Code](https://github.com/lartpang/HDFNet) |
|   2020    |   ECCV   | Asymmetric Two-Stream Architecture for Accurate RGB-D Saliency Detection <br> <sub><sup>*Miao Zhang, Huchuan Lu, et al.*</sup></sub>                | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730375.pdf)/[Code](https://github.com/OIPLab-DUT/ATSA) |
|   2020    |    MM    | Is Depth Really Necessary for Salient Object Detection?      <br> <sub><sup>*Jiawei Zhao, Yifan Zhao, Jia Li, Xiaowu Chen*</sup></sub>           | [Paper](https://arxiv.org/pdf/2006.00269.pdf)/[Code](https://github.com/iCVTEAM/DASNet)<br>[Proj](http://cvteam.net/projects/2020/DASNet/) |
|   2020    |   CVPR   | Select, Supplement and Focus for RGB-D Saliency Detection    <br> <sub><sup>*Miao Zhang, Yongri Piao, et al.*</sup></sub>                | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Select_Supplement_and_Focus_for_RGB-D_Saliency_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/OIPLab-DUT/CVPR_SSF-RGBD) |
|   2020    |   CVPR   | JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection <br> <sub><sup>*Keren Fu, Deng-Ping Fan, Ge-Peng Ji, Qijun Zhao*</sup></sub>         | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf)/[arXiv](https://arxiv.org/abs/2004.08515)/<br>[PAMI21](https://arxiv.org/pdf/2008.12134.pdf)/[Code](https://github.com/kerenfu/JLDCF/) |
|   2020    |   CVPR   | A2dele: Adaptive and Attentive Depth Distiller for Efficient RGB-D Salient Object Detection <br> <sub><sup>*Yongri Piao, Huchuan Lu, et al.*</sup></sub>                     | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Piao_A2dele_Adaptive_and_Attentive_Depth_Distiller_for_Efficient_RGB-D_Salient_CVPR_2020_paper.html)/Code |
|   2020    |   CVPR   | UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders <br> <sub><sup>*Jing Zhang, Deng-Ping Fan, et al.*</sup></sub>                   | [Paper](https://arxiv.org/abs/2009.03075)/[Code](https://github.com/JingZhang617/UCNet?utm_source=catalyzex.com) |
|   2020    |   CVPR   | Multi-Scale Interactive Network for Salient Object Detection <br> <sub><sup>*Youwei Pang, Huchuan Lu, et al.*</sup></sub>       | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/lartpang/MINet) |
|   2020    |   CVPR   | Interactive Two-Stream Decoder for Accurate and Fast Saliency Detection <br> <sub><sup>*H. Zhou, Xiaohua Xie, Jian-Huang Lai, et al.*</sup></sub>        | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/moothes/ITSD-pytorch) |
|   2020    |   CVPR   | Label Decoupling Framework for Salient Object Detection      <br> <sub><sup>*Jun Wei, et al.*</sup></sub>                        | Paper/[Code](https://github.com/weijun88/LDF)                |
|   2020    |   CVPR   | <span style="white-space:nowrap;">Learning Selective Self-Mutual Attention for RGB-D Saliency Detection&emsp;</span>   <br> <sub><sup>*Nian Liu, Ni Zhang, et al.*</sup></sub>    | Paper/[Code](https://github.com/nnizhang/SMAC?utm_source=catalyzex.com)<br>[Ext](https://arxiv.org/abs/2010.05537) |
|      ---     |    ---      |    ---       |     ---      |
|   2020    |   TIP    | RGBD Salient Object Detection via Disentangled Cross-Modal Fusion <br> <sub><sup>*Hao Chen, et al.*</sup></sub>      | Paper/Code                                                   |
|   2020    |   TIP    | RGBD Salient Object Detection via Deep Fusion           <br> <sub><sup>*Liangqiong Qu, et al.*</sup></sub>  | [Paper](http://www.shengfenghe.com/qfy-content/uploads/2019/12/4853ab33ac9a11e019f165388e57acf1.pdf)/Code |
|   2020    |   TIP    | Boundary-Aware RGBD Salient Object Detection With Cross-Modal Feature Sampling  <br> <sub><sup>*Yuzhen Niu, et al.*</sup></sub>              | Paper/Code                                                   |
|   2020    |   TIP    | ICNet: Information Conversion Network for RGB-D Based Salient Object Detection  <br> <sub><sup>*Gongyang Li, Zhi Liu, Haibin Ling*</sup></sub>           | Paper/Code                                                   |
|   2020    |   TIP    | Improved Saliency Detection in RGB-D Images Using Two-Phase Depth Estimation and Selective Deep Fusion <br> <sub><sup>*Chenglizhao Chen, et al.*</sup></sub>                     | Paper/Code                                                   |
|   2020    |   TCYB   | Discriminative Cross-Modal Transfer Learning and Densely Cross-Level Feedback Fusion for RGB-D Salient Object Detection <br> <sub><sup>*Hao Chen, Youfu Li, Dan Su*</sup></sub>                   | Paper/Code                                                   |
|   2020    |   TCYB   | Going From RGB to RGBD Saliency: A Depth-Guided Transformation Model <br> <sub><sup>*Runmin Cong, et al.*</sup></sub>  | Paper/Code                                                   |
|   2020    |   TMM    | cmSalGAN: RGB-D Salient Object Detection With Cross-View Generative Adversarial Networks <br> <sub><sup>*Bo Jiang, et al.*</sup></sub>                    | [Paper](https://arxiv.org/pdf/1912.10280.pdf)/[Code](https://github.com/wangxiao5791509/cmSalGAN_PyTorch)<br>[Proj](https://sites.google.com/view/cmsalgan/) |
|   2020    |   TMM    | <span style="white-space:nowrap;">Joint Cross-Modal and Unimodal Features for RGB-D Salient Object Detection &emsp;</span>   <br> <sub><sup>*N. Huang, Yi Liu, Qiang Zhang, Jungong Han*</sup></sub>    | Paper/Code 




### 2019 

| **Year** | **Pub.** | **Title**         | **Links**                               |
| :-----: | :------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
|   2019    |   ICCV   | EGNet: Edge Guidance Network for Salient Object Detection    <br> <sub><sup>*Jia-Xing Zhao, Ming-Ming Cheng, et al.*</sup></sub>             | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/JXingZhao/EGNet) |
|   2019    |   ICCV   | Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection   <br> <sub><sup>*Yongri Piao, Wei Ji Miao Zhang, et al.*</sup></sub>                | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Piao_Depth-Induced_Multi-Scale_Recurrent_Attention_Network_for_Saliency_Detection_ICCV_2019_paper.pdf)/[Code](https://github.com/jiwei0921/DMRA) |
|   2019    |   CVPR   | <span style="white-space:nowrap;">Contrast Prior and Fluid Pyramid Integration for RGBD Salient Object Detection&emsp;</span>   <br> <sub><sup>*Jia-Xing Zhao, Ming-Ming Cheng, et al.*</sup></sub>           | [Paper](http://mftp.mmcheng.net/Papers/19cvprRrbdSOD.pdf)/[Code](https://github.com/JXingZhao/ContrastPrior)<br>[Proj](https://mmcheng.net/rgbdsalpyr/) |
|   2019    |   CVPR   | BASNet: Boundary-Aware Salient Object Detection  <br> <sub><sup>*Xuebin Qin, Zichen Zhang, Chenyang Huang, Chao Gao, et al.*</sup></sub>         | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf)/[Code](https://github.com/xuebinqin/BASNet)<br>[Ext](https://arxiv.org/abs/2101.04704)        |
|      ---     |    ---      |    ---       |     ---      |
|   2019    |   TIP    | Three-Stream Attention-Aware Network for RGB-D Salient Object Detection <br> <sub><sup>*Hao Chen, Youfu Li*</sup></sub>   | [Paper](https://ieeexplore.ieee.org/document/8603756)/Code   |
|   2019    |   TIP    | RGB-'D' Saliency Detection With Pseudo Depth      <br> <sub><sup>*Xiaolin Xiao, Yicong Zhou, Yue-Jiao Gong*</sup></sub>                 |   [Paper](https://ieeexplore.ieee.org/document/8540072)/[Code](https://github.com/shellyXXL/RGB-D-saliency-detection-with-pseudo-depth)         |




### 2018

| **Year** | **Pub.** | **Title**                                                   | **Links** |
| :-----: | :------: | :----------------------------------------------------------- | :-------- |
|   2018    |   CVPR   | Progressively Complementarity-Aware Fusion Network for RGB-D Salient Object Detection <br> <sub><sup>*Hao Chen, Youfu Li*</sup></sub>   |     Paper/Code      |
|   2018    |   CVPR   | <span style="white-space:nowrap;">PiCANet: Learning Pixel-wise Contextual Attention for Saliency Detection &emsp;</span>              |     Paper/Code      |
|   2018    |   TCYB   | CNNs-Based RGB-D Saliency Detection via Cross-View Transfer and Multiview Fusion  <br> <sub><sup>*Junwei Han, et al.*</sup></sub>        |   Paper/Code        |




### 2017

| **Year** | **Pub.** | **Title**                                                   | **Links** |
| :-----: | :------: | :----------------------------------------------------------- | :-------- |
|   2017    |   CVPR   | Instance-Level Salient Object Segmentation       |   Paper/Code      |
|   2017    |   TIP    | Depth-Aware Salient Object Detection and Segmentation via Multiscale Discriminative Saliency Fusion and Bootstrap Learning  <br> <sub><sup>*Hangke Song, et al.*</sup></sub>      |      Paper/Code     |
|   2017    |   TIP    | <span style="white-space:nowrap;">Edge Preserving and Multi-Scale Contextual Neural Network for Salient Object Detection &emsp;</span>   <br> <sub><sup>*Xiang Wang, Huimin Ma, Xiaozhi Chen, Shaodi You*</sup></sub>   |  Paper/Code  |



### Beyond

| **Year** | **Pub.** | **Title**                                                    | **Links**      |
| :------: | :------: | :----------------------------------------------------------- |  :-------|
| 2021 | arXiv02 | CPP-Net: Context-aware Polygon Proposal Network for Nucleus Segmentation <br> <sub><sup>*Shengcong Chen, Changxing Ding, Minfeng Liu, Dacheng Tao*</sup></sub> | [Paper](https://arxiv.org/abs/2102.06867)/Code |
|   2019   | NeurIPS  | <span style="white-space:nowrap;">One-Shot Object Detection with Co-Attention and Co-Excitation &emsp;</span>   <br> <sub><sup>*Ting-I Hsieh, et al.*</sup></sub>                                         | [Paper](https://papers.nips.cc/paper/2019/file/92af93f73faf3cefc129b6bc55a748a9-Paper.pdf)/[Code](https://github.com/timy90022/One-Shot-Object-Detection) |
|   2020   | NeurIPS  | Deep Multimodal Fusion by Channel Exchanging                 <br> <sub><sup>*Yikai Wang, et al.*</sup></sub>      | [Paper](https://papers.nips.cc/paper/2020/file/339a18def9898dd60a634b2ad8fbbd58-Paper.pdf)/[Code](https://github.com/yikaiw/CEN) |
|   2021   |    MM    | Occlusion-aware Bi-directional Guided Network for Light Field Salient Object Detection <br> <sub><sup>*Dong Jing, Runmin Cong, et al.*</sup></sub>                               | Paper/Code                  |
|   2021   |   ICCV   | **Contrastive Multimodal Fusion with TupleInfoNCE**          <br> <sub><sup>*Yunze Liu, Li Yi, et al.*</sup></sub>                                     | [Paper](https://arxiv.org/pdf/2107.02575.pdf)/Code           |
|   2019   |   ICCV   | RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment <br> <sub><sup>*Guan'an Wang, et al.*</sup></sub>              | Paper/[Code](https://github.com/wangguanan/AlignGAN)         |
| 2021 | CVPR | Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation <br> <sub><sup>*Chufeng Tang, Xiaolin Hu, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html)/[Code](https://github.com/tinyalpha/BPR) |
| 2021 | CVPR | Boundary IoU: Improving Object-Centric Image Segmentation Evaluation <br> <sub><sup>*Bowen Cheng, Ross Girshick, Piotr Dollár, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)/[Code](https://bowenc0221.github.io/boundary-iou/) | 
| 2021 | AAAI | Active Boundary Loss for Semantic Segmentation  <br> <sub><sup>*Chi Wang, Yunke Zhang, Miaomiao Cui, Peiran Ren, et al.*</sup></sub> | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-2277.WangC.pdf)/Code |
|      ---     |    ---      |    ---       |     ---      |
|   2021   |   TCYB   | PANet: Patch-Aware Network for Light Field Salient Object Detection  <br> <sub><sup>*Yongri Piao, Huchuan Lu, et al.*</sup></sub>                              | Paper/Code                                                   |
| 2021 | TIP | Progressive Self-Guided Loss for Salient Object Detection <br> <sub><sup>*Sheng Yang, Weisi Lin, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2101.02412)/Code |
|   2020   |   TIP    | RGB-T Salient Object Detection via Fusing Multi-Level CNN Features <br> <sub><sup>*Qiang Zhang, Dingwen Zhang, et al.*</sup></sub>                           | Paper/Code                                                   |
|   2020   |   TIP    | LFNet: Light Field Fusion Network for Salient Object Detection <br> <sub><sup>*Miao Zhang, Huchuan Lu, et al.*</sup></sub>                               | Paper/Code                                                   |
|   2019   |   TIP    | Cross-Modal Attentional Context Learning for RGB-D Object Detection <br> <sub><sup>*G. Li, Liang Lin, et al.*</sup></sub> | Paper/Code                                                   |
|   2019   |   TMM    | RGB-T Image Saliency Detection via Collaborative Graph Learning   <br> <sub><sup>*Zhengzheng Tu, et al.*</sup></sub>                                        | Paper/Code                                                   |







## Image SOD

#### Preprint

| **Year.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
2022 | DFTR: Depth-supervised Fusion Transformer for Salient Object Detection | Heqin Zhu, et al. | [Paper](https://arxiv.org/abs/2203.06429)/Code




#### 2022

| **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
CVPR | Pyramid Grafting Network for One-Stage High Resolution Saliency Detection | Chenxi Xie, Jia Li, et al. | [Paper](https://arxiv.org/abs/2204.05041)/[Code](https://github.com/iCVTEAM/PGNet)
AAAI | Unsupervised Domain Adaptive Salient Object Detection Through Uncertainty-Aware Pseudo-Label Learning | Pengxiang Yan, Liang Lin, et al. | [Paper](https://arxiv.org/abs/2202.13170)/[Code](https://github.com/Kinpzz/UDASOD-UPL)
AAAI | Weakly-Supervised Salient Object Detection Using Point Supervison | Shuyong Gao, Wei Zhang, et al. | [Paper](https://arxiv.org/pdf/2203.11652.pdf)/[Code](https://github.com/shuyonggao/PSOD)



#### 2021 

| **Pub.** | **Title**                                                    | **Author**                                                   | **Links**                                                    |
| :------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| NeurIPS | Learning Generative Vision Transformer with Energy-Based Latent Space for Saliency Prediction | Jing Zhang, Jianwen Xie, et al. | [Paper](https://papers.nips.cc/paper/2021/file/8289889263db4a40463e3f358bb7c7a1-Paper.pdf)/Code
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

