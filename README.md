# DASMN
![](https://img.shields.io/badge/language-python-orange.svg)
[![](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/fyancy/DASMN/blob/main/LICENSE)
[![](https://img.shields.io/badge/CSDN-燕策西-blue.svg)](https://blog.csdn.net/weixin_43543177?spm=1001.2101.3001.5343)
<!-- 如何设计GitHub badge: https://lpd-ios.github.io/2017/05/03/GitHub-Badge-Introduction/ -->

The PyTorch implementation for `Domain-adversarial similarity-based meta-learning network (DASMN)` in [Similarity-based meta-learning network with adversarial domain adaptation for cross-domain fault identification](https://www.sciencedirect.com/science/article/pii/S0950705121000927).

## Abstract
  With wide applications of intelligent methods in mechanical fault diagnosis, satisfactory results have been achieved. However, complicated and diverse practical working conditions would significantly reduce the performance of the diagnostic model that works well in the laboratory, i.e. domain shift occurs. To address the problem, this paper proposed a novel `similarity-based meta-learning network with adversarial domain adaptation for cross-domain fault identification`. The proposed `domain-adversarial similarity-based meta-learning network (DASMN)` consists of three modules: a feature encoder, a classifier and a domain discriminator. First, the encoder and the classifier implement the similarity-based meta-learning algorithm, in while the good generalization ability for unseen tasks is obtained. Then, adversarial domain adaptation is conducted by minimizing and maximizing the domain-discriminative error adversarially, which takes unlabeled source data and target data as inputs. The effectiveness of DASMN is evaluated by multiple cross-domain cases using three bearing vibration datasets and is compared with five well-established methods. Experimental results demonstrate the availability and outstanding generalization ability of the proposed method for cross-domain fault identification.
  
## Citation
If you have used our codes, or got help here or from our article, please cited our work in your own work. The BibTex format is as follows.
```
@article{FENG2021106829,
title = {Similarity-based meta-learning network with adversarial domain adaptation for cross-domain fault identification},
journal = {Knowledge-Based Systems},
volume = {217},
pages = {106829},
year = {2021},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2021.106829},
url = {https://www.sciencedirect.com/science/article/pii/S0950705121000927},
author = {Yong Feng and Jinglong Chen and Zhuozheng Yang and Xiaogang Song and Yuanhong Chang and Shuilong He and Enyong Xu and Zitong Zhou},
keywords = {Meta-learning, Domain adaptation, Adversarial learning, Fault identification}
```
