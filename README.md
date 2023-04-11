# A Confusion Matrix for Evaluating Feature Attribution Methods

This repository contains the code needed to replicate the
experiments in our paper: [A Confusion Matrix for Evaluating Feature Attribution Methods](). 

The four feature attribution techniques used are:
* Layer-wise Relevance Propagation (LRP) [[1]](#1): the implementation used is based on the [work](https://github.com/kazuto1011/grad-cam-pytorch) of Nakashima *et al.*
* GradCAM [[6]](#6): the implementation used is based on the [work](https://github.com/jacobgil/pytorch-grad-cam)
of Gildenblat *et al.*
* LIME [[5]](#5): the implementation used is based on the [work](https://github.com/marcotcr/lime) of Tulio *et al.*
* Integrated Gradients (IG) [[7]](#7): the implementation used is based on the [work](https://github.com/pytorch/captum) of Kokhlikyan *et al.* [[2]](#2).


### Requirements

This code runs under Python 3.7.1. The python dependencies are defined in `requirements.txt`. 


### Available mosaics

These are the mosaics used in our experiments:

* [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/catsdogs_mosaic.zip)
* MIT67 [[4]](#4) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/mit670_mosaic.zip).
* MAMe [[3]](#3) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/mame_mosaic.zip).



## How to run the experiments

These are the bash scripts needed to compute the different scores.

#### Step 1. Mosaics explainability is computed and saved in: ```$PROJECT_PATH/data/explainability/```
> `cd $PROJECT_PATH/explainability/scripts/explainability_dataset/`
     
> `sh explainability_dataset_architecture_method.sh`

#### Step 2. The different scores are computed (Attribute-Accuracy, Attribute-Precision, Attribute-Recall and Attribute-F1) and saved in ```$PROJECT_PATH/data/explainability/```

> `cd $PROJECT_PATH/evaluation/scripts/evaluation_dataset/`

> `sh evaluation_dataset_architecture_method.sh`

#### Step 3. Visualize the scores distribution results.

> `cd $PROJECT_PATH/plots/scripts/plot_dataset/`

> `sh plot_dataset_architecture_method.sh`

where:
  * **dataset** must be exchanged by 
  one of the following datasets: **catsdogs**, **mit67** or **mame**. 
  * **architecture** must be **vgg16** or **resnet18**.
  * **method** must be **lrp**, **gradcam**, **lime** or **intgrad**.


For example, to get the scores for the Dogs vs. Cats dataset,
using the ResNet18 architecture and the GradCAM method,
run the following:

#### Step 1
> `cd $PROJECT_PATH/explainability/scripts/explainability_catsdogs/`

> `sh explainability_catsdogs_resnet18_gradcam.sh`

#### Step 2
> `cd $PROJECT_PATH/evaluation/scripts/evaluation_dataset/`

> `sh evaluation_catsdogs_resnet18_gradcam.sh`

#### Step 3
> `cd $PROJECT_PATH/plots/scripts/evaluation_dataset/`

> `sh plot_catsdogs_resnet18_gradcam.sh`

## Cite
Please cite our paper when using this code. 




## References
<a id="1">[1]</a>
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek,
W. (2015). On pixel-wise explanations for non-linear classifier decisions
by layer-wise relevance propagation. PloS one, 10(7), e0130140.


<a id="2">[2]</a>
Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., ... & Reblitz-Richardson, O. (2020). Captum: A unified and generic model interpretability library for pytorch. arXiv preprint arXiv:2009.07896.


<a id="3">[3]</a>
Parés, F., Arias-Duart, A., Garcia-Gasulla, D., Campo-Francés, G., Viladrich, N.,
Ayguadé, E., & Labarta, J. (2020). A Closer Look at Art Mediums: 
The MAMe Image Classification Dataset. arXiv preprint arXiv:2007.13693.

<a id="4">[4]</a>
Quattoni, A., & Torralba, A. (2009, June). Recognizing indoor scenes. 
In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 413-420). 
IEEE.

<a id="5">[5]</a>
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

<a id="6">[6]</a>
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra,
D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based
localization. In Proceedings of the IEEE international conference on computer
vision (pp. 618-626).


<a id="7">[7]</a>
Sundararajan, M., Taly, A., & Yan, Q. (2017, July).
Axiomatic attribution for deep networks. In International Conference on Machine Learning (pp. 3319-3328). PMLR.

