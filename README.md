# Coronal Jet detection and localisation

**by [Adrien Joliat](https://github.com/adrienjoliat)**  

**[[Paper]](./MLO_Semester_project_first_sub.pdf)**

Machine Learning model to effectively detect and locate Solar Coronal Jet


## Main Results
First model based on the backbone SwinB and the TransVod Lite architecture to capture coronal jets.

![Results](TransVOD/TransVOD_Lite-main/Animations/animation_690.gif)



*Note:*
1. This model is trained  with pre-trained weights on COCO dataset.


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [TransVOD](https://github.com/SJTU-LuHe/TransVOD). An easier installation for Defromable DETR is available here : [Deformable DETR](https://github.com/adrienjoliat/Deformable-DETR)

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n TransVOD python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate TransVOD
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/)

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage
Below, I provide the final selected model:

[DownLoad Link of Google Drive](https://drive.google.com/file/d/1mrA1RFCxGWZrM9RSylFH9PVzlTUHb2Ra/view?usp=sharing)

This model should be put in TransVOD_Lite-main/exps/multi_model_jet/FINAL_PROJECT folder to be use by the notebook available in the main TransVOD lite folder, named : **Use_model_with_new_data.ipynb**

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
* [TransVOD Lite](https://github.com/qianyuzqy/TransVOD_Lite)