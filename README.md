# Joint compression of multi-model systems for edge devices

This repository is organized as follows : 
- [**Project Abstract**](#Abstract)
- [**Repository Structure**](#repository-structure)
- [**Data**](#data)

For specific information/documentation on a step of our pipeline, consult the folder corresponding to the project stage, according to the flowchart:

![flowchart](assets/project_flowchart.png?raw=true "Project Steps Flowchart")

## Environment Set-up

**Requirements**

If you wish to reproduce our results with PTQ and single-shot QAT, you are subject to OS requirements imposed by AIMET, which is compatible with either:

* Ubuntu LTS 22.06 and Python 3.11
* Ubuntu LTS 20.06 and Python 3.8

Please be aware that these requirements might change in the future, more information here.

**Setting up the environment**

We strongly advise conda/miniconda to manage the different Python versions that might be required by some frameworks used in this project. After cloning this repository, you may:

```
conda create --name my_env python=3.8
conda activate my_env
python -m pip install torch==1.13.1+cu116  --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install torchvision==0.14.1+cu116  --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
python -m pip install openmim
python -m mim install mmcv-full

```

Note that:

* The Torch and CUDA versions are fixed by AIMET. If you plan on running experiments on CPU only (not advised), you must used the same torch/torchvision versions, without the ```cu116``` reference
* The original YuNET codebase depends on an outdate version of mmcv. In order to install the correct versions, one must install ```openmim``` and then ```mim install mmcv-full```. From our experiments, all attempts to ```pip install mmcv``` fail to reproduce YuNET dependencies.

## Abstract

Deploying deep learning models for computer vision (CV) often requires a careful trade-off between performance, efficiency and cost. In the case of edge inference of machine learning systems, this trade-off is specially delicate, as computing and memory constraints imposed by the hardware severely limit the size and architecture of implementable models. These constraints are further aggravated in settings in which the same edge device must concurrently host several vision models running in series or in parallel. 

In this work, we benchmark model compression strategies and hardware platforms for the joint deployment of multiple CV models in three steps. First, we consider the problem of detecting human faces in unfavorable imaging conditions as a prototypical CV task requiring the concurrent implementation of multiple image restoration and detection models. Second, we evaluate the performance of pruning and quantization techniques for model compression in the context of our prototypical restoration and detection multi-model system, and propose **Joint Multi-Model Compression (JMMC)**, an adaptation of Quantization Aware Training (QAT) and pruning techniques in which the multi-model system is fine-tuned as a single unit with an adapted loss function. Lastly, we port the proposed multi-model system to an edge device containing a Hailo-8 accelerator, we explore the opportunities of parallel inference of the multi-model system and evaluate its efficiency in terms of power consumption and latency. We further discuss challenges in porting our system to memory-constrained microcontroller systems, taking the model porting framework available to Espressif's ESP32-S3 chip as a case study.
## Repository Structure
```
├───inference_experiments
│   ├───esp_eye
│   └───hailo
├───compression_experiments
│   ├───pruning
│   └───quantization
│       ├───INQ
│       └───single_shot_QAT 
├───logynthetic
├───model_artifacts 
├───report_artifacts 
├───dataset
└───utils
```


**Inference experiments** : Contain code and/or instructions for inference experiments in the Hailo and ESP devices, corresponding to Chapter 4 of the report.

**Compression experiments** : Contain code for compression experiments, corresponding to Chapter 3 of the report.

**logynthetic** : The dataset used in this work 

**model_artifacts** : weights and configurations for building models. Due to storage constraints, these are available only in the Google Drive backup of this repository.

**report_artifacts** : code and result files needed for generation of figures present in the report. 

**dataset** : dataset used for trained and evaluation, as well as code for generating it.

**utils** : Combination of utility functions that are repeteadly used in this project

## Data

All training, evaluation and calibration data used in this project is in the ```logynthetic``` folder. Due to git upload and storage constraints, the `logynthetic` folder is available only in the Google Drive backup of this repository.

The dataset splits are as follows:

* ***train*** : Used for training, QAT and pruning experiments (3500 images).
* ***test*** : Used in all evaluation pipelines, results are reported with respect to this split (571 images).
* ***calibration*** : In certain circunstances, compression algorithms might require additional data. In order to avoid test set spillage and to sanity check, we use this split in such algorithms (2000 images).

- **`Downstream tasks`**

- `Image Restoration`   
 - Models are always trained with image pairs, containing a noisy version and a clean version of the datapoint. All noisy images are suffixed with `_noisy`.
- `Low light enhancement`   
 - Models are always trained with image pairs, containing a moderately low light evrsion and a normal light version version of the datapoint. Normal light images are suffixed with `_0`, moderately low light versions are suffixed with `_1` and severely low light version are suffixed with `_2`.

 **Note on ground truth data for detector training and evaluation** : YuNET training/fine-tuning requires bboxes and landmarks in the RetinaNET format. These labels are available in the ```yunet_labels``` subfolder. If these must be edited, one should also edit the YuNET model constructor configuration file, available in ```model_artifacts/configs```.


