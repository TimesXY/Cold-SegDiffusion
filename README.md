# Cold SegDiffusion
This repository is an official implementation of the paper "Cold SegDiffusion: A Novel 
Diffusion Model for Medical Image Segmentation."

## Dataset
These medical images utilized in the experiments are collected from three public datasets: ISIC [1], TN3K [2], and REFUGE [3].
The references for the experimental datasets are given below:

[1] D. Gutman, N. C. Codella, E. Celebi, B. Helba, M. Marchetti, N. Mishra, A. Halpern, Skin lesion analysis toward melanoma detection: A challenge at the international symposium 
on biomedical imaging (isbi) 2016, hosted by the international skin imaging collab- oration (isic), arXiv preprint arXiv:1605.01397 (2016). 

[2] H. Gong, J. Chen, G. Chen, H. Li, G. Li, F. Chen, Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules,
Computers in Biology and Medicine 155 (2023) 106389. 

[3] J. I. Orlando, H. Fu, J. B. Breda, K. Van Keer, D. R. Bathula, A. DiazPinto, R. Fang, P.-A. Heng, J. Kim, J. Lee, et al., Refuge challenge: A unified 
framework for evaluating automated methods for glaucoma assessment from fundus photographs, Medical image analysis 59 (2020) 101570.

## Code Usage

## Installation

### Requirements

* Linux, CUDA>=11.3, GCC>=7.5.0
  
* Python>=3.8

* PyTorch>=1.11.0, torchvision>=0.12.0 (following instructions [here](https://pytorch.org/))

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```
  
### Dataset preparation

Please organize the dataset as follows:

```
ISIC_Med/
└── ISBI2016_ISIC_Dataset/
      ├── ISIC_0000000.jpg
      ├── ISIC_0000000_Segmentation.png
      ├── ISIC_0000001.jpg
      ├── ISIC_0000001_Segmentation.png
      ...
└── train.txt
└── valid.txt
└── test.txt
```

### Training

For example, the command for the training Cold SegDiffusion is as follows:

```bash
python driver.py
```
The configs in model_train.py or other files can be changed.

### Evaluation

After obtaining the trained Cold SegDiffusion, then run the following command to evaluate it on the validation set:

```bash
python sample.py
```

## Notes
The code of this repository is built on
https://github.com/TimesXY/Cold-SegDiffusion.
