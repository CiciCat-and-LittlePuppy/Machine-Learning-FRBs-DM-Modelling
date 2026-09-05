This project uses three neural network based architecture to predict the dispersion measure (DM) of Fast Radio Burst (FRB) signals from waterfall data.
This repository is the model training part of the project.

# Repository Structure

The repository is organized into three main components:


Machine-Learning-FRBs-DM-Modelling/
│
├── Time_Inference_New/
│   ├── CNN.py                        # Measure Inference Time for the baseline CNN model
│   ├── LSTM.py                       # Measure Inference Time for the CNN-LSTM model
│   └── res50.py                      # Measure Inference Time for the ResNet50 model
│
├── Training and quick run/
│   ├── quick_run_trained_CNN.ipynb            # Load weights and quick test the baseline CNN model
│   ├── quick_run_trained_Hybrid.ipynb         # Load weights and quick test the CNN-LSTM model
│   ├── quick_run_trained_ResNet50.ipynb       # Load weights and quick test the ResNet50 model
│   ├── CNN_LSTM.py                            # Train the CNN-LSTM model
│   ├── CNN_faster.py                          # Train the baseline CNN model
│   └── res50.py                               # Train the ResNet50 model
│
├── bash file/run training and inference
│   ├── CNN.sh                        # Submit Task for training the baseline CNN model
│   ├── LSTM.sh                       # Submit Task for training the CNN-LSTM model
│   ├── res50.sh                      # Submit Task for training the ResNet50 model
│   ├── CNN_Inference_time.sh         # Submit Task for testing the inference time of the baseline CNN model
│   ├── LSTM_Inference_time.sh        # Submit Task for testing the inference time of the CNN-LSTM model
│   └── Resnet50_Inference_time.sh    # Submit Task for testing the inference time of the ResNet50 model
│
└── README.md

# Explanation
The input for each model shall have shape (1024, 512) for (time, frequency) as explained in the paper, with time resolution of about 1.67*7340/1024 ms.
Those bash files are used to run the training/testing tasks in the linux based clusters.

# Some Corrections
We found the following issues and unclear points in our scripts during further validation :(
1. We set up a learning rate scheduler for the ResNet50 Model (line 180) and the CNN-LSTM model (line 157), but did not call it in the later iteration loop, so it was not implemented successfully :(.
2. We replaced the original classification output block of the ResNet-50 model with a standard regression head (line 159). We then freeze all blocks, and unfreeze the last 2 convolutional blocks for gradient descent (line 165 - line 169)

# Reference
If you use this code in your research, please kindly cite our paper:

Rajabi, H., Liu, Z., Rajabi, F., & Houde, M. (2026). Machine-learning approaches to dispersion measure estimation for fast radio bursts. Astronomy and Computing, 101148.

BibTeX:

```bibtex
@article{RAJABI2026101148,
title = {Machine-learning approaches to dispersion measure estimation for fast radio bursts},
journal = {Astronomy and Computing},
volume = {57},
pages = {101148},
year = {2026},
issn = {2213-1337},
doi = {https://doi.org/10.1016/j.ascom.2026.101148},
url = {https://www.sciencedirect.com/science/article/pii/S2213133726000909},
author = {Hosein Rajabi and Zhejian Liu and Fereshteh Rajabi and Martin Houde},
keywords = {Transients: fast radio bursts, Methods: data analysis, Relativistic processes, Radiation: dynamics, Radiation mechanisms: non-thermal},
abstract = {Fast radio bursts (FRBs) are bright, mostly millisecond-duration transients of extragalactic origin whose emission mechanisms remain unknown. As FRB signals propagate through ionised media, they experience frequency-dependent delays quantified by the dispersion measure (DM), a key parameter for inferring source distances and local plasma conditions. Accurate DM estimation is therefore essential for characterising FRB sources and testing physical models, yet current dedispersion methods can be computationally intensive and prone to human bias. In this proof-of-concept study, we develop and benchmark three deep-learning architectures, a conventional convolutional neural network (CNN), a fine-tuned ResNet-50, and a hybrid CNN–LSTM model, for automated DM estimation. All models are trained and validated on a large set of synthetic FRB dynamic spectra generated using CHIME/FRB-like specifications. The hybrid CNN–LSTM achieves the highest accuracy and stability while maintaining low computational cost across the investigated DM range. Although trained on simulated data, these models can be fine-tuned on real CHIME/FRB observations and extended to future facilities, providing a pathway towards real-time, data-driven DM estimation in large FRB surveys with further development.}
}

Thank you for your visiting and interesting! ≽^•⩊•^≼ U・ᴥ・U
