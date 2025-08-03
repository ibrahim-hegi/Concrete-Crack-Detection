# Concrete-Crack-Detection

## Overview

This project implements a crack recognition system using a Convolutional Neural Network (CNN) built with PyTorch. The system is designed to detect cracks in images, specifically from the **Cracks Dataset** available on Mendeley. The dataset contains images categorized into:

- **Positive**: Images with cracks  
- **Negative**: Images without cracks

---

## Dataset

The dataset used in this project is the **Cracks Dataset**. It includes:

- **20,000 Crack Images** (`Positive/` directory)  
- **20,000 No-Crack Images** (`Negative/` directory)

These images are used for training and evaluating the model to distinguish between surfaces with and without cracks.

---

## Requirements

To run the code in the provided Jupyter notebook (`Crack-recognition.ipynb`), you need the following dependencies:

- Python 3.11.5  
- PyTorch  
- Torchvision  
- NumPy  
- OpenCV (cv2)  
- Matplotlib  
- PIL (Pillow)  
- Other standard Python libraries: `os`, `time`, `copy`, `random`, `shutil`, `re`

Install required packages with:

```bash
pip install torch torchvision numpy opencv-python matplotlib pillow

