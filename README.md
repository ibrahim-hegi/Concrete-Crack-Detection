# Concrete-Crack-Detection

Overview
This project implements a crack recognition system using a convolutional neural network (CNN) built with PyTorch. The system is designed to detect cracks in images, specifically from the Cracks Dataset available on Mendeley. The dataset contains images categorized into "Positive" (images with cracks) and "Negative" (images without cracks).
Dataset
The dataset used in this project is the Cracks Dataset. It includes:

20,000 Crack Images (Positive directory)
20,000 No-Crack Images (Negative directory)

These images are used for training and evaluating the model to distinguish between surfaces with and without cracks.
Requirements
To run the code in the provided Jupyter notebook (Crack-recognition.ipynb), you need the following dependencies:

Python 3.11.5
PyTorch
Torchvision
NumPy
OpenCV (cv2)
Matplotlib
PIL (Pillow)
Other standard Python libraries (os, time, copy, random, shutil, re)

You can install the required packages using:
pip install torch torchvision numpy opencv-python matplotlib pillow

Project Structure

Crack-recognition.ipynb: The main Jupyter notebook containing the code for loading the dataset, visualizing images, training the model, and making predictions.
Positive/: Directory containing images with cracks.
Negative/: Directory containing images without cracks.
real_images/: Directory containing sample images for testing (e.g., road_surface_crack3.jpg).
pretrained_net_G.pth: The saved model weights after training.
base_model.pth: Another saved model file referenced in the notebook.

Key Components

Data Loading and Visualization:

The notebook loads images from the Positive and Negative directories and prints the number of images in each.
It visualizes a sample image (road_surface_crack3.jpg) after processing it with the predict_on_crops function, displaying the result using Matplotlib.


Model:

The notebook uses a PyTorch-based CNN (base_model) for crack detection.
The model is saved to `


