# Fashion MNIST Classification using CNN


This repository contains a simple Convolutional Neural Network (CNN) model for classifying images from the Fashion MNIST dataset. 
The model is implemented in Python using TensorFlow and Keras.

## Overview


Fashion MNIST is a dataset of Zalando's article images, consisting of 60,000 28x28 grayscale images of 10 fashion categories. 
The goal is to train a CNN model to accurately classify these images.

## Getting Started

### Prerequisites


Make sure you have the following dependencies installed:

numpy
pandas
matplotlib
scikit-learn
tensorflow


You can  install them using the following command:
```bash
pip install -r requirements.txt
```

### Download dataset


[Fashion MNIST Dataset] (https://github.com/zalandoresearch/fashion-mnist)

###  Model


The model is trained using the Fashion MNIST training dataset.
Training parameters such as learning rate, batch size, and dropout rate can be adjusted in the script.
Evaluate the model on the validation set to assess its performance. Evaluation results (including loss and accuracy) are printed at the end of the training process.
The trained model is saved to the model_checkpoint directory.
You can clone this project and run the code below to load this model for further evaluation or use in other applications.

```bash
load_model = tf.keras.models.load_model('model_checkpoint')
```

###  Unit test


You can unit test by running the following command：
```bash
python test_model.py
```
###  About cnn


CNN is favored in computer vision applications due to its good feature extraction capabilities, parameter sharing and sparse connections, strong adaptability, end-to-end learning and deep structure.

About the main contributions and characteristics of several classic CNN networks

Inception: By improving the width of the network, convolution kernels of different scales are introduced for feature extraction, as well as global average pooling and 1x1 convolution layers to reduce the number of parameters.

ResNet: Introduces a residual structure to solve the vanishing gradient problem. The network is deeper and has fewer parameters, while improving training efficiency and performance.

ResNeXt: Introduces the concept of cardinality, increases the width of the network through group convolution, and improves network performance.

DenseNet: Introducing a dense connection structure, each layer is connected to all subsequent layers, effectively utilizing features and reducing the number of parameters, alleviating the vanishing gradient problem and over-fitting.

SENet: By modeling the relationship between feature channels, it enhances useful features and suppresses useless features, thereby improving network performance.

##  Reference


【Google Team】Inception
[2014.09] Inception v1: [Link](https://arxiv.org/pdf/1409.4842.pdf)
[2015.02] Inception v2: [Link](https://arxiv.org/pdf/1502.03167.pdf)
[2015.12] Inception v3: [Link](https://arxiv.org/pdf/1512.00567.pdf)
[2016.02] Inception v4: [Link](https://arxiv.org/pdf/1602.07261.pdf)

【Microsoft】ResNet
[2015.12] ResNet: [Link](https://arxiv.org/pdf/1512.03385v1.pdf)

【Facebook】ResNeXt
[2016.11] ResNeXt: [Link](https://arxiv.org/pdf/1611.05431.pdf)

【Cornell & Tsinghua & Facebook】DenseNet
[2016.08] DenseNet: [Link](https://arxiv.org/pdf/1608.06993.pdf)

【Momenta】SENet
[2017.09] SENet: [Link](https://arxiv.org/pdf/1709.01507.pdf)


###### Author

xinkai fan
