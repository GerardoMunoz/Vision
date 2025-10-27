# Vision

## Overview
This document provides an overview of different computer vision techniques used for image processing, including their objectives, neural networks used, and real-world applications.

## Summary of how these techniques are applied in image processing:

| **Task**                 | **Objective**                                        | **Networks Used**                            | **Use Cases**                                                 |
| ------------------------ | ---------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------- |
| **Image Classification** | Assign a label to the entire image                   | Traditional CNNs (LeNet, AlexNet)            | Medical diagnosis, object classification in photos            |
| **Object Detection**     | Locate and classify multiple objects within an image | R-CNN, YOLO (v1–v9), Faster/Mask R-CNN, DETR | Autonomous vehicles, surveillance, retail analytics, robotics |
| **Image Segmentation**   | Assign labels to each pixel                          | FCN, U-Net, DeepLab                          | Medical imaging, agriculture, autonomous vehicles             |
| **Image Generation**     | Create realistic or transformed images               | GANs, VAEs, Diffusion Models                 | Generative art, deepfakes, text-to-image generation           |
| **Pose Estimation**      | Detect keypoints or skeletons of humans/objects      | OpenPose, HRNet, PoseNet, MediaPipe                     | Sports analytics, motion capture, AR/VR applications          |

## Examples of Object Detection
Here are a couple of examples of object detection applications:
* [Hand Detection](https://gerardomunoz.github.io/Vision/Hand_Detection.html) - A demo of detecting hands in real-time.
* [Hand Detection Landmark](https://gerardomunoz.github.io/Vision/Hand_Detection_LandMark.html) - Example showcasing hand landmark detection for gesture recognition.


## Challenge
Check out this video on building an object detection: [Object Detection Challenge](https://www.youtube.com/watch?v=sGDJwilP-oo).

## Image Classification

### Comparison of LeNet, AlexNet, VGG, ResNet, and MobileNet

| Feature                  | **LeNet**                          | **AlexNet**                                | **VGG**                                  | **ResNet**                                | **MobileNet**                           |
|--------------------------|-------------------------------------|--------------------------------------------|------------------------------------------|-------------------------------------------|-----------------------------------------|
| **Year**                 | 1989                               | 2012                                       | 2014                                     | 2015                                      | 2017                                    |
| **Developed By**         | Yann LeCun and team                | Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton | Visual Geometry Group (Oxford)          | Kaiming He and team (Microsoft Research) | Google                                  |
| **Input Image Size**     | 32x32 grayscale                    | 224x224 RGB                                | 224x224 RGB                              | 224x224 RGB                               | 224x224 RGB                             |
| **Number of Parameters** | ~60,000                            | ~60 million                                | ~138 million (VGG-16)                    | ~25.5 million (ResNet-50)                 | ~4.2 million (MobileNet-V1 α=1.0)       |
| **Layers**               | 2 Conv, 2 Pooling, 2 FC            | 5 Conv, 3 Pooling, 3 FC                    | Deep Conv Stacks, 3 FC                   | Residual Blocks, Global Avg Pool, FC      | Depthwise Separable Conv, Global Avg Pool, FC |
| **Activation Function**  | Tanh                               | ReLU                                       | ReLU                                     | ReLU                                      | ReLU6                                   |
| **Pooling**              | Average Pooling                    | Max Pooling                                | Max Pooling                              | Max Pooling                               | Global Average Pooling                  |
| **Regularization**       | None                               | Dropout                                    | Dropout                                  | Batch Normalization                       | Batch Normalization                     |
| **Skip Connections**     | No                                 | No                                         | No                                       | Yes                                       | No                                      |
| **Training**             | CPU                                | GPU                                        | GPU                                      | GPU                                       | Mobile-optimized                        |
| **Dataset**              | MNIST (10 classes)                 | ImageNet (1000 classes)                    | ImageNet (1000 classes)                  | ImageNet (1000 classes)                   | ImageNet (1000 classes)                 |
| **Top-5 Accuracy**       | ~99% (on MNIST)                    | ~83%                                       | ~90.0% (VGG-16)                          | ~93.3% (ResNet-50)                        | ~89.5%                                  |
| **Model Size**           | ~0.25 MB                           | ~240 MB                                    | ~528 MB (VGG-16)                         | ~97 MB (ResNet-50)                        | ~16 MB                                  |
| **Legacy**               | First practical CNN                | Sparked deep learning revolution           | Showed the value of deep networks        | Solved vanishing gradient problem         | Efficient for mobile and edge devices   |

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

A brief introduction to CNN:
* https://www.kaggle.com/code/gerardomunoz/seminario-2019-x/notebook

Challenge: Investigate the 90% Accuracy Limit
* https://gerardomunoz.github.io/Vision/mnist_90_.html

## Object Detection

| **Feature**                | **YOLO**                                    | **SSD**                                    | **Faster R-CNN**                          | **RetinaNet**                              |
|----------------------------|---------------------------------------------|-------------------------------------------|-------------------------------------------|--------------------------------------------|
| **Architecture**           | Single convolutional neural network (CNN)  | Single-stage detector with multi-scale predictions | Two-stage detector (RPN + classifier)     | Single-stage detector with Focal Loss     |
| **Speed (FPS)**            | Very fast (real-time detection)             | Fast (but slower than YOLO)               | Slower (requires region proposals)        | Slower than YOLO/SSD, faster than Faster R-CNN |
| **Accuracy**               | High, but less accurate than Faster R-CNN   | Moderate to High                          | Very high                                 | High, close to Faster R-CNN               |
| **Output Bounding Boxes**  | Anchor boxes + grid cell prediction         | Anchor boxes                              | Region proposals + bounding boxes         | Anchor boxes                              |
| **Training Complexity**    | Relatively simple                          | Moderate                                  | Complex                                   | Moderate to complex                       |
| **Key Strengths**          | Speed and efficiency for real-time tasks   | Good balance of speed and accuracy        | High accuracy and robust for small objects | Handles class imbalance well with Focal Loss |
| **Limitations**            | Struggles with small objects, dense scenes | Sensitive to scale and aspect ratio       | Computationally expensive                 | Computationally expensive                 |
| **Best Use Cases**         | Real-time applications (e.g., drones, surveillance) | Mobile devices and applications with limited resources | Tasks requiring high accuracy (e.g., medical imaging) | Datasets with high class imbalance        |
| **Anchor-based?**          | Yes                                         | Yes                                       | Yes                                       | Yes                                        |
| **Key Innovations**        | Grid cell object prediction, end-to-end model | Multi-scale feature maps                  | Region Proposal Network (RPN)             | Focal Loss for class imbalance            |
| **Versions**               | YOLOv1 to YOLOv8                           | SSD300, SSD512                            | Faster R-CNN                              | RetinaNet                                  |



* https://github.com/GerardoMunoz/Vision/blob/main/GroundingDINO.ipynb
* https://github.com/GerardoMunoz/Vision/blob/main/Zero_shot_object_detection_with_grounding_dino.ipynb




## Image Segmentation
* https://github.com/GerardoMunoz/Vision/blob/main/Automatic_mask_generator_example.ipynb
* https://github.com/GerardoMunoz/Vision/blob/main/segmentation_UNet.ipynb
## Image Generation
* https://github.com/GerardoMunoz/Vision/blob/main/cvae.ipynb
* https://github.com/GerardoMunoz/LinearCombinationGrade/blob/master/CalifCombLin_2020_4_09_wea001.ipynb

## Comparison of Lightweight TensorFlow.js Models for Efficient Image Processing
| **Model Name**         | **Image Suitability**                    | **Speed for Prediction**        | **Speed for Retraining Last Layers** | **Why Choose It**                                                                                          |
|-------------------------|------------------------------------------|----------------------------------|---------------------------------------|------------------------------------------------------------------------------------------------------------|
| **COCO-SSD**           | Best for object detection (e.g., people, objects) | Fast (real-time on most devices) | Moderate                              | Lightweight and optimized for speed.                                                                      |
| **MobileNet**          | Best for image classification            | Very Fast                        | Fast                                  | Pre-trained on ImageNet; great for classification tasks, lightweight, and fast inference.                 |
| **YOLO-Tiny**          | Best for simple object detection         | Fast (real-time on mobile)       | Moderate                              | Optimized for smaller devices; good balance between speed and object detection accuracy.                   |
| **PoseNet** (Optional) | Best for pose estimation and keypoints   | Moderate                         | Slow                                  | Useful for pose estimation; lightweight and well-suited for quick applications like tracking skeletons.    |

## MobileNet: Summary of Innovations and Architecture

* [MobileNet Detection Example Using WebCam](https://gerardomunoz.github.io/Vision/simple_MobileNet_ejem.html)

### 1. **Key Innovations**
MobileNet was designed for efficient deep learning on mobile and embedded devices, emphasizing low computational cost and memory usage.

#### Innovations:
- **Depthwise Separable Convolutions:**
  - Breaks standard convolution into two steps:
    1. **Depthwise Convolution:** Applies a single filter per input channel.
    2. **Pointwise Convolution:** Uses a 1x1 kernel to combine features from the depthwise convolution.
  - Reduces computational complexity significantly compared to standard convolutions.

- **Width Multiplier (α):**
  - Controls the number of channels in each layer.
  - Allows trade-off between model size/speed and accuracy.

- **Resolution Multiplier (ρ):**
  - Scales input image resolution (e.g., 224x224 → smaller sizes).
  - Reduces computational cost at the expense of accuracy.

- **ReLU6 Activation:**
  - A modified ReLU function that caps values at 6.
  - Improves stability in low-precision environments like mobile devices.

---

### 2. **MobileNet Architecture**
MobileNet consists of a series of depthwise separable convolutions, culminating in global average pooling and a fully connected layer for classification.

#### Architecture (e.g., MobileNet-V1):
1. **Input Layer:** 224x224 RGB image.
2. **Convolutional Layers:**
   - Initial standard convolution: 32 filters, 3x3 kernel, stride 2.
   - Followed by 13 Depthwise Separable Convolution blocks.
     - Each block consists of:
       - **Depthwise Convolution:** Spatial filtering (3x3 kernel).
       - **Pointwise Convolution:** Combines features (1x1 kernel).
       - Followed by BatchNorm and ReLU6 activation.
   - Feature maps reduce progressively through striding.
3. **Global Average Pooling:**
   - Reduces spatial dimensions to a single vector per channel.
4. **Fully Connected Layer:**
   - Outputs class probabilities (e.g., 1000 for ImageNet).

#### Example of Layer Breakdown:
| Layer Type                | Output Size | Details                                  |
|---------------------------|-------------|------------------------------------------|
| Input                     | 224x224x3   | RGB Image                               |
| Convolution (Standard)    | 112x112x32  | 32 filters, 3x3 kernel, stride 2        |
| Depthwise + Pointwise #1  | 112x112x64  | Depthwise (3x3), Pointwise (64 filters) |
| Depthwise + Pointwise #2  | 56x56x128   | Depthwise (3x3), Pointwise (128 filters, stride 2) |
| Depthwise + Pointwise #3  | 28x28x256   | Depthwise (3x3), Pointwise (256 filters, stride 2) |
| Depthwise + Pointwise #4  | 14x14x512   | Repeated 5 times with no striding       |
| Depthwise + Pointwise #5  | 7x7x1024    | Depthwise (3x3), Pointwise (1024 filters, stride 2) |
| Global Average Pooling    | 1x1x1024    | Compresses spatial dimensions           |
| Fully Connected (Output)  | 1000        | Softmax for classification              |

---

### 3. **Advantages of MobileNet**
- **Low Computational Cost:** Depthwise separable convolutions drastically reduce FLOPs (floating point operations).
- **Customizable:** Width and resolution multipliers allow scaling based on resource constraints.
- **Efficient for Mobile Devices:** Optimized for low-power hardware with minimal accuracy trade-offs.

---

### Summary Table: MobileNet Innovations and Architecture
| Feature                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Depthwise Separable Conv** | Reduces computation by splitting spatial and channel filtering tasks.      |
| **Width Multiplier (α)**      | Scales the number of channels, controlling model size and speed.          |
| **Resolution Multiplier (ρ)** | Adjusts input image resolution, trading accuracy for efficiency.          |
| **ReLU6 Activation**          | Enhances stability in mobile-friendly environments.                       |
| **Architecture**              | Series of depthwise separable convolutions, global avg pooling, FC layer. |


# Neural Network Architectures Overview

## 1. **LeNet-5 (1998)**
LeNet-5, developed by Yann LeCun, is one of the first Convolutional Neural Networks (CNNs) designed for digit recognition (e.g., MNIST dataset).

### Architecture:
- **Input Layer:** 32x32 grayscale image.
- **Layer 1:** Convolutional Layer (6 filters, 5x5 kernel, stride 1) → Activation (Sigmoid) → Subsampling (Average Pooling 2x2, stride 2).
- **Layer 2:** Convolutional Layer (16 filters, 5x5 kernel, stride 1) → Activation (Sigmoid) → Subsampling (Average Pooling 2x2, stride 2).
- **Layer 3:** Fully Connected Layer (120 neurons) → Activation (Sigmoid).
- **Layer 4:** Fully Connected Layer (84 neurons) → Activation (Sigmoid).
- **Output Layer:** Fully Connected Layer (10 neurons, one per class).  

---

## 2. **AlexNet (2012)**
Developed by Alex Krizhevsky, AlexNet revolutionized deep learning by leveraging GPUs for large-scale image classification on ImageNet.

### Architecture:
- **Input Layer:** 224x224 RGB image.
- **Layer 1:** Convolutional Layer (96 filters, 11x11 kernel, stride 4) → ReLU → Max Pooling (3x3, stride 2).
- **Layer 2:** Convolutional Layer (256 filters, 5x5 kernel, stride 1) → ReLU → Max Pooling (3x3, stride 2).
- **Layer 3:** Convolutional Layer (384 filters, 3x3 kernel, stride 1) → ReLU.
- **Layer 4:** Convolutional Layer (384 filters, 3x3 kernel, stride 1) → ReLU.
- **Layer 5:** Convolutional Layer (256 filters, 3x3 kernel, stride 1) → ReLU → Max Pooling (3x3, stride 2).
- **Fully Connected Layers:** 
  - FC1: 4096 neurons → ReLU → Dropout.
  - FC2: 4096 neurons → ReLU → Dropout.
  - Output: 1000 neurons (Softmax).

---

## 3. **VGG (2014)**
Proposed by the Visual Geometry Group (VGG) at Oxford, this network emphasizes simplicity by stacking small (3x3) convolutional filters.

### Architecture (e.g., VGG-16):
- **Input Layer:** 224x224 RGB image.
- **Convolutional Blocks:** 
  - Block 1: 2x[Conv (64 filters, 3x3, stride 1) → ReLU] → Max Pooling (2x2, stride 2).
  - Block 2: 2x[Conv (128 filters, 3x3, stride 1) → ReLU] → Max Pooling (2x2, stride 2).
  - Block 3: 3x[Conv (256 filters, 3x3, stride 1) → ReLU] → Max Pooling (2x2, stride 2).
  - Block 4: 3x[Conv (512 filters, 3x3, stride 1) → ReLU] → Max Pooling (2x2, stride 2).
  - Block 5: 3x[Conv (512 filters, 3x3, stride 1) → ReLU] → Max Pooling (2x2, stride 2).
- **Fully Connected Layers:** 
  - FC1: 4096 neurons → ReLU → Dropout.
  - FC2: 4096 neurons → ReLU → Dropout.
  - Output: 1000 neurons (Softmax).

---

## 4. **ResNet (2015)**
ResNet, or Residual Network, introduced the concept of skip connections to address the vanishing gradient problem in deep networks.

### Architecture (e.g., ResNet-50):
- **Input Layer:** 224x224 RGB image.
- **Initial Block:** Conv (64 filters, 7x7, stride 2) → BatchNorm → ReLU → Max Pooling (3x3, stride 2).
- **Residual Blocks:** (Each block contains convolutional layers + skip connection)
  - Block 1: 3x[1x1 Conv → 3x3 Conv → 1x1 Conv] with identity mapping (64 filters).
  - Block 2: 4x[1x1 Conv → 3x3 Conv → 1x1 Conv] with identity mapping (128 filters).
  - Block 3: 6x[1x1 Conv → 3x3 Conv → 1x1 Conv] with identity mapping (256 filters).
  - Block 4: 3x[1x1 Conv → 3x3 Conv → 1x1 Conv] with identity mapping (512 filters).
- **Fully Connected Layer:** 
  - Global Average Pooling → FC (1000 neurons, Softmax).

### Key Innovation:
- **Skip Connections:** Bypasses certain layers, enabling gradients to flow directly through the network, mitigating the vanishing gradient problem.

---

## Summary Table

| Architecture | Input Size | Key Features                             | Fully Connected Layers |
|--------------|------------|------------------------------------------|-------------------------|
| **LeNet-5**  | 32x32      | Small kernels, average pooling           | 2 FC layers            |
| **AlexNet**  | 224x224    | Large kernels, ReLU, Dropout             | 2 FC layers (4096)     |
| **VGG-16**   | 224x224    | Small 3x3 kernels, deep stack of layers  | 2 FC layers (4096)     |
| **ResNet-50**| 224x224    | Skip connections, identity mapping       | Global Avg Pool + FC   |

