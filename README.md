# AI Upscale
This project uses a convolutional neural network to upscale images in patches of 64x64 -> 128x128 pixels.

# Dependencies
  - Python 3.6+

Tested on
 - Ubuntu 18.04
 - Python 3.8.13
 - Tensorflow 2.8.0
 - CUDA 11
 - OpenCV 4.5.5

# Dataset
This project scrapes data from google images.

# Usage
- Download this repository
 - Open Terminal/Command Prompt in the main GenderAndAgeRecognition directory
 - Install all the necessery libraries with:
```bash
pip install -r requirements.txt
```

- run the demo for image 
```bash
python demo.py <path_to_image>
```