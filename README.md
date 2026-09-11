# PRODIGY_GA_04
Task 04 – Image-to-Image Translation with cGAN

An image-to-image translation project using a **Conditional Generative Adversarial Network (cGAN)** based on the **pix2pix architecture** to transform images from one visual domain into another.

## 📌 Overview

This project explores **image-to-image translation** using a Conditional GAN.

The pix2pix architecture learns a mapping between a source image and its corresponding target image. Unlike an unconditional GAN that generates images from random noise, a cGAN is conditioned on an input image, allowing the generated output to preserve important structures from the source.

## 🎯 Objectives

- Understand Conditional GANs (cGANs)
- Implement the pix2pix architecture
- Learn image-to-image translation
- Understand Generator and Discriminator networks
- Train a model using paired image datasets
- Generate translated images from input images
- Explore adversarial training

## 🧠 What is a cGAN?

A **Conditional Generative Adversarial Network** generates data based on additional information called a condition.

In this project:

```text
Input Image → Condition → Generator → Translated Image
````

The discriminator receives both the input image and generated/real target image and determines whether the translation is realistic.

## 🔄 Project Workflow

```text
        Paired Dataset
              ↓
      Image Preprocessing
              ↓
       Input Image + Target
              ↓
        ┌──────────────┐
        │   Generator  │
        └──────┬───────┘
               ↓
       Generated Image
               ↓
      ┌─────────────────┐
      │   Discriminator │
      └────────┬────────┘
               ↓
     Real or Generated?
               ↓
      Adversarial Training
               ↓
      Improved Generator
               ↓
     Translated Image
```

## 🏗️ pix2pix Architecture

The pix2pix framework consists primarily of two neural networks.

### Generator

The Generator receives an input image and attempts to produce a realistic target-domain image.

```text
Input Image
     ↓
Encoder
     ↓
Latent Representation
     ↓
Decoder
     ↓
Generated Image
```

### Discriminator

The Discriminator compares the input image with the target image and determines whether the image pair is real or generated.

```text
Input Image + Target Image
            ↓
       Discriminator
            ↓
      Real / Fake
```

## ⚙️ Technologies Used

* Python
* PyTorch / TensorFlow
* Conditional GAN
* pix2pix
* Deep Learning
* Computer Vision
* Image Processing
* NumPy
* OpenCV
* Matplotlib

## ✨ Key Features

* Conditional image generation
* Paired image-to-image translation
* Generator–Discriminator architecture
* Adversarial training
* Visual comparison of input and output
* Deep-learning-based image transformation

## 📂 Project Structure

```text
Task-04-cGAN-Image-Translation/
│
├── dataset/
│   ├── train/
│   ├── test/
│   └── val/
│
├── generated_images/
│
├── models/
│   ├── generator.py
│   └── discriminator.py
│
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

## 🚀 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-LINK>
cd Task-04-cGAN-Image-Translation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
torchvision
numpy
opencv-python
matplotlib
Pillow
```

## ▶️ Training

Prepare the paired dataset and run:

```bash
python train.py
```

The model learns the mapping between the source and target image domains through adversarial training.

## 🖼️ Image Translation

After training, provide an input image:

```bash
python test.py
```

The trained Generator produces the corresponding translated image.

## 📊 Example Applications

pix2pix-style models can be applied to tasks such as:

* Sketch → Realistic Image
* Edge Map → Photograph
* Black & White → Color Image
* Satellite Image → Map
* Day → Night Translation
* Label Map → Scene Image

The actual application depends on the paired dataset used for training.

## 📚 Learning Outcomes

Through this task, I learned:

* Fundamentals of GANs
* Conditional GAN architecture
* Generator and Discriminator networks
* Image-to-image translation
* Adversarial training
* Paired image datasets
* Deep-learning-based computer vision
* pix2pix architecture

## 🔮 Future Improvements

* Experiment with different datasets
* Add a Streamlit interface
* Compare different GAN architectures
* Implement image quality evaluation metrics
* Experiment with CycleGAN for unpaired translation
* Improve training stability
* Explore high-resolution image translation

## 🆚 GAN vs cGAN

| GAN                                  | cGAN                                 |
| ------------------------------------ | ------------------------------------ |
| Generates without explicit condition | Generation depends on a condition    |
| Usually starts from random noise     | Uses additional input information    |
| Less control over output             | More control over output             |
| General image generation             | Conditional generation / translation |
