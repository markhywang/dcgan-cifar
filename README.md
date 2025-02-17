# Deep Convolutional Generative Adversarial Networks

## Background

Deep Convolutional Generative Adversarial Networks (DCGANs) are a class of GANs that incorporate deep convolutional layers to enhance the stability and performance of generative models. Introduced by Radford et al. (2015), DCGANs have become a widely used framework for generating realistic images from random noise.

Like standard GANs, a DCGAN consists of two adversarially trained neural networks: a generator and a discriminator. The generator takes random noise as input and progressively upsamples it through transposed convolutional layers to produce realistic images. The discriminator, on the other hand, is a convolutional neural network that classifies images as either real or generated. 

DCGANs introduced key architectural improvements over traditional GANs, including the removal of fully connected layers, the use of batch normalization for stable training, and the application of ReLU activations in the generator while using LeakyReLU in the discriminator. These modifications improved the training process and enabled DCGANs to generate high-quality images with finer details. 

Due to their effectiveness, DCGANs have been widely applied in image synthesis, super-resolution, and domain adaptation, influencing the development of more advanced GAN architectures such as StyleGAN and BigGAN.

## Description

In this repository, DCGANs will be used to generate images contained in the CIFAR-10 dataset by Alex Krizhevsky. The model architecture mimics that of the original DCGAN paper.

CIFAR Datasets: https://www.cs.toronto.edu/~kriz/cifar.html \
DCGAN Original Paper: https://arxiv.org/abs/1511.06434