# TransPIR
This repository contains the implementation for our paper Polarimetric Inverse Rendering for Transparent Shapes Reconstruction.

### Abstract

In this work, we propose a novel method for the detailed reconstruction of transparent objects by exploiting polarimetric cues. Most of the existing methods usually lack sufficient constraints and suffer from the over-smooth problem. Hence, we introduce polarization information as a complementary cue. We implicitly represent the object’s geometry as a neural network, while the polarization render is capable of rendering the object’s polarization images from the given shape and illumination configuration. Direct comparison of the rendered polarization images to the real-world captured images will have additional errors due to the transmission in the transparent object. To address this issue, the concept of reflection percentage which represents the proportion of the reflection component is introduced. The reflection percentage is calculated by a ray tracer and then used for weighting the polarization loss. We build a polarization dataset for multi-view transparent shapes reconstruction to verify our method. The experimental results show that our method is capable of recovering detailed shapes and improving the reconstruction quality of transparent objects.  

![1](https://raw.githubusercontent.com/s1752729916/githubsshaomq.github.iogithub/master/1.png)

### Dataset

Dataset will be available soon.

### 
