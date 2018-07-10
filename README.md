# JejuNet

Real-Time Video Segmentation on Mobile Devices

##### Keywords

Video Segmentation, Mobile, Tensorflow Lite

## Introduction

Running vision tasks such as object detection, segmentation in real time on mobile devices. Our goal is to implement video segmentation in real time at least 24 fps on Google Pixel 2. We use effiicient deep learning netwrok specialized in mobile/embedded devices and exploit data redundancy between consecutive frames to reduce unaffordable computational cost. Moreover, the network can be optimized with 8-bits quantization provided by tf-lite. More detail in [proposal](https://drive.google.com/open?id=1-HtlV1fkZKYup4Pw0ROzmj5eLMvPvhkE)

![Real-Time Video Segmentation(Credit: Google AI)](https://raw.githubusercontent.com/tantara/JejuNet/master/docs/real_time_video_segmentation_google_ai.gif)

*Example: Reai-Time Video Segmentation(Credit: Google AI)*

## Expected Result

- Video Segmentation in Real-Time(at least 24fps) on Google Pixel 2

## Architecture

#### Video Segmentation

- Compressed [DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)
  - Backbone: [MobileNetv2](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)

## Experiments

* Video Segmentation on Google Pixel 2
* Datasets
  * PASCAL VOC 2012
  * Youtube-VOS

## Plan @Deep Learning Camp Jeju 2018

### July, 2018

- [x] DeepLabv3+ on tf-lite
- [ ] Use data redundancy between frames
- Optimization
  - [x] Quantization
  - [ ] Reduce the number of layers, filters and input size
- [ ] Dynamic inference path

## References

1. **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br>

   Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. arXiv: 1802.02611.<br>

   [[link]](https://arxiv.org/abs/1802.02611). arXiv: 1802.02611, 2018.

2. **Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation**<br />Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br />[[link]](https://arxiv.org/abs/1801.04381). arXiv:1801.04381, 2018.

## Authors

- Taekmin Kim(Mentee) [@tantara](https://www.linkedin.com/in/taekminkim/)
- Jisung Kim(Mentor) [@runhani](https://github.com/runhani)

## Acknowledgement

This work was partially supported by Deep Learning Jeju Camp and sponsors such as Google, SK Telecom. Thank you for the generous support for TPU and Google Pixel 2.