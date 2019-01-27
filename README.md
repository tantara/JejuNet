# JejuNet

Real-Time Video Segmentation on Mobile Devices

##### Keywords

Video Segmentation, Mobile, Tensorflow Lite

##### Tutorials

* Benchmarks: Tensorflow Lite on GPU
  * A Post on Medium [Link](https://medium.com/@tantara/benchmarks-tensorflow-lite-on-gpu-769bff8afa6d)
  * Detail results [Link](https://www.dropbox.com/sh/6mtyfwhfasvfaun/AADG52s-5Q4aCjC8BmL1cA4xa?dl=0)

## Introduction

Running vision tasks such as object detection, segmentation in real time on mobile devices. Our goal is to implement video segmentation in real time at least 24 fps on Google Pixel 2. We use effiicient deep learning netwrok specialized in mobile/embedded devices and exploit data redundancy between consecutive frames to reduce unaffordable computational cost. Moreover, the network can be optimized with 8-bits quantization provided by tf-lite.

![Real-Time Video Segmentation(Credit: Google AI)](https://raw.githubusercontent.com/tantara/JejuNet/master/docs/real_time_video_segmentation_google_ai.gif)

*Example: Reai-Time Video Segmentation(Credit: Google AI)*

## Architecture

#### Video Segmentation

- Compressed [DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)[1]
  - Backbone: [MobileNetv2](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)[2]

#### Optimization

* 8-bits Quantization on [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/)

## Experiments

* Video Segmentation on Google Pixel 2
* Datasets
  * PASCAL VOC 2012

## Plan @Deep Learning Camp Jeju 2018

### July, 2018

- [x] DeepLabv3+ on tf-lite
- [x] Use data redundancy between frames
- Optimization
  - [x] Quantization
  - [x] Reduce the number of layers, filters and input size

## Results

More results here [bit.ly/jejunet-output](bit.ly/jejunet-output)

#### Demo

![DeepLabv3+ on tf-lite](https://raw.githubusercontent.com/tantara/JejuNet/master/docs/20180726-current-results-deeplabv3_on_tf-lite.gif)

*Video Segmentation on Google Pixel 2*

#### Trade-off Between Speed(FPS) and Accuracy(mIoU) 

![Trade-off Between Speed(FPS) and Accuracy(mIoU)](https://raw.githubusercontent.com/tantara/JejuNet/master/docs/trade-off-between-speed-fps-and-accuracy-miou.png)

#### Low Bits Quantization

| Network                | Input   | Stride     | Quantization(w/a) | PASCAL mIoU | Runtime(.tflite) | File Size(.tflite) |
| ---------------------- | ------- | ---- | ----------------- | ----------- | ---------------- | ------------------ |
| DeepLabv3, MobileNetv2 | 512x512 | 16     | 32/32             | 79.9%       | 862ms            | 8.5MB              |
| DeepLabv3, MobileNetv2 | 512x512 | 16     | 8/8               | 79.2%       | 451ms            | 2.2MB              |
| DeepLabv3, MobileNetv2 | 512x512 | 16     | 6/6               | 70.7%       | -                | -                  |
| DeepLabv3, MobileNetv2 | 512x512 | 16     | 6/4               | 30.3%       | -                | -                  |

![Low Bits Quantization](https://raw.githubusercontent.com/tantara/JejuNet/master/docs/low-bits-quantization.png)

## References

1. **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br>

   Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. arXiv: 1802.02611.<br>

   [[link]](https://arxiv.org/abs/1802.02611). arXiv: 1802.02611, 2018.

2. **Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation**<br />Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br />[[link]](https://arxiv.org/abs/1801.04381). arXiv:1801.04381, 2018.

## Authors

- [Taekmin Kim](https://www.linkedin.com/in/taekminkim/)(Mentee) [@tantara](https://www.linkedin.com/in/taekminkim/)
- Jisung Kim(Mentor) [@runhani](https://github.com/runhani)

## Acknowledgement

This work was partially supported by Deep Learning Jeju Camp and sponsors such as Google, SK Telecom. Thank you for the generous support for TPU and Google Pixel 2, and thank [Hyungsuk](https://github.com/corea) and all the mentees for tensorflow impelmentations and useful discussions.

## License

© [Taekmin Kim](https://www.linkedin.com/in/taekminkim/), 2018. Licensed under the [MIT](LICENSE) License.

