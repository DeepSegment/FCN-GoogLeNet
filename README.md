# FCN-GoogLeNet
This is a repo for course project of [DD2424 Deep Learning in Data Science](https://www.kth.se/social/course/DD2424/) at KTH.

This project is a GoogLeNet Implementation of [Fully Convolutional Networks for Semantic Segmentation, CVPR 2015](https://github.com/shelhamer/fcn.berkeleyvision.org) in TensorFlow. Another Tensorflow implementation: [FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow). 

Our project is mainly based on these previous works and we performed several changes from them.

## Changes from previous work

- Pre-trained model: [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) -> [GoogLeNet (inception v3)](https://github.com/tensorflow/models/tree/master/slim)
- Framework: Caffe -> TensorFlow
- Datasets: [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) (20 classes) + [MIT Scene Parsing](http://sceneparsing.csail.mit.edu/) (150 classes)

## Pipeline

- Convolutionalize GoogLeNet into FCN-GoogLeNet
- Add upsampling layers on the top
- Fuse skip layers in network
- Fine-tune whole net from end to end

## Skip architectures

#### Skip FCNs
![skip FCNs](https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/skip.png)

## Results

#### PASCAL VOC
<img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/pic_0.png" width="430"/> <img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/pic_1.png" width="430">

#### MIT Scene Parsing
<img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/pic_2.png" width="430"/> <img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/pic_3.png" width="430">

## Related materials
### Here is the presentation given by the authors of the original paper.
http://techtalks.tv/talks/fully-convolutional-networks-for-semantic-segmentation/61606/

<!-- ## Notes from this presentation
- Step 1: reinterpret fully connected layer as conv layers with 1x1 output. (No weight changing)
- Step 2: add conv layer at the very end to do upsample.
- Step 3: put a pixelwise loss in the end

			along the way we have stack of features.

			closer to the input - higher resolution - shallow, local - where

			closer to the output - lower resolution - deep, global - what
- Step 4: skip to fuse layers. interpolate and sum.
- Step 5: Fine tune on per-pixel dataset, PASCAL

			I stopped at 8:30 in the video -->

### Extra Reading
#### This is about CONVERT fully connected layer to convolutional layer:
http://cs231n.github.io/convolutional-networks/#convert

#### This is someone basically did what we want to do
This guy's [Blog](http://warmspringwinds.github.io/blog/) and his [TensorFlow Image Segmentation](https://github.com/warmspringwinds/tf-image-segmentation) can be useful. 

#### This is a helpful tensorflow tutorial about transfer learning
https://github.com/Hvass-Labs/TensorFlow-Tutorials

<!-- Blog posts worth mentioning are: (some of this can also be found by the end of his project README)

[TFrecords Guide](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)

[Convert Classification Network to FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/)

[His Implementation on FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2017/01/23/fully-convolutional-networks-(fcns)-for-image-segmentation/)

[About Upsampling](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)

#### Some links about previous people asking about this but with no success. LOL:
http://stackoverflow.com/questions/38536202/how-to-use-inception-v3-as-a-convolutional-network
http://stackoverflow.com/questions/38565497/tensorflow-transfer-learning-implementation-semantic-segmentation -->


