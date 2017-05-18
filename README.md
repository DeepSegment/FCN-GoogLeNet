# ConvNet-Segment
This is an initial repo for Deep Learning Project.

This is a Tensorflow Implementation of [Fully Convolutional Networks for Semantic Segmentation, CVPR 2015](https://github.com/shelhamer/fcn.berkeleyvision.org).

To do things a bit differently, we would like to take a [GoogLeNet (Inception v3)](https://github.com/tensorflow/models/tree/master/slim) as our pre-trained model, add convolutional layers on the top (upsampling) and fine-tune it.

There is also a Tensorflow implementation here: [FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow).

This project would mostly based on these previous work.


## Here is the presentation given by the authors of the original paper.
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

## Avaliable Datasets
[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), 
[MS-Coco](http://mscoco.org/dataset/#overview),
[MIT Scene Parsing](http://sceneparsing.csail.mit.edu/)

## Extra Reading
### This is about CONVERT fully connected layer to convolutional layer:
http://cs231n.github.io/convolutional-networks/#convert

### This is someone basically did what we want to do
This guy's [Blog](http://warmspringwinds.github.io/blog/) and his [TensorFlow Image Segmentation](https://github.com/warmspringwinds/tf-image-segmentation) can be useful. 

### This is a helpful tutorial with some basic tensorflow implementations
https://github.com/Hvass-Labs/TensorFlow-Tutorials

<!-- Blog posts worth mentioning are: (some of this can also be found by the end of his project README)

[TFrecords Guide](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)

[Convert Classification Network to FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/)

[His Implementation on FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2017/01/23/fully-convolutional-networks-(fcns)-for-image-segmentation/)

[About Upsampling](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)

### Some links about previous people asking about this but with no success. LOL:
http://stackoverflow.com/questions/38536202/how-to-use-inception-v3-as-a-convolutional-network
http://stackoverflow.com/questions/38565497/tensorflow-transfer-learning-implementation-semantic-segmentation -->


