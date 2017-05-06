# ConvNet-Segment
This is an initial repo for Deep Learning Project.

This is a Tensorflow Implementation of [Fully Convolutional Networks for Semantic Segmentation, CVPR 2015](https://github.com/shelhamer/fcn.berkeleyvision.org).

To do things a bit differently, we would like to take a [GoogLeNet (Inception v3)](https://github.com/tensorflow/models/tree/master/slim) and do this.

There is also a Tensorflow implementation here: [FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow).

This project would mostly based on these previous work.

# Edit this on Github Directly is WRONG!

# Things that Yang found interesting
## Here is the presentation given by the authors of the original paper.
http://techtalks.tv/talks/fully-convolutional-networks-for-semantic-segmentation/61606/

## Notes from this presentation
- Step 1: reinterpret fully connected layer as conv layers with 1x1 output. (No weight changing)
- Step 2: add conv layer at the very end to do upsample.
- Step 3: put a pixelwise loss in the end

			along the way we have stack of features.

			closer to the input - higher resolution - shallow, local - where

			closer to the output - lower resolution - deep, global - what
- Step 4: skip to fuse layers. interpolate and sum.
- Step 5: Fine tune on per-pixel dataset, PASCAL

			I stopped at 8:30 in the video

## This is about CONVERT fully connected layer to convolutional layer:
http://cs231n.github.io/convolutional-networks/#convert

## Some links about previous people asking about this but with no success. LOL:
http://stackoverflow.com/questions/38536202/how-to-use-inception-v3-as-a-convolutional-network
http://stackoverflow.com/questions/38565497/tensorflow-transfer-learning-implementation-semantic-segmentation
