# FCN-GoogLeNet
This is a repo for course project of [DD2424 Deep Learning in Data Science](https://www.kth.se/social/course/DD2424/) at KTH.

This project is a GoogLeNet Implementation of [*Fully Convolutional Networks for Semantic Segmentation, CVPR 2015*](https://github.com/shelhamer/fcn.berkeleyvision.org) in TensorFlow. Another Tensorflow implementation: [FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow). 

Our project is mainly based on these previous works and we performed several changes from them. We attach our [report](https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/doc/report.pdf) and [slides](https://docs.google.com/presentation/d/18vOFGMYoj6pqJS9KlWr9exc44tFza1LMqx5PiebpAF4/edit?usp=sharing) (with several introductory pages skipped for presentation) here for reference. 

**Model Downloads**

We provide models trained on two different datasets - [PASCAL VOC 2012](https://drive.google.com/open?id=0B4wMcSmM_17Ya25lY0pabmNRSUE) and [MIT Scene Parsing](https://drive.google.com/open?id=0B4wMcSmM_17YcEVka3BCT09mTUU). Please download the corresponding folder, rename it to ```logs``` and put them in your local repo to replace the old one. For more details, please read the subsection [**Visualize and test results**](#here).

**Detailed Origins**

- The "main" function was from [FCN.tensorflow/FCN.py](https://github.com/shekkizh/FCN.tensorflow/blob/master/FCN.py) with a little bit adaptation from [slim/train_image_classifier.py](https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py) with a few FLAGs and the "two-step" training procedure.
- The network took [inception_v3](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) directly and warm-started at [checkpoint](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz). 
- The upsampling layers were defined using [slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).
- The utility functions came from various projects with corresponding datasets.
- The bash scripts were written for [PDC](https://www.pdc.kth.se/) with GPU acceleration nodes.

## Why this repo?

In the original paper [*Fully Convolutional Networks for Semantic Segmentation*](https://arxiv.org/pdf/1411.4038.pdf), the authors mentioned several results of FCN-GoogLeNet and compared them with FCN-VGG16. The results showed a worse performance of GoogLeNet than VGG16 in semantic segmentation tasks. Two things make this conclusion questionable:

- Their GoogLeNet implementation is, however, still not open-sourced though they mentioned in their repo documentation that it is coming soon. 
- When the authors performed their training, they used their own reimplementation of GoogLeNet as the pre-trained model since there was no publicly available version of GoogLeNet at that time. 

Given the above two points, we are quite curious about how it would perform if a public version of GoogLeNet is actually put into use, and it would also be a good practice to fill the vacancy of open-source FCN-GoogLeNet. That's basically why we make this repo.

## Changes from previous work

- Pre-trained model: [VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) -> [GoogLeNet (inception v3)](https://github.com/tensorflow/models/tree/master/research/slim)
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

## How to run

### Requirements

- Python 3.5.0+
- TensorFlow 1.0+
- matplotlib 2.0.2+
- Cython 0.22+
- [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf) v2

<a name="here"></a>
### Visualize and test results 

First download model checkpoints ([PASCAL VOC](https://drive.google.com/open?id=0B4wMcSmM_17Ya25lY0pabmNRSUE) and [MIT Scene Parsing](https://drive.google.com/open?id=0B4wMcSmM_17YcEVka3BCT09mTUU)) we've trained and put it in folder ```logs``` and replace any other checkpoints if exist. Note if directory ```/logs/all``` doesn't exist, please create it by ```mkdir FCN-GoogLeNet/logs/all```. Then change the flags ```tf.flags.DEFINE_string('mode', "visualize", "Mode: train/ test/ visualize")``` at the beginning of the script ```inception_FCN.py``` to set the mode to visulize or test results. After that, run ```python inception_FCN.py``` from terminal to start running. The segmentation results are saved in the folder ```results```.

### View loss graph using TensorBoard

After training FCN (or downloading our models), you can launch Tensorboard by typing in ```tensorboard --logdir=logs/all``` in terminal when you are inside the folder ```FCN-GoogLeNet```. Then open your web browser and navigate to ```localhost:6060```. Graph of pixelwise training loss and validation loss is expected to view now.

### Fine-tune whole net

The following operations are needed if you want to train your own model from scratch. 

First delete all the files in ```/logs``` and ```/logs/all```. After this, you need to provide the path to a checkpoint from which to fine-tune. You can download the [checkpoint](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) of inception v3 model and correspondingly change ```tf.flags.DEFINE_string('checkpoint_path', '/path/to/checkpoint', 'The path to a checkpoint from which to fine-tune.')```. To avoid problems, it's better to directly copy the inception v3 model to ```/logs``` and change the above flag to ```tf.flags.DEFINE_string('checkpoint_path', 'logs/inception_v3.ckpt', ...)```, although this seems to be an unclever way. To train whole net, we need two steps:

**(1) Add upsampling layer on the top of inception v3; freeze lower layers and just train the output layer of the pretrained model and the upsampling layers:**

To acheive this, change ```tf.flags.DEFINE_string('trainable_scopes', ...)``` to ```'InceptionV3/Logits,InceptionV3/Upsampling'```. Make sure you've set the flag of ```skip_layers``` to the architecture you want. Set mode to train and run ```inception_FCN.py```. If the code is planned to run on PDC clusters, run ```sbatch ./batchpyjobunix.sh``` to submit your job to the queuing system [Slurm](https://www.pdc.kth.se/resources/computers/beskow/how-to/run).

**(2) Fine-tune all the variables:**

Change ```tf.flags.DEFINE_string('trainable_scopes', ...)``` to be None. Also remember to change ```tf.flags.DEFINE_string('checkpoint_path', ...)``` to ```'logs'```. Run ```inception_FCN.py``` again. If the code is planned to run on PDC clusters, run ```sbatch ./batchpyjobunix.sh``` to submit your job to the queuing system [Slurm](https://www.pdc.kth.se/resources/computers/beskow/how-to/run).

### Change dataset to MIT Scene Parsing

To train and test FCN on MIT Scene Parsing, two scripts should be changed manually as follow. Afterwards, you can play around with this new dataset according to the steps mentioned [above](#here).

**(1) Script ```inception_FCN.py```:**

- Import module ```read_MITSceneParsingData``` and comment out ```read_PascalVocData``` *(line 8-9)*;
- Change the flag of ```data_dir``` to the path of MIT Scene Parsing *(line 20-21)*;
- Set variable ```NUM_OF_CLASSES = 151``` *(line 59)*.

**(2) Script ```BatchDatSetReader.py```:**

- Change ```...[np.expand_dims(self._transform(filename['annotation'], True), ...)``` to ```...[np.expand_dims(self._transform(filename['annotation'], False), ...)``` *(line 39)*.

## Results

#### Training (orange) and validation (red) loss of FCN-8s trained on PASCAL VOC 2012
<p align="center">
<img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/losses_pascal.png"/>
</p>

#### Training (orange) and validation (red) loss of FCN-8s trained on MIT Scene Parsing
<p align="center">
<img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/losses_scene.png"/>
</p>

#### PASCAL VOC
<p align="center">
<img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/pic_0.png" width="430"/> <img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/pic_1.png" width="430">
</p>

#### MIT Scene Parsing
<p align="center">
  <img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/pic_2.png" width="390"/>   <img src="https://github.com/DeepSegment/FCN-GoogLeNet/blob/master/results/pic_3.png" width="400">
</p>

## Conclusions

- VGG16 net outperforms GoogLeNet when dealt with semantic segmentation tasks;
- A performance drop is expected when FCN is fed with a dataset containing large number of object classes;
- Objects which have more examples in the training set, e.g. vehicles, humans, are easier to be correctly identified.

## Future work

- Rewrite code so that some manual operations (like copying pretrained model, changing file path) can be avoided;
- Play around with parameters of the FCN trained on PASCAL VOC and try to find a better initialization; try to implement grid search or random search for some major parameters;
- Train FCNs on other segmentation dataset such as [MS COCO](http://cocodataset.org/#home).

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


