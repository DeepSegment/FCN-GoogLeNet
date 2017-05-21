# Update Log
## Major Updates: #1
uploaded modified inception_v3_fcn (generate the model) based on slim/nets/inception_v3.py

uploaded modified inception_FCN (training and visualize script) based on tensorflow.FCN/FCN.py and slim/train_image_classifier.py

uploaded inception_utils.py from slim/nets because I think it's needed

More will be uploaded later if needed

Possible work for next update

cleanup inception_v3_fcn: use slim/nets/inception_v3.py as much as possible and separate upsampling part

minor mod for inception_FCN: i dont know if it will work 

## Minor Update: #1.1
added .gitignore to ignore data, model, and log folder

added .sh to note what params to use when run the inception_FCN

inception_FCN is our train/visualize script

when run it, it can take some params to specify mode, scope, and such

## Major Updates: #2
changed the default value of those params to run without specify them

used the whole inception_v3 for the model without space squeeze

(when it's written, it's already fully convolutional)

Bug: shape of upsampling is a problem

## Major Updates: #3
added a file that contain the shape for each layer for checking

selected arbitrary layer to do skip

Bug: never really train it but everything before that should be OK


