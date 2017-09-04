import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax


def post_process_crf(image, final_probabilities, num_cl):
	softmax = final_probabilities.squeeze()
	softmax = softmax.transpose((2, 0, 1))

	# The input should be the negative of the logarithm of probability values
	# Look up the definition of the unary_from_softmax for more information
	unary = unary_from_softmax(softmax, scale=None, clip=1e-5)

	# The inputs should be C-continious -- we are using Cython wrapper
	unary = np.ascontiguousarray(unary)

	d = dcrf.DenseCRF(image.shape[0] * image.shape[1], num_cl)

	d.setUnaryEnergy(unary)

	# This potential penalizes small pieces of segmentation that are
	# spatially isolated -- enforces more spatially consistent segmentations
	feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

	d.addPairwiseEnergy(feats, compat=3,
                    	kernel=dcrf.DIAG_KERNEL,
                    	normalization=dcrf.NORMALIZE_SYMMETRIC)

	# This creates the color-dependent features --
	# because the segmentation that we get from CNN are too coarse
	# and we can use local color features to refine them
	feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
									img=image, chdim=2)

	d.addPairwiseEnergy(feats, compat=10,
                     	kernel=dcrf.DIAG_KERNEL,
                     	normalization=dcrf.NORMALIZE_SYMMETRIC)
	Q = d.inference(10)
	res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

	return res