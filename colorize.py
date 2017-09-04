import sys
import os
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import post_crf


def save_result(pic_id, image, final_probabilities, num_cl, save_img = True):
	folder_path = os.getcwd()
	pic_path = '/logs/all'
	result_path = '/results'
	class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']

	inp = image
	gt = misc.imread(folder_path + pic_path + '/gt_' + str(pic_id) + '.png')
	pred = misc.imread(folder_path + pic_path + '/pred_' + str(pic_id) + '.png')
	res = post_crf.post_process_crf(image, final_probabilities, num_cl)

	if save_img:
		fig, ax = plt.subplots(2, 2)
		cmap = plt.cm.jet
		bounds = np.linspace(0, num_cl, num_cl + 1)
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		plt.subplot(2, 2, 1)
		plt.imshow(inp)
		plt.title('image')
		plt.axis('off')
		plt.subplot(2, 2, 2)
		plt.imshow(gt, cmap=cmap, vmin=0, vmax=num_cl)
		plt.title('ground truth')
		plt.axis('off')
		plt.subplot(2, 2, 3)
		plt.imshow(pred, cmap=cmap, vmin=0, vmax=num_cl)
		plt.title('prediction')
		plt.axis('off')
		plt.subplot(2, 2, 4)
		plt.imshow(res, cmap=cmap, vmin=0, vmax=num_cl)
		plt.title('prediction with CRF')
		plt.axis('off')
	
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.7])
		labels = np.arange(0, num_cl, 1)
		loc = labels + .5
		cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
		cbar.set_ticks(loc)
		cbar.set_ticklabels(class_names)

		fig.savefig(folder_path + result_path + '/pic_' + str(pic_id) +'.png')

	return single_metrics(gt, pred, num_cl)


def save_compare_results(pic_id, num_cl):
	folder_path = os.getcwd()
	pic_path = '/logs/all'
	result_path = '/results'
	class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']

	inp = misc.imread(folder_path + pic_path + '/inp_' + str(pic_id) + '.png')
	gt = misc.imread(folder_path + pic_path + '/gt_' + str(pic_id) + '.png')
	pred_32s = misc.imread(folder_path + pic_path + '/pred_' + str(pic_id) + '.png')
	pred_16s = misc.imread(folder_path + pic_path + '/pred_' + str(pic_id + 1) + '.png')
	pred_8s = misc.imread(folder_path + pic_path + '/pred_' + str(pic_id + 2) + '.png')

	fig, ax = plt.subplots(2, 3)
	cmap = plt.cm.jet
	bounds = np.linspace(0, num_cl, num_cl + 1)
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	plt.subplot(2, 3, 1)
	plt.imshow(inp)
	plt.title('image')
	plt.axis('off')
	plt.subplot(2, 3, 2)
	plt.imshow(gt, cmap=cmap, vmin=0, vmax=num_cl)
	plt.title('ground truth')
	plt.axis('off')
	plt.subplot(2, 3, 3)
	plt.axis('off')
	plt.subplot(2, 3, 4)
	plt.imshow(pred_32s, cmap=cmap, vmin=0, vmax=num_cl)
	plt.title('FCN-32s')
	plt.axis('off')
	plt.subplot(2, 3, 5)
	plt.imshow(pred_16s, cmap=cmap, vmin=0, vmax=num_cl)
	plt.title('FCN-16s')
	plt.axis('off')
	plt.subplot(2, 3, 6)
	plt.imshow(pred_8s, cmap=cmap, vmin=0, vmax=num_cl)
	plt.title('FCN-8s')
	plt.axis('off')
	
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
	labels = np.arange(0, num_cl, 1)
	loc = labels + .5
	cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
	cbar.set_ticks(loc)
	cbar.set_ticklabels(class_names)

	fig.savefig(folder_path + result_path + '/pic_' + str(pic_id) +'.png')


def single_metrics(gt, pred, num_cl):
	t_px = np.zeros(num_cl)
	n_px = np.zeros(num_cl)
	n1_px = np.zeros(num_cl)
	px_class = np.unique(gt) 
	error = np.subtract(gt, pred)

	for i in px_class:
		t_px[i] += (np.where(gt == i)[0]).shape[0]
		n_px[i] += (np.where((gt == i) & (error == 0))[0]).shape[0]
		n1_px[i] += (np.where(pred == i)[0]).shape[0]

	return t_px, n_px, n1_px


# if __name__ == "__main__":
# 	pic_start = int(sys.argv[1])
# 	pic_end = int(sys.argv[2])
# 	num_cl = 22
# 	t_px = np.zeros(num_cl)
# 	n_px = np.zeros(num_cl)
# 	n1_px = np.zeros(num_cl)
	
# 	for idx in range(pic_start, pic_end+1):
# 		tmp_t, tmp_n, tmp_n1 = save_result(str(idx), num_cl, True)
# 		t_px += tmp_t
# 		n_px += tmp_n
# 		n1_px += tmp_n1

# 	t_sum = np.sum(t_px)
# 	n_sum = np.sum(n_px)
# 	px_acc = n_sum/t_sum
# 	condition_1 = t_px != 0
# 	c_n1 = np.extract(condition_1, n_px)
# 	c_t1 = np.extract(condition_1, t_px)
# 	condition_2 = (np.subtract(np.add(t_px, n1_px), n_px)) != 0
# 	c_n2 = np.extract(condition_2, n_px)
# 	c_d2 = np.extract(condition_2, (np.subtract(np.add(t_px, n1_px), n_px)))
# 	mean_acc = np.sum(np.divide(c_n1, c_t1))/num_cl
# 	mean_IU = np.sum(np.divide(c_n2, c_d2))/num_cl
# 	fw_IU = np.sum(np.divide(np.extract(condition_2, np.multiply(t_px, n_px)), c_d2))/t_sum
	
# 	print("========= metrics =========")
# 	print("pixel accuracy: " + str(px_acc))
# 	print("mean accuracy: " + str(mean_acc))
# 	print("mean IU: " + str(mean_IU))
# 	print("frequency weighted IU: " + str(fw_IU))
# 	print("")

# if __name__ == "__main__":
# 	num_cl = 22
# 	pic_id = int(sys.argv[1])
# 	save_compare_results(pic_id, num_cl)