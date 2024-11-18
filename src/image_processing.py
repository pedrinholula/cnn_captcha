import cv2
import numpy as np
from PIL import Image

def image_processing(
		image_path,
		gaussian_kernel=None, sigma=0,
		median_kernel=None,
		closing_k=(5,5),
		dilation_k=(3,5),
		method = 'dilation'):
	kernel_d = np.ones(dilation_k, np.uint8)
	kernel_c = np.ones(closing_k, np.uint8)

	img = cv2.imread(image_path, 0)
	(h, w) = img.shape[:2]
	img = cv2.resize(img, (int(w*1.8), int(h*1.8)))
	ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

	if median_kernel != None:
		thresh = cv2.medianBlur(thresh, median_kernel)

	if gaussian_kernel != None:
		thresh = cv2.GaussianBlur(thresh, (gaussian_kernel, gaussian_kernel), sigma)

	tmp_path = "./data/tmp/" + image_path[-9:]
	if method == 'dilation' and dilation_k != None:
		dilation = cv2.dilate(thresh, kernel_d, iterations=1)
		dilation_image = Image.fromarray(dilation, mode="L")
		# dilation_image.save(tmp_path,format='PNG')
		# dilation_buffer = io.BytesIO()
		# dilation_image.save(dilation_buffer,format='PNG')
		return dilation
	elif method == 'closing' and closing_k != None:
		closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
		closing_image = Image.fromarray(closing, mode="L")
		# closing_image.save(tmp_path,format='PNG')
		# closing_buffer = io.BytesIO()
		# closing_image.save(closing_buffer,format='PNG')
		return closing
	else:
		return thresh

	# cv2.imshow('Original', img)
	# # cv2.imshow('Blur', blur)
	# cv2.imshow('Median', median)
	# cv2.imshow('Dilation', dilation)
	# cv2.imshow('Closing', closing)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()