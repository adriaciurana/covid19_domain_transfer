import numpy as np
import sys
import random
import os
import cv2
import traceback
import glob
import torch
from torch.utils import data
import math
from resizeimage import resizeimage
from PIL import Image
#pip install python-resize-image

class Covid19Dataset(data.Dataset):
	def __init__(self, 
		folder,
		do_aug, 
	    mean,
	    std, 
	    pad=32,
	    size=(224, 224)):
		self.folder = folder
		self.do_aug = do_aug
		self.size = size
		self.mean = mean
		self.std = std

		self.samples = [(p, 1.) for p in glob.glob(os.path.join(self.folder, "positives", "*"))] + \
			[(n, 0.) for n in glob.glob(os.path.join(self.folder, "negatives", "*"))]

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		im_path, label = self.samples[index]
		im = Image.open(im_path)
		im = resizeimage.resize_contain(im, self.size)
		im_r = np.array(im, dtype=np.float32)[..., :3]
		im_r[..., 0] -= self.mean[0]
		im_r[..., 1] -= self.mean[1]
		im_r[..., 2] -= self.mean[2]
		im_r[..., 0] /= self.std[0]
		im_r[..., 1] /= self.std[1]
		im_r[..., 2] /= self.std[2]
		
		im_t = torch.from_numpy(im_r)
		return im_t.permute(2, 0, 1), torch.tensor([label], dtype=im_t.dtype)

class Covid19DatasetUnlabeled(data.Dataset):
	def __init__(self, 
		folder,
		do_aug, 
	    mean,
	    std, 
	    pad=32,
	    size=(224, 224)):
		self.folder = folder
		self.do_aug = do_aug
		self.mean = mean
		self.std = std
		self.size = size

		self.samples = glob.glob(os.path.join(self.folder, "*"))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		im_path = self.samples[index]
		im = Image.open(im_path)
		im = resizeimage.resize_contain(im, self.size)
		im_r = np.array(im, dtype=np.float32)[..., :3]
		im_r[..., 0] -= self.mean[0]
		im_r[..., 1] -= self.mean[1]
		im_r[..., 2] -= self.mean[2]
		im_r[..., 0] /= self.std[0]
		im_r[..., 1] /= self.std[1]
		im_r[..., 2] /= self.std[2]


		im_t = torch.from_numpy(im_r)
		return im_t.permute(2, 0, 1)




