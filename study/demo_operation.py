import torch
import numpy as np
import cv2

im_data = '/Users/great/Documents/姚圣/Pictures/IMG_1185.JPG'
cv2.imread(im_data)
cv2.waitKey(0)

a = np.zeros([2, 2])
out = torch.from_numpy(im_data)
print(out)
out = torch.flip(out, dims=[0])
out.to(torch.device('cpu'))
print(out)
