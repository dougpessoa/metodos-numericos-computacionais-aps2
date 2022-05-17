# Rode as importações
import numpy as np
from numpy.linalg import svd
import cv2
import os
from skimage import io
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

# Questão (a)
origem = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
image = io.imread(origem) 

img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(img)

cv2_imshow(img)
X = np.array(img)

# Questão (B)

U, Singular, V = svd(X)
print("U: ",U)
print("Valores Singulares:", Singular)
print("V^{T}",V.T)

# Questão (C)
PERCENTAGE = 80
size = int(PERCENTAGE * img.shape[0] / 100)

rr = np.dot(U[:, :size], np.dot(np.diag(Singular[:size]), V[:size, :]))

print("rr",rr)

# Questão (D)
cv2_imshow(rr)
