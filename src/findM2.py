import numpy as np 
import cv2
from main import eightpoint, essentialMatrix, triangulate
from helper import camera2, displayEpipolarF

I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")

h, w, c = I1.shape
M = max(h, w)

data = np.load("../data/some_corresp.npz")
intr = np.load("../data/intrinsics.npz")
K1 = intr["K1"]
K2 = intr["K2"]

pts1 = data["pts1"]
pts2 = data["pts2"]

F = eightpoint(pts1, pts2, M)

# displayEpipolarF(I1, I2, F)

E = essentialMatrix(F, K1, K2)
M2s = camera2(E)
zeros = np.zeros((3, 1))
C1 = np.hstack((K1, zeros))

fin_M2 = None
fin_C2 = None
p = None

for i in range(4):
    M2 = M2s[:, :, i]
    C2 = np.dot(K2, M2)
    w, err = triangulate(C1, pts1, C2, pts2)
    print(err)
    if (w[:, -1] <= 0).sum() == 0:
        fin_M2 = M2
        fin_C2 = C2
        p = w

