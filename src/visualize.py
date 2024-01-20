import numpy as np 
import cv2
from submission import eightpoint, essentialMatrix, triangulate, epipolarCorrespondence
from helper import displayEpipolarF, epipolarMatchGUI, camera2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")

h, w, c = I1.shape
M = max(h, w)

data = np.load("../data/some_corresp.npz")
pts1 = data["pts1"]
pts2 = data["pts2"]

intr = np.load("../data/intrinsics.npz")
K1 = intr["K1"]
K2 = intr["K2"]

F = eightpoint(pts1, pts2, M)
E = essentialMatrix(F, K1, K2)
M2s = camera2(E)
zeros = np.zeros((3, 1))
C1 = np.hstack((K1, zeros))
M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

hp_coords = np.load("../data/templeCoords.npz")
N = len(hp_coords["x1"])

new_pts1 = np.zeros((N, 2))
new_pts2 = np.zeros((N, 2))
for i in range(N):
    x1, y1 = hp_coords["x1"][i][0], hp_coords["y1"][i][0]
    x2, y2 = epipolarCorrespondence(I1, I2, F, x1, y1)
    new_pts2[i][0] = x2
    new_pts2[i][1] = y2
    new_pts1[i][0] = x1
    new_pts1[i][1] = y1

fin_M2 = None
fin_C2 = None
p = None

for i in range(4):
    M2 = M2s[:, :, i]
    C2 = np.dot(K2, M2)
    w, err = triangulate(C1, new_pts1, C2, new_pts2)
    print(err)
    if (w[:, -1] <= 0).sum() == 0:
        fin_M2 = M2
        fin_C2 = C2
        p = w

np.savez('q4_2.npz', F=F, M1=M1, M2=fin_M2, C1=C1, C2=fin_C2) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='r', marker='.')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

