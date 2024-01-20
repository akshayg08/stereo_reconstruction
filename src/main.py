import numpy as np 
from util import refineF
from helper import camera2
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def eigenSolve(A):
	u, d, v = np.linalg.svd(A, full_matrices=True)
	w = v[-1, :]
	return w

def enforce(x):
	u, d, v = np.linalg.svd(x, full_matrices=True)
	d[-1] = 0
	new_x = np.dot(u, np.dot(np.diag(d), v))
	return new_x

def calc_patch_diff(patch1, patch2, gauss_f):
	_, _, c = patch1.shape
	gauss_f = np.repeat(gauss_f[:, :, np.newaxis], c, axis=2)
	fp1 = patch1 * gauss_f 
	fp2 = patch2 * gauss_f
	return np.linalg.norm(fp1 - fp2)

def calc_inliers(pts1, pts2, F, tol):
	N = len(pts1)
	ones = np.ones((N, 1))
	pl = np.hstack((pts1, ones))
	lines = np.dot(pl, F.T)
	mask = [False]*N
	for i in range(N):
		a, b, c = lines[i]
		x2, y2 = pts2[i]
		dist = abs(a*x2 + b*y2 + c) / np.sqrt(a**2 + b**2)
		if dist <= tol:
			mask[i] = True

	return mask

def min_f(residuals):
	return (residuals**2).sum()

def eightpoint(pts1, pts2, M):
	# Replace pass by your implementation
	T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
	N = len(pts1)

	A = np.zeros((N, 9))
	for i in range(N):
		xl, yl = pts1[i]/M
		xr, yr = pts2[i]/M
		A[i] = np.array([xr*xl, xr*yl, xr, yr*xl, yr*yl, yr, xl, yl, 1])
	
	u, d, v = np.linalg.svd(A, full_matrices=True)
	h = v.T[:, -1]
	F_norm = h.reshape(3, 3)
	F_norm = refineF(F_norm, pts1/M, pts2/M)
	F = np.dot(T.T, np.dot(F_norm, T))

	np.savez('q2_1.npz', F=F, M=M)
	return F

def essentialMatrix(F, K1, K2):
	# Replace pass by your implementation
	E = np.dot(K2.T, np.dot(F, K1))
	np.savez("./q3_1.npz", E=E, F=F)
	return E

def triangulate(C1, pts1, C2, pts2):
	# Replace pass by your implementation
	N = len(pts1)
	w = []
	for i in range(N):
		x1, y1 = pts1[i][0], pts1[i][1]
		x2, y2 = pts2[i][0], pts2[i][1] 
		row1 = x1*C1[2] - C1[0]
		row2 = y1*C1[2] - C1[1]
		row3 = x2*C2[2] - C2[0]
		row4 = y2*C2[2] - C2[1]
		A = np.vstack((row1, row2, row3, row4))
		wi = eigenSolve(A)
		wi = wi / wi[-1]
		w.append(wi.reshape(1, -1))

	w = np.concatenate(w, axis=0)
	reconstucted_pts1 = np.dot(w, C1.T)
	reconstucted_pts2 = np.dot(w, C2.T)

	reconstucted_pts1 /= reconstucted_pts1[:, -1].reshape(-1, 1)
	reconstucted_pts2 /= reconstucted_pts2[:, -1].reshape(-1, 1)
	err = np.linalg.norm(pts1 - reconstucted_pts1[:, :2])**2 + \
				np.linalg.norm(pts2 - reconstucted_pts2[:, :2])**2 
	
	return w[:, :3], err

def epipolarCorrespondence(im1, im2, F, x1, y1):
	# Replace pass by your implementation

	window_size = 21
	sigma = 11
	gauss_f = gaussian(window_size, sigma).reshape(window_size, 1)
	gauss_f = np.outer(gauss_f, gauss_f)

	x1, y1 = int(x1), int(y1)
	patch1 = im1[y1 - window_size//2 : y1 + window_size//2 + 1, x1 - window_size//2 : x1 + window_size//2 + 1]

	pt1 = np.array([x1, y1, 1])	
	line = np.dot(F, pt1)

	h, w, _ = im2.shape
	min_patch_diff = 1e9
	match_x2 = None
	match_y2 = None
	for x2 in range(w):
		if x2 - window_size//2 < 0 or x2 + window_size//2 >= w:
			continue
		y2 = int(-1 * (line[0] * x2 + line[2])/line[1])
		if y2 - window_size//2 < 0 or y2 + window_size//2 >= h:
			continue

		dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
		if dist >= 50:
			continue

		patch2 = im2[y2 - window_size//2 : y2 + window_size//2 + 1, x2 - window_size//2 : x2 + window_size//2 + 1]
		patch_diff = calc_patch_diff(patch1, patch2, gauss_f)
		
		if patch_diff < min_patch_diff:
			min_patch_diff = patch_diff
			match_x2 = x2
			match_y2 = y2

		elif patch_diff == min_patch_diff:
			dist1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
			dist2 = np.sqrt((match_x2 - x1)**2 + (match_y2 - y1)**2)
			if dist1 < dist2:
				match_x2 = x2
				match_y2 = y2

	return match_x2, match_y2

def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
	# Replace pass by your implementation
	N = len(pts1)
	bestF = None
	bestInliers = None
	best = 0
	for i in range(nIters):
		indices = np.random.permutation(N)[:8]
		pot_F = eightpoint(pts1[indices], pts2[indices], M)
		inliers = calc_inliers(pts1, pts2, pot_F, tol)
		if sum(inliers) > best:
			best = sum(inliers)
			bestInliers = inliers
			bestF = pot_F
	
	# F = eightpoint(pts1[bestInliers], pts2[bestInliers], M)
	# final_inliers = calc_inliers(pts1, pts2, F, M)
	# return F, final_inliers

	return bestF, bestInliers

def rodrigues(r):
	# Replace pass by your implementation
	theta = np.linalg.norm(r)
	I = np.eye(3)
	if theta != 0:
		r = r / theta
	cross = np.array([[0, -r[2][0], r[1][0]], [r[2][0], 0, -r[0][0]], [-r[1][0], r[0][0], 0]])
	squared = np.dot(r, r.reshape(1, -1))
	R = np.cos(theta) * I + (1 - np.cos(theta)) * squared + np.sin(theta) * cross
	return R

def invRodrigues(R):
	# Replace pass by your implementation
	theta = np.arccos((np.trace(R) - 1)/2)
	w = np.array([[R[2][1] - R[1][2]], [R[0][2] - R[2][0]], [R[1][0] - R[0][1]]]) / (2*np.sin(theta))		
	r = theta * w
	return r

def rodriguesResidual(K1, M1, p1, K2, p2, x):
	# Replace pass by your implementation
	C1 = np.dot(K1, M1)
	t2 = x[-3:]
	r2 = x[-6:-3]
	R = rodrigues(r2.reshape(-1, 1))
	M2 = np.hstack((R, t2.reshape(-1, 1)))
	C2 = np.dot(K2, M2)

	P = x[:-6]
	N = len(P) // 3
	P = P.reshape(N, 3)
	ones = np.ones((N, 1))
	P = np.hstack((P, ones))

	proj_1 = np.dot(P, C1.T)
	proj_1 = proj_1 / proj_1[:, -1].reshape(-1, 1)
	proj_1 = proj_1[:, :2]

	proj_2 = np.dot(P, C2.T)
	proj_2 = proj_2 / proj_2[:, -1].reshape(-1, 1)
	proj_2 = proj_2[:, :2]

	residuals = np.concatenate([(p1-proj_1).reshape([-1]), (p2-proj_2).reshape([-1])])

	return residuals

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
	# Replace pass by your implementation
	N = len(P_init)
	R = M2_init[:, :3]
	t = M2_init[:, -1]
	r = invRodrigues(R)
	
	x = np.concatenate((P_init.flatten(), r.reshape(-1), t))

	residuals = rodriguesResidual(K1, M1, p1, K2, p2, x)
	sol = least_squares(min_f, x).x

	new_P = sol[:-6]
	new_r = sol[-6:-3]
	new_t = sol[-3:]

	new_R = rodrigues(new_r.reshape(-1, 1))
	M2 = np.hstack((new_R, new_t.reshape(-1, 1)))

	return M2, new_P.reshape(N, 3)
