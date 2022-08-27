import numpy as np
from scipy.spatial.distance import euclidean as euclidean_dis
import cv2


def trilateration(P1, P2, P3, r1, r2, r3):
	
	#ref:https://github.com/noomrevlis/trilateration/blob/master/trilateration3D.py
	#vector transformation: circle 1 at origin, circle 2 on x axis
	ex = (P2 - P1)/(np.linalg.norm(P2 - P1))
	i = np.dot(ex, P3 - P1)
	ey = (P3 - P1 - i*ex)/(np.linalg.norm(P3 - P1 - i*ex))
	ez = np.cross(ex,ey)
	d = np.linalg.norm(P2 - P1)
	j = np.dot(ey, P3 - P1)

	#plug and chug using above values
	x = (pow(r1,2) - pow(r2,2) + pow(d,2))/(2*d)
	y = ((pow(r1,2) - pow(r3,2) + pow(i,2) + pow(j,2))/(2*j)) - ((i/j)*x)

	# only one case shown here
	z = np.sqrt(pow(r1,2) - pow(x,2) - pow(y,2))
	
	#triPt is an array with ECEF x,y,z of trilateration point
	T1 = P1 + x*ex + y*ey + z*ez
	T2 = P1 + x*ex + y*ey - z*ez

	return T1, T2


def P3P(points3D, points2D, cameraMatrix, distCoeffs):
	# points3D 3x4
	# points2D 2x4

	#print("P3P start")
	
	#step1 transform 2D points from image coordinate sys to camera coordinate sys
	inv_cameraMatrix = np.linalg.pinv(cameraMatrix)
	homo_points2D = np.concatenate((points2D, np.array([[1,1,1,1]])), axis=0) #3x4
	ccs_points2D = inv_cameraMatrix @ homo_points2D  #3x3 x 3x4 = 3x4
	ccs_points2D = ccs_points2D / np.linalg.norm(ccs_points2D, axis=0) #change to unit vector
	#print(ccs_points2D)
	
	#step2 calculate angle and distanc for 3D points (Cab, Cac, Cbc, Rab, Rac, Rbc)

	Cab = np.dot(ccs_points2D[:, 0], ccs_points2D[:, 1])
	Cac = np.dot(ccs_points2D[:, 0], ccs_points2D[:, 2])
	Cbc = np.dot(ccs_points2D[:, 1], ccs_points2D[:, 2])

	Rab = euclidean_dis(points3D[:, 0], points3D[:, 1])
	Rac = euclidean_dis(points3D[:, 0], points3D[:, 2])
	Rbc = euclidean_dis(points3D[:, 1], points3D[:, 2])


	#step3 calculate distance ||x1 - T|| = a, ||x2 - T|| = b, ||x3 - T|| = c

	K1 = (Rbc / Rac) ** 2
	K2 = (Rbc / Rab) ** 2
	
	G4 = (K1 * K2 - K1 - K2) ** 2 - 4 * K1 * K2 * (Cbc ** 2)
	G3 = 4 * (K1 * K2 - K1 - K2) * K2 * (1 - K1) * Cab \
		+ 4 * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2 * K2 * Cab * Cbc)
	G2 = (2 * K2 * (1 - K1) * Cab) ** 2 \
        + 2 * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) \
        + 4 * K1 * ((K1 - K2) * (Cbc**2) + K1 * (1 - K2) * (Cac ** 2) - 2 * (1 + K1) * K2 * Cab * Cac * Cbc)
	G1 = 4 * (K1 * K2 + K1 - K2) * K2 * (1 - K1) * Cab \
        + 4 * K1 * ((K1 * K2 - K1 + K2) * Cac * Cbc \
		+ 2 * K1 * K2 * Cab * (Cac ** 2))
	G0 = (K1 * K2 + K1 - K2) ** 2 \
        - 4 * (K1 ** 2) * K2 * (Cac ** 2)

	quartic_poly = [G4, G3, G2, G1, G0]
	#print("qp",quartic_poly)
	poly_root = np.roots(quartic_poly)
	x = np.array([np.real(r) for r in poly_root if np.isreal(r)])

	#solve y by 2 equation
	m1, p1, q1 = (1 - K1), 2 * (K1 * Cac - x * Cbc), (x ** 2 - K1)
	m2, p2, q2 = 1, 2 * (-x * Cbc), (x ** 2) * (1 - K2) + 2 * x * K2 * Cab - K2
	y = -1 * (m2 * q1 - m1 * q2) / (p1 * m2 - p2 * m1)
	#print("Rab",Rab)
	#print("Cab",Cab)
	#b = xa , c = ya
	#print((Rab ** 2) / (1 + (x ** 2) - 2 * x * Cab))
	
	a = np.sqrt((Rab ** 2) / (1 + (x ** 2) - 2 * x * Cab))

	b = x * a
	c = y * a
	#print(a)
	#step4 get 2 possible camera center(T)
	centers = []
	for i in range(len(a)):
		tmp_c1, tmp_c2 = trilateration(points3D[:, 0], points3D[:, 1], points3D[:, 2], a[i], b[i], c[i])
		centers.append(tmp_c1)
		centers.append(tmp_c2)
	

	#step5 calculate lambda and R
	possible_sol = []
	
	for center in centers:
		center = center.reshape((3,1))
		for sign in [1,-1]:
			lambda_ = sign * np.linalg.norm((points3D[:, :3] - center), axis=0)
			R = (lambda_ * ccs_points2D[:, :3]) @ np.linalg.pinv(points3D[:, :3] - center )
			possible_sol.append([R, center, lambda_])		
			
	best_R = possible_sol[0][0]
	best_T = possible_sol[0][1]
	min_error = np.inf

	#step6 use 4th point to choose best result
	for R, T, lambda_ in possible_sol:
		projected_2D = cameraMatrix @ R @ (points3D[:, 3].reshape(3, 1) - T)
		projected_2D /= projected_2D[-1]
		error = np.linalg.norm(projected_2D[:2, :] - points2D[:, 3].reshape(2, 1))
		if error < min_error:
			best_R = R
			best_T = T
			min_error = error

	return best_R, best_T
	




	
	