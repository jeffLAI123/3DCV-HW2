
import os

from scipy.spatial.transform import Rotation
import open3d as o3d
import warnings

import pandas as pd
import numpy as np
import cv2
from p3p import P3P


def load_point_cloud(points3D_df: pd.DataFrame):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd

def create_camera_position(camera_intrinsic, R, T, img_size, color=[1, 0, 0]):
    #print(R,T)
    R = R.reshape((3,3))
    #image plane initialization  3x4
    camera_plane = np.array([[0, 0, 1],
                              [0, img_size[0], 1],
                              [img_size[1], img_size[0], 1],
                              [img_size[1], 0, 1]]).T
    camera_plane = camera_plane / 5 # just because origin one is too big
    v = np.linalg.pinv(camera_intrinsic) @ camera_plane #3x4
    camera_model_3d = (np.linalg.pinv(R) @ v + T) # 3x4
    # add center
    camera_model_3d = np.concatenate((camera_model_3d, T), axis=1) #3x5
    camera_model_3d = camera_model_3d.T # 5x3
 
    # create o3d object
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(camera_model_3d),
        lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    )
    colors = np.tile(color, (8, 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set
def image_undistortion(points, distCoeffs, size):

	
    points /= np.array([size[1], size[0]]).reshape((2, 1))

    center = np.array([0.5, 0.5]).reshape((2, 1))
    r = np.linalg.norm((points - center), axis=0)

    xc, yc = center[0], center[1]
    x, y = points[0], points[1]
    k1, k2, p1, p2 = distCoeffs[0], distCoeffs[1], distCoeffs[2], distCoeffs[3]

    xu = x + (x - xc) * (k1 * (r**2) + k2 * (r**4)) + \
	    (p1 * (r**2 + 2*((x - xc)**2)) + 2 * p2 * (x - xc) * (y - yc))
    yu = y + (y - yc) * (k1 * (r**2) + k2 * (r**4)) + \
	    (p2 * (r**2 + 2*((y - yc)**2)) + 2 * p1 * (x - xc) * (y - yc))
    undistorted_points = np.vstack((xu, yu)) * np.array([size[1], size[0]]).reshape((2, 1))
	
    return undistorted_points

def get_transform_mat(rotation, translation, scale):
    r_mat = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

#np.degrees: Convert angles from radians to degrees.
#np.clip: Given an interval, values outside the interval are clipped to the interval edges.
def calculate_error(Rs, Ts, gt_Rs, gt_Ts):
    error_Rs = []
    error_Ts = []
    for i in range(len(Rs)):
        error_Ts.append(np.linalg.norm(Ts[i] - gt_Ts[i], 2))
        norm_R = Rs[i] / np.linalg.norm(Rs[i])
        norm_gt_R = gt_Rs[i] / np.linalg.norm(gt_Rs[i])
        diff_R = np.clip(np.sum(norm_R * norm_gt_R), 0, 1)
        error_Rs.append(np.degrees(np.arccos(2 * diff_R * diff_R - 1)))        
    error_Rs = np.array(error_Rs)
    error_Ts = np.array(error_Ts)

    return np.median(error_Rs), np.median(error_Ts)
    

def ransac(points3D, points2D, cameraMatrix, distCoeffs):
    # parm: [N,3] [N,2] [3,3] [4,]

    
    img_size = [1920, 1080]
    N = np.log((1-0.99)) / np.log(1 - np.power((1 - 0.3), 3))
    N = round(N)
    min_outlier = np.inf
    best_R, best_T = None, None
    for i in range(N):
        # image undistortion
        mask = np.random.randint(points2D.shape[0], size = 4)
        chosen_points2D = points2D[ mask, :].T
        chosen_points3D = points3D[ mask, :].T #2x4
        chosen_points2D = image_undistortion(chosen_points2D, distCoeffs, img_size)
        try:
            R, T = P3P(chosen_points3D, chosen_points2D, cameraMatrix, distCoeffs)

            projected2D = cameraMatrix @ (R @ (points3D.T - T.reshape(3, 1)))
            projected2D /= projected2D[-1, :].reshape((1,-1))
            error = np.linalg.norm(projected2D[:2, :] - points2D.T, axis=0)
            #print(error)
            outliers = len(error[np.where(error > 10)])
            if outliers < min_outlier:
                best_R = R
                best_T = T
                min_outlier = outliers
        except:
            print("No solution")
    best_R = Rotation.from_matrix(best_R).as_quat()
    best_T = best_T.reshape(-1)

    return best_R, best_T
def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    #print(points2D.shape) #(N,2)
    #print(points3D.shape) #(N,3)
    #return self_ransacP3P
    R, T = ransac(points3D, points2D, cameraMatrix, distCoeffs)
    
    #opencv_ans =  cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
    #R, T = opencv_ans[1], opencv_ans[2]

    

    return R, T

def main():
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")
    print("data loaded.")
    image_idx = images_df["IMAGE_ID"]
    image_id = images_df["IMAGE_ID"].to_list()
    img_size = [1920, 1080] #real size of images

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    result_Rs = []
    result_Ts = []
    
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)
    
    R_list = []
    T_list = []
    R_groundtruth = []
    T_groundtruth = []
    #for i in range(1):
    for i in range(len(image_id)):
        
        idx = image_id[i]
        print("now in :",idx)
        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        R, T = pnpsolver((kp_query, desc_query),(kp_model, desc_model))
        
        # Get camera pose groudtruth 
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values
        #for opencv
        #R = Rotation.from_rotvec(R.reshape(1,3)).as_quat()

        #T = T.reshape(1,3)
        #=============
        R_list.append(R)
        T_list.append(T)
        R_groundtruth.append(np.squeeze(rotq_gt))
        T_groundtruth.append(np.squeeze(tvec_gt))

    R_list = np.array(R_list)
    T_list = np.array(T_list)
    R_groundtruth = np.array(R_groundtruth)
    T_groundtruth = np.array(T_groundtruth)
    
    np.save("result/Undistorted_Rotation.npy", R_list)
    np.save("result/Undistorted_Translation.npy", T_list)
    np.save("result/Rotation_groundtruth.npy", R_groundtruth)
    np.save("result/Translation_groundtruth.npy", T_groundtruth)

    error_R, error_T = calculate_error(R_list, T_list, R_groundtruth, T_groundtruth)

    print("rotation error: {}, pose error: {}".format(error_R, error_T))


    """
    R_list = np.load("result/Rotation.npy",allow_pickle=True)
    T_list = np.load("result/Translation.npy",allow_pickle=True)
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)
    #print(R_result.shape)
    for i in range(0, R_list.shape[0]):
        idx = image_id[i]

        R = Rotation.from_quat(R_list[i]).as_matrix()
        T = T_list[i].reshape(3, 1)
        line_set = create_camera_position(cameraMatrix, R, T, img_size)
        vis.add_geometry(line_set)


    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()


"""
[[ 942.66907853  793.51321135  778.75098025  813.43103656]
 [  40.82875506  532.62713162 1202.27622822  815.88853446]]
"""