import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import sys
import os
import pandas as pd
import os
from copy import deepcopy



class Point():
    def __init__(self, position, color):
        self.position = position
        self.color = color


def make_points_for_cube(cube_vertices):
    
    points = []

    #front
    front = [[0,0,0], [1,0,0], [0,0,1], [1,0,1]]
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , 0 , (j+1)*0.1]
            #print(point_pos)
            points.append(Point(point_pos, (0, 0, 255)))

    #back
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , 1 , (j+1)*0.1]
            #print(point_pos)
            points.append(Point(point_pos, (0, 0, 255)))

    #top
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , (j+1)*0.1, 1]
            #print(point_pos)
            points.append(Point(point_pos, (0, 255, 0)))


    #bottom
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , (j+1)*0.1, 0]
            #print(point_pos)
            points.append(Point(point_pos, (0, 255, 0)))

    #left
    for i in range(9):
        for j in range(9):
            point_pos = [0, (i+1)*0.1 , (j+1)*0.1]
            #print(point_pos)
            points.append(Point(point_pos, (255, 0, 0)))

    #right
    for i in range(9):
        for j in range(9):
            point_pos = [1, (i+1)*0.1 , (j+1)*0.1]
            #print(point_pos)
            points.append(Point(point_pos, (255, 0, 0)))

    return points


def draw_points(img, R, T, cube_vertices, cameraMatrix):

    #for opencv
    #R = R.reshape((4,))
    #T = T.reshape((3,))
    R = Rotation.from_quat(R).as_matrix()
    points = make_points_for_cube(cube_vertices)
    points.sort(key=lambda point: np.linalg.norm((point.position-T)), reverse=True)

    for i,p in enumerate(points):
        pos = (cameraMatrix @ (R @ (points[i].position - T).T)).T
        pos = (pos/pos[2])
        if((pos < 0).any()):
            continue
        img = cv2.circle(img, (int(pos[0]), int(pos[1])), radius=5, color=points[i].color, thickness=-1)

    return img



def main():
    images_df = pd.read_pickle("data/images.pkl")
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    

    # load cube
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube_vertices = np.asarray(cube.vertices).copy()
    #vis.add_geometry(cube)

    R_list = np.load("result/Undistorted_Rotation.npy")
    T_list = np.load("result/Undistorted_Translation.npy")

    fname = ['valid_img{}.jpg'.format(i) for i in range(5,655,5)]

    video_imgs = []
    for f in fname:
        #fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        print("fname",f)
        idx = ((images_df.loc[images_df["NAME"] == f])["IMAGE_ID"].values)[0]
        print(idx)
        rimg = cv2.imread("data/frames/" + f,cv2.IMREAD_COLOR)
        video_imgs.append(draw_points(rimg, R_list[idx-1], T_list[idx-1], cube_vertices, cameraMatrix))
    img_shape = (rimg.shape[1], rimg.shape[0])

    video = cv2.VideoWriter("Video.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, img_shape)

    for img in video_imgs:
        #cv2.imshow("img",img)
        video.write(img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    video.release()
if __name__ == '__main__':
    main()
