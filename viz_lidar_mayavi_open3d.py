


import open3d
import numpy as np




def viz_lidar_open3d(pointcloud, colors=None, width=None, height=None):

    x = pointcloud[:,0]  # x position of point
    y = pointcloud[:,1]  # y position of point
    z = pointcloud[:,2]  # z position of point

    pcd = open3d.geometry.PointCloud()
    points = np.hstack([x[:,None],y[:,None],z[:,None]])
    points = open3d.utility.Vector3dVector(points)
    pcd.points = points


    if colors is not None:
        pcd.colors = open3d.utility.Vector3dVector(colors)
    


    if (width is not None) & (height is not None):
        open3d.visualization.draw_geometries([pcd], width=width, height=height)
    else:

        open3d.visualization.draw_geometries([pcd])

