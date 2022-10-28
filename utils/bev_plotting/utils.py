import cv2
import numpy as np
from enum import Enum


#Custom plotting funcion and classes

def plot_bird_view(birdview_im, dimensions, alpha, theta_ray,location, color, x_max, z_max):
    # color = cv_colors.RED.value
    pix_x = 1000
    pix_z = 2000
    # x_max = 50
    # z_max = 30

    if location[0] == -1000:
        return -1

    orient = alpha + theta_ray
    R = rotation_matrix(orient)

    corners = create_corners(dimensions, location = location, R = R)
    cv2.circle(birdview_im, (int(pix_x/2),int(pix_z)), 4, color, thickness=1)
    
    
    #Print objects center
    # print(f'Processing object in {location}\n')

    x_bird = (pix_x/x_max)*location[0]+(pix_x/2)
    z_bird = pix_z-(pix_z/z_max)*location[2]
    bird_point = (int(x_bird),int(z_bird))
    cv2.circle(birdview_im, bird_point, 3, color, thickness=5)


    #Print 3D box bird view corners
    line_points = []
    for pt in corners:
        x_bird = (pix_x/x_max)*pt[0]+(pix_x/2)
        z_bird = pix_z-(pix_z/z_max)*pt[2]
        bird_point = (int(x_bird),int(z_bird))

        cv2.circle(birdview_im, bird_point, 1, color, thickness=3)
        line_points.append((x_bird,z_bird))

    #Print 3D box bird view
    if line_points:
        cv2.line(birdview_im,\
            (int(line_points[0][0]),int(line_points[0][1])), (int(line_points[1][0]),int(line_points[1][1])), color, 1)
        cv2.line(birdview_im,\
            (int(line_points[0][0]),int(line_points[0][1])), (int(line_points[4][0]),int(line_points[4][1])), color, 1)
        cv2.line(birdview_im,\
            (int(line_points[4][0]),int(line_points[4][1])), (int(line_points[5][0]),int(line_points[5][1])), color, 1)
        cv2.line(birdview_im,\
            (int(line_points[1][0]),int(line_points[1][1])), (int(line_points[5][0]),int(line_points[5][1])), color, 1)


    

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

def constraint_to_color(constraint_idx):
    return {
        0 : cv_colors.PURPLE.value, #left
        1 : cv_colors.ORANGE.value, #top
        2 : cv_colors.MINT.value, #right
        3 : cv_colors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4




def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)

def rotation_matrix(yaw, pitch=0, roll=0):
    tx = roll
    ty = yaw
    tz = pitch

    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])


    return Ry.reshape([3,3])
    # return np.dot(np.dot(Rz,Ry), Rx)

# option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners

