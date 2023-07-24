from parseTrackletXML import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED

from pathlib import Path

import os
import numpy as np
import readline   

import pdb

kittiDir = '/home/robesafe/Datasets/'
drive = '2011_09_26_drive_0013_sync'

DEFAULT_DRIVE = drive
twoPi = 2.*np.pi

# Create gt files
if not os.path.exists(os.path.join(kittiDir,drive,'label_2')):
    os.makedirs(os.path.join(kittiDir,drive,'label_2'))
if not os.path.exists(os.path.join(kittiDir,drive,'calib')):
    os.makedirs(os.path.join(kittiDir,drive,'calib'))


for name in os.listdir(os.path.join(kittiDir,drive,'image_02')):
    frame_name = Path(name).stem
    if len(frame_name) > 6:
        print('Renaming frame: ',frame_name)
        frame_name = '{:06d}'.format(int(frame_name))
        os.rename(os.path.join(kittiDir,drive,'image_02',name),os.path.join(kittiDir,drive,'image_02',frame_name+'.png'))

    open(os.path.join(kittiDir,drive,'label_2',frame_name+'.txt'),'w+').close()

for name in os.listdir(os.path.join(kittiDir,drive,'image_02')):
    frame_name = Path(name).stem
    with open(os.path.join(kittiDir,drive,'calib',frame_name+'.txt'),'w+') as fd:
        fd.writelines(
            [
            'P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00\n',
            'P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00\n',
            'P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03\n',
            'P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03\n',
            'R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01\n',
            'Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01\n',
            'Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01\n'
            ]
        )


# get dir names
if kittiDir is None:
    kittiDir = os.path.expanduser(raw_input('please enter kitti base dir (e.g. ~/path/to/kitti): ').strip())
if drive is None:
    drive    = raw_input('please enter drive name (default {0}): '.format(DEFAULT_DRIVE)).strip()
if len(drive) == 0:
    drive = DEFAULT_DRIVE

# read tracklets from file
myTrackletFile = os.path.join(kittiDir, drive, 'tracklet_labels.xml')
tracklets = parseXML(myTrackletFile)

for iTracklet, tracklet in enumerate(tracklets):
    print ('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

    # this part is inspired by kitti object development kit matlab code: computeBox3D
    h,w,l = tracklet.size
    trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
        [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
        [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
        [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

    # loop over all data in tracklet
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber \
        in tracklet:

        # determine if object is in the image; otherwise continue
        if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
            continue

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]   # other rotations are 0 in all xml files I checked
        assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
        rotMat = np.array([\
            [np.cos(yaw), -np.sin(yaw), 0.0], \
            [np.sin(yaw),  np.cos(yaw), 0.0], \
            [        0.0,          0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T

        # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to 
        #   car-centered yaw (i.e. 0 degree = same orientation as car).
        #   makes quite a difference for objects in periphery!
        # Result is in [0, 2pi]
        x, y, z = translation
        yawVisual = ( yaw - np.arctan2(y, x) ) % twoPi

        cl = tracklet.objectType
        alpha = 0                       # TODO (not used yet)
        bbox = [0,0,0,0]                # TODO (not used yet)

        # print ('3D Bounding Box for frame {0}:'.format(absoluteFrameNumber))
        # print ('  size: {0}, {1}, {2}'.format(l, w, h))
        # print ('  translation: ({0}, {1}, {2})'.format(x, y, z))
        # print ('  rotation:    {0} (y-axis in camera coordinates)'.format(yawVisual))
        # print ('  yaw:         {0} (y-axis in velodyne coordinates)'.format(yaw))
        
        # Transform to camera coordinates
        calib_file = open(os.path.join(kittiDir,drive,'calib',frame_name+'.txt'),"r").readlines()

        p2 = calib_file[2].strip('\n').strip("P2: ").split(' ')
        p2 = np.matrix([float(x) for x in p2]).reshape(3,4)
        p2 = np.vstack((p2,np.array((0,0,0,1))))
        
        R0_rect = calib_file[4].strip('\n').strip('R0_rect: ').split(' ')
        R0_rect = np.matrix([float(x) for x in R0_rect]).reshape(3,3)
        R0_rect = np.hstack((R0_rect,np.array(([0],[0],[0]))))
        R0_rect = np.vstack((R0_rect,np.array((0,0,0,1))))

        Tr_velo_to_cam = calib_file[5].strip('\n').strip('Tr_velo_to_cam: ').split(' ')
        Tr_velo_to_cam = np.matrix([float(x) for x in Tr_velo_to_cam]).reshape(3,4)
        Tr_velo_to_cam = np.vstack((Tr_velo_to_cam,np.array([0,0,0,1])))

        cam_coord = (Tr_velo_to_cam @ np.array([x,y,z,1])).T
        
        x_c, y_c, z_c = cam_coord[0,0], cam_coord[1,0], cam_coord[2,0]

        '''
        if yaw > -np.pi/2 and yaw < np.pi/2:
            yaw += np.pi/2
            
        elif yaw < -np.pi/2:
            yaw += np.pi/2
        else:
            yaw -= 3*np.pi/2
        ''' 
        print('yaw: ',yaw)
        # Save data in kitti format
        with open(os.path.join(kittiDir,drive,'label_2','{:06d}.txt'.format(absoluteFrameNumber)),'a') as fd:
            fd.write("{} 0.00 0 {} {} {} {} {} {} {} {} {} {} {} {}".format(cl,alpha,bbox[0],bbox[1],bbox[2],bbox[3],\
                                h,w,l,x_c,y_c,z_c,-yaw-np.pi/2))
            fd.write('\n')

        

