#!/usr/bin/env python
import cv2
import math
import numpy as np
import torch
import random
import time


class MotionDataset():
    def __init__(self,sequence_length):

        self.sequence_length = sequence_length
        self.img_size = (128,128)
        self.background_count = 100


    def __iter__(self):
        return self

    def next(self):
        background_speed = 1.0
        target_speed = 1.0

        background_direction = random.random()*360.0
        target_direction = background_direction + 90.0 + random.random()*180.0

        h,w = self.img_size
        velocities = np.zeros((self.background_count + 1, 2)) #vx,vy
        velocities[0,0] = target_speed * math.cos(math.radians(target_direction)) #target x speed
        velocities[0,1] = target_speed * math.sin(math.radians(target_direction)) #target y speed

        velocities[1:,0] = background_speed * math.cos(math.radians(background_direction)) #background x speed
        velocities[1:,1] = background_speed * math.sin(math.radians(background_direction)) #background y speed


        positions = np.random.random((self.background_count + 1, 2)) #x,y
        positions[:,0] *= w
        positions[:,1] *= h




        sequence = np.zeros((self.sequence_length,2,h,w)) #1 current frame, 0 last frame
        target_location = np.zeros((self.sequence_length,2))
        positions_floor = np.floor(positions).astype(np.int)

        for point_i in range(self.background_count + 1):
            x = positions_floor[point_i,0]
            y = positions_floor[point_i,1]
            sequence[0,0,y,x] = 1.0 #first last frame

        for frame_i in range(self.sequence_length):
            positions += velocities
            positions[:,0] = np.remainder(positions[:,0],w)
            positions[:,1] = np.remainder(positions[:,1],h)
            positions_floor = np.floor(positions).astype(np.int)

            target_location[frame_i,0] = positions_floor[0,0]
            target_location[frame_i,1] = positions_floor[0,1]

            for point_i in range(self.background_count + 1):
                x = positions_floor[point_i,0]
                y = positions_floor[point_i,1]
                sequence[frame_i,1,y,x] = 1.0

            if frame_i > 0:
                sequence[frame_i,0,:,:] = sequence[frame_i-1,1,:,:]


        return {"images": torch.from_numpy(sequence),"targets": torch.from_numpy(target_location)}









if __name__ == "__main__":
    md = MotionDataset(100)
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    for batch in md:
        sequence = batch["images"].numpy()
        n,c,h,w = sequence.shape
        for frame_i in range(n):
            cv2.imshow("image",sequence[frame_i,1,:,:]+sequence[frame_i,0,:,:]*0.5)
            cv2.waitKey(50)
        time.sleep(0.1)
