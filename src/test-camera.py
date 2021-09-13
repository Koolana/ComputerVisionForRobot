import cv2

from realsenseD415 import RealsenseD415

if __name__ == '__main__':
    camera = RealsenseD415(640, 480);

    while(True):
        depthImg, colorImg = camera.getColorAndDepthImg();

        cv2.imshow('RealSense depth', depthImg)
        cv2.imshow('RealSense color', cv2.cvtColor(colorImg, cv2.COLOR_RGB2BGR))
        print(depthImg.shape, colorImg.shape)
        
        cv2.waitKey(1)
