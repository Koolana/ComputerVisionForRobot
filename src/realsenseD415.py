import pyrealsense2 as rs
import numpy as np
import cv2

class RealsenseD415(object):
    """Class for connecting with realsense camera D415,
        depth, color = getColorAndDepthImg() - return grayscale image to represent depth (w, h, 1) and RGB image (w, h, 3) from the camera"""

    def __init__(self, width, height, framerate=30):
        self.width = width
        self.height = height
        self.framerate = framerate

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        self.isValid = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.isValid = True
                break

        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.framerate)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.framerate)

        self.pipeline.start(self.config)

    def getColorAndDepthImg(self):
        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)

        if depth_image.shape != color_image.shape:
            color_image = cv2.resize(color_image, dsize=(depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_AREA)

        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return depth_image, color_image

    def isValid(self):
        return self.isValid
