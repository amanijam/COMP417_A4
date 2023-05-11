import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

class VisualTracker():
    def __init__(self, penddims):
        self.pend_width = penddims[0]
        self.pend_height = penddims[1]

    def observe_angle(self, surface_array, debugMode=False):
        """
        :param img: cropped image with pendulum color extracted
        """
        # Get the image and extract red channel (because pendulum is drawn in red)
        red_im = Image.fromarray(surface_array[:,:,2]) 

        # Crop image
        left = 0
        top = 80
        right = surface_array.shape[1]
        bottom = surface_array.shape[0] - 50
        red_cropped = red_im.crop((left, top, right, bottom)) 
        red_cropped.save("pendulum.png")
        img = cv2.imread("pendulum.png")
        if(debugMode):
            cv2.imshow('img', img)
            cv2.waitKey(0)

        # Convert frame from BGR to GRAY
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # if (debugMode):
        #     cv2.imshow('gray', gray)
        #     cv2.waitKey(0)

        # Edge detection using Canny function
        img_edges = cv2.Canny(gray, 50, 190, 3)

        # if (debugMode):
        #     cv2.imshow('img_edges', img_edges)
        #     cv2.waitKey(0)

        # Convert to black and white image
        _, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)

        if (debugMode):
            cv2.imshow('img_thresh', img_thresh)
            cv2.waitKey(0)

        # Find contours
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_c = contours[0]

        # If there are more than one contour shapes, get the one 
        #  closest to the pendulum
        if len(contours) > 1:
            #print("More than 1 contour")
            dict_areas = {}
            for i in range(len(contours)):
                c = contours[i]
                rect = cv2.minAreaRect(c)
                width, height = rect[1]
                dict_areas[i] = width*height
            pend_area = self.pend_width*self.pend_height
            best_c_index = min(dict_areas, key=lambda x:abs(dict_areas[x]-pend_area))
            best_c = contours[best_c_index]

        # Find angle
        # (vx, vy) is a normalized vector collinear to the line
        [vx,vy,x,y] = cv2.fitLine(best_c, cv2.DIST_L2, 0, 0.01, 0.01)
        line_angle = math.atan2(vy, vx)
        if line_angle > 0:
            angle = math.pi/2 - line_angle
        else:
            angle = -math.pi/2 - line_angle

        # Draw line
        _, cols = img.shape[:2]
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(img, (cols-1, righty), (0, lefty), (0,255,0), 1)

        if(debugMode):
            cv2.imshow('img_line', img)
            cv2.waitKey(0)

        return angle


# def main():
#     img = cv2.imread("timestep_100_current_state.png")
#     print(observe_angle(img, False, (6.0, 150.0)))

# if __name__ == "__main__":
#     # execute main
#     main()
