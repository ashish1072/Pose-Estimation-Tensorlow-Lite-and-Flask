import tensorflow as tf
import numpy as np
import cv2
import os


keypoint_id = {0:"nose", 
               1:"left_eye", 2:"right_eye",
               3:"left_ear", 4:"right_ear",
               5:"left_shoulder", 6:"right_shoulder",
               7:"left_elbow", 8:"right_elbow",
               9:"left_wrist", 10:"right_wrist",
               11:"left_hip", 12:"right_hip",
               13:"left_knee", 14:"right_knee",
               15:"left_ankle", 16:"right_ankle",
               }

# Every tuple contains the two keypoints forming an edge, mimicking human skeleton
edge_endpoints = [(0,1), (0,2), (1,3), (2,4), (5,7), (5,11), (5,6), (6,12), (6,8), (7,9), (8,10), 
                  (11,13), (11,12), (12,14), (13,15), (14,16)]

# For each tuple, first value corresponds to the keypoint angle location and the next two correspond to the edge neighbors
keypoint_angle_record = [ (5, 11, 7), (6, 12, 8), (7, 9, 5), (8, 6, 10), (11, 5, 13), 
                          (12, 6, 14), (13, 11, 15), (14, 12, 16)]


# Angle computation at 8 keypoint locations considering three (i, j, k) joints at a time 
def compute_angles(output_data):
    angle_collection = []
    for i, j, k in keypoint_angle_record:
        p0 = output_data[i][0:2]
        p1 = output_data[j][0:2]
        p2 = output_data[k][0:2]      
        vec1 = np.subtract(p1, p0)
        vec2 = np.subtract(p2, p0)
        vec1_normalized = vec1 / np.linalg.norm(vec1)
        vec2_normalized = vec2 / np.linalg.norm(vec2)
        angle = np.arccos(np.dot(vec1_normalized, vec2_normalized))*(180/np.pi)
        angle_collection.append(angle)
    return angle_collection

    
def interpolate(kp_0, kp_1, kp_2):
     # If current keypoint(kp_0) score is smaller than kp_1 and kp_2, apply interpolation
    kp_final = []
    for idx in range(kp_0.shape[0]):
        if kp_0[idx][2] >= kp_1[idx][2] or kp_0[idx][2] >= kp_2[idx][2]:
            kp_final.append(kp_0[idx])
        else:
            kp_new_y = kp_1[idx][0] + (kp_1[idx][0] - kp_2[idx][0])
            kp_new_x = kp_1[idx][1] + (kp_1[idx][1] - kp_2[idx][1])
            kp_final.append([kp_new_y, kp_new_x, kp_1[idx][2]])
    kp_final = np.array(kp_final)
    return kp_final


#Inference from pre-trained MoveNet Model with Tensorflow Lite 
movenet_path = os.getcwd() + "/movenet_lightning16.tflite"
input_size = 192
def movenet(image):
    interpreter = tf.lite.Interpreter(model_path=movenet_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input image
    img = tf.image.resize_with_pad(image, input_size, input_size)

    # Set input tensor 
    input_data = np.expand_dims(img, axis=0)
    input_data = tf.cast(input_data, dtype=tf.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Model Output 
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    keypt_data = output_data.reshape(output_data.shape[2], output_data.shape[3])

    # Scale coordinates to original image without padding 
    larger_dim = max(image.shape[0], image.shape[1])
    smaller_dim = min(image.shape[0], image.shape[1])
    image_size_ratio = smaller_dim / larger_dim
    Scale_factor = larger_dim / input_size 
    Padding_size = 0.5 * input_size * (1 - image_size_ratio)
    scaled_coords = []
    for i in range(keypt_data.shape[0]):
        if larger_dim == image.shape[1]: # larger dim is the width of the image
            px = int(keypt_data[i][1] * input_size * Scale_factor)
            py = int((keypt_data[i][0] * input_size - Padding_size) * Scale_factor)
        else:
            px = int((keypt_data[i][1] * input_size - Padding_size) * Scale_factor)
            py = int(keypt_data[i][0] * input_size * Scale_factor)
        scaled_coords.append((px, py))
        keypt_data[i][1] = px
        keypt_data[i][0] = py
    return keypt_data


def display(keypt_data, image):
    # Draw the edge between keypoints i, j, only if confidence score is above threshold
    score_threshold = 0.2
    coords = keypt_data[:, 0:2]
    coords = coords.astype(int)
    for i, j in edge_endpoints:
        if keypt_data[i][2] < score_threshold:
            continue
        cv2.line(image, (coords[i][1], coords[i][0]), (coords[j][1], coords[j][0]), (255, 0, 0), 2)

    # Display the angle at the keypoint i if confidence score is above threshold and print on the terminal
    angle_collection = compute_angles(keypt_data)
    print("\n\nAngle between the joints..........")
    for  n, (i, j, k) in enumerate(keypoint_angle_record):
        if keypt_data[i][2] < score_threshold:
            continue
        print(keypoint_id[i], "\t", keypoint_id[j], "\t",keypoint_id[k], "\t=  ", f"{int(angle_collection[n])}\N{DEGREE SIGN}")
        cv2.putText(image, str(int(angle_collection[n])), (coords[i][1], coords[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2)
    print("\n")
    return image


def init_crop_area(image):
    width = image.shape[1]
    height = image.shape[0]
    return(width/2, height/2, width, height)


def crop_image(mid_x, mid_y, max_x, max_y, image):
    safety_margin = 1.5
    scaled_max_x = max_x * safety_margin
    scaled_max_y = max_y * safety_margin
    img_width = image.shape[1]
    img_height = image.shape[0]
    x_min = max(mid_x - scaled_max_x/2, 0)
    x_max = min(mid_x + scaled_max_x/2, img_width)
    y_min = max(mid_y - scaled_max_y/2, 0)
    y_max = min(mid_y + scaled_max_y/2, img_height)
    cropped_image = image[int(y_min):int(y_max),int(x_min):int(x_max)]
    return cropped_image


def rescaling(keypt, mid_x, mid_y, max_x, max_y):
    for i in range(keypt.shape[0]):
        keypt[i][1] = mid_x - (max_x/2) + keypt[i][1]
        keypt[i][0] = mid_y - (max_y/2) + keypt[i][0]
    return keypt


def generate_crop_region(keypt):
    max_x = 0
    max_y = 0
    mid_x = 0
    mid_y = 0
    diff_x = 0
    diff_y = 0
    for i in range(keypt.shape[0]):
        for j in range(keypt.shape[0]):
            diff_x = abs(keypt[i][1] - keypt[j][1])
            diff_y = abs(keypt[i][0] - keypt[j][0])

            if diff_x > max_x:
                max_x = diff_x
                mid_x = (keypt[i][1] + keypt[j][1]) / 2

            if diff_y > max_y:
                max_y = diff_y
                mid_y = (keypt[i][0] + keypt[j][0]) / 2
    return mid_x, mid_y, max_x, max_y








