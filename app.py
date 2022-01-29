from flask import Flask, render_template, Response
from utils import *
import os

app=Flask(__name__)
camera_feed = cv2.VideoCapture(-1)

@app.route('/')
def index():
    return render_template('index.html')


# Endpoints for output on captured images ....................................................
def process_stream():
    #camera_feed = cv2.VideoCapture(cv2.CAP_V4L2)
    global output_frame
    global FLAG 
    FLAG = 1
    while FLAG:
        ret, frame = camera_feed.read()
        output_frame = frame
        frame = cv2.imencode('.jpg',frame)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Trigger the webcam for streaming and capture the image
@app.route('/video_stream1')
def video_stream1():
    return Response(process_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Send the captured image to MoveNet model
@app.route('/process_image', methods = ["POST", "GET"])
def process_image():
    return render_template('result_image.html')

# Obtain the pose from the model and display the output on webpage
@app.route('/model_output_image', methods = ["POST", "GET"])
def model_output_image():
    global output_frame
    global FLAG
    #FLAG = 0
    # call MoveNet model
    keypoint_data = movenet(output_frame)
    image = display(keypoint_data, output_frame)
    frame = cv2.imencode('.JPEG', image)[1].tobytes()
    out_frame = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
    return Response(out_frame, mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoints for output on video stream ....................................................
@app.route('/video_stream2', methods = ["POST", "GET"])
def video_stream2():
    return render_template('result_video.html')

def process_video():
    # Initialize previous two frames
    #camera_feed = cv2.VideoCapture(cv2.CAP_V4L2)
    keyp_1 = np.zeros((17, 3))
    keyp_2 = np.zeros((17, 3))
    while True:
        ret, img = camera_feed.read()
        # call MoveNet model
        keyp_0 = movenet(img)
        # Apply interpolation
        keyp_curr = interpolate(keyp_0, keyp_1, keyp_2)
        image = display(keyp_curr, img)
        keyp_2 = keyp_1
        keyp_1 = keyp_0
        frame = cv2.imencode('.jpg',image)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/model_output_video', methods = ["POST", "GET"])
def model_output_video():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()

