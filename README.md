# Landing Test

I used a custom trained TensorFlow object detection model and OpenCV
to detect the landing zone and the aircraft's orientation relative to that
landing zone.

## Usage

This program can be run on any Linux machine with Python3, TensorFlow, and OpenCV installed.

You will also need a webcam in the `/dev/video0` slot. If accessing that mount point is not an option, you may use a
different mount point and update the call to LandingDetection accordingly.

Eg. If you want to use a webcam mounted at `/dev/video1` instead of `/dev/video0`, change:

    detection = LandingDetection(0, './frozen_inference_graph.pb')
    
to
    
    detection = LandingDetection(1, './frozen_inference_graph.pb')
    
in `main.py`

You can either construct a landing zone that resembles the Sustral landing zone or pull up one of 
the training images on your screen.

Assuming these prerequisites are satisfied:

1. Clone the directory

2. Open a terminal and navigate to the cloned directory

3. Run `python main.py`

4. Wait for the OpenCV window to appear

5. Direct the webcam toward the landing zone or wave it around as you please

6. When you are finished, hit enter on the terminal
