'''
Driver script that runs the landing detection program.
'''

from landing_detection import LandingDetection


def main():

    detection = LandingDetection(0, './frozen_inference_graph.pb')
    detection.detect_start()

    while not detection.initialized:
        pass

    # Hit return to halt the program
    _ = input("Press return to stop: ")
    detection.kill = True


if __name__ == '__main__':
    main()
