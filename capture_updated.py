import time
import collections
import pyautogui
from PIL import Image

import cv2
# GNU/Linux
from mss.linux import MSS as mss
import numpy
import pickle  # to save/load Q-Tables
import random

TOP = 432    
LEFT = 180#172 dino touch value
WIDTH = 70 
HEIGHT = 30

CHECKPOINT_TOP = 320
CHECKPOINT_LEFT = 620
CHECKPOINT_WIDTH = 60
CHECKPOINT_HEIGHT = 50
MIN_DURATION_BETWEEN_100_MARKS = 3

END_TOP = 422
END_LEFT = 376
END_WIDTH = 23
END_HEIGHT = 19
END_ARRAY_FILE = "end_icon_view_arrray.npy"
CANNY_EDGE_THRESHOLD_1 = 100
CANNY_EDGE_THRESHOLD_2 = 200
start_q_table = "q_table.pickle"  
# start_q_table = None  

# if we have a pickled Q table, we'll put the filename of it here.

SPEED_MIN = 10
SPEED_MAX = 250                 

DISTANCE_MIN = 0
DISTANCE_MAX = WIDTH


ACTION_JUMP = 1
ACTION_NOTHING = 0

CAUGHT_PANELTY = 1000
SERVIVE_PANELTY = 200
SUCESSFULL_JUMP = 1000
LEARNING_RATE = 0.01
DISCOUNT = 0.8          

q_table = {}

# Part of the screen to capture
monitor = {"top": TOP, "left": LEFT, "width": WIDTH, "height": HEIGHT}

check_point_monitor = {"top": CHECKPOINT_TOP, "left": CHECKPOINT_LEFT, "width": CHECKPOINT_WIDTH, "height": CHECKPOINT_HEIGHT}

end_monitor = {"top": END_TOP, "left": END_LEFT, "width": END_WIDTH, "height": END_HEIGHT}

sct = mss()

game_over_image = numpy.load(END_ARRAY_FILE)
# #array of END_HEIGHT arrays each sub array has END_WIDTH values

previous_start_time = 0


previous_checkpoint_img = numpy.array(sct.grab(check_point_monitor))
previous_time = int(round(time.time() * 1000))
previous_time_sec = round(time.time())
previous_100_point_time = 0


if start_q_table is None:
    # initialize the q-table#
    for distance in range(DISTANCE_MIN, DISTANCE_MAX):
        for q_speed in range(SPEED_MIN, SPEED_MAX):
            q_table[(distance, q_speed)] =  [numpy.random.uniform(-2, 0) for i in range(2)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
        #load q table from file


def get_distance(img) :
    
    if last_distance == DISTANCE_MIN or last_distance is None :
        width= DISTANCE_MAX
    else :
        width = last_distance

    for y in range(0, HEIGHT-1) :
        for x in range(0, width) :   
            pixel = img[y][x]
            if pixel == 255 :
                return x  

def pil_frombytes(im):
    """ Efficient Pillow version. """
    return Image.frombytes('RGB', im.size, im.bgra, 'raw', 'BGRX').tobytes()

last_distance = DISTANCE_MIN
last_speed = SPEED_MIN

while True:

    speed = SPEED_MIN
    distance = 0

        # Press "q" to quit
    # if cv2.waitKey(25) & 0xFF == ord("q"):
    #     cv2.destroyAllWindows()
    #     break

    checkpoint_img = numpy.array(sct.grab(check_point_monitor))
    checkpoint_img = cv2.Canny(checkpoint_img,CANNY_EDGE_THRESHOLD_1,CANNY_EDGE_THRESHOLD_2)
    nonzero_pixel_count = numpy.count_nonzero(checkpoint_img == 255) 
    mark_100_point = nonzero_pixel_count == 0

    current_time_sec = round(time.time())

    duration_100_marks = current_time_sec - previous_100_point_time
    if mark_100_point and duration_100_marks > MIN_DURATION_BETWEEN_100_MARKS :
        speed = 1/duration_100_marks
        speed = round(speed,3)*1000
        if speed == 0 :
            speed = SPEED_MIN
        previous_100_point_time = current_time_sec
        # print(speed)

    previous_checkpoint_img = checkpoint_img


    # Get raw pixels from the screen, save it to a Numpy array
    img = numpy.array(sct.grab(monitor))
    img = cv2.Canny(img,CANNY_EDGE_THRESHOLD_1,CANNY_EDGE_THRESHOLD_2)
    # cv2.imshow('monitor screen',img)
    
    distance = get_distance(img)
    

    end_img = numpy.array(sct.grab(end_monitor))
    end_img = cv2.Canny(end_img,CANNY_EDGE_THRESHOLD_1,CANNY_EDGE_THRESHOLD_2)
    # cv2.imshow('game over screen ',end_img)

    is_over = (game_over_image==end_img).all()
    # print(is_over)


    # Q learn start from here

    new_obs = (distance,speed)  # new observation

    if is_over :
       
        reward = -CAUGHT_PANELTY

    else :

        if q_table.get(new_obs) and last_distance is not None :
             
            reward = SERVIVE_PANELTY

            max_action = numpy.argmax(q_table[new_obs])  # max Q value for this new obs

            # print("max_action ",max_action)
            if max_action == ACTION_JUMP : 
                pyautogui.press("space")
                print("jumping")

            
            obs = (last_distance,last_speed)
            action = numpy.argmax(q_table[obs])

            if action == ACTION_JUMP :
                reward = SUCESSFULL_JUMP

            max_future_q = numpy.max(q_table[new_obs])  # max Q value for this new obs
            current_q = q_table[obs][action]  # current Q for our chosen action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[obs][action] = new_q
            print("q table update done")                         



    if is_over :
        last_distance = DISTANCE_MIN
        last_speed = SPEED_MIN
        print("caught---------")
        with open('q_table.pickle', 'wb') as f:
            pickle.dump(q_table, f)
        pyautogui.press("space")
    else : 
        last_distance = distance
        last_speed = speed
    # Q learn stops











    



