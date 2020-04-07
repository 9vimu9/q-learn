import time
import collections
import pyautogui

import cv2
import mss
import numpy
import pickle  # to save/load Q-Tables
import random

TOP = 428
LEFT = 230
WIDTH = 50
HEIGHT = 36

CHECKPOINT_TOP = TOP
CHECKPOINT_LEFT = 400
CHECKPOINT_WIDTH = WIDTH
CHECKPOINT_HEIGHT = HEIGHT

END_TOP = 422
END_LEFT = 376
END_WIDTH = 23
END_HEIGHT = 19
END_ARRAY_FILE = "end_icon_view_arrray.npy"
CANNY_EDGE_THRESHOLD_1 = 100
CANNY_EDGE_THRESHOLD_2 = 200
# start_q_table = "q_table.pickle"  
start_q_table = None  

# if we have a pickled Q table, we'll put the filename of it here.

ACTIVE_PIXEL_MIN_VALUE = 5  
ACTIVE_PIXEL_MAX_VALUE = 300 
SPEED_MIN = 5
SPEED_MAX = 100

ACTION_JUMP = 1
ACTION_NOTHING = 0

CAUGHT_PANELTY = 500
SERVIVE_PANELTY = 500
LEARNING_RATE = 0.1
DISCOUNT = 0.95

q_table = {}

# Part of the screen to capture
monitor = {"top": TOP, "left": LEFT, "width": WIDTH, "height": HEIGHT}
check_point_monitor = {"top": CHECKPOINT_TOP, "left": CHECKPOINT_LEFT, "width": CHECKPOINT_WIDTH, "height": CHECKPOINT_HEIGHT}

end_monitor = {"top": END_TOP, "left": END_LEFT, "width": END_WIDTH, "height": END_HEIGHT}

speed = 0


if start_q_table is None:
    # initialize the q-table#
    for active_pixels in range(ACTIVE_PIXEL_MIN_VALUE, ACTIVE_PIXEL_MAX_VALUE):
        for q_speed in range(SPEED_MIN, SPEED_MAX):
            q_table[(active_pixels, q_speed)] =  [numpy.random.uniform(-2, 0) for i in range(2)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
        #load q table from file


with mss.mss() as sct:

    game_over_image = numpy.load(END_ARRAY_FILE)
    # #array of END_HEIGHT arrays each sub array has END_WIDTH values

    previous_start_time = 0

    speed = 0
    last_active_pixels = 0
    last_speed = 0

    while True:

        checkpoint_img = numpy.array(sct.grab(check_point_monitor))
        checkpoint_img = cv2.Canny(checkpoint_img,CANNY_EDGE_THRESHOLD_1,CANNY_EDGE_THRESHOLD_2)
        cv2.imshow('checkpoint screen ',checkpoint_img)

        start_pixel = 0
        end_pixel = 0
        rounded_speed = 0
        new_obs = 0

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        img = cv2.Canny(img,CANNY_EDGE_THRESHOLD_1,CANNY_EDGE_THRESHOLD_2)
        cv2.imshow('monitor screen',img)


        nonzero_pixel_count = numpy.count_nonzero(img == 255)  
       
        for i in range(0, HEIGHT-1):
            end_pixel += img[i][WIDTH-1]
            start_pixel += checkpoint_img[i][WIDTH-1]

        current_time = int(round(time.time() * 1000))

        if start_pixel > 0 and end_pixel == 0 and previous_start_time == 0 :
            previous_start_time = current_time

        if end_pixel > 0 and previous_start_time > 0 :
            duration = current_time - previous_start_time
            
            if duration > 0 :
                new_speed = (CHECKPOINT_LEFT - LEFT)/duration
                new_speed = round(new_speed,2)*100
                speed_dif = new_speed - globals()["speed"]

                if abs(speed_dif) < 15 or globals()["speed"] == 0 :
                    globals()["speed"] = new_speed
                    
                previous_start_time = 0

        
        end_img = numpy.array(sct.grab(end_monitor))
        end_img = cv2.Canny(end_img,CANNY_EDGE_THRESHOLD_1,CANNY_EDGE_THRESHOLD_2)
        cv2.imshow('game over screen ',end_img)


        is_over = (game_over_image==end_img).all()
        

        if speed >= SPEED_MIN and nonzero_pixel_count >= ACTIVE_PIXEL_MIN_VALUE :
            speed = round(speed,2)
            new_obs = (nonzero_pixel_count,speed)  # new observation

        if is_over :
            print("caught---------")
            with open('q_table.pickle', 'wb') as f:
                pickle.dump(q_table, f)
            speed = 0
            pyautogui.press("space")
            reward = - CAUGHT_PANELTY
        else :
            reward = SERVIVE_PANELTY

            if new_obs != 0 :
                max_action = numpy.argmax(q_table[new_obs])  # max Q value for this new obs
                print("max_action ",max_action)
                if max_action == ACTION_JUMP : 
                    pyautogui.press("space")
                    print("jumping")

        
        if last_active_pixels > 0 and last_speed > 0 and new_obs != 0 :    
            
            obs = (last_active_pixels,last_speed)
            action = numpy.argmax(q_table[obs])
            max_future_q = numpy.max(q_table[new_obs])  # max Q value for this new obs
            current_q = q_table[obs][action]  # current Q for our chosen action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[obs][action] = new_q
            print("q table update done")



        if is_over or new_obs == 0 :
            last_active_pixels = 0
            last_speed = 0
        else : 
            last_active_pixels = nonzero_pixel_count
            last_speed = speed


        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
