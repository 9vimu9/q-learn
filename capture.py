import time

import cv2
import mss
import numpy

TOP = 427
LEFT = 190
WIDTH = 30
HEIGHT = 42

END_TOP = 422
END_LEFT = 376
END_WIDTH = 23
END_HEIGHT = 19
END_ARRAY_FILE = "game_over.npy"
CANNY_EDGE_THRESHOLD_1 = 100
CANNY_EDGE_THRESHOLD_2 = 200

# Part of the screen to capture
monitor = {"top": TOP, "left": LEFT, "width": WIDTH, "height": HEIGHT}
end_monitor = {"top": END_TOP, "left": END_LEFT, "width": END_WIDTH, "height": END_HEIGHT}



with mss.mss() as sct:

    game_over_image = numpy.load(END_ARRAY_FILE)
    print(tuple(game_over_image)) #array of END_HEIGHT arrays each sub array has END_WIDTH values

    while True:
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        img = cv2.Canny(img,CANNY_EDGE_THRESHOLD_1,CANNY_EDGE_THRESHOLD_2)

        end_img = numpy.array(sct.grab(end_monitor))
        end_img = cv2.Canny(end_img,CANNY_EDGE_THRESHOLD_1,CANNY_EDGE_THRESHOLD_2)

        is_over = (game_over_image==end_img)
        
        cv2.imshow('monitor screen',img)
        cv2.imshow('game over screen ',end_img)

        print(is_over)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
    