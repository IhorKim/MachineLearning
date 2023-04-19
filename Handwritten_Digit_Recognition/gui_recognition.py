import numpy as np
import cv2
import pygame
import sys
from pygame.locals import *
from keras.models import load_model

WINDOWSIZE_X = 640
WINDOWSIZE_Y = 480
WHITE = (255, 255, 255)
WHITE_INT = 255
BLACK = (0, 0, 0)
RED = (255, 0, 0)
SAVE_IMAGE = False
MODEL = load_model("Handwritten_Digit_Recognition//bestmodel.h5")
LABELS = {0: "Zero", 1: "One",
          2: "Two", 3: "Three",
          4: "Four", 5: "Five",
          6: "Six", 7: "Seven",
          8: "Eight", 9: "Nine"}
IS_WRITING = False
NUMBER_xcord = []
NUMBER_ycord = []
BOUNDARY = 5
IMAGE_COUNT = 1
PREDICT = True

pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAY_SURFACE = pygame.display.set_mode((WINDOWSIZE_X, WINDOWSIZE_Y))
pygame.display.set_caption("Board")

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and IS_WRITING:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAY_SURFACE, WHITE, (xcord, ycord), 4, 0)

            NUMBER_xcord.append(xcord)
            NUMBER_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            IS_WRITING = True

        if event.type == MOUSEBUTTONUP:
            IS_WRITING = False
            NUMBER_xcord = sorted(NUMBER_xcord)
            NUMBER_ycord = sorted(NUMBER_ycord)

            rect_min_x, rect_max_x = max(NUMBER_xcord[0] - BOUNDARY, 0), min(WINDOWSIZE_X, NUMBER_xcord[-1] + BOUNDARY)
            rect_min_y, rect_max_y = max(0, NUMBER_ycord[0] - BOUNDARY), min(NUMBER_ycord[-1] + BOUNDARY, WINDOWSIZE_Y)

            NUMBER_xcord = []
            NUMBER_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAY_SURFACE))[rect_min_x: rect_max_x,
                      rect_min_y: rect_max_y].T.astype(np.float32)

            if SAVE_IMAGE:
                cv2.imwrite("image.png")
                IMAGE_COUNT += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))  # our model was trained with size (28, 28)
                image = np.pad(image, (10, 10), "constant", constant_values=0)
                image = cv2.resize(image, (28, 28)) / WHITE_INT

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))]).title()

                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                DISPLAY_SURFACE.blit(textSurface, textRectObj)

            pygame.draw.rect(DISPLAY_SURFACE, RED,
                             (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 3)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAY_SURFACE.fill(BLACK)

        pygame.display.update()
