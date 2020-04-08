# -*- coding: utf-8 -*-
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

import visualization_utils as vis_util
from button_detection import ButtonDetector
from chars_recognition import CharacterRecognizer

class ButtonRecognition:
    def __init__(self):
        self.detector = ButtonDetector()
        self.recognizer = CharacterRecognizer(verbose=False)
        self.warmup()

    def warmup(self, panel_path='./test_panels/image1.jpg', button_path='./test_panels/test_button.png'):
        image = imageio.imread(panel_path)
        button = imageio.imread(button_path)
        self.detector.predict(image)
        self.recognizer.predict(button)

    def button_candidates(self, boxes, scores, image_np, score_threshold=0.5, press_threshold=19):
        # TBD: press detection needs to be optimized or modified based on environments
        avg_col = np.mean(image_np[:,:,2])
        #print("average color", avg_col, " ##################")
        #if avg_col > 125: avg_col = 160
        img_height = image_np.shape[0]
        img_width = image_np.shape[1]

        button_scores = []
        button_patches = []
        button_positions = []
        button_presses = []

        for box, score in zip(boxes, scores):
            if score < score_threshold: continue

            y_min = int(box[0] * img_height)
            x_min = int(box[1] * img_width)
            y_max = int(box[2] * img_height)
            x_max = int(box[3] * img_width)

            button_patch = image_np[y_min: y_max, x_min: x_max]
            
            button_scores.append(score)
            button_patches.append(button_patch)
            button_positions.append([x_min, y_min, x_max, y_max])

            buf = np.copy(button_patch)#[:,:,2])
            buf[buf != 255] = 0 #buf[buf > 0] = 1
            v = np.sum(buf)/(max(img_height,img_width))
            #print(np.mean(button_patch[:,:,2]), v)
            
            button_presses.append(False)
            if (avg_col < 60 and v > press_threshold*10) or \
               (avg_col >= 60 and v > press_threshold): button_presses[-1] = True
            
        return button_patches, button_positions, button_scores, button_presses

    def draw_detection_result(self, image_np, boxes, classes, scores, category, predict_chars=None):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category,
            max_boxes_to_draw=100,
            use_normalized_coordinates=True,
            line_thickness=5,
            predict_chars=predict_chars
        )

    def draw_recognition_result(self, image_np, recognitions):
        
        for (button_patch,_,chars,_,pos,press) in recognitions:    
            # generate image layer for drawing
            img_pil = Image.fromarray(button_patch)            
            img_show = ImageDraw.Draw(img_pil)
            # draw at a proper location
            x_center = button_patch.shape[1] / 2.0
            y_center = button_patch.shape[0] / 2.0
            font_size = min(x_center, y_center)*.8
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", int(font_size))
            text_center = int(x_center), int(y_center)
            text_color = (0, 0, 255) if press == True else (255, 0, 0)
            #print(text_color, press)
            img_show.text(text_center, text=chars, fill=text_color, font=font)
            image_np[pos[1]: pos[3], pos[0]: pos[2]] = np.array(img_pil)

    def predict(self, image_np, draw=False, image_f=False):
        recognitions = []
        boxes, scores, _, classes = self.detector.predict(image_np)
        button_patches, button_positions, button_scores, button_presses = self.button_candidates(boxes, scores, image_np)
        for i, button_img in enumerate(button_patches):
            button_text, text_score, _ = self.recognizer.predict(button_img)
            if image_f: button_presses[i] = 1 - button_presses[i]
            recognitions.append((button_img, button_scores[i], button_text, text_score, button_positions[i], button_presses[i]))

        if draw:
            self.draw_detection_result(image_np, boxes, classes, scores, self.detector.category_index)
            self.draw_recognition_result(image_np, recognitions)

        return image_np, recognitions

    def clear_session(self):
        self.detector.clear_session()
        self.recognizer.clear_session()

# end of file 