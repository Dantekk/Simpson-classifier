from classifier import Classifier
import cv2 as cv
from keras.preprocessing import image
import numpy as np

# Pre processing necessary for input net
def pre_processing_img(img, SCALE):
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor

# Simple image post processing for put text label on image
def post_processing_img(img, class_predicted):

    y = img.shape[0]
    x = img.shape[1]
    start_point = (0, int(y*0.9))
    end_point = (x, y)
    cv.rectangle(img, start_point, end_point, (0, 0, 255), cv.FILLED)
    cv.rectangle(img, start_point, end_point, (255, 0, 0), thickness = 3)
    cv.putText(img, class_predicted, ( int(x * 0.08), int(y * 0.975)), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2,
               cv.LINE_AA)

    return img


def main():

    SCALE = (200, 200)

    c = Classifier()
    c.model_load(path_model = "simpson_classifier.h5")
    c.set_image_size(SCALE)

    # Load and pre processing image

    # Insert image path
    img_path = "./sideshowbob-whitebg.jpg"

    img = image.load_img(img_path, target_size=SCALE)
    input_img = pre_processing_img(img = img, SCALE = SCALE)

    # For predict class image
    class_predicted = c.model_predict_img(input_img)

    print(class_predicted)

    # Processing output image
    img = cv.imread(img_path)
    img = post_processing_img(img, class_predicted.upper())
    cv.imshow('Simpson Image Classification', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite(class_predicted+"_pred.jpg",img)

if __name__ == "__main__":
    main()
