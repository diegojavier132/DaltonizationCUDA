import cv2 as cv
import numpy as np

def daltonize(image, level, deficiency_type):
    
    RGB_to_LMS = np.array([[17.8824,43.5161,4.11935],[3.45565,27.1554,3.86714],[0.0299566,0.184309,1.46709]])

    LMS_to_RGB = np.linalg.inv(RGB_to_LMS)

    daltonization_matrices = {
        "Deuteranomaly": np.array([[1, 0, 0],
                                    [0.4942, 0, 1.2483],
                                    [0, 0, 1]]),
        "Protanomaly": np.array([[0, 2.0234, -2.5258],
                                  [0, 1, 0],
                                  [0, 0, 1]]),
        "Tritanomaly": np.array([[0.967, 0.033, 0],
                                  [0, 0.733, 0.267],
                                  [0, 0.183, 0.817]])
    }

    daltonization_matrix = daltonization_matrices.get(deficiency_type, None)

    if daltonization_matrix is None:
        raise ValueError("Invalid deficiency type. Supported types are: Deuteranomaly, Protanomaly, Tritanomaly")


    image_rgb = image / 255.0
    
    # 1. Convert RGB to LMS
    image_lms = np.dot(image_rgb, RGB_to_LMS.T)
    
    # 2. Simulate color blindness
    simulated_lms = np.dot(image_lms, daltonization_matrix.T)
    
    # 3. Apply the daltonization correction
    corrected_lms = image_lms + level * (image_lms - simulated_lms)
    
    # 4. Convert LMS back to RGB
    corrected_rgb = np.dot(corrected_lms, LMS_to_RGB.T)
    corrected_rgb = np.clip(corrected_rgb, 0, 1)
    corrected_image = (corrected_rgb * 255).astype(np.uint8)
    
    return corrected_image



correction_level = 100
correction_types = ["Deuteranomaly", "Protanomaly", "Tritanomaly"]
current_correction_type = "Deuteranomaly"

window_name = 'Daltonization for Color Deficiency Correction'
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

cv.createTrackbar("Correction Level (%)", window_name, correction_level, 200, lambda x: x)
cv.createTrackbar("Correction Type", window_name, 0, len(correction_types) - 1, lambda x: x)


vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()

    level = cv.getTrackbarPos("Correction Level (%)", window_name) / 100.0
    correction_type_index = cv.getTrackbarPos("Correction Type", window_name)
    
    current_correction_type = correction_types[correction_type_index]

    corrected = daltonize(frame, level, current_correction_type)

    cv.putText(frame, "Original", (10,30),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    cv.imshow(window_name, np.hstack([frame, corrected]))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
