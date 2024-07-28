import cv2 as cv
import numpy as np

# Colorspace matrices
RGB_to_LMS = np.array([[17.8824,43.5161,4.11935],
                        [3.45565,27.1554,3.86714],
                        [0.0299566,0.184309,1.46709]])
LMS_to_RGB = np.array([[0.0809,-0.1305,0.11672],
                        [-0.01025,0.054019327,-0.11361],
                        [-0.0003653,-0.004122,0.69351]]) # Inverse of RGB_to_LMS
# Shift matrix
shift = np.array([[0,0,0],
                    [0.7,1,0],
                    [0.7,0,1]])

def daltonize(img, intensity, deficiency):
    
    # Image RGB
    RGB = np.asarray(img, dtype=float)
    
    # Deficiency matrices
    daltonization_matrices = {
        'd': np.array([ [1, 0, 0],
                        [0.4942, 0, 1.2483],
                        [0, 0, 1]]),
        'p': np.array([ [0, 2.0234, -2.5258],
                        [0, 1, 0],
                        [0, 0, 1]]),
        't': np.array([ [1,0,0],
                        [0,1,0],
                        [-0.395913,0.801109,0]])
    }

    daltonization_matrix = daltonization_matrices.get(deficiency, None)
    if daltonization_matrix is None:
        raise ValueError("Invalid deficiency type. Supported types are: Deuteranomaly(d), Protanomaly(p), Tritanomaly(t)")
    
    # RGB to LMS
    LMS = np.tensordot(RGB[...,:3], RGB_to_LMS.T, axes=1)
    
    # LMS to simul LMS
    _LMS = np.tensordot(LMS[...,:3], daltonization_matrix.T, axes=1)

    # simul LMS to compensated LMS
    error = (LMS - _LMS) * intensity
    cLMS = np.tensordot(error[...,:3], shift.T, axes=1)
    
    # compensated LMS to compensated RGB
    _RGB = np.tensordot(cLMS[...,:3], LMS_to_RGB.T, axes=1)
    cRGB = _RGB + RGB

    # Clip the values to be within [0, 255]
    result = np.clip(cRGB, 0, 255).astype('uint8')
    
    return result

# Default parameters
correction_level = 100
correction_types = ["d", "p", "t"]
default_correction_type = "d"

# Get image data
path = "Images/colorblind.jpg" # Should work, if it doesn't, use full path to img
frame = cv.imread(path)

# GUI
window_name = 'Daltonization for Color Deficiency Correction'
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
cv.createTrackbar("Correction Level (%)", window_name, correction_level, 500, lambda x: x)
cv.createTrackbar("Correction Type", window_name, 0, len(correction_types) - 1, lambda x: x)

while True:
    
    # Get window size
    window_width, window_height = cv.getWindowImageRect(window_name)[2:4]
    
    # Calculate new size to maintain aspect ratio
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    if window_width / aspect_ratio <= window_height:
        new_width = window_width
        new_height = int(window_width / aspect_ratio)
    else:
        new_height = window_height
        new_width = int(window_height * aspect_ratio)

    resized_frame = cv.resize(frame, (new_width, new_height))

    level = cv.getTrackbarPos("Correction Level (%)", window_name) / 100.0
    correction_type_index = cv.getTrackbarPos("Correction Type", window_name)
    
    default_correction_type = correction_types[correction_type_index]

    corrected = daltonize(resized_frame, level, default_correction_type)

    cv.putText(resized_frame, "Original", (10,30),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    cv.imshow(window_name, np.hstack([resized_frame, corrected]))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv.destroyAllWindows()