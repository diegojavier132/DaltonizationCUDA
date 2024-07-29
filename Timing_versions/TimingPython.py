import cv2 as cv
import numpy as np
from timeit import default_timer as timer

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

# Get image data
path = "Images/colorblind.jpg" # Should work, if it doesn't, use full path to img
frame = cv.imread(path)

exec_time = 0

for i in range(0,10):
    start = timer()
    daltonize(frame, 1, 'd')
    end = timer()
    # Function execution time
    print(f"[{i}]Execution Time: {end - start}")
    exec_time = exec_time + end - start

exec_time = exec_time / 10   
print(f"Avg Execution Time: {exec_time}")