import cv2 as cv
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Colorspace constant matrices
RGB_to_LMS = np.array([ [17.8824,43.5161,4.11935],
                        [3.45565,27.1554,3.86714],
                        [0.0299566,0.184309,1.46709]]).astype(np.float32)
LMS_to_RGB = np.array([ [0.0809,-0.1305,0.11672],
                        [-0.01025,0.054019327,-0.11361],
                        [-0.0003653,-0.004122,0.69351]]).astype(np.float32)
# Shift matrix
shift = np.array([  [0,0,0],
                    [0.7,1,0],
                    [0.7,0,1]]).astype(np.float32)

kernel_code = """
__constant__ float RGB_to_LMS[9];
__constant__ float LMS_to_RGB[9];
__constant__ float shift[9];

__global__ void daltonize(float *img, float intensity, char deficiency, float *result, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * 3; // 1D addressing
    
    if (x >= width || y >= height) return;
    
    float daltonization_matrices[3][3];
    
    // Set daltonization matrices based on deficiency
    if (deficiency == 'd') {  // Deuteranomaly
        daltonization_matrices[0][0] = 1; daltonization_matrices[0][1] = 0; daltonization_matrices[0][2] = 0;
        daltonization_matrices[1][0] = 0.4942; daltonization_matrices[1][1] = 0; daltonization_matrices[1][2] = 1.2483;
        daltonization_matrices[2][0] = 0; daltonization_matrices[2][1] = 0; daltonization_matrices[2][2] = 1;
    } else if (deficiency == 'p') {  // Protanomaly
        daltonization_matrices[0][0] = 0; daltonization_matrices[0][1] = 2.0234; daltonization_matrices[0][2] = -2.5258;
        daltonization_matrices[1][0] = 0; daltonization_matrices[1][1] = 1; daltonization_matrices[1][2] = 0;
        daltonization_matrices[2][0] = 0; daltonization_matrices[2][1] = 0; daltonization_matrices[2][2] = 1;
    } else if (deficiency == 't') {  // Tritanomaly
        daltonization_matrices[0][0] = 1; daltonization_matrices[0][1] = 0; daltonization_matrices[0][2] = 0;
        daltonization_matrices[1][0] = 0; daltonization_matrices[1][1] = 1; daltonization_matrices[1][2] = 0;
        daltonization_matrices[2][0] = -0.395913; daltonization_matrices[2][1] = 0.801109; daltonization_matrices[2][2] = 0;
    } else {
        printf("Invalid deficiency type. Supported types are: Deuteranomaly(d), Protanomaly(p), Tritanomaly(t)");
        return;
    }
    
    // RGB to LMS
    float R = img[idx];
    float G = img[idx + 1];
    float B = img[idx + 2];

    float L = RGB_to_LMS[0] * R + RGB_to_LMS[1] * G + RGB_to_LMS[2] * B;
    float M = RGB_to_LMS[3] * R + RGB_to_LMS[4] * G + RGB_to_LMS[5] * B;
    float S = RGB_to_LMS[6] * R + RGB_to_LMS[7] * G + RGB_to_LMS[8] * B;

    // LMS to simul LMS
    float _L = daltonization_matrices[0][0] * L + daltonization_matrices[0][1] * M + daltonization_matrices[0][2] * S;
    float _M = daltonization_matrices[1][0] * L + daltonization_matrices[1][1] * M + daltonization_matrices[1][2] * S;
    float _S = daltonization_matrices[2][0] * L + daltonization_matrices[2][1] * M + daltonization_matrices[2][2] * S;

    // Error calculation
    float eL = (L - _L) * intensity;
    float eM = (M - _M) * intensity;
    float eS = (S - _S) * intensity;

    // Compensated LMS
    float cL = eL * shift[0] + eM * shift[1] + eS * shift[2];
    float cM = eL * shift[3] + eM * shift[4] + eS * shift[5];
    float cS = eL * shift[6] + eM * shift[7] + eS * shift[8];

    // Compensated LMS to compensated RGB
    float _R = LMS_to_RGB[0] * cL + LMS_to_RGB[1] * cM + LMS_to_RGB[2] * cS;
    float _G = LMS_to_RGB[3] * cL + LMS_to_RGB[4] * cM + LMS_to_RGB[5] * cS;
    float _B = LMS_to_RGB[6] * cL + LMS_to_RGB[7] * cM + LMS_to_RGB[8] * cS;

    // Final RGB
    result[idx] = min(max(R + _R, 0.0), 255.0);
    result[idx + 1] = min(max(G + _G, 0.0), 255.0);
    result[idx + 2] = min(max(B + _B, 0.0), 255.0);
}
"""

mod = SourceModule(kernel_code)

# Copy constants to device
RGB_to_LMS_gpu = mod.get_global('RGB_to_LMS')[0]
LMS_to_RGB_gpu = mod.get_global('LMS_to_RGB')[0]
shift_gpu = mod.get_global('shift')[0]

# Host to Device
cuda.memcpy_htod(RGB_to_LMS_gpu, RGB_to_LMS)
cuda.memcpy_htod(LMS_to_RGB_gpu, LMS_to_RGB)
cuda.memcpy_htod(shift_gpu, shift)

# Define the kernel function
daltonize_kernel = mod.get_function("daltonize")

def daltonize_gpu(img, intensity, deficiency):
    height, width, channels = img.shape
    img = img.astype(np.float32).flatten()
    result = np.zeros_like(img)

    img_gpu = cuda.mem_alloc(img.nbytes)
    result_gpu = cuda.mem_alloc(result.nbytes)

    cuda.memcpy_htod(img_gpu, img)
    
    block = (16, 16, 1)
    grid = (int(np.ceil(width / block[0])), int(np.ceil(height / block[1])), 1)
    
    daltonize_kernel(img_gpu, np.float32(intensity), np.int8(ord(deficiency)), result_gpu, np.int32(width), np.int32(height), block=block, grid=grid)
    
    cuda.memcpy_dtoh(result, result_gpu)
    result = result.reshape((height, width, channels)).astype(np.uint8)
    
    img_gpu.free()
    result_gpu.free()

    return result

# Default parameters
correction_level = 100
correction_types = ["d", "p", "t"]
default_correction_type = "d"

# Input video
vid = cv.VideoCapture(0)

# GUI
window_name = 'Daltonization for Color Deficiency Correction'
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
cv.createTrackbar("Correction Level (%)", window_name, correction_level, 500, lambda x: x)
cv.createTrackbar("Correction Type", window_name, 0, len(correction_types) - 1, lambda x: x)

while True:
    ret, frame = vid.read()
    
    level = cv.getTrackbarPos("Correction Level (%)", window_name) / 100.0
    correction_type_index = cv.getTrackbarPos("Correction Type", window_name)
    
    default_correction_type = correction_types[correction_type_index]

    corrected = daltonize_gpu(frame, level, default_correction_type)

    cv.putText(frame, "Original", (10,30),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    cv.imshow(window_name, np.hstack([frame, corrected]))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()