import cv2 as cv
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDA kernel for converting RGB to LMS
kernel_code = """
__global__ void rgb_to_lms(float *rgb, float *lms, int width, int height, float *RGB_to_LMS) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * 3;

    if (x < width && y < height) {
        float r = rgb[idx];
        float g = rgb[idx + 1];
        float b = rgb[idx + 2];

        lms[idx]     = RGB_to_LMS[0] * r + RGB_to_LMS[1] * g + RGB_to_LMS[2] * b;
        lms[idx + 1] = RGB_to_LMS[3] * r + RGB_to_LMS[4] * g + RGB_to_LMS[5] * b;
        lms[idx + 2] = RGB_to_LMS[6] * r + RGB_to_LMS[7] * g + RGB_to_LMS[8] * b;
    }
}

__global__ void simulate_color_blindness(float *lms, float *simulated_lms, int width, int height, float *daltonization_matrix) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * 3;

    if (x < width && y < height) {
        float l = lms[idx];
        float m = lms[idx + 1];
        float s = lms[idx + 2];

        simulated_lms[idx]     = daltonization_matrix[0] * l + daltonization_matrix[1] * m + daltonization_matrix[2] * s;
        simulated_lms[idx + 1] = daltonization_matrix[3] * l + daltonization_matrix[4] * m + daltonization_matrix[5] * s;
        simulated_lms[idx + 2] = daltonization_matrix[6] * l + daltonization_matrix[7] * m + daltonization_matrix[8] * s;
    }
}

__global__ void apply_daltonization_correction(float *lms, float *simulated_lms, float *corrected_lms, int width, int height, float level) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * 3;

    if (x < width && y < height) {
        corrected_lms[idx]     = lms[idx]     + level * (lms[idx]     - simulated_lms[idx]);
        corrected_lms[idx + 1] = lms[idx + 1] + level * (lms[idx + 1] - simulated_lms[idx + 1]);
        corrected_lms[idx + 2] = lms[idx + 2] + level * (lms[idx + 2] - simulated_lms[idx + 2]);
    }
}

__global__ void lms_to_rgb(float *lms, float *rgb, int width, int height, float *LMS_to_RGB) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * 3;

    if (x < width && y < height) {
        float l = lms[idx];
        float m = lms[idx + 1];
        float s = lms[idx + 2];

        rgb[idx]     = LMS_to_RGB[0] * l + LMS_to_RGB[1] * m + LMS_to_RGB[2] * s;
        rgb[idx + 1] = LMS_to_RGB[3] * l + LMS_to_RGB[4] * m + LMS_to_RGB[5] * s;
        rgb[idx + 2] = LMS_to_RGB[6] * l + LMS_to_RGB[7] * m + LMS_to_RGB[8] * s;
    }
}
"""

mod = SourceModule(kernel_code)

rgb_to_lms = mod.get_function("rgb_to_lms")
simulate_color_blindness = mod.get_function("simulate_color_blindness")
apply_daltonization_correction = mod.get_function("apply_daltonization_correction")
lms_to_rgb = mod.get_function("lms_to_rgb")

def daltonize(image, level, deficiency_type):
    RGB_to_LMS = np.array([[17.8824, 43.5161, 4.11935],
                           [3.45565, 27.1554, 3.86714],
                           [0.0299566, 0.184309, 1.46709]], dtype=np.float32)

    LMS_to_RGB = np.linalg.inv(RGB_to_LMS).astype(np.float32)

    daltonization_matrices = {
        "Deuteranomaly": np.array([[1, 0, 0],
                                   [0.4942, 0, 1.2483],
                                   [0, 0, 1]], dtype=np.float32),
        "Protanomaly": np.array([[0, 2.0234, -2.5258],
                                 [0, 1, 0],
                                 [0, 0, 1]], dtype=np.float32),
        "Tritanomaly": np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [-0.395913, 0.801109, 0]], dtype=np.float32)
    }

    daltonization_matrix = daltonization_matrices.get(deficiency_type, None)
    if daltonization_matrix is None:
        raise ValueError("Invalid deficiency type. Supported types are: Deuteranomaly, Protanomaly, Tritanomaly")

    image_rgb = image.astype(np.float32) / 255.0
    height, width, channels = image_rgb.shape

    rgb = image_rgb.flatten()
    lms = np.empty_like(rgb)
    simulated_lms = np.empty_like(rgb)
    corrected_lms = np.empty_like(rgb)
    corrected_rgb = np.empty_like(rgb)

    rgb_gpu = cuda.mem_alloc(rgb.nbytes)
    lms_gpu = cuda.mem_alloc(lms.nbytes)
    simulated_lms_gpu = cuda.mem_alloc(simulated_lms.nbytes)
    corrected_lms_gpu = cuda.mem_alloc(corrected_lms.nbytes)
    corrected_rgb_gpu = cuda.mem_alloc(corrected_rgb.nbytes)

    cuda.memcpy_htod(rgb_gpu, rgb)
    cuda.memcpy_htod(lms_gpu, lms)
    cuda.memcpy_htod(simulated_lms_gpu, simulated_lms)
    cuda.memcpy_htod(corrected_lms_gpu, corrected_lms)
    cuda.memcpy_htod(corrected_rgb_gpu, corrected_rgb)

    block_size = (16, 16, 1)
    grid_size = (int((width + block_size[0] - 1) / block_size[0]), int((height + block_size[1] - 1) / block_size[1]))

    rgb_to_lms(rgb_gpu, lms_gpu, np.int32(width), np.int32(height), cuda.In(RGB_to_LMS.flatten()), block=block_size, grid=grid_size)
    simulate_color_blindness(lms_gpu, simulated_lms_gpu, np.int32(width), np.int32(height), cuda.In(daltonization_matrix.flatten()), block=block_size, grid=grid_size)
    apply_daltonization_correction(lms_gpu, simulated_lms_gpu, corrected_lms_gpu, np.int32(width), np.int32(height), np.float32(level), block=block_size, grid=grid_size)
    lms_to_rgb(corrected_lms_gpu, corrected_rgb_gpu, np.int32(width), np.int32(height), cuda.In(LMS_to_RGB.flatten()), block=block_size, grid=grid_size)

    cuda.memcpy_dtoh(corrected_rgb, corrected_rgb_gpu)

    corrected_rgb = np.clip(corrected_rgb, 0, 1)
    corrected_image = (corrected_rgb.reshape((height, width, channels)) * 255).astype(np.uint8)

    return corrected_image

correction_level = 100
correction_types = ["Deuteranomaly", "Protanomaly", "Tritanomaly"]
current_correction_type = "Deuteranomaly"

window_name = 'Daltonization for Color Deficiency Correction'
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

cv.createTrackbar("Correction Level (%)", window_name, correction_level, 500, lambda x: x)
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
