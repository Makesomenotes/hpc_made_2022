from pycuda.compiler import SourceModule


histogramModule = SourceModule("""
  __global__ void histogram(const int binCount, const int *bins, const int rows, const int cols, const int *img, int *hist)
  {
    unsigned int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
    if (yIdx < rows && xIdx < cols) {
        int val = img[yIdx * cols + xIdx];
        for (int i = 0; i < binCount; i++) {
            if (val >= bins[i] && val < bins[i + 1]) {
                atomicAdd(&(hist[i]), 1);
            }
        }
    }
  }
  """)
  