#define BLOCK_AMOUNT 3
#define THREADS_PER_BLOCK 128 //max 1024, should be multiple of warp size (32)
#define THREADS_IN_GRID (BLOCK_AMOUNT*THREADS_PER_BLOCK)

__global__ void TestDevice(float* data){
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	data[id] = OtherDeviceFunction(data[id]);
}

__device__ int OtherDeviceFunction(int input) {
	return input + 50;
}