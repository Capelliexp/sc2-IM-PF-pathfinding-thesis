#pragma once

#include "../examples/CUDA/cuda_header.cuh"

//DEVICE SYMBOL VARIABLES (const & global)
__device__ __constant__ UnitInfoDevice* device_unit_lookup;

__device__ float GetFloatMapValue(cudaPitchedPtr map, int x, int y) {
	char* ptr = (char*)map.ptr;
	size_t pitch = map.pitch;

	return *((float*)((char*)ptr + y * pitch) + x);
}

//__device__ float GetFloatMapValue(cudaPitchedPtr map, int global_id) {
//	int x = (global_id % (MAP_X_R)) + 1;
//	int y = (global_id / (float)(MAP_X_R)) + 1;
//
//	char* ptr = (char*)map.ptr;
//	size_t pitch = map.pitch;
//
//	return *((float*)((char*)ptr + y * pitch) + x);
//}

__device__ bool GetBoolMapValue(cudaPitchedPtr map, int x, int y) {
	char* ptr = (char*)map.ptr;
	size_t pitch = map.pitch;

	return *((bool*)((char*)ptr + y * pitch) + x);
}

//__device__ bool GetBoolMapValue(cudaPitchedPtr map, int global_id) {
//	int x = (global_id % (MAP_X_R)) + 1;
//	int y = (global_id / (float)(MAP_X_R)) + 1;
//
//	char* ptr = (char*)map.ptr;
//	size_t pitch = map.pitch;
//
//	return *((bool*)((char*)ptr + y * pitch) + x);
//}

/* check if the id is present in the given list. This could possibly be sped up by fetching
many entries at once... */
__device__ int IDInList(int id, node* list, int list_length){
	for (int i = 0; i < list_length; ++i) {
		if (list[i].pos == id) {
			return i;
		}
	}

	return -1;
}

__device__ void SetMapValue(cudaPitchedPtr map, int x, int y, float value) {
	char* ptr = (char*)map.ptr;
	size_t pitch = map.pitch;

	float* row = (float*)((char*)ptr + y * pitch);
	row[x] = value;
}


//returnes the distance, not divided for grid sub-division
__device__ float FloatDistance(float posX1, float posY1, float posX2, float posY2) {
	float a = powf(posX2 - posX1, 2);
	float b = powf(posY2 - posY1, 2);
	return sqrtf(a + b)/* / GRID_DIVISION*/;
}

//returnes the distance, not divided for grid sub-division
//__device__ float FloatDistanceFromIDRelative(int ID, IntPoint2D destination) {
//	float a = powf(destination.x - ((ID % MAP_X_R) + 1), 2);
//	float b = powf(destination.y - ((ID / (float)(MAP_X_R)) + 1), 2);
//	return sqrt(a + b);
//}

__device__ int BlockDistance(int posX1, int posY1, int posX2, int posY2) {
	int a = fabsf(posX1 - posX2);
	int b = fabsf(posY1 - posY2);
	return a + b;
}

//__device__ int BlockDistance(int ID, IntPoint2D destination) {
//	int a = fabsf(destination.x - ((ID % MAP_X_R) + 1));
//	int b = fabsf(destination.y - ((ID / (float)(MAP_X_R)) + 1));
//	return a + b;
//}

__device__ int PosToID(IntPoint2D pos) {
	return (pos.x) + ((pos.y) * MAP_X_R);
}

//__device__ int PosToID(IntPoint2D pos) {
//	return (pos.x) + ((pos.y) * (gridDim.x * blockDim.x));
//}

__device__ IntPoint2D IDToPos(int ID) {
	IntPoint2D res;
	res.x = (ID % MAP_X_R);
	res.y = (ID / (float)MAP_X_R);
	return res;
}

//__device__ IntPoint2D IDToPos(int ID) {
//	IntPoint2D res;
//	res.x = (ID % (gridDim.x * blockDim.x));
//	res.y = (ID / (float)(gridDim.x * blockDim.x));
//	return res;
//}