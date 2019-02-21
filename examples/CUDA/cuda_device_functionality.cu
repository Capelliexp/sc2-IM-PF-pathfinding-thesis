#pragma once

#include "../examples/CUDA/cuda_header.cuh"
#include "../examples/CUDA/cuda_device_utility.cu"

/*
PF Todo:
* Quad-tree for units
* Compare simultaneous global write vs non-simultaneous
* Compare different block sizes & dimensions
*/

__global__ void DeviceRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map_ground, cudaPitchedPtr device_map_air) {
	extern __shared__ Entity unit_list_s[];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_block = threadIdx.x + threadIdx.y * blockDim.x;
	int id_global = x + y * blockDim.x;

	//move unit list to shared memory
	if (id_block < nr_of_units) unit_list_s[id_block] = device_unit_list_pointer[id_block];

	//cull threads outside of tex
	if (x > MAP_X_R || y > MAP_Y_R) return;

	__syncthreads();

	//add upp all fields affecting the owned coord to ground_charge and air_charge
	float ground_charge = 0;
	float air_charge = 0;
	float dist = 0;
	for (int i = 0; i < nr_of_units; ++i) {
		UnitInfoDevice unit = device_unit_lookup[unit_list_s[i].id];
		float range_sub = unit.range;

		if ((dist = (FloatDistance(unit_list_s[i].pos.x, unit_list_s[i].pos.y, x, y) + 0.0001)) < range_sub) {
			ground_charge += ((range_sub / dist) * unit.can_attack_ground * unit_list_s[i].enemy);
			air_charge += ((range_sub / dist) * unit.can_attack_air * unit_list_s[i].enemy);
		}
	}

	//__syncthreads();
	
	//write ground_charge and air_charge to global memory in owned coord
	((float*)(((char*)device_map_ground.ptr) + y * device_map_ground.pitch))[x] = ground_charge;
	((float*)(((char*)device_map_air.ptr) + y * device_map_ground.pitch))[x] = air_charge;
}

__global__ void DeviceAttractingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, int owner_type_id, cudaPitchedPtr device_map){

}

__global__ void DeviceGroundIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map, cudaPitchedPtr dynamic_map, list_entry* global_memory_im_list_storage) {
	int block_size = blockDim.x*blockDim.y;
	int grid_size = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);

	//int original_x = threadIdx.x + blockIdx.x * blockDim.x;
	//int original_y = threadIdx.y + blockIdx.y * blockDim.y;

	int id_block = blockIdx.x + blockIdx.y * gridDim.x;
	int id_global = threadIdx.x + id_block * block_size + threadIdx.y * blockDim.x;
	int new_id_global = (id_global + (id_global % block_size) * block_size) % grid_size;
	
	int x = new_id_global % (gridDim.x * blockDim.x);
	int y = new_id_global / (float)(gridDim.x * blockDim.x);

	//if (x > MAP_X_R || y > MAP_Y_R) return;
	//((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = id_global;

	//cull threads outside of tex
	if (x > MAP_X_R || y > MAP_Y_R) return;
	if (((float*)(((char*)dynamic_map.ptr) + y * dynamic_map.pitch))[x] == 0) return;

	//array max size: 150 000
	list_entry* open_list = &global_memory_im_list_storage[new_id_global * (300000/2)];
	list_entry* closed_list = &global_memory_im_list_storage[new_id_global * (300000/2) + (512000000/2)];
	int open_list_it = 0, closed_list_it = 0;

	open_list[0] = new_id_global;



}

__global__ void DeviceAirIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map) {

}