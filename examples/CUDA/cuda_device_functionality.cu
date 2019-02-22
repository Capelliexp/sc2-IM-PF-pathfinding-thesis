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

__global__ void DeviceGroundIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map, cudaPitchedPtr dynamic_map, list_double_entry* global_memory_im_list_storage) {
	int block_size = blockDim.x*blockDim.y;
	int grid_size = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);

	int id_block = blockIdx.x + blockIdx.y * gridDim.x;
	int original_x = threadIdx.x + blockIdx.x * blockDim.x;
	int original_y = threadIdx.y + blockIdx.y * blockDim.y;
	int original_id = threadIdx.x + id_block * block_size + threadIdx.y * blockDim.x;

	int start_id = (original_id + (original_id % block_size) * block_size) % grid_size;
	
	//overwritten later
	int x = start_id % (gridDim.x * blockDim.x);
	int y = start_id / (float)(gridDim.x * blockDim.x);

	//if (x > MAP_X_R || y > MAP_Y_R) return;
	//((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = id_global;

	//cull threads outside of tex
	if (x > MAP_X_R || y > MAP_Y_R) return;
	if (GetMapValue(dynamic_map, x, y) == 0) return;

	list_double_entry* open_list = (list_double_entry*)malloc(1000 * sizeof(list_double_entry));
	list_double_entry* closed_list = (list_double_entry*)malloc(1000 * sizeof(list_double_entry));
	//array max size: 75 000
	//list_double_entry* open_list = &global_memory_im_list_storage[start_id * (150000/2)];
	//list_double_entry* closed_list = &global_memory_im_list_storage[start_id * (150000/2) + (256000000/2)];
	int open_list_it = 0, closed_list_it = 0, open_list_size = 1000, closed_list_size = 1000;

	open_list[0] = { start_id, -1 };
	++open_list_it;

	while (1) {
		//find the next cell to expand
		float closest_distance_found = 999999999;
		float curr_node_dist = 0;
		list_double_entry closest_entry;
		int closest_coord_found = -1;
		list_double_entry entry;

		for (int i = 0; i < open_list_it; ++i) {
			entry = open_list[i];
			if (entry.node != -1) {
				if ((curr_node_dist = FloatDistanceFromIDRelative(entry.node, destination)) < closest_distance_found) {
					closest_distance_found = curr_node_dist;
					closest_coord_found = i;
					closest_entry.node = entry.node;
					closest_entry.backtrack_iterator = entry.backtrack_iterator;
				}
			}
		}

		if (closest_coord_found == -1) return;	//open list is empty and no path to the destination is found, RIP

		//add the expanded coord to the closed list
		closed_list[closed_list_it] = { closest_entry.node, closest_entry.backtrack_iterator };
		++closed_list_it;

		//add the expanded nodes neighbours to the open list
		x = closest_entry.node % (gridDim.x * blockDim.x);
		y = closest_entry.node / (float)(gridDim.x * blockDim.x);

		int new_open_list_entries = 0;
		for (int x_it = -1; x_it < 2; ++x_it) {
			int x_coord = x + x_it;
			for (int y_it = -1; y_it < 2; ++y_it) {
				int y_coord = y + y_it;
				if (x_it == 0 && y_it == 0) continue;
				int coord_global = closest_entry.node + x_it + (y_it * gridDim.x * blockDim.x);

				if (x_coord == destination.x && y_coord == destination.y) {	//destination has been found! HYPE
					Backtrack(device_map, closed_list, closed_list_it - 1);
					free(open_list);
					free(closed_list);
					return;
				}

				if (GetMapValue(dynamic_map, x_coord, y_coord) != 0) {	//coord not in terrain
					if (IDInList(coord_global, open_list, open_list_it) == -1 && IDInList(coord_global, closed_list, closed_list_it) == -1) {	//coord not already in open or closed list
						open_list[open_list_it + new_open_list_entries] = { coord_global, (closed_list_it - 1) };
						++new_open_list_entries;
					}
				}
			}
		}

		open_list[closest_coord_found].node = -1;	//mark expanded node as invalid in the open list
	}
}

__device__ void Backtrack(cudaPitchedPtr device_map, list_double_entry* closed_list, int start_it) {
	int loop_count = 1;
	list_double_entry node = closed_list[start_it];

	while (1) {
		int x = node.node % (gridDim.x * blockDim.x);
		int y = node.node / (float)(gridDim.x * blockDim.x);

		((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = loop_count;

		node = closed_list[node.backtrack_iterator];
		if (node.backtrack_iterator == -1) return;
		++loop_count;
	}
}


__global__ void DeviceAirIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map) {

}