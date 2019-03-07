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
	if (x >= MAP_X_R || y >= MAP_Y_R) return;

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
	int x = start_id % (MAP_X_R);
	int y = start_id / (float)(MAP_X_R);

	//int start_id = original_id;
	//int x = original_x;
	//int y = original_y;

	if (x >= MAP_X_R || y >= MAP_Y_R || x < 0 || y < 0 || destination.x >= MAP_X_R || destination.y >= MAP_Y_R) {	//return if start tex or destination is out of bounds
		return;
	}

	if (GetBoolMapValue(dynamic_map, destination.x, destination.y) == 0) {	//return if destination is unreachable
		((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -1;
		return;
	}

	if (GetBoolMapValue(dynamic_map, x, y) == 0) {	//return if start tex is in terrain
		((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -2;
		return;
	}

	node* open_list = (node*)malloc(3000 * sizeof(node));
	node* closed_list = (node*)malloc(3000 * sizeof(node));
	int open_list_it = 0, closed_list_it = 0, open_list_size = 3000, closed_list_size = 3000;

	if (open_list == NULL || closed_list == NULL) {
		printf("Device heap limit to low for lists\n");
		return;
	}

	open_list[0] = { start_id, -1, 0, FloatDistanceFromIDRelative(start_id, destination) };
	//++open_list_it;
	open_list_it = 1;

	int size_check_counter = 0;
	for (int step_iterator = 0; step_iterator < MAP_SIZE_R; ++step_iterator) {
		//find the next cell to expand
		float closest_distance_found = 99999;
		node closest_entry;
		int closest_coord_found = -1;
		node entry;

		for (int i = 0; i < open_list_it; ++i) {
			entry = open_list[i];
			if (entry.pos != -1) {
				if (entry.est_dist_start_to_dest_via_pos < closest_distance_found) {
					closest_distance_found = entry.est_dist_start_to_dest_via_pos;
					closest_coord_found = i;
					closest_entry = entry;
				}
			}
		}

		if (closest_coord_found == -1) {	//open list is empty and no path to the destination is found, RIP
			((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -1;
			return;
		}

		//add the expanded coord to the closed list
		closed_list[closed_list_it] = closest_entry;
		++closed_list_it;

		int pos_x = closest_entry.pos % MAP_X_R;
		int pos_y = closest_entry.pos / (float)(MAP_X_R);

		if ((pos_x == destination.x) && (pos_y == destination.y)) {	//destination has been found! HYPE
			Backtrack(device_map, closed_list, closed_list_it - 1);
			free(open_list);
			free(closed_list);
			return;
		}

		//expand the size of the open and closed list if necessary
		if (size_check_counter == 30) {
			size_check_counter = 0;
			if ((open_list_size - open_list_it) < 200) {
				node* open_list_new = (node*)malloc(open_list_size * 2 * sizeof(node));
				memcpy(open_list_new, open_list, open_list_size);
				open_list_size *= 2;
				free(open_list);
				open_list = open_list_new;
			}
			if ((closed_list_size - closed_list_it) < 200) {
				node* closed_list_new = (node*)malloc(closed_list_size * 2 * sizeof(node));
				memcpy(closed_list_new, closed_list, closed_list_size);
				closed_list_size *= 2;
				free(closed_list);
				closed_list = closed_list_new;
			}
		}

		//add the expanded nodes neighbours to the open list
		short_coord neighbour_coords[4];
		neighbour_coords[0] = { pos_x, pos_y - 1 };
		neighbour_coords[1] = { pos_x - 1, pos_y };
		neighbour_coords[2] = { pos_x + 1, pos_y };
		neighbour_coords[3] = { pos_x, pos_y + 1 };

		int new_open_list_entries = 0;
		for (int i = 0; i < 4; ++i) {
			int coord_global = neighbour_coords[i].x + (neighbour_coords[i].y * MAP_X_R);

			if (neighbour_coords[i].x <= MAP_X_R && neighbour_coords[i].y <= MAP_Y_R && neighbour_coords[i].x > 0 && neighbour_coords[i].y > 0) {	//coord in map
				if (GetBoolMapValue(dynamic_map, neighbour_coords[i].x, neighbour_coords[i].y) != 0) {	//coord not in terrain
					if (IDInList(coord_global, open_list, open_list_it) == -1 && IDInList(coord_global, closed_list, closed_list_it) == -1) {	//coord not already in open or closed list
						node new_list_entry = {
							coord_global,
							closed_list_it - 1,
							closest_entry.steps_from_start + 0.8,
							closest_entry.steps_from_start + 0.8 + FloatDistance(neighbour_coords[i].x, neighbour_coords[i].y, destination.x, destination.y)
						};
						open_list[open_list_it + new_open_list_entries] = new_list_entry;
						++new_open_list_entries;
					}
				}
			}
		}
		open_list_it += new_open_list_entries;

		open_list[closest_coord_found].pos = -1;	//mark expanded node as invalid in the open list
		++size_check_counter;
	}
}

__device__ void Backtrack(cudaPitchedPtr device_map, node* closed_list, int start_it) {
	node curr = closed_list[start_it];

	for (int loop_count = 1; loop_count < MAP_SIZE_R + 1; ++loop_count) {
		int x = curr.pos % MAP_X_R;
		int y = curr.pos / (float)(MAP_X_R);

		((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = loop_count;

		if (curr.backtrack_it == -1) return;
		curr = closed_list[curr.backtrack_it];
	}
}


__global__ void DeviceAirIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map) {

}