#pragma once

#include "../examples/CUDA/cuda_header.cuh"
#include "../examples/CUDA/cuda_device_utility.cu"

__global__ void DeviceRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map_ground, cudaPitchedPtr device_map_air) {
	extern __shared__ Entity unit_list_s[];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_block = threadIdx.x + threadIdx.y * blockDim.x;
	int id_global = x + y * blockDim.x;

	//move unit list to shared memory
	if (id_block < nr_of_units) unit_list_s[id_block] = device_unit_list_pointer[id_block];

	if (x >= MAP_X_R || y >= MAP_Y_R || x < 0 || y < 0) {	//return if start tex is out of bounds 
		return;
	}

	__syncthreads();

	//add upp all fields affecting the owned coord to ground_charge and air_charge
	float ground_charge = 0;
	float air_charge = 0;
	float dist = 0;
	for (int i = 0; i < nr_of_units; ++i) {
		UnitInfoDevice unit = device_unit_lookup[unit_list_s[i].id];
		float range_sub = unit.range + 2;
		dist = (FloatDistance((int)unit_list_s[i].pos.x, (int)unit_list_s[i].pos.y, x, y) + 0.0001);

		if (unit_list_s[i].enemy) {	//avoid enemies
			if (dist < range_sub) {
				ground_charge += ((range_sub / dist) * unit.can_attack_ground) + 50;
				air_charge += ((range_sub / dist) * unit.can_attack_air) + 50;
			}
		}
		else {	//avoid friendlies
			int res = 1 - (int)dist + (int)(unit.radius + 0.5);
			if (res > 0) {
				ground_charge += (res/2.f) * !(unit.is_flying);
				air_charge += (res/2.f) * unit.is_flying;
			}
		}
	}

	//__syncthreads();
	
	//write ground_charge and air_charge to global memory in owned coord
	((float*)(((char*)device_map_ground.ptr) + y * device_map_ground.pitch))[x] = ground_charge;
	((float*)(((char*)device_map_air.ptr) + y * device_map_ground.pitch))[x] = air_charge;
}

__global__ void DeviceAttractingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, int owner_type_id, cudaPitchedPtr device_map){
	extern __shared__ Entity unit_list_s[];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_block = threadIdx.x + threadIdx.y * blockDim.x;
	int id_global = x + y * blockDim.x;

	//move unit list to shared memory
	if (id_block < nr_of_units) unit_list_s[id_block] = device_unit_list_pointer[id_block];

	if (x >= MAP_X_R || y >= MAP_Y_R || x < 0 || y < 0) {	//return if start tex is out of bounds 
		return;
	}

	__syncthreads();

	UnitInfoDevice self_info = device_unit_lookup[owner_type_id];

	float tot_charge = 0;
	Entity unit;
	for (int i = 0; i < nr_of_units; ++i) {
		UnitInfoDevice other_info = device_unit_lookup[unit_list_s[i].id];
		Entity other_entity = unit_list_s[i];

		float dist = (FloatDistance((int)other_entity.pos.x, (int)other_entity.pos.y, x, y) + 0.0001);
		bool self_can_attack_other = (other_info.is_flying && self_info.can_attack_air) || (!other_info.is_flying && self_info.can_attack_ground);

		if (other_entity.enemy) {	//attack enemy
			if (self_can_attack_other) {	//can attack unit
				if (self_info.range < 1.1) {	//self is melee
					if (dist < 10) {	//attack enemy
						//tot_charge += 10 / dist;
						tot_charge -= 10 / dist;
					}
				}
				else {	//self is ranged
					float range_diff = self_info.range - other_info.range;
					if (range_diff > 0) {	//self more range than other
						if (dist < (other_info.range + (self_info.radius/* + 1*/))) {	//avoid area close to enemy
							tot_charge += 10 / dist;
						}
						else if (dist < self_info.range * 1.2 || dist < 10) {	//attack enemy
							tot_charge -= 10 / dist;
						}
					}
					else {	//attack other with larger range than self
						tot_charge -= 10 / dist;
					}
					//if(dist < (self_info.range - 3)){	//avoid area close to enemy
					//	//tot_charge -= 10 / dist;
					//	tot_charge += 10 / dist;
					//}
					//else if (dist > (self_info.range - 3) && (dist < self_info.range * 1.2 || dist < 10)) {	//attack enemy
					//	//tot_charge += 10 / dist;
					//	tot_charge -= 10 / dist;
					//}
				}
			}
		}
		else {	//avoid friend
			if (self_info.is_flying == other_info.is_flying) {
				/*if (dist < (other_info.radius * 1.2)) {
					tot_charge += 10 / dist;
				}*/
				int res = 1 - (int)dist + (int)(other_info.radius + 0.5);
				if (res > 0) {
					tot_charge += (res / 2.f);
				}
			}
		}
	}
	//__syncthreads();

	((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = tot_charge;
}

//Maximum number of 32-bit registers per thread block: 64k
//Maximum number of 32-bit registers per thread: 255
//Maximum amount of shared memory per multiprocessor: 48kB (49152B)
//Maximum amount of shared memory per thread block: 48kB (49152B)
//Number of shared memory banks: 32
//Amount of local memory per thread: 512KB
//Constant memory size: 64KB

__global__ void DeviceGroundIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map, cudaPitchedPtr dynamic_map) {
	int block_size = blockDim.x * blockDim.y;
	int grid_size = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
	int grid_thread_width = gridDim.x * blockDim.x;

	int thread_id_in_block = threadIdx.x + (threadIdx.y * blockDim.x);
	int id_block = blockIdx.x + (blockIdx.y * gridDim.x);
	int original_x = threadIdx.x + (blockIdx.x * blockDim.x);
	int original_y = threadIdx.y + (blockIdx.y * blockDim.y);
	int original_id = threadIdx.x + (id_block * block_size) + (threadIdx.y * blockDim.x);

	//thread spreading
	//int start_id = (original_id + (original_id % block_size) * block_size) % grid_size;
	//int x = (start_id % MAP_X_R);
	//int y = (start_id / (float)MAP_X_R);

	//original
	int start_id = original_id;
	int x = (start_id % grid_thread_width);
	int y = (start_id / (float)grid_thread_width);

	//int print_x = 5, print_y = 50;
	//if (x == print_x && y == print_y) printf("<x,y> start\n");

	if (destination.x >= MAP_X_R || destination.y >= MAP_Y_R) {	//return if destination is out of bounds
		((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -1;
		if (x == 0 && y == 0) printf("CUDA PRINT: destination out of bounds\n");
		return;
	}

	if (GetBoolMapValue(dynamic_map, destination.x, destination.y) == 0) {	//return if destination is unreachable
		((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -1;
		if (x == 0 && y == 0) printf("CUDA PRINT: destination unreachable\n");
		return;
	}

	if (x >= MAP_X_R || y >= MAP_Y_R || x < 0 || y < 0) {	//return if start tex is out of bounds 
		return;
	}

	if (GetBoolMapValue(dynamic_map, x, y) == 0) {	//return if start tex is in terrain
		((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -2;
		return;
	}

	//REGISTER ARRAY
	//we use about 60 of 255 register bytes, 195 bytes allocated to thread register array
	//sizeof(integer) = 2
	//sizeof(node) = 8 + 2 * sizeof(integer) = 12
	//195 / 12 = 16
	const int register_list_size = 16;
	node register_list[16];
	memset(register_list, -1, register_list_size * sizeof(node));

	//SHARED ARRAY
	//sizeof(integer) = 2
	//sizeof(node) = 8 + 2 * sizeof(integer) = 12
	//49152 / 12 = 4096
	//5461 / 32 = 128
	//int open_list_shared_it = 0;
	//const int open_list_shared_size = 128;
	//__shared__ node open_list_shared[open_list_shared_size * 32];
	//node* open_list_shared_pointer = &open_list_shared[open_list_shared_size * thread_id_in_block];

	//GLOBAL ARRAY
	int open_list_it = 0, closed_list_it = 0, open_list_size = 1000, closed_list_size = 1000;
	node* open_list = (node*)malloc(1000 * sizeof(node));
	node* closed_list = (node*)malloc(1000 * sizeof(node));

	if (open_list == NULL || closed_list == NULL) {
		printf("Device heap limit to low for lists (init)\n");
		return;
	}

	open_list[0] = { start_id, -1, 0, FloatDistance(x, y, destination.x, destination.y) };
	open_list_it = 1;

	int size_check_counter = 0;
	for (int step_iterator = 0; step_iterator < MAP_SIZE_R; ++step_iterator) {
		//find the next cell to expand
		float closest_distance_found = 99999;
		node closest_entry;
		int closest_coord_found = -1;
		node entry;
		int copy_amount = min(register_list_size, open_list_it);

		int it = 0;
		bool run_loop = true;
		while (it < open_list_it) {
			memcpy(register_list, &open_list[it], copy_amount * sizeof(node));	//copy 16 entries to register array

			for (int i = 0; i < register_list_size; ++i) {
				entry = register_list[i];

				//if (entry.pos == 0) {	//if end of open list
				//	run_loop = false;
				//	break;
				//}
				if (entry.pos != -1) {	//if valid node
					if (entry.est_dist_start_to_dest_via_pos </*=*/ closest_distance_found) {	//if closest node
						closest_distance_found = entry.est_dist_start_to_dest_via_pos;
						closest_coord_found = it + i;
						closest_entry = entry;
					}
				}
			}
			//memset(register_list, -1, register_list_size * sizeof(node));	//clear register array
			it += 16;
		}

		//for (int i = 0; i < open_list_it; ++i) {
		//	entry = open_list[i];
		//	if (entry.pos != -1) {
		//		if (entry.est_dist_start_to_dest_via_pos </*=*/ closest_distance_found) {
		//			closest_distance_found = entry.est_dist_start_to_dest_via_pos;
		//			closest_coord_found = i;
		//			closest_entry = entry;
		//		}
		//	}
		//}

		if (closest_coord_found == -1) {	//open list is empty and no path to the destination is found, RIP
			((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -3;
			return;
		}

		//add the expanded coord to the closed list
		closed_list[closed_list_it] = closest_entry;
		++closed_list_it;

		IntPoint2D pos = IDToPos(closest_entry.pos, grid_thread_width);

		//if (x == print_x && y == print_y) printf("<x,y> expanded (%d,%d)\n", pos.x, pos.y);

		if ((pos.x == destination.x) && (pos.y == destination.y)) {	//destination has been found! HYPE
			bool print = false;
			//if (x == print_x && y == print_y) {
			//	printf("<x,y> DESTINATION FOUND!\n");
			//	print = true;
			//}
			Backtrack(device_map, closed_list, closed_list_it - 1, grid_thread_width, print);
			free(open_list);
			free(closed_list);
			return;
		}

		//expand the size of the open and closed list if necessary
		if (size_check_counter == 30) {

			size_check_counter = 0;
			if ((open_list_size - open_list_it) < 200) {
				node* open_list_new = (node*)malloc(open_list_size * 2 * sizeof(node));
				memcpy(&open_list_new[0], &open_list[0], open_list_size * sizeof(node));
				open_list_size *= 2;
				free(open_list);
				open_list = open_list_new;

				if (open_list == NULL) {
					printf("Device heap limit to low for lists (expand)\n");
					return;
				}

				//if (x == print_x && y == print_y) printf("<x,y> EXPANDED size of open list, new max_size: %d\n", open_list_size);

			}
			if ((closed_list_size - closed_list_it) < 200) {
				node* closed_list_new = (node*)malloc(closed_list_size * 2 * sizeof(node));
				memcpy(&closed_list_new[0], &closed_list[0], closed_list_size * sizeof(node));
				closed_list_size *= 2;
				free(closed_list);
				closed_list = closed_list_new;

				if (closed_list == NULL) {
					printf("Device heap limit to low for lists (expand)\n");
					return;
				}

				//if (x == print_x && y == print_y) printf("<x,y> EXPANDED size of closed list, new max_size: %d\n", closed_list_size);

			}
		}

		//add the expanded nodes neighbours to the open list
		short_coord neighbour_coords[4];
		neighbour_coords[0] = { pos.x, pos.y - 1 };	//up
		neighbour_coords[1] = { pos.x - 1, pos.y };	//left
		neighbour_coords[2] = { pos.x + 1, pos.y };	//right
		neighbour_coords[3] = { pos.x, pos.y + 1 };	//down

		int new_open_list_entries = 0;
		for (int i = 0; i < 4; ++i) {
			int coord_global = PosToID({ neighbour_coords[i].x, neighbour_coords[i].y }, grid_thread_width);

			if (neighbour_coords[i].x <= MAP_X_R && neighbour_coords[i].y <= MAP_Y_R && neighbour_coords[i].x > 0 && neighbour_coords[i].y > 0) {	//coord in map
				if (GetBoolMapValue(dynamic_map, neighbour_coords[i].x, neighbour_coords[i].y) != 0) {	//coord not in terrain
					if (IDInList(coord_global, open_list, open_list_it) == -1 && IDInList(coord_global, closed_list, closed_list_it) == -1) {	//coord not already in open or closed list
						//if (x == print_x && y == print_y) printf("<x,y> added neighbour (%d) to open list\n", i);
						node new_list_entry = {
							coord_global,
							closed_list_it - 1,
							closest_entry.steps_from_start + /*0.8*/ 1,
							closest_entry.steps_from_start + /*0.8*/ 1 + FloatDistance(neighbour_coords[i].x, neighbour_coords[i].y, destination.x, destination.y)
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

__device__ void Backtrack(cudaPitchedPtr device_map, node* closed_list, int start_it, int width, bool print) {
	node curr = closed_list[start_it];

	IntPoint2D pos;
	for (int loop_count = 1; loop_count < MAP_SIZE_R + 1; ++loop_count) {
		pos = IDToPos(curr.pos, width);

		//if (print) printf("<backtrack> drawing %d to <%d,%d>\n", loop_count, pos.x, pos.y);

		((float*)(((char*)device_map.ptr) + pos.y * device_map.pitch))[pos.x] = loop_count;

		if (curr.backtrack_it == -1) return;
		curr = closed_list[curr.backtrack_it];
	}
}

__global__ void DeviceAirIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map) {
	int block_size = blockDim.x*blockDim.y;
	int grid_size = (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);

	int id_block = blockIdx.x + blockIdx.y * gridDim.x;
	int original_x = threadIdx.x + blockIdx.x * blockDim.x;
	int original_y = threadIdx.y + blockIdx.y * blockDim.y;
	int original_id = threadIdx.x + id_block * block_size + threadIdx.y * blockDim.x;

	//original
	int start_id = original_id;
	int x = original_x;
	int y = original_y;

	if (x >= MAP_X_R || y >= MAP_Y_R || x < 0 || y < 0) {	//return if start tex is out of bounds 
		return;
	}

	if (destination.x >= MAP_X_R || destination.y >= MAP_Y_R) {	//return if destination is out of bounds
		((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -1;
		if (x == 0 && y == 0) printf("CUDA PRINT: destination out of bounds\n");
		return;
	}

	((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = BlockDistance(x, y, destination.x, destination.y) + 1;
}

__global__ void DeviceUpdateDynamicMap(IntPoint2D top_left, IntPoint2D bottom_right, IntPoint2D center, float radius, int new_value, cudaPitchedPtr dynamic_map_device_pointer) {
	int local_x = threadIdx.x + blockIdx.x * blockDim.x;
	int local_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (local_x > bottom_right.x || local_y > bottom_right.y) {
		return;
	}

	int global_x = local_x + top_left.x;
	int global_y = local_y + top_left.y;

	FloatPoint2D center_r, corners[4];
	center_r = { ((float)center.x + (0.5 / GRID_DIVISION)) , ((float)center.y + (0.5 / GRID_DIVISION)) };
	corners[0] = { global_x, global_y };
	corners[1] = { global_x + 1, global_y };
	corners[2] = { global_x, global_y + 1 };
	corners[3] = { global_x + 1, global_y + 1 };
	
	float a, b, dist;
	for (int i = 0; i < 4; ++i) {
		a = powf(corners[0].x - center_r.x, 2);
		b = powf(corners[0].y - center_r.y, 2);
		dist = sqrtf(a + b) / GRID_DIVISION;

		if (dist < radius) {
			((bool*)(((char*)dynamic_map_device_pointer.ptr) + global_y * dynamic_map_device_pointer.pitch))[global_x] = new_value;
			return;
		}
	}
}