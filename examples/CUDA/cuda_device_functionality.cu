#pragma once

#include "../examples/CUDA/cuda_header.cuh"
#include "../examples/CUDA/cuda_device_utility.cu"

__global__ void DeviceRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map_ground, cudaPitchedPtr device_map_air) {
	extern __shared__ Entity unit_list_s[];
	memset(unit_list_s, 0, (nr_of_units * sizeof(Entity)) + ((32 * sizeof(Entity)) - ((nr_of_units * sizeof(Entity))) % 32));

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_block = threadIdx.x + threadIdx.y * blockDim.x;

	//move unit list to shared memory
	if (id_block < nr_of_units) unit_list_s[id_block] = device_unit_list_pointer[id_block];

	if (x >= MAP_X_R || y >= MAP_Y_R || x < 0 || y < 0) {	//return if start tex is out of bounds 
		return;
	}

	const int register_list_size = 32;
	Entity register_list[register_list_size];

	__syncthreads();

	float ground_charge = 0;
	float air_charge = 0;
	float largest_ground_charge = 0;
	float largest_air_charge = 0;

	int nr_of_slice_loops = ((float)((int)(((float)nr_of_units / (float)register_list_size) - 0.00001)) + 1);
	for (int slice = 0; slice < nr_of_slice_loops; ++slice) {
		for (int i = 0; i < register_list_size; ++i) {
			register_list[i] = unit_list_s[i + (slice * register_list_size)];
		}

		for (int i = 0; i < register_list_size; ++i) {
			if (register_list[i].id < 1 || register_list[i].id > 120) break;	//break if we reach end of list

			UnitInfoDevice unit = device_unit_lookup[register_list[i].id];
			float range_sub = fmaxf(unit.range, 3) + 2;
			float dist = (FloatDistance((int)register_list[i].pos.x, (int)register_list[i].pos.y, x, y) + 0.0001);

			if (register_list[i].enemy) {	//avoid enemies
				if (dist < range_sub) {
					float curr_ground_charge = ((range_sub / dist) * unit.can_attack_ground) + 50;
					float curr_air_charge = ((range_sub / dist) * unit.can_attack_air) + 50;
					if (curr_ground_charge > largest_ground_charge) largest_ground_charge = curr_ground_charge;
					if (curr_air_charge > largest_air_charge) largest_air_charge = curr_air_charge;
				}
			}
			else {	//avoid friendlies
				int res = 1 - (int)dist + (int)(unit.radius + 0.5);
				if (res > 0) {
					ground_charge += (res / 2.f) * !(unit.is_flying);
					air_charge += (res / 2.f) * unit.is_flying;
				}
			}
		}
	}

	//write ground_charge and air_charge to global memory in owned coord
	((float*)(((char*)device_map_ground.ptr) + y * device_map_ground.pitch))[x] = ground_charge + largest_ground_charge;
	((float*)(((char*)device_map_air.ptr) + y * device_map_ground.pitch))[x] = air_charge + largest_air_charge;
}

__global__ void DeviceLargeRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map_ground, cudaPitchedPtr device_map_air) {
	extern __shared__ Entity unit_list_s[];
	memset(unit_list_s, 0, (nr_of_units * sizeof(Entity)) + ((32 * sizeof(Entity)) - ((nr_of_units * sizeof(Entity))) % 32));

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_block = threadIdx.x + threadIdx.y * blockDim.x;
	int id_global = x + y * blockDim.x;

	//move unit list to shared memory
	if (id_block < nr_of_units) unit_list_s[id_block] = device_unit_list_pointer[id_block];

	if (x >= MAP_X_R || y >= MAP_Y_R || x < 0 || y < 0) {	//return if start tex is out of bounds 
		return;
	}

	const int register_list_size = 32;
	Entity register_list[register_list_size];

	__syncthreads();

	float max_value = 100;
	float falloff = 2;
	float range_sub = 14;
	float ground_charge = 0;
	float air_charge = 0;
	float largest_ground_charge = 0;
	float largest_air_charge = 0;

	int nr_of_slice_loops = ((float)((int)(((float)nr_of_units / (float)register_list_size) - 0.00001)) + 1);
	for (int slice = 0; slice < nr_of_slice_loops; ++slice) {
		for (int i = 0; i < register_list_size; ++i) {
			register_list[i] = unit_list_s[i + (slice * register_list_size)];
		}

		for (int i = 0; i < register_list_size; ++i) {
			if (register_list[i].id < 1 || register_list[i].id > 120) break;

			UnitInfoDevice unit = device_unit_lookup[register_list[i].id];
			float dist = (FloatDistance((int)register_list[i].pos.x, (int)register_list[i].pos.y, x, y) + 0.0001);

			if (register_list[i].enemy) {	//avoid enemies
				if (dist < range_sub) {
					float curr_ground_charge = ((max_value - (falloff * dist)) * unit.can_attack_ground);
					float curr_air_charge = ((max_value - (falloff * dist)) * unit.can_attack_air);
					if (curr_ground_charge > largest_ground_charge) largest_ground_charge = curr_ground_charge;
					if (curr_air_charge > largest_air_charge) largest_air_charge = curr_air_charge;
				}
			}
			else {	//avoid friendlies
				int res = 1 - (int)dist + (int)(unit.radius + 0.5);
				if (res > 0) {
					ground_charge += (res / 2.f) * !(unit.is_flying);
					air_charge += (res / 2.f) * unit.is_flying;
				}
			}
		}
	}

	//write ground_charge and air_charge to global memory in owned coord
	((float*)(((char*)device_map_ground.ptr) + y * device_map_ground.pitch))[x] = ground_charge + largest_ground_charge;
	((float*)(((char*)device_map_air.ptr) + y * device_map_ground.pitch))[x] = air_charge + largest_air_charge;
}

__global__ void DeviceAttractingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, int owner_type_id, cudaPitchedPtr device_map){
	extern __shared__ Entity unit_list_s[];
	memset(unit_list_s, 0, (nr_of_units * sizeof(Entity)) + ((32 * sizeof(Entity)) - ((nr_of_units * sizeof(Entity))) % 32));

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_block = threadIdx.x + threadIdx.y * blockDim.x;

	//move unit list to shared memory
	if (id_block < nr_of_units) unit_list_s[id_block] = device_unit_list_pointer[id_block];

	if (x >= MAP_X_R || y >= MAP_Y_R || x < 0 || y < 0) {	//return if start tex is out of bounds 
		return;
	}

	const int register_list_size = 32;
	Entity register_list[register_list_size];

	__syncthreads();

	UnitInfoDevice self_info = device_unit_lookup[owner_type_id];

	float tot_charge = 0;
	UnitInfoDevice other_info;
	Entity other_entity;

	int nr_of_slice_loops = ((float)((int)(((float)nr_of_units / (float)register_list_size) - 0.00001)) + 1);
	for (int slice = 0; slice < nr_of_slice_loops; ++slice) {
		for (int i = 0; i < register_list_size; ++i) {
			if (i + slice * register_list_size >= nr_of_units) break;
			register_list[i] = unit_list_s[i + (slice * register_list_size)];
		}

		for (int i = 0; i < register_list_size; ++i) {
			if (register_list[i].id < 1 || register_list[i].id > 120) break;

			other_info = device_unit_lookup[register_list[i].id];
			other_entity = register_list[i];

			float dist = (FloatDistance((int)other_entity.pos.x, (int)other_entity.pos.y, x, y) + 0.0001);
			bool self_can_attack_other = (other_info.is_flying && self_info.can_attack_air) || (!other_info.is_flying && self_info.can_attack_ground);

			if (other_entity.enemy) {	//attack enemy
				if (self_can_attack_other) {	//can attack unit
					if (self_info.range < 1.1) {	//self is melee
						if (dist < 10) {	//attack enemy
							tot_charge -= 10 / dist;
						}
					}
					else {	//self is ranged
						float range_diff = self_info.range - other_info.range;
						if (range_diff >= 0) {	//self more range than other
							if (dist <= (self_info.range + (self_info.radius /*+ 1*/))) {	//avoid area close to enemy
								tot_charge += 40 - (dist * 2);
							}
							else if (dist < self_info.range * 1.2 || dist < 10) {	//attack enemy
								tot_charge -= 10 - (dist);
							}
						}
						else {	//attack other with larger range than self
							tot_charge -= 10 / dist;
						}
					}
				}
			}
			else {	//avoid friend
				if (self_info.is_flying == other_info.is_flying) {
					int res = 1 - (int)dist + (int)(other_info.radius + 0.5);
					if (res > 0) {
						tot_charge += (res / 2.f);
					}
				}
			}
		}
	}

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
	integer start_id = (integer)original_id;
	integer x = (start_id % grid_thread_width);
	integer y = (start_id / (float)grid_thread_width);

	IntPoint2D debug_coord = {10, 49};
	bool debug = false;

	//if (debug && debug_coord.x == x && debug_coord.y == y) printf(" \n");

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

	//CREATE REGISTER ARRAY
	const int register_list_size = 32;
	node register_list[register_list_size];

	//CREATE SHARED ARRAY
	const int mem_per_SM = 47616;	//normal: 49152, 1536 (384 per block) reserved for block intra-communication
	const int mem_per_block = mem_per_SM / 4;
	const int nodes_per_block = mem_per_block / sizeof(node);	// = 992 (should be 1024) 
	const int nodes_per_thread = nodes_per_block / 32;	// = 31 (should be 32)
	int thread_array_start_index = nodes_per_thread * thread_id_in_block;
	__shared__ node shared_list[nodes_per_block];
	node* __restrict__ shared_list_thread_pointer = &shared_list[thread_array_start_index];
	int /*open*/   shared_open_it = 0,
		/*closed*/ shared_closed_it = nodes_per_thread - 1;

	const int array_size_diff = (int)(((float)nodes_per_thread / (float)register_list_size) + 0.99);

	//CREATE GLOBAL ARRAY
	int open_list_it = 0, closed_list_it = 0, open_list_size = 2000, closed_list_size = 1400;
	node* __restrict__ open_list = (node*)malloc(2000 * sizeof(node));
	node* __restrict__ closed_list = (node*)malloc(1400 * sizeof(node));

	if (open_list == NULL || closed_list == NULL) {
		printf("Device heap limit to low for lists\n");
		free(open_list);
		free(closed_list);
		return;
	}

	__shared__ bool block_check;
	bool solution_found = false;

	open_list[0] = { start_id, -1, 0, FloatDistance(x, y, destination.x, destination.y) };
	open_list_it = 1;

	if (debug && debug_coord.x == x && debug_coord.y == y) printf("start \n");

	//-----------------------------

	const int max_step_loops = 2800;
	const int max_open_list_size = max_step_loops * 3 + 1;
	const int max_closed_list_size = max_step_loops + 1;
	for (int step_iterator = 0; step_iterator < max_step_loops; ++step_iterator) {
		//~1400 is the nr of iterations it takes for the longest path to be calculated in the complex experiment map
		
		block_check = false;
		if (shared_closed_it - shared_open_it < 6) block_check = true;	//check if 1 or more threads need to move data from shared to global
		if (block_check) {	//transfer shared data to global memory
			if (debug && debug_coord.x == x && debug_coord.y == y) printf("transfering data from shared to global mem\n");

			//GLOBAL READ/WRITE
			memcpy(&open_list[open_list_it], &shared_list_thread_pointer[0], shared_open_it * sizeof(node));
			open_list_it += shared_open_it;
			for (int i = 0; i < nodes_per_thread - 1; ++i) {
				if (i >= (nodes_per_thread - shared_closed_it - 1)) break;
				closed_list[closed_list_it + i] = shared_list_thread_pointer[nodes_per_thread - 1 - i];
			}
			closed_list_it += (nodes_per_thread - shared_closed_it - 1);
			memset(&shared_list_thread_pointer[0], 0, nodes_per_thread * sizeof(node));	//reset shared array
			shared_open_it = 0;
			shared_closed_it = nodes_per_thread - 1;
		}

		//-----------------------------

		//find the next cell to expand
		bool closest_node_in_shared_mem;
		float closest_distance_found = 99999;
		node closest_entry;
		int closest_coord_found = -1;

		//search in shared open list
		for (int slice = 0; slice < array_size_diff; ++slice) {
			for (int i = 0; i < register_list_size; ++i) {
				if ((i + slice * register_list_size) >= shared_open_it) {
					register_list[i].pos = -1;
					register_list[i].backtrack_it = -1;
					register_list[i].steps_from_start = -1;
					register_list[i].est_dist_start_to_dest_via_pos = -1.f;
				}
				else register_list[i] = shared_list_thread_pointer[i + (slice * register_list_size)];
			}

			for (int i = 0; i < register_list_size; ++i) {	//loop over register array
				if ((i + slice * register_list_size) >= shared_open_it) /*break*/ goto search_shared_open_list_end;
				if (register_list[i].pos > 0) {	//if valid node
					if (register_list[i].est_dist_start_to_dest_via_pos </*=*/ closest_distance_found) {	//if closest node
						closest_distance_found = register_list[i].est_dist_start_to_dest_via_pos;
						closest_node_in_shared_mem = true;
						closest_coord_found = i + slice * register_list_size;	//maybe...
						closest_entry = register_list[i];
					}
				}
			}
		}
		search_shared_open_list_end:

		//search in global open list
		int it = 0;
		while (it < open_list_it) {
			//GLOBAL READ/WRITE
			for (int i = 0; i < register_list_size; ++i) {
				if (i >= (open_list_it - it + i)) {
					register_list[i].pos = -2;
					register_list[i].backtrack_it = -2;
					register_list[i].steps_from_start = -2;
					register_list[i].est_dist_start_to_dest_via_pos = -2.f;
				}
				else register_list[i] = open_list[it + i];
			}

			for (int i = 0; i < register_list_size; ++i) {
				if (register_list[i].pos == -258 || register_list[i].pos == -2) /*break*/ goto search_global_open_list_end;
				if (register_list[i].pos > 0 ) {	//if valid node
					if (register_list[i].est_dist_start_to_dest_via_pos </*=*/ closest_distance_found) {	//if closest node
						closest_distance_found = register_list[i].est_dist_start_to_dest_via_pos;
						closest_node_in_shared_mem = false;
						closest_coord_found = it + i;
						closest_entry = register_list[i];
					}
				}
			}
			it += register_list_size;
		}
		search_global_open_list_end:

		//-----------------------------

		if (closest_coord_found == -1) {	//open list is empty and no path to the destination is found, RIP
			((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -3;
			memset(open_list, 0, open_list_it * sizeof(node));
			memset(closed_list, 0, open_list_it * sizeof(node));
			free(open_list);
			free(closed_list);
			return;
		}

		//-----------------------------

		//add the expanded coord to the closed list
		//SHARED READ/WRITE
		shared_list_thread_pointer[shared_closed_it] = closest_entry;
		--shared_closed_it;

		IntPoint2D pos = IDToPos(closest_entry.pos, grid_thread_width);

		if (debug && debug_coord.x == x && debug_coord.y == y) printf("expanded (%d,%d) on step %d \n", pos.x, pos.y, step_iterator);

		//-----------------------------

		if ((pos.x == destination.x) && (pos.y == destination.y)) {	//destination has been found! HYPE
			if (debug && debug_coord.x == x && debug_coord.y == y) printf("destination found! \n");
			solution_found = true;
			break;
		}

		//-----------------------------

		//expand the size of the open and closed list if necessary
		if (step_iterator%30 == 0) {
			block_check = false;
			if ((open_list_size - (open_list_it + shared_open_it)) < 200) block_check = true;
			if (block_check) {	//expand open list
				//GLOBAL READ/WRITE
				node* __restrict__ open_list_new = (node*)malloc(open_list_size * 2 * sizeof(node));
				memcpy(&open_list_new[0], &open_list[0], open_list_size * sizeof(node));
				open_list_size *= 2;
				free(open_list);
				open_list = open_list_new;

				if (debug && debug_coord.x == x && debug_coord.y == y) printf("expanding open list, new size: %d \n", open_list_size);

				if (open_list == NULL) {
					printf("Device heap limit to low for lists (expand)\n");
					memset(open_list, 0, open_list_it * sizeof(node));
					memset(closed_list, 0, open_list_it * sizeof(node));
					free(open_list);
					free(closed_list);
					return;
				}
			}

			block_check = false;
			if ((closed_list_size - (closed_list_it + (nodes_per_thread - 1 - shared_closed_it))) < 200) block_check = true;
			if (block_check) {	//expand closed list
				//GLOBAL READ/WRITE
				node* __restrict__ closed_list_new = (node*)malloc(closed_list_size * 2 * sizeof(node));
				memcpy(&closed_list_new[0], &closed_list[0], closed_list_size * sizeof(node));
				closed_list_size *= 2;
				free(closed_list);
				closed_list = closed_list_new;

				if (debug && debug_coord.x == x && debug_coord.y == y) printf("expanding closed list \n");

				if (closed_list == NULL) {
					printf("Device heap limit to low for lists (expand)\n");
					memset(open_list, 0, open_list_it * sizeof(node));
					memset(closed_list, 0, open_list_it * sizeof(node));
					free(open_list);
					free(closed_list);
					return;
				}
			}
		}

		//-----------------------------

		//add the expanded nodes neighbours to the open list
		IntPoint2D neighbour_coords[4];
		neighbour_coords[0] = { pos.x, pos.y - 1 };	//up
		neighbour_coords[1] = { pos.x + 1, pos.y };	//right
		neighbour_coords[2] = { pos.x, pos.y + 1 };	//down
		neighbour_coords[3] = { pos.x - 1, pos.y };	//left

		integer neighbour_coord_global[4];
		neighbour_coord_global[0] = PosToID({ neighbour_coords[0].x, neighbour_coords[0].y }, grid_thread_width);
		neighbour_coord_global[1] = PosToID({ neighbour_coords[1].x, neighbour_coords[1].y }, grid_thread_width);
		neighbour_coord_global[2] = PosToID({ neighbour_coords[2].x, neighbour_coords[2].y }, grid_thread_width);
		neighbour_coord_global[3] = PosToID({ neighbour_coords[3].x, neighbour_coords[3].y }, grid_thread_width);

		bool neighbour_coord_validity[4];
		neighbour_coord_validity[0] = true;
		neighbour_coord_validity[1] = true;
		neighbour_coord_validity[2] = true;
		neighbour_coord_validity[3] = true;

		if (debug && debug_coord.x == x && debug_coord.y == y) printf("checking neighbours to (%d,%d)<%d>:\n   (%d,%d)<%d>\n   (%d,%d)<%d>\n   (%d,%d)<%d>\n   (%d,%d)<%d> \n",
			pos.x, pos.y, closest_entry.pos, 
			neighbour_coords[0].x, neighbour_coords[0].y, neighbour_coord_global[0], 
			neighbour_coords[1].x, neighbour_coords[1].y, neighbour_coord_global[1],
			neighbour_coords[2].x, neighbour_coords[2].y, neighbour_coord_global[2],
			neighbour_coords[3].x, neighbour_coords[3].y, neighbour_coord_global[3]);

		//-----------------------------

		//use register list for terrain map values
		register_list[0].pos = GetBoolMapValue(dynamic_map, neighbour_coords[0].x, neighbour_coords[0].y);
		register_list[1].pos = GetBoolMapValue(dynamic_map, neighbour_coords[1].x, neighbour_coords[1].y);
		register_list[2].pos = GetBoolMapValue(dynamic_map, neighbour_coords[2].x, neighbour_coords[2].y);
		register_list[3].pos = GetBoolMapValue(dynamic_map, neighbour_coords[3].x, neighbour_coords[3].y);

		//Check the neighbours for invalid positions
		for (int i = 0; i < 4; ++i) {
			if (!(neighbour_coords[i].x <= MAP_X_R) || !(neighbour_coords[i].y <= MAP_Y_R) || !(neighbour_coords[i].x > 0) || !(neighbour_coords[i].y > 0)) {	//coord not in map (FIX UGLINESS!)
				if (debug && debug_coord.x == x && debug_coord.y == y) printf("   neighbour %d failed map bound check \n", i);
				neighbour_coord_validity[i] = false;
			}

			//GLOBAL READ/WRITE
			if (neighbour_coord_validity[i] && !(register_list[i].pos != 0)) {	//coord in terrain
				if (debug && debug_coord.x == x && debug_coord.y == y) printf("   neighbour %d failed terrain check \n", i);
				neighbour_coord_validity[i] = false;
			}
		}

		//-----------------------------

		//Search for id in shared closed_list
		//SHARED READ/WRITE
		for (int slice = 0; slice < array_size_diff; ++slice) {
			for (int i = 0; i < register_list_size; ++i) {
				if ((i + slice * register_list_size) >= (nodes_per_thread - shared_closed_it + 1)) {
					register_list[i].pos = -1;
					register_list[i].backtrack_it = -1;
					register_list[i].steps_from_start = -1;
					register_list[i].est_dist_start_to_dest_via_pos = -1.f;
				}
				else register_list[i] = shared_list_thread_pointer[(shared_closed_it + 1 + i) + (slice * register_list_size)];
			}

			for (int i = 0; i < register_list_size; ++i) {	//loop over register array
				if ((i + register_list_size * slice) + shared_closed_it >= nodes_per_thread - 1) /*break*/ goto shared_closed_list_end;
				if (register_list[i].pos > 0) {	//if valid list node
					for (int j = 0; j < 4; ++j) {	//loop over the 4 neighbours
						if (neighbour_coord_validity[j]) {	//if neighbour is valid	(remove?)
							if (neighbour_coord_global[j] == register_list[i].pos) {	//node already in closed list
								if (debug && debug_coord.x == x && debug_coord.y == y) printf("   neighbour %d failed (shared) closed list check \n", j);
								neighbour_coord_validity[j] = false;
							}
						}
					}
				}
			}
		}
		shared_closed_list_end:

		//Search for id in shared open_list
		//SHARED READ/WRITE
		for (int slice = 0; slice < array_size_diff; ++slice) {
			for (int i = 0; i < register_list_size; ++i) {
				if ((i + slice * register_list_size) >= shared_open_it) {
					register_list[i].pos = -1;
					register_list[i].backtrack_it = -1;
					register_list[i].steps_from_start = -1;
					register_list[i].est_dist_start_to_dest_via_pos = -1.f;
				}
				else register_list[i] = shared_list_thread_pointer[i + (slice * register_list_size)];
			}

			//for (int i = 0; i < (nodes_per_thread - 1); ++i) {	//loop over register array
			for (int i = 0; i < register_list_size; ++i) {	//loop over register array
				if ((i + register_list_size * slice) > (shared_open_it - 1)) /*break*/ goto shared_open_list_end;	//(HINDERS REGISTER ARRAY)
				if (register_list[i].pos > 0) {	//if valid list node
					for (int j = 0; j < 4; ++j) {	//loop over the 4 neighbours
						if (neighbour_coord_validity[j]) {	//if neighbour is valid	(remove?)
							if (neighbour_coord_global[j] == register_list[i].pos) {	//node already in open list
								if (debug && debug_coord.x == x && debug_coord.y == y) printf("   neighbour %d failed (shared) closed list check \n", j);
								neighbour_coord_validity[j] = false;
							}
						}
					}
				}
			}
		}
		shared_open_list_end:

		//Search for id in global closed_list
		it = 0;
		while (it < open_list_it) {
			if ((neighbour_coord_validity[0] + neighbour_coord_validity[1] + neighbour_coord_validity[2] + neighbour_coord_validity[3]) < 1) break;
			//GLOBAL READ/WRITE
			for (int i = 0; i < register_list_size; ++i) {
				if (i >= (closed_list_it - it + i)) {
					register_list[i].pos = -2;
					register_list[i].backtrack_it = -2;
					register_list[i].steps_from_start = -2;
					register_list[i].est_dist_start_to_dest_via_pos = -2.f;
				}
				else register_list[i] = closed_list[it + i];
			}

			for (int i = 0; i < register_list_size; ++i) {	//loop over register array
				if (register_list[i].pos == -258 || register_list[i].pos == -2) /*break*/ goto global_closed_list_end;
				if (register_list[i].pos > 0) {	//if valid list node
					for (int j = 0; j < 4; ++j) {	//loop over the 4 neighbours
						if (neighbour_coord_validity[j]) {	//if neighbour is valid	(remove?)
							if (neighbour_coord_global[j] == register_list[i].pos) {	//node already in closed list
								if (debug && debug_coord.x == x && debug_coord.y == y) printf("   neighbour %d failed (global) closed list check \n", j);
								neighbour_coord_validity[j] = false;
							}
						}
					}
				}
			}
			it += register_list_size;
		}
		global_closed_list_end:

		//Search for id in global open_list
		it = 0;
		while (it < open_list_it) {
			if ((neighbour_coord_validity[0] + neighbour_coord_validity[1] + neighbour_coord_validity[2] + neighbour_coord_validity[3]) < 1) break;
			//GLOBAL READ/WRITE
			for (int i = 0; i < register_list_size; ++i) {
				if (i >= (open_list_it - it + i)) {
					register_list[i].pos = -2;
					register_list[i].backtrack_it = -2;
					register_list[i].steps_from_start = -2;
					register_list[i].est_dist_start_to_dest_via_pos = -2.f;
				}
				else register_list[i] = open_list[it + i];
			}

			for (int i = 0; i < register_list_size; ++i) {	//loop over register array
				if (register_list[i].pos == -258 || register_list[i].pos == -2) /*break*/ goto global_open_list_end;
				if (register_list[i].pos > 0) {	//if valid list node
					for (int j = 0; j < 4; ++j) {	//loop over the 4 neighbours
						if (neighbour_coord_validity[j]) {	//if neighbour is valid	(remove?)
							if (neighbour_coord_global[j] == register_list[i].pos) {	//node already in open list
								if (debug && debug_coord.x == x && debug_coord.y == y) printf("   neighbour %d failed (global) open list check \n", j);
								neighbour_coord_validity[j] = false;
							}
						}
					}
				}
			}
			it += register_list_size;
		}
		global_open_list_end:

		//-----------------------------

		if (debug && debug_coord.x == x && debug_coord.y == y) printf("   nodes to add: ");

		//Add the valid neighbours to the open list
		int new_open_list_entries = 0;
		struct { node node; bool valid = false; } nodes_to_add[4];
		for (int i = 0; i < 4; ++i) {	//loop over the 4 neighbours
			if (neighbour_coord_validity[i]) {	//the neighbour is valid
				node new_list_entry = {
					neighbour_coord_global[i],
					closed_list_it - 1 + (nodes_per_thread - 1 - shared_closed_it),
					closest_entry.steps_from_start + 1,
					closest_entry.steps_from_start + 1 + FloatDistance(neighbour_coords[i].x, neighbour_coords[i].y, destination.x, destination.y)
				};
				nodes_to_add[i].node = new_list_entry;
				nodes_to_add[i].valid = true;
				++new_open_list_entries;

				if (debug && debug_coord.x == x && debug_coord.y == y) printf("(%d, %d, %d, %f) ", new_list_entry.pos, 
					new_list_entry.backtrack_it, new_list_entry.steps_from_start, new_list_entry.est_dist_start_to_dest_via_pos);
			}
		}

		if (debug && debug_coord.x == x && debug_coord.y == y) printf("\n   %d valid neighbours to (%d,%d) \n", new_open_list_entries, pos.x, pos.y);

		//-----------------------------

		//SHARED READ/WRITE
		for (int i = 0; i < 4; ++i) {
			if (nodes_to_add[i].valid) {
				shared_list_thread_pointer[shared_open_it] = nodes_to_add[i].node;
				shared_open_it++;
			}
		}
		
		if (!closest_node_in_shared_mem) {
			//GLOBAL READ/WRITE
			open_list[closest_coord_found].pos = -1;	//mark expanded node as invalid in the open list (global)
		}
		else {
			//SHARED READ/WRITE
			shared_list_thread_pointer[closest_coord_found].pos = -1;	//mark expanded node as invalid in the open list (shared)
		}

		if (debug && debug_coord.x == x && debug_coord.y == y) printf("open_list_it: %d\nclosed_list_it: %d\nshared_open_it: %d\nshared_closed_it: %d\nsize_check_counter: ?\n",
			open_list_it, closed_list_it, shared_open_it, shared_closed_it/*, size_check_counter*/);

	}	//END OF A* LOOP

	__syncthreads();	//unnecessary?

	if (!solution_found) {
		//GLOBAL READ/WRITE
		if(((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] == 0)	//shit solution, but it works...
			((float*)(((char*)device_map.ptr) + y * device_map.pitch))[x] = -2;
		memset(open_list, 0, open_list_it * sizeof(node));
		memset(closed_list, 0, open_list_it * sizeof(node));
		free(open_list);
		free(closed_list);
		return;
	}

	__syncthreads();

	if (debug && debug_coord.x == x && debug_coord.y == y) printf("transfering data from shared to global mem\n");

	//GLOBAL READ/WRITE
	for (int i = 0; i < nodes_per_thread - 1; ++i) {
		if (i >= (nodes_per_thread - shared_closed_it - 1)) break;
		closed_list[closed_list_it + i] = shared_list_thread_pointer[nodes_per_thread - 1 - i];
	}
	closed_list_it += (nodes_per_thread - shared_closed_it - 1);
	shared_open_it = 0;
	shared_closed_it = nodes_per_thread - 1;

	//--------

	node curr = closed_list[closed_list_it - 1];
	IntPoint2D pos;
	for (int loop_count = 1; loop_count < MAP_SIZE_R + 1; ++loop_count) {
		pos = IDToPos(curr.pos, grid_thread_width);

		if (debug && debug_coord.x == x && debug_coord.y == y) printf("backtrack: printing %d to (%d,%d), node <%d, %d, %d, %f>\n", loop_count, pos.x, pos.y,
			curr.pos, curr.backtrack_it, curr.steps_from_start, curr.est_dist_start_to_dest_via_pos);

		if (curr.steps_from_start < 4) ((float*)(((char*)device_map.ptr) + pos.y * device_map.pitch))[pos.x] = loop_count;

		if (curr.backtrack_it == -1) {
			break;
		}
		curr = closed_list[curr.backtrack_it];
	}

	if (debug && debug_coord.x == x && debug_coord.y == y) printf("backtracking done\n");

	//--------

	memset(open_list, 0, open_list_it * sizeof(node));
	memset(closed_list, 0, open_list_it * sizeof(node));
	free(open_list);
	free(closed_list);
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