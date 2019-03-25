#pragma once

#include "../examples/CUDA/cuda_header.cuh"

#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <windows.h>

#include "../examples/CUDA/cuda_device_functionality.cu"
#include "../examples/CUDA/cuda_device_tests.cu"

__host__ CUDA::CUDA() {
}

__host__ CUDA::~CUDA() {

}

__host__ void CUDA::PrintGenInfo() {
	SYSTEM_INFO siSysInfo;
	GetSystemInfo(&siSysInfo);

	int deviceCount = 0, setDevice = 0;
	Check(cudaGetDeviceCount(&deviceCount));
	Check(cudaSetDevice(setDevice));

	int Rv, Dv;
	cudaRuntimeGetVersion(&Rv);
	cudaDriverGetVersion(&Dv);

	std::cout <<
		"MAP_DIM: <" << MAP_X << ", " << MAP_Y << "> " << std::endl <<
		"GRID_DIVISION: " << GRID_DIVISION << std::endl <<
		std::endl << "Available devices: " << deviceCount << std::endl <<
		"Device ID in use: <" << setDevice << ">" << std::endl <<
		"Runtime API version: " << Rv << std::endl <<
		"Driver API version: " << Dv << std::endl;

	std::cout << std::endl;
}

__host__ void CUDA::InitializeCUDA(const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions, float ground_PF[][MAP_Y_R][1], float air_PF[][MAP_Y_R][1]){
	std::cout << "Initializing CUDA object" << std::endl;
	
	size_t size;
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	std::cout << "CUDA base heap size: " << size << std::endl;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, size*128);
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	std::cout << "CUDA new heap size: " << size << std::endl;

	this->observation = observations;
	this->debug = debug;
	this->actions = actions;
	this->actions_feature_layer = actions_feature_layer;

	dim_block_high = { 32, 32, 1 };
	unsigned int x1 = (unsigned int)(ceil((MAP_X_R - 1) / (float)dim_block_high.x) + 0.5);
	unsigned int y1 = (unsigned int)(ceil((MAP_Y_R - 1) / (float)dim_block_high.y) + 0.5);
	dim_grid_high = { x1, y1, 1 };
	threads_in_grid_high = (dim_block_high.x * dim_block_high.y) * (dim_grid_high.x * dim_grid_high.y);

	dim_block_low = { 8, 4, 1 };
	unsigned int x2 = (unsigned int)(ceil((MAP_X_R - 1) / (float)dim_block_low.x) + 0.5);
	unsigned int y2 = (unsigned int)(ceil((MAP_Y_R - 1) / (float)dim_block_low.y) + 0.5);
	dim_grid_low = { x2, y2, 1 };
	threads_in_grid_low = (dim_block_low.x * dim_block_low.y) * (dim_grid_low.x * dim_grid_low.y);

	next_id = 0;
	unit_list_max_length = 800;

	this->ground_PF = (float*)ground_PF;
	this->air_PF = (float*)air_PF;

	PopErrorsCheck("CUDA Initialization base");

	//analysis
	PrintGenInfo();

	//device_malloc
	//std::string file = "unitInfo.txt";
	//CreateUnitLookupOnHost(file);

	//AllocateDeviceMemory();

	PopErrorsCheck("CUDA Initialization malloc");

	//IMGeneration(IntPoint2D{ 18, 29 }, false);

	//Check(cudaPeekAtLastError(), "init check 7", true);
}

__host__ void CUDA::DeviceTransfer(bool dynamic_terrain[][MAP_Y_R][1]) {
	//device_transfer
	TransferDynamicMapToDevice(dynamic_terrain);
	TransferUnitLookupToDevice();
	TransferUnitsToDevice();

	Check(cudaPeekAtLastError(), "init check 5");
}

__host__ void CUDA::Tests(float ground_avoidance_PF[][MAP_Y_R][1], float air_avoidance_PF[][MAP_Y_R][1]) {
	//tests
	//TestLookupTable();
	Check(cudaPeekAtLastError(), "init check 6a");

	//Test3DArrayUsage();
	Check(cudaPeekAtLastError(), "init check 6b");

	RepellingPFGeneration(ground_avoidance_PF, air_avoidance_PF);
	Check(cudaPeekAtLastError(), "init check 6c");
}

__host__ const sc2::ObservationInterface* CUDA::GetObservation(){
	return observation;
}

__host__ sc2::DebugInterface* CUDA::GetDebug(){
	return debug;
}

__host__ sc2::ActionInterface* CUDA::GetAction(){
	return actions;
}

__host__ sc2::ActionFeatureLayerInterface* CUDA::GetActionFeature(){
	return actions_feature_layer;
}

__host__ void CUDA::CreateUnitLookupOnHost(std::string file){
	sc2::UnitTypes types = observation->GetUnitTypeData();

	int host_iterator = 0;
	for (int i = 1; i < types.size(); ++i) {
		sc2::UnitTypeData data = types.at(i);
		//Check for units that are not considered valid.
		std::string str = sc2::UnitTypeToName(data.unit_type_id.ToType());
		if (str.find("PROTOSS") != std::string::npos || str.find("TERRAN") != std::string::npos || str.find("ZERG") != std::string::npos) {
			std::vector<sc2::Attribute> att = data.attributes;
			if (data.weapons.size() == 0 && std::find(att.begin(), att.end(), sc2::Attribute::Structure) != att.end()) continue;
			host_unit_info.push_back(UnitInfo());

			std::vector<sc2::Weapon> weapons = data.weapons;
			int longest_weapon_range;
			longest_weapon_range = 0;
			for (auto& const weapon : weapons) {
				if (weapon.range > longest_weapon_range)
					longest_weapon_range = weapon.range;

				if (weapon.type == sc2::Weapon::TargetType::Ground)
					host_unit_info.at(host_iterator).can_attack_air = false;
				else if (weapon.type == sc2::Weapon::TargetType::Air)
					host_unit_info.at(host_iterator).can_attack_ground = false;
			}
			host_unit_info.at(host_iterator).range = longest_weapon_range;
			host_unit_info.at(host_iterator).device_id = host_iterator;
			host_unit_info.at(host_iterator).id = data.unit_type_id;


			if (std::find(att.begin(), att.end(), sc2::Attribute::Hover) != att.end())
				host_unit_info.at(host_iterator).is_flying = true;

			host_unit_transform.insert({ { data.unit_type_id, host_iterator } });

			/*
			//failed attempt att doing it the easy way... :(

			debug->DebugCreateUnit(data.unit_type_id, sc2::Point2D(0,0));
			debug->SendDebug();
			actions->SendActions();
			actions_feature_layer->SendActions();
			sc2::Units u = observation->GetUnits();
			for (int i = 0; i < u.size() + 1; ++i) {
				if (i == u.size()) {
					std::cout << "FAILED to get radius of unit: ''" << data.unit_type_id << "''" << std::endl;
					for (int j = 0; j < u.size(); ++j) debug->DebugKillUnit(u.at(j));
					break;
				}
				if (u.at(i)->unit_type == data.unit_type_id) {
					host_unit_info.at(host_iterator).radius = u.at(i)->radius;
					debug->DebugKillUnit(u.at(i));
					break;
				}
			}
			*/

			++host_iterator;
		}
	}
	PrintUnitInfoToFile(file);
	std::cout << "Created unit data table on host. Nr of elements: " << host_iterator << ". " << std::endl;

	for (int i = 0; i < host_unit_info.size(); ++i) {
		device_unit_lookup_on_host.push_back({ host_unit_info.at(i).range, host_unit_info.at(i).radius,
			host_unit_info.at(i).is_flying, host_unit_info.at(i).can_attack_air,
			host_unit_info.at(i).can_attack_ground });
	}
	std::cout << std::endl;
	std::cout << "device_unit_lookup array filled on host" << std::endl;
}

__host__ void CUDA::TransferStaticMapToHost(){}

__host__ void CUDA::TransferUnitLookupToDevice(){
	Check(cudaMemcpy(unit_lookup_device_pointer, device_unit_lookup_on_host.data(), device_unit_lookup_on_host.size() * sizeof(UnitInfoDevice), cudaMemcpyHostToDevice), "lookup_memcpy");
	Check(cudaMemcpyToSymbol(device_unit_lookup, &unit_lookup_device_pointer, sizeof(UnitInfoDevice*)), "lookup_symbol_memcpy");
}

__host__ void CUDA::AllocateDeviceMemory(){
	//cudaMalloc3D(&static_map_device_pointer, cudaExtent{ MAP_X_R * sizeof(bool), MAP_Y_R, 1 });	//static map
	cudaMalloc3D(&dynamic_map_device_pointer, cudaExtent{ MAP_X_R * sizeof(bool), MAP_Y_R, 1 });	//dynamic map
	Check(cudaMalloc(&unit_lookup_device_pointer, device_unit_lookup_on_host.size() * sizeof(UnitInfoDevice)), "lookup alloc");	//lookup table (global on device)
	cudaMalloc((void**)&device_unit_list_pointer, unit_list_max_length * sizeof(Entity));	//unit list (might extend size during runtime)
	cudaMalloc3D(&repelling_pf_ground_map_pointer, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });	//repelling on ground
	cudaMalloc3D(&repelling_pf_air_map_pointer, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });	//repelling in air
	Check(cudaMalloc((void**)&global_memory_im_list_storage, 256000000 * sizeof(list_double_entry)), "big AF allocation");	//big AF list for A* open/closed list

	Check(cudaPeekAtLastError(), "cuda allocation peek");
}

__host__ void CUDA::TransferUnitsToDevice() {

	if (host_unit_list.size() > unit_list_max_length) {
		std::cout << "WARNING: too many units! Increase allocation size, overflow discarded" << std::endl;
	}

	Check(cudaMemcpyAsync(device_unit_list_pointer, host_unit_list.data(), 
		std::min((int)host_unit_list.size(), unit_list_max_length) * sizeof(Entity),
		cudaMemcpyHostToDevice),
		"TransferUnitsToDevice");
}

__host__ void CUDA::TransferDynamicMapToDevice(bool dynamic_terrain[][MAP_Y_R][1]) {
	cudaMemcpy3DParms par = { 0 };
	par.srcPtr = make_cudaPitchedPtr((void*)dynamic_terrain, MAP_X_R * sizeof(bool), MAP_X_R, MAP_Y_R);
	par.dstPtr.ptr = dynamic_map_device_pointer.ptr;
	par.dstPtr.pitch = dynamic_map_device_pointer.pitch;
	par.dstPtr.xsize = MAP_X_R;
	par.dstPtr.ysize = MAP_Y_R;
	par.extent.width = MAP_X_R * sizeof(bool);
	par.extent.height = MAP_Y_R;
	par.extent.depth = 1;
	par.kind = cudaMemcpyHostToDevice;

	Check(cudaMemcpy3DAsync(&par), "Dynamic map transfer");
}

__host__ int CUDA::QueueDeviceJob(int owner_id, float* map){
	//check for available slots in vector
	int storage_found = -1;
	for (int i = 0; i < PF_mem.size(); ++i) {
		if (PF_mem.at(i).status == DeviceMemoryStatus::EMPTY) {
			storage_found = i;
			cudaEventDestroy(PF_mem.at(i).begin);	//destroy previous events
			cudaEventDestroy(PF_mem.at(i).done);
			break; 
		}
	}
	if (storage_found == -1) {
		storage_found = PF_mem.size();
		AttractingFieldMemory mem;
		cudaMalloc3D(&mem.device_map_ptr, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });
		PF_mem.push_back(mem);
	}

	Check(cudaEventCreateWithFlags(&PF_mem.at(storage_found).begin, cudaEventDisableTiming), "PF event begin create");
	Check(cudaEventCreateWithFlags(&PF_mem.at(storage_found).done, cudaEventDisableTiming), "PF event done create");
	PF_mem.at(storage_found) = { owner_id, next_id, DeviceMemoryStatus::OCCUPIED, false, PF_mem.at(storage_found).begin, PF_mem.at(storage_found).done, map, PF_mem.at(storage_found).device_map_ptr };
	PF_queue.push(next_id);
	next_id++;
	
	PopErrorsCheck();

	return next_id - 1;
}

__host__ int CUDA::QueueDeviceJob(IntPoint2D destination, bool air_path, float* map){
	//check for available slots in vector
	int storage_found = -1;
	for (int i = 0; i < IM_mem.size(); ++i) {
		if (IM_mem.at(i).status == DeviceMemoryStatus::EMPTY) {
			storage_found = i;
			Check(cudaEventDestroy(IM_mem.at(i).begin), "event begin reset");	//destroy previous events
			Check(cudaEventDestroy(IM_mem.at(i).done), "event done reset");
			break;
		}
	}
	if (storage_found == -1) {
		storage_found = IM_mem.size();
		InfluenceMapMemory mem;
		cudaMalloc3D(&mem.device_map_ptr, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });	//alloc new device mem
		IM_mem.push_back(mem);
	}

	Check(cudaEventCreateWithFlags(&IM_mem.at(storage_found).begin, cudaEventDisableTiming), "IM event begin create");
	Check(cudaEventCreateWithFlags(&IM_mem.at(storage_found).done, cudaEventDisableTiming), "IM event done create");
	IM_mem.at(storage_found) = { destination, air_path, next_id, DeviceMemoryStatus::OCCUPIED, false, IM_mem.at(storage_found).begin, IM_mem.at(storage_found).done, map, IM_mem.at(storage_found).device_map_ptr };
	IM_queue.push(next_id);
	next_id++;

	PopErrorsCheck();

	return next_id - 1;
}

__host__ Result CUDA::ExecuteDeviceJobs(){
	//cudaDeviceSynchronize();

	//start PF-repelling job
	RepellingPFGeneration((float(*)[MAP_Y_R][1])ground_PF, (float(*)[MAP_Y_R][1])air_PF);

	//start PF-attracting jobs
	for (int i = 0; i < std::min((int)PF_queue.size(), 5); ++i) {
		AttractingFieldMemory* mem = &PF_mem.at(PF_queue.front());
		cudaEventRecord(mem->begin, 0);
		AttractingPFGeneration(mem->owner_id, (float(*)[MAP_Y_R][1])mem->map, mem->device_map_ptr);
		mem->initialized = true;
		//mem->status = DeviceMemoryStatus::BUSY;
		cudaEventRecord(mem->done, 0);
		PF_queue.pop();
	}

	//start IM job
	if (IM_queue.size() > 0) {
		InfluenceMapMemory* mem = &IM_mem.at(IM_queue.front());
		cudaEventRecord(mem->begin, 0);
		IMGeneration(mem->destination, (float(*)[MAP_Y_R][1])mem->map, mem->air_path, mem->device_map_ptr);
		mem->initialized = true;
		//mem->status = DeviceMemoryStatus::BUSY;
		cudaEventRecord(mem->done, 0);
		IM_queue.pop();
	}

	PopErrorsCheck();
	return Result::OK;
}

__host__ Result CUDA::TransferMapToHost(int id){
	DeviceMemoryStatus* status;
	float* map;
	cudaPitchedPtr map_ptr;
	bool found = false;

	for (int i = 0; i < IM_mem.size(); ++i) {
		if (IM_mem.at(i).queue_id == id) {
			IM_mem.at(i).status = CheckJobStatus(&IM_mem.at(i));
			status = &IM_mem.at(i).status;
			map = IM_mem.at(i).map;
			map_ptr = IM_mem.at(i).device_map_ptr;
			found = true;
			break;
		}
	}
	if (!found) {
		for (int i = 0; i < PF_mem.size(); ++i) {
			if (PF_mem.at(i).queue_id == id) {
				PF_mem.at(i).status = CheckJobStatus(&PF_mem.at(i));
				status = &PF_mem.at(i).status;
				map = PF_mem.at(i).map;
				map_ptr = PF_mem.at(i).device_map_ptr;
				found = true;
				break;
			}
		}
	}

	if (!found || *status == DeviceMemoryStatus::EMPTY || *status == DeviceMemoryStatus::UNKNOWN) {
		return Result::BAD_ARG;
	}

	if (*status == DeviceMemoryStatus::BUSY || *status == DeviceMemoryStatus::OCCUPIED) {
		return Result::BAD_RES;
	}

	cudaMemcpy3DParms par = { 0 };
	par.srcPtr.ptr = map_ptr.ptr;
	par.srcPtr.pitch = map_ptr.pitch;
	par.srcPtr.xsize = MAP_X_R;
	par.srcPtr.ysize = MAP_Y_R;
	par.dstPtr.ptr = map;
	par.dstPtr.pitch = MAP_X_R * sizeof(float);
	par.dstPtr.xsize = MAP_X_R;
	par.dstPtr.ysize = MAP_Y_R;
	par.extent.width = MAP_X_R * sizeof(float);
	par.extent.height = MAP_Y_R;
	par.extent.depth = 1;
	par.kind = cudaMemcpyDeviceToHost;

	cudaError_t err = cudaMemcpy3DAsync(&par);	//transfer
	Check(err, "Transfer queued map to host");
	if (err != cudaSuccess) {
		*status = DeviceMemoryStatus::EMPTY;
		return Result::BAD_RES;
	}

	*status = DeviceMemoryStatus::EMPTY;

	PopErrorsCheck();

	return Result::OK;
}

__host__ DeviceMemoryStatus CUDA::CheckJobStatus(int id){
	for (int i = 0; i < IM_mem.size(); ++i) {
		if (IM_mem.at(i).queue_id == id) {
			return CheckJobStatus(&IM_mem.at(i));
		}
	}
	for (int i = 0; i < PF_mem.size(); ++i) {
		if (PF_mem.at(i).queue_id == id) {
			return CheckJobStatus(&PF_mem.at(i));
		}
	}

	return DeviceMemoryStatus::UNKNOWN;
}

__host__ DeviceMemoryStatus CUDA::CheckJobStatus(AttractingFieldMemory* mem){
	if (mem->initialized == true) {
		if (mem->status == DeviceMemoryStatus::EMPTY) {
			return DeviceMemoryStatus::EMPTY;
		}
		if (cudaEventQuery(mem->done) == cudaSuccess) {
			mem->status == DeviceMemoryStatus::DONE;
			return DeviceMemoryStatus::DONE;
		}
		if (cudaEventQuery(mem->begin) == cudaSuccess) {
			mem->status == DeviceMemoryStatus::BUSY;
			return DeviceMemoryStatus::BUSY;
		}
	}

	return DeviceMemoryStatus::OCCUPIED;
}

__host__ DeviceMemoryStatus CUDA::CheckJobStatus(InfluenceMapMemory* mem){
	if (mem->initialized == true) {
		if (mem->status == DeviceMemoryStatus::EMPTY) {
			return DeviceMemoryStatus::EMPTY;
		}
		if (cudaEventQuery(mem->done) == cudaSuccess) {
			mem->status == DeviceMemoryStatus::DONE;
			return DeviceMemoryStatus::DONE;
		}
		if (cudaEventQuery(mem->begin) == cudaSuccess) {
			mem->status == DeviceMemoryStatus::BUSY;
			return DeviceMemoryStatus::BUSY;
		}
	}

	return DeviceMemoryStatus::OCCUPIED;
}


/*KERNAL LAUNCHES START*/

__host__ void CUDA::RepellingPFGeneration(float ground_avoidance_PF[][MAP_Y_R][1], float air_avoidance_PF[][MAP_Y_R][1]) {
	if (host_unit_list.size() > 0) {
		//cudaDeviceSynchronize();
		DeviceRepellingPFGeneration << <dim_grid_high, dim_block_high, (host_unit_list.size() * sizeof(Entity)) >> >
			(device_unit_list_pointer, host_unit_list.size(), repelling_pf_ground_map_pointer, repelling_pf_air_map_pointer);
		//cudaDeviceSynchronize();
	}

	/*cudaMemcpy3DParms par = { 0 };
	par.srcPtr.ptr = repelling_pf_ground_map_pointer.ptr;
	par.srcPtr.pitch = repelling_pf_ground_map_pointer.pitch;
	par.srcPtr.xsize = MAP_X_R;
	par.srcPtr.ysize = MAP_Y_R;
	par.dstPtr.ptr = ground_avoidance_PF;
	par.dstPtr.pitch = MAP_X_R * sizeof(float);
	par.dstPtr.xsize = MAP_X_R;
	par.dstPtr.ysize = MAP_Y_R;
	par.extent.width = MAP_X_R * sizeof(float);
	par.extent.height = MAP_Y_R;
	par.extent.depth = 1;
	par.kind = cudaMemcpyDeviceToHost;

	Check(cudaPeekAtLastError(), "PF generation peek 1", true);
	Check(cudaDeviceSynchronize(), "PF generation sync", true);
	Check(cudaPeekAtLastError(), "PF generation peek 2", true);

	Check(cudaMemcpy3D(&par), "ground PF memcpy3D");

	par.srcPtr.ptr = repelling_pf_air_map_pointer.ptr;
	par.srcPtr.pitch = repelling_pf_air_map_pointer.pitch;
	par.dstPtr.ptr = air_avoidance_PF;

	Check(cudaMemcpy3D(&par), "air PF memcpy3D");*/

}

__host__ void CUDA::AttractingPFGeneration(int owner_type_id, float map[][MAP_Y_R][1], cudaPitchedPtr device_map){

	//cudaPitchedPtr device_map;
	//cudaMalloc3D(&device_map, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });
	//cudaDeviceSynchronize();
	DeviceAttractingPFGeneration << <dim_grid_high, dim_block_high, (host_unit_list.size() * sizeof(Entity)) >> >
		(device_unit_list_pointer, host_unit_list.size(), owner_type_id, device_map);
	//cudaDeviceSynchronize();

	/*cudaMemcpy3DParms par = { 0 };
	par.srcPtr.ptr = device_map.ptr;
	par.srcPtr.pitch = device_map.pitch;
	par.srcPtr.xsize = MAP_X_R;
	par.srcPtr.ysize = MAP_Y_R;
	par.dstPtr.ptr = map;
	par.dstPtr.pitch = MAP_X_R * sizeof(float);
	par.dstPtr.xsize = MAP_X_R;
	par.dstPtr.ysize = MAP_Y_R;
	par.extent.width = MAP_X_R * sizeof(float);
	par.extent.height = MAP_Y_R;
	par.extent.depth = 1;
	par.kind = cudaMemcpyDeviceToHost;

	Check(cudaPeekAtLastError(), "PF generation peek 1", true);
	Check(cudaDeviceSynchronize(), "PF generation sync", true);
	Check(cudaPeekAtLastError(), "PF generation peek 2", true);

	Check(cudaMemcpy3D(&par), "ground PF memcpy3D");*/

}

__host__ void CUDA::IMGeneration(IntPoint2D destination, float map[][MAP_Y_R][1], bool air_path, cudaPitchedPtr device_map) {

	/*cudaPitchedPtr device_map;
	cudaMalloc3D(&device_map, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });

	IntPoint2D destination_R = {destination.x * GRID_DIVISION, destination.y * GRID_DIVISION};*/

	//cudaDeviceSynchronize();
	if (!air_path) {
		DeviceGroundIMGeneration <<<dim_grid_low, dim_block_low>>>
			(destination, device_map, dynamic_map_device_pointer, global_memory_im_list_storage);
	}
	else {
		DeviceAirIMGeneration <<<dim_grid_low, dim_block_high>>> (destination, device_map);
	}
	//cudaDeviceSynchronize();

	/*Check(cudaPeekAtLastError(), "IM generation peek 1", true);
	Check(cudaDeviceSynchronize(), "IM generation sync", true);

	PopErrorsCheck("IMGen");

	cudaMemcpy3DParms par = { 0 };
	par.srcPtr.ptr = device_map.ptr;
	par.srcPtr.pitch = device_map.pitch;
	par.srcPtr.xsize = MAP_X_R;
	par.srcPtr.ysize = MAP_Y_R;
	par.dstPtr.ptr = map;
	par.dstPtr.pitch = MAP_X_R * sizeof(float);
	par.dstPtr.xsize = MAP_X_R;
	par.dstPtr.ysize = MAP_Y_R;
	par.extent.width = MAP_X_R * sizeof(float);
	par.extent.height = MAP_Y_R;
	par.extent.depth = 1;
	par.kind = cudaMemcpyDeviceToHost;

	Check(cudaMemcpy3D(&par), "IM memcpy3D", true);

	Check(cudaDeviceSynchronize(), "IM print sync", true);*/
}

__host__ void CUDA::UpdateDynamicMap(IntPoint2D center, float radius, int value) {
	IntPoint2D top_left = { center.x - radius - 1, center.y - radius - 1};
	IntPoint2D bottom_right = { center.x + radius + 1, center.y + radius + 1};

	//cudaDeviceSynchronize();
	DeviceUpdateDynamicMap <<< {((bottom_right.x - top_left.x) / dim_block_high.x) + 1, ((bottom_right.y - top_left.y) / dim_block_high.y) + 1, 1},
		dim_block_high >> > (top_left, bottom_right, center, radius, value, dynamic_map_device_pointer);
	//cudaDeviceSynchronize();

}

__host__ void CUDA::TestLookupTable() {
	int table_length = device_unit_lookup_on_host.size();

	float* write_data_d;
	cudaMalloc((void**)&write_data_d, table_length * sizeof(float));
	
	TestDeviceLookupUsage << <1, table_length>> > (write_data_d);

	float* return_data = new float[table_length];
	cudaMemcpy(return_data, write_data_d, table_length * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();

	for (int i = 0; i < table_length; ++i) {
		float a = return_data[i];
		float b = device_unit_lookup_on_host[i].range;
		if (std::abs(return_data[i] - device_unit_lookup_on_host[i].range) > 0.01) {
			std::cout << "lookup table test FAILED" << std::endl;
			delete return_data;
			cudaFree(write_data_d);
			return;
		}
	}
	std::cout << "lookup table test SUCCESS" << std::endl;

	delete return_data;
	cudaFree(write_data_d);
}

__host__ void CUDA::Test3DArrayUsage() {
	cudaPitchedPtr device_map;
	Check(cudaMalloc3D(&device_map, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 }), "PFGeneration malloc3D");

	TransferUnitsToDevice();	//unnecessary for the test

	TestDevice3DArrayUsage << <1, MAP_SIZE_R, (host_unit_list.size() * sizeof(Entity)) >> >
		(device_unit_list_pointer, host_unit_list.size(), device_map);

	float *return_data = new float(MAP_X_R * MAP_Y_R);
	cudaMemcpy3DParms par = { 0 };
	par.srcPtr.ptr = device_map.ptr;
	par.srcPtr.pitch = device_map.pitch;
	par.srcPtr.xsize = MAP_X_R;
	par.srcPtr.ysize = MAP_Y_R;
	par.dstPtr.ptr = return_data;
	par.dstPtr.pitch = MAP_X_R * sizeof(float);
	par.dstPtr.xsize = MAP_X_R;
	par.dstPtr.ysize = MAP_Y_R;
	par.extent.width = MAP_X_R * sizeof(float);
	par.extent.height = MAP_Y_R;
	par.extent.depth = 1;
	par.kind = cudaMemcpyDeviceToHost;

	Check(cudaMemcpy3D(&par), "memcpy3D");

	//Check(cudaDeviceSynchronize());

	//check
	int it = 0;
	for (int i = 0; i < MAP_X_R; ++i) {
		for (int j = 0; j < MAP_Y_R; ++j) {
			if (return_data[i + j * MAP_X_R] != i * MAP_X_R + j) {
				std::cout << "3D Array Usage test FAILED" << std::endl;
				return;
			}
		}
	}
	std::cout << "3D Array Usage test SUCCESS" << std::endl;

	cudaFree(device_map.ptr);
}

/*KERNAL LAUNCHES END*/

__host__ void CUDA::Check(cudaError_t blob, std::string location, bool print_res){
	if (blob != cudaSuccess) {
		std::cout << "CUDA ERROR: (" << location << ") " << cudaGetErrorString(blob) << std::endl;
		blob = cudaDeviceReset(); //might be drastic...
	}
	else if (print_res) {
		std::cout << "CUDA STATUS (" << location << ") SUCESS: " << cudaGetErrorString(blob) << std::endl;
	}
}

__host__ void CUDA::PopErrorsCheck(std::string location) {
	int it = 0;
	while (cudaPeekAtLastError() != cudaSuccess) {
		Check(cudaGetLastError(), ("error pop repeat <" + location + "> " + std::to_string(it)));
		++it;
	}
}

__host__ void CUDA::PrintUnitInfoToFile(std::string filename) {
	std::stringstream str(std::stringstream::out);
	str << "UnitID, DeviceID, Radius, WeaponRange, CanAttackGround, CanAttackAir, IsFlying" << std::endl;

	for (UnitInfo unit : this->host_unit_info) {
		str << unit.id << "," << unit.device_id << ","
			<< unit.radius << "," << unit.range << ","
			<< unit.can_attack_ground << "," << unit.can_attack_air << ","
			<< unit.is_flying << std::endl;
	}

	std::ofstream file;
	file.open(filename);
	file.write(str.str().c_str(), str.str().length());
	file.close();
}

__host__ void CUDA::ReadUnitInfoFromFile(std::string filename) {
	this->host_unit_info.clear();
	std::ifstream inFile(filename);
	std::string line;
	std::getline(inFile, line);	//Remove the first line
	int host_iterator = 0;
	while (std::getline(inFile, line)) {
		UnitInfo unit;
		int pos = line.find(",");
		unit.id = std::stoi(line.substr(0, pos));
		line.erase(0, pos + 1);
		
		pos = line.find(",");
		unit.device_id = std::stoi(line.substr(0, pos));
		line.erase(0, pos + 1);

		pos = line.find(",");
		unit.radius = std::stof(line.substr(0, pos));
		line.erase(0, pos + 1);

		pos = line.find(",");
		unit.range = std::stof(line.substr(0, pos));
		line.erase(0, pos + 1);

		pos = line.find(",");
		unit.can_attack_ground = std::stoi(line.substr(0, pos));
		line.erase(0, pos + 1);

		pos = line.find(",");
		unit.can_attack_air = std::stoi(line.substr(0, pos));
		line.erase(0, pos + 1);

		pos = line.find(",");
		unit.is_flying = std::stoi(line.substr(0, pos));
		line.erase(0, pos + 1);

		this->host_unit_info.push_back(unit);
		this->host_unit_transform.insert({ { sc2::UNIT_TYPEID(unit.id), host_iterator } });
		++host_iterator;
	}

	for (int i = 0; i < host_unit_info.size(); ++i) {
		device_unit_lookup_on_host.push_back({ host_unit_info.at(i).range, host_unit_info.at(i).radius,
			host_unit_info.at(i).is_flying, host_unit_info.at(i).can_attack_air,
			host_unit_info.at(i).can_attack_ground });
	}
}

__host__ std::vector<int> CUDA::GetUnitsID() {
	std::vector<int> unit_IDs;
	for (UnitInfo unit : host_unit_info) {
		unit_IDs.push_back(unit.id);
	}
	return unit_IDs;
}

__host__ void CUDA::SetRadiusForUnits(std::vector<float> radius) {
	for (int i = 0; i < radius.size(); ++i) {
		host_unit_info[i].radius = radius[i];
	}
	PrintUnitInfoToFile("unitInfo.txt");
}

__host__ void CUDA::SetIsFlyingForUnits(std::vector<bool> is_flying) {
	for (int i = 0; i < is_flying.size(); ++i) {
		host_unit_info[i].is_flying = is_flying[i];
	}
	PrintUnitInfoToFile("unitInfo.txt");
}

__host__ int CUDA::GetPosOFUnitInHostUnitVec(sc2::UNIT_TYPEID typeID) {
	return host_unit_transform.at(typeID);
}

__host__ int CUDA::GetUnitIDInHostUnitVec(sc2::UnitTypeID unit_id) {
	return host_unit_transform[unit_id.ToType()];
}

__host__ int CUDA::GetSizeOfUnitInfoList() {
	return host_unit_info.size();
}

__host__ int CUDA::TranslateSC2IDToDeviceID(sc2::UnitTypeID sc2_id) {
	return host_unit_transform.at(sc2_id);
}

__host__ void CUDA::SetHostUnitList(std::vector<Entity>& host_unit_list) {
	this->host_unit_list = host_unit_list;
}