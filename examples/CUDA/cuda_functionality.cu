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

__host__ CUDA::CUDA(){	
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

__host__ void CUDA::Update(clock_t dt_ticks, sc2::Units units, float ground_avoidance_PF[][MAP_Y_R][1], float air_avoidance_PF[][MAP_Y_R][1]) {
	//float dt = ((float)dt_ticks) / CLOCKS_PER_SEC;	//get dt in seconds
	
	PopErrorsCheck("CUDA Update pre");	//run first

	//if (map_storage->update_terrain) {
	//	TransferDynamicMapToDevice();
	//	//DeleteAllIMs();	//this might be drastic. should search for which require update and delete those...
	//}


	FillDeviceUnitArray(units);
	TransferUnitsToDevice();
	RepellingPFGeneration(ground_avoidance_PF, air_avoidance_PF);

	PopErrorsCheck("CUDA Update post");	//run last

	//run generation of PFs
}

__host__ void CUDA::InitializeCUDA(const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions){
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

	unit_list_max_length = 800;
	unit_type_attracting_pf_pointers.reserve(100);
	im_pointers.reserve(100);

	PopErrorsCheck("CUDA Initialization base");

	//analysis
	PrintGenInfo();

	//device_malloc
	AllocateDeviceMemory();

	PopErrorsCheck("CUDA Initialization malloc");

	//IMGeneration(IntPoint2D{ 18, 29 }, false);

	//Check(cudaPeekAtLastError(), "init check 7", true);
}

__host__ void CUDA::HostTransfer(sc2::Units units) {
	//host_transfer
	TransferStaticMapToHost();
	FillDeviceUnitArray(units);

	Check(cudaPeekAtLastError(), "init check 4", true);
}

__host__ void CUDA::DeviceTransfer(bool dynamic_terrain[][MAP_Y_R][1]) {
	//device_transfer
	TransferDynamicMapToDevice(dynamic_terrain);
	TransferUnitLookupToDevice();
	TransferUnitsToDevice();

	Check(cudaPeekAtLastError(), "init check 5", true);
}

__host__ void CUDA::Tests(float ground_avoidance_PF[][MAP_Y_R][1], float air_avoidance_PF[][MAP_Y_R][1]) {
	//tests
	//TestLookupTable();
	Check(cudaPeekAtLastError(), "init check 6a", true);

	//Test3DArrayUsage();
	Check(cudaPeekAtLastError(), "init check 6b", true);

	RepellingPFGeneration(ground_avoidance_PF, air_avoidance_PF);
	Check(cudaPeekAtLastError(), "init check 6c", true);
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
	std::cout << "device_unit_lookup array transfered to device" << std::endl;
}

__host__ void CUDA::AllocateDeviceMemory(){
	//cudaMalloc3D(&static_map_device_pointer, cudaExtent{ MAP_X_R * sizeof(bool), MAP_Y_R, 1 });	//static map
	cudaMalloc3D(&dynamic_map_device_pointer, cudaExtent{ MAP_X_R * sizeof(bool), MAP_Y_R, 1 });	//dynamic map
	cudaMalloc(&unit_lookup_device_pointer, device_unit_lookup_on_host.size() * sizeof(UnitInfoDevice));	//lookup table (global on device)
	cudaMalloc((void**)&device_unit_list_pointer, unit_list_max_length * sizeof(Entity));	//unit list (might extend size during runtime)
	cudaMalloc3D(&repelling_pf_ground_map_pointer, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });	//repelling on ground
	cudaMalloc3D(&repelling_pf_air_map_pointer, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });	//repelling in air
	Check(cudaMalloc((void**)&global_memory_im_list_storage, 256000000 * sizeof(list_double_entry)), "big AF allocation", true);	//big AF list for A* open/closed list

	Check(cudaPeekAtLastError(), "cuda allocation peek", true);
}

__host__ void CUDA::FillDeviceUnitArray(sc2::Units units) {
	host_unit_list.clear();
	host_unit_list.resize(units.size());

	//int device_list_length = map_storage->units.size();
	int device_list_length = 0;
	for (const sc2::Unit* unit : units) {
		std::unordered_map<sc2::UNIT_TYPEID, unsigned int>::const_iterator it = host_unit_transform.find(unit->unit_type);
		if (it == host_unit_transform.end()) {
			host_unit_list.resize(host_unit_list.size() - 1);
			std::cout << "WARNING: invalid entity in map_storage unit vector" << std::endl;
			continue;
		}

		host_unit_list.at(device_list_length).id = it->second;
		host_unit_list.at(device_list_length).pos = { unit->pos.x * GRID_DIVISION, unit->pos.y * GRID_DIVISION };
		switch (unit->alliance)
		{
		case sc2::Unit::Alliance::Self:
			host_unit_list.at(device_list_length).enemy = false;
			break;
		case sc2::Unit::Alliance::Ally:
			host_unit_list.at(device_list_length).enemy = false;
			break;
		case sc2::Unit::Alliance::Neutral:
			host_unit_list.at(device_list_length).enemy = false;
				break;
		case sc2::Unit::Alliance::Enemy:
			host_unit_list.at(device_list_length).enemy = true;
			break;
		default:
			host_unit_list.at(device_list_length).enemy = false;
			break;
		}

		device_list_length++;
	}	
}

__host__ void CUDA::TransferUnitsToDevice() {

	if (host_unit_list.size() > unit_list_max_length) {
		std::cout << "WARNING: too many units! Increase allocation size, overflow discarded" << std::endl;
	}

	Check(cudaMemcpy(device_unit_list_pointer, host_unit_list.data(), 
		min(host_unit_list.size(), unit_list_max_length) * sizeof(Entity),
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

	Check(cudaMemcpy3D(&par), "Dynamic map transfer");
}

/*KERNAL LAUNCHES START*/

__host__ void CUDA::RepellingPFGeneration(float ground_avoidance_PF[][MAP_Y_R][1], float air_avoidance_PF[][MAP_Y_R][1]) {
	DeviceRepellingPFGeneration<<<dim_grid_high, dim_block_high, (host_unit_list.size() * sizeof(Entity))>>>
		(device_unit_list_pointer, host_unit_list.size(), repelling_pf_ground_map_pointer, repelling_pf_air_map_pointer);

	cudaMemcpy3DParms par = { 0 };
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

	Check(cudaMemcpy3D(&par), "air PF memcpy3D");

	//Check(cudaDeviceSynchronize());
}

__host__ void CUDA::AttractingPFGeneration(int owner_type_id, float map[][MAP_Y_R][1]){

	cudaPitchedPtr device_map;
	cudaMalloc3D(&device_map, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });

	DeviceAttractingPFGeneration << <dim_grid_high, dim_block_high, (host_unit_list.size() * sizeof(Entity)) >> >
		(device_unit_list_pointer, host_unit_list.size(), owner_type_id, device_map);

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

	Check(cudaPeekAtLastError(), "PF generation peek 1", true);
	Check(cudaDeviceSynchronize(), "PF generation sync", true);
	Check(cudaPeekAtLastError(), "PF generation peek 2", true);

	Check(cudaMemcpy3D(&par), "ground PF memcpy3D");

	//Check(cudaDeviceSynchronize());
}

__host__ void CUDA::IMGeneration(IntPoint2D destination, float map[][MAP_Y_R][1], bool air_path) {

	cudaPitchedPtr device_map;
	cudaMalloc3D(&device_map, cudaExtent{ MAP_X_R * sizeof(float), MAP_Y_R, 1 });

	IntPoint2D destination_R = {destination.x * GRID_DIVISION, destination.y * GRID_DIVISION};

	if (!air_path) {
		DeviceGroundIMGeneration <<<dim_grid_low, dim_block_low>>>
			(destination_R, device_map, dynamic_map_device_pointer, global_memory_im_list_storage);
	}
	else {
		DeviceAirIMGeneration <<<dim_grid_low, dim_block_high>>> (destination_R, device_map);
	}

	Check(cudaPeekAtLastError(), "IM generation peek 1", true);
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

	Check(cudaDeviceSynchronize(), "IM print sync", true);
}

__host__ void CUDA::UpdateDynamicMap(IntPoint2D center, float radius, int value) {
	IntPoint2D top_left = { center.x - radius - 1, center.y - radius - 1};
	IntPoint2D bottom_right = { center.x + radius + 1, center.y + radius + 1};

	DeviceUpdateDynamicMap <<< {((bottom_right.x - top_left.x) / dim_block_high.x) + 1, ((bottom_right.y - top_left.y) / dim_block_high.y) + 1, 1},
		dim_block_high >> > (top_left, bottom_right, center, radius, value, dynamic_map_device_pointer);
}

__host__ void CUDA::TestLookupTable() {
	int table_length = device_unit_lookup_on_host.size();

	float* write_data_d;
	cudaMalloc((void**)&write_data_d, table_length * sizeof(float));

	TestDeviceLookupUsage << <1, table_length >> > (write_data_d);

	float* return_data = new float[table_length];
	cudaMemcpy(return_data, write_data_d, table_length * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < table_length; ++i) {
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

	Check(cudaDeviceSynchronize());

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

	//cudaFree(device_map);	//do not free, space will be used next frame
}

__host__ void CUDA::TestAttractingPFGeneration() {

}

__host__ void CUDA::TestIMGeneration(sc2::Point2D destination, bool air_route) {
	//TransferDynamicMapToDevice();

	//TestDevice << <BLOCK_AMOUNT, THREADS_PER_BLOCK >> > ();

	//cudaMemcpy();
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

__host__ int CUDA::GetSizeOfUnitInfoList() {
	return host_unit_info.size();
}
