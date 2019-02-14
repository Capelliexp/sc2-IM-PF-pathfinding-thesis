#pragma once

#include "../examples/CUDA/cuda_header.cuh"
//#include "../examples/CUDA/map_storage.hpp"

#include <stdio.h>
#include <string>
#include <iostream>
#include <cmath>

#include "../examples/CUDA/cuda_device_functionality.cu"

__host__ CUDA::CUDA(MapStorage* maps, const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions,
	sc2::ActionFeatureLayerInterface* actions_feature_layer) : map_storage(maps), observation(observations), debug(debug), actions(actions),
	actions_feature_layer(actions_feature_layer){
	
	InitializeCUDA();
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
		"BLOCK_AMOUNT: " << BLOCK_AMOUNT << std::endl <<
		"THREADS_PER_BLOCK: " << THREADS_PER_BLOCK << std::endl <<
		std::endl << "Available devices: " << deviceCount << std::endl <<
		"Device ID in use: <" << setDevice << ">" << std::endl <<
		"Runtime API version: " << Rv << std::endl <<
		"Driver API version: " << Dv << std::endl;

	std::cout << std::endl;
}

__host__ void CUDA::Update(clock_t dt_ticks) {
	//float dt = ((float)dt_ticks) / CLOCKS_PER_SEC;	//get dt in seconds

	if (map_storage->update_terrain) {
		TransferDynamicMapToDevice();
		//DeleteAllIMs();	//this might be drastic. should search for which require update and delete those...
	}

	FillDeviceUnitArray();
	TransferUnitsToDevice();
	//run generation of PFs
}

__host__ void CUDA::InitializeCUDA() {
	std::cout << "Initializing CUDA object" << std::endl;
	
	// analysis -> host_transfer -> device_malloc -> device_transfer -> tests

	//analysis
	PrintGenInfo();

	//host_transfer
	CreateUnitLookupOnHost();
	TransferStaticMapToHost();

	//device_malloc
	AllocateDeviceMemory();

	//device_transfer
	TransferStaticMapToDevice();
	TransferUnitLookupToDevice();
	
	//tests
	TestLookupTable();
	Test3DArrayUsage();

}

__host__ void CUDA::CreateUnitLookupOnHost(){
	sc2::UnitTypes types = observation->GetUnitTypeData();

	int host_iterator = 0;
	for (int i = 1; i < types.size(); ++i) {
		sc2::UnitTypeData data;
		data = types.at(i);
		if (data.unit_type_id == sc2::UNIT_TYPEID::INVALID) continue;

		std::vector<sc2::Weapon> weapons;
		weapons = data.weapons;
		if (weapons.size() > 0 || data.movement_speed > 0) {
			//add to "avoid-and-attack" list

			host_unit_info.push_back(UnitInfo());

			sc2::Weapon longest_weapon;
			longest_weapon.range = 0;

			for (auto& const weapon : weapons) {
				if (weapon.range > longest_weapon.range) longest_weapon = weapon;

				if (weapon.type == sc2::Weapon::TargetType::Ground) host_unit_info.at(host_iterator).can_attack_air = false;
				else if (weapon.type == sc2::Weapon::TargetType::Air) host_unit_info.at(host_iterator).can_attack_ground = false;
			}

			host_unit_info.at(host_iterator).range = longest_weapon.range;
			host_unit_info.at(host_iterator).device_id = host_iterator;
			host_unit_info.at(host_iterator).id = data.unit_type_id;

			std::vector<sc2::Attribute> att = data.attributes;
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
	cudaMalloc((void**)&static_map_device_pointer, MAP_X * MAP_Y * sizeof(bool));
	cudaMalloc((void**)&dynamic_map_device_pointer, MAP_X * MAP_Y * sizeof(bool));
	cudaMalloc(&unit_lookup_device_pointer, device_unit_lookup_on_host.size() * sizeof(UnitInfoDevice));
	//cudaMalloc((void**)&unit_lookup_device_pointer, 156 * sizeof(UnitInfoDevice));
	cudaMalloc((void**)&device_unit_list_pointer, 800 * sizeof(Entity));	//might extend size during runtime
}

__host__ void CUDA::FillDeviceUnitArray() {
	host_unit_list.clear();
	host_unit_list.resize(map_storage->units.size());

	int device_list_length = map_storage->units.size();
	for (int i = 0; i < map_storage->units.size(); ++i) {
		std::unordered_map<sc2::UNIT_TYPEID, unsigned int>::const_iterator it = host_unit_transform.find(map_storage->units.at(i).id);
		if (it == host_unit_transform.end()) {
			host_unit_list.resize(host_unit_list.size() - 1);
			std::cout << "WARNING: invalid entity in map_storage unit vector" << std::endl;
			continue;
		}

		host_unit_list.at(device_list_length).id = it->second;
		host_unit_list.at(device_list_length).pos = { map_storage->units.at(i).position.x, map_storage->units.at(i).position.y };
		host_unit_list.at(device_list_length).enemy = map_storage->units.at(i).enemy;

		device_list_length++;
	}	
}

__host__ void CUDA::TestLookupTable(){
	int table_length = device_unit_lookup_on_host.size();

	float* write_data_d;
	cudaMalloc((void**)&write_data_d, table_length * sizeof(float));

	TestDeviceLookupUsage<<<1, table_length >>>(write_data_d);

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
	Check(cudaMalloc3D(&device_map, cudaExtent{ MAP_X * GRID_DIVISION * sizeof(float), MAP_Y * GRID_DIVISION, 1 }), "PFGeneration malloc3D");

	TransferUnitsToDevice();	//unnecessary for the test

	TestDevice3DArrayUsage<<<1, MAP_SIZE, (host_unit_list.size() * sizeof(Entity))>>>
		(device_unit_list_pointer, host_unit_list.size(), device_map);

	float return_data[MAP_X][MAP_Y][1];
	cudaMemcpy3DParms par = { 0 };
	par.srcPtr.ptr = device_map.ptr;
	par.srcPtr.pitch = device_map.pitch;
	par.srcPtr.xsize = MAP_X;
	par.srcPtr.ysize = MAP_Y;
	par.dstPtr.ptr = return_data;
	par.dstPtr.pitch = MAP_X * sizeof(float);
	par.dstPtr.xsize = MAP_X;
	par.dstPtr.ysize = MAP_Y;
	par.extent.width = MAP_X * sizeof(float);
	par.extent.height = MAP_Y;
	par.extent.depth = 1;
	par.kind = cudaMemcpyDeviceToHost;

	Check(cudaMemcpy3D(&par), "memcpy3D", true);
	
	Check(cudaDeviceSynchronize());
	
	//check
	int it = 0;
	for (int i = 0; i < MAP_X; ++i) {
		for (int j = 0; j < MAP_Y; ++j) {
			if (return_data[i][j][0] != i * MAP_X + j) {
				std::cout << "Repelling PF Generation test FAILED" << std::endl;
				return;
			}
		}
	}
	std::cout << "Repelling PF Generation test SUCCESS" << std::endl;
	
	//cudaFree(device_map);	//do not free, space will be used next frame
}

__host__ void CUDA::TestAttractingPFGeneration() {
	
}

__host__ void CUDA::TestIMGeneration(sc2::Point2D destination, bool air_route) {
	//TransferDynamicMapToDevice();

	//TestDevice << <BLOCK_AMOUNT, THREADS_PER_BLOCK >> > ();

	//cudaMemcpy();
}

__host__ void CUDA::TransferUnitsToDevice() {
	Check(cudaMemcpy(device_unit_list_pointer, host_unit_list.data(), host_unit_list.size() * sizeof(Entity),
		cudaMemcpyHostToDevice), "TransferUnitsToDevice");
}

__host__ void CUDA::TransferStaticMapToDevice() {

}

__host__ bool CUDA::TransferDynamicMapToDevice() {

	return true;
}

__host__ void CUDA::Check(cudaError_t blob, std::string location, bool print_res){
	if (blob != cudaSuccess) {
		std::cout << "CUDA ERROR: (" << location << ") " << cudaGetErrorString(blob) << std::endl;
		blob = cudaDeviceReset(); //might be drastic...
	}
	else if (print_res) {
		std::cout << "CUDA STATUS (" << location << ") SUCESS: " << cudaGetErrorString(blob) << std::endl;
	}
}
