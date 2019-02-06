#pragma once

#include "../examples/CUDA/cuda_header.cuh"
//#include "../examples/CUDA/map_storage.hpp"

#include <stdio.h>
#include <string>
#include <iostream>

__host__ CUDA::CUDA(MapStorage* maps, const sc2::ObservationInterface* observations) :
	map_storage(maps), observation(observations) {

	if (!InitializeCUDA())
		std::cout << "shit b fucked yo" << std::endl;
}

__host__ CUDA::~CUDA() {

}

__host__ void CUDA::PrintGenInfo() {
	SYSTEM_INFO siSysInfo;
	GetSystemInfo(&siSysInfo);
	cudaError_t blob;

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
	//run generation of PFs
}

__host__ bool CUDA::InitializeCUDA() {
	std::cout << "Initializing CUDA object" << std::endl;
	
	PrintGenInfo();
	AllocateDeviceMemory();
	TransferStaticMapToDevice();
	CreateDeviceLookup();

	TestLookupTable();

	return true;
}

__host__ void CUDA::AllocateDeviceMemory(){
	//THIS NEEDS TO BE DEFERRED TO AFTER WE KNOW ARRAY SIZES

	cudaMalloc((void**)&static_map_device_pointer, MAP_X * MAP_Y * sizeof(bool));
	cudaMalloc((void**)&dynamic_map_device_pointer, MAP_X * MAP_Y * sizeof(bool));
	//cudaMalloc((void**)&unit_lookup_device_pointer, 156 * sizeof(UnitInfoDevice));
	//cudaMalloc((void**)&unit_array_device_pointer, 800 * sizeof(UnitStructInDevice));
}

__host__ void CUDA::CreateDeviceLookup() {

	//host_unit_info[0] = {sc2::UNIT_TYPEID::TERRAN_WIDOWMINEBURROWED, 0, sc2::UnitTypeID::};

	sc2::UnitTypes types = observation->GetUnitTypeData();

	std::cout << "starting unit search" << std::endl;

	int host_iterator = 0;
	for (int i = 1; i < types.size(); ++i) {
		sc2::UnitTypeData data;
		data = types.at(i);
		if (data.unit_type_id == sc2::UNIT_TYPEID::INVALID) continue;

		std::vector<sc2::Weapon> weapons;
		weapons = data.weapons;
		if(weapons.size() > 0 || data.movement_speed > 0){
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

			++host_iterator;
		}
	}
	std::cout << "Created unit lookup table on host. Nr of elements: " << host_iterator << ". " << std::endl;
	if (host_iterator != 86) 
		std::cout << "This " << (host_iterator < 86 ? "might" : "will") << " end badly..." << std::endl;
	

	for (int i = 0; i < host_iterator; ++i) {
		/*device_unit_lookup_on_host.push_back({ host_unit_info.at(i).range, host_unit_info.at(i).is_flying,
			host_unit_info.at(i).can_attack_air, host_unit_info.at(i).can_attack_ground });*/
		device_unit_lookup_on_host[i] = { host_unit_info.at(i).range, host_unit_info.at(i).is_flying,
			host_unit_info.at(i).can_attack_air, host_unit_info.at(i).can_attack_ground };
	}

	std::cout << "device_unit_lookup array filled on host" << std::endl;
	TransferSymbolsToDevice();
	std::cout << "device_unit_lookup array copied to device (i think)" << std::endl;
}

__host__ bool CUDA::FillDeviceUnitArray() {

	return true;
}

__host__ void CUDA::TestLookupTable(){

	float* device_write_data;
	cudaMalloc((void**)&device_write_data, 156 * sizeof(float));

	TestDeviceLookupUsage<<<1, 156>>>(/*unit_lookup_device_pointer, */device_write_data);

	float* device_return_data = new float[156];
	cudaMemcpy(device_return_data, device_write_data, 156 * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "TestLookupTable() device return data:" << std::endl;
	for (int i = 0; i < 156; ++i) {
		std::cout << device_return_data[i] << ", ";
	}
}

__host__ void CUDA::TestRepellingPFGeneration() {
	float* device_map;
	float* new_map = new float[THREADS_IN_GRID];

	cudaMalloc((void**)&device_map, THREADS_IN_GRID * sizeof(float));	//allocate space for map on device

	TransferUnitsToDevice();

	TestDevicePFGeneration << <BLOCK_AMOUNT, THREADS_PER_BLOCK >> > (device_map);

	cudaMemcpy(new_map, device_map, THREADS_IN_GRID * sizeof(float), cudaMemcpyDeviceToHost);	//transfer map to host
	//the memcpy should copy to a host 2D array directly, not like this!

	//cudaFree(device_map);	//do not free, space will be used next frame

}

__host__ void CUDA::TestAttractingPFGeneration(float range, bool is_flying, bool can_attack_air, bool can_attack_ground) {
	
}

__host__ void CUDA::TestIMGeneration(sc2::Point2D destination, bool air_route) {
	//TransferDynamicMapToDevice();

	//TestDevice << <BLOCK_AMOUNT, THREADS_PER_BLOCK >> > ();

	//cudaMemcpy();
}

__host__ bool CUDA::TransferUnitsToDevice() {
	//std::vector<UnitStructInDevice> vec;
	//vec.reserve(map_storage->units.size());


	for (auto const& unit : map_storage->units) {

	}

	//transfer to GPU ...

	return true;
}

__host__ bool CUDA::TransferStaticMapToDevice() {

	return true;
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