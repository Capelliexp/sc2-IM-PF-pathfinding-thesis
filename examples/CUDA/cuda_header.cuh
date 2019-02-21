#pragma once

#ifndef CUDA_HEADER
#define CUDA_HEADER

#include <stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>
#include <unordered_map>
#include <windows.h>
#include <cmath>
#include <utility>
#include <tuple>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"
#include "../examples/CUDA/map_storage.hpp"

//https://devtalk.nvidia.com/default/topic/476201/passing-structures-into-cuda-kernels/
//https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
//http://www.ce.jhu.edu/dalrymple/classes/602/Class13.pdf
//http://disi.unal.edu.co/~gjhernandezp/HeterParallComp/GPU/memory-hardware.pdf
//https://www.reddit.com/r/CUDA/comments/9fkb6n/fastest_way_to_implement_small_lookup_table/
//https://devtalk.nvidia.com/default/topic/482986/how-do-you-copy-an-array-into-constant-memory-/
//https://stackoverflow.com/questions/28821743/sharing-roots-and-weights-for-many-gauss-legendre-quadrature-in-gpus/28822918#28822918
//https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory
//https://stackoverflow.com/questions/5531247/allocating-shared-memory
//https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/shared-memory/shared-memory.cu
//http://www.orangeowlsolutions.com/archives/817
//Check(cudaPeekAtLastError());
//Check(cudaDeviceSynchronize());

#define BLOCK_AMOUNT 1 
#define THREADS_PER_BLOCK 1024
#define THREADS_IN_GRID (BLOCK_AMOUNT*THREADS_PER_BLOCK)

typedef struct {
	int x;
	int y;
} IntPoint2D;

typedef struct {
	int id = 0;		//sc2::UNIT_TYPEID::INVALID
	unsigned int device_id = 0;
	float radius = 0;
	float range = 0;
	bool is_flying = false;
	bool can_attack_air = true;
	bool can_attack_ground = true;
} UnitInfo;

typedef struct {
	float range;
	float radius;
	bool is_flying;
	bool can_attack_air;
	bool can_attack_ground;
} UnitInfoDevice;

typedef struct {
	unsigned int id;	//this is NOT the UNIT_TYPEID, it is the converted device_id
	struct { float x; float y; } pos;
	bool enemy;
} Entity;

typedef struct {
	int id;
	cudaPitchedPtr map_ptr;
	cudaMemcpy3DParms parameters;
} AttractingFieldPointer;

typedef struct {
	IntPoint2D destination;
	cudaPitchedPtr map_ptr;
} InfluenceMapPointer;

//DEVICE FUNCTIONS
__global__ void DeviceAttractingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, int owner_type_id, cudaPitchedPtr device_map);
__global__ void DeviceRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map_ground, cudaPitchedPtr device_map_air);
__global__ void DeviceGroundIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map, cudaPitchedPtr dynamic_map/*, cudaPitchedPtr static_map*/);
__global__ void DeviceAirIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map);

__global__ void TestDevice3DArrayUsage(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map);
__global__ void TestDeviceLookupUsage(float* result);

class CUDA {
	//friend class MapStorage;

public:
	__host__ CUDA();
	__host__ ~CUDA();

	//Initialization (requires cleanup)
	__host__ void InitializeCUDA(MapStorage* maps, const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions, sc2::ActionFeatureLayerInterface* actions_feature_layer);
	__host__ void PrintGenInfo();
	__host__ void CreateUnitLookupOnHost();
	__host__ void TransferStaticMapToHost();
	__host__ void AllocateDeviceMemory();
	__host__ void TransferStaticMapToDevice();
	__host__ void TransferUnitLookupToDevice();
	
	//Runtime functionality
	__host__ void Update(clock_t dt_ticks);
	__host__ void FillDeviceUnitArray();
	__host__ void TransferUnitsToDevice();
	__host__ bool TransferDynamicMapToDevice();

	//Other functionality
	__host__ const sc2::ObservationInterface* GetObservation();
	__host__ sc2::DebugInterface* GetDebug();
	__host__ sc2::ActionInterface* GetAction();
	__host__ sc2::ActionFeatureLayerInterface* GetActionFeature();
	__host__ void CreateUnitRadiusTable();
	__host__ bool DeleteAllIMs();
	__host__ void PrintUnitInfoToFile(std::string filename);
	__host__ void ReadUnitInfoFromFile(std::string filename);
	__host__ std::vector<int> GetUnitsID();
	__host__ void SetRadiusForUnits(std::vector<float> radius);
	__host__ void SetIsFlyingForUnits(std::vector<bool> is_flying);
	__host__ int GetPosOFUnitInHostUnitVec(sc2::UNIT_TYPEID typeID);
	__host__ int GetSizeOfUnitInfoList();

	//Kernel launches
	__host__ void RepellingPFGeneration();
	__host__ void IMGeneration(IntPoint2D destination, bool air_path);
	__host__ void Test3DArrayUsage(); 
	__host__ void TestAttractingPFGeneration();
	__host__ void TestIMGeneration(sc2::Point2D destination, bool air_route);
	__host__ void TestLookupTable();

	//Error checking
	__host__ void Check(cudaError_t blob, std::string location = "unknown", bool print_res = false);	//should not be used in release

private:
	MapStorage* map_storage;
	const sc2::ObservationInterface* observation;
	sc2::DebugInterface* debug;
	sc2::ActionInterface* actions;
	sc2::ActionFeatureLayerInterface* actions_feature_layer;

	dim3 dim_block;
	dim3 dim_grid;
	int threads_in_grid;

	//device memory pointers
	//cudaPitchedPtr static_map_device_pointer;
	cudaPitchedPtr dynamic_map_device_pointer;
	UnitInfoDevice* unit_lookup_device_pointer;
	Entity* device_unit_list_pointer;
	cudaPitchedPtr repelling_pf_ground_map_pointer;
	cudaPitchedPtr repelling_pf_air_map_pointer;
	std::vector<AttractingFieldPointer> unit_type_attracting_pf_pointers;
	std::vector<InfluenceMapPointer> im_pointers;

	//data
	std::vector<UnitInfo> host_unit_info;
	std::vector<UnitInfoDevice> device_unit_lookup_on_host;
	std::unordered_map<sc2::UNIT_TYPEID, unsigned int> host_unit_transform;
	std::vector<Entity> host_unit_list;
	int unit_list_max_length;
};
#endif