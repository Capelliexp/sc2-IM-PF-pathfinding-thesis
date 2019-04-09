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
#include <queue>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"
//#include "../examples/CUDA/map_storage.hpp"

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


//! Maps: empty20, wall20 (18,29) or (8,8), (10,16)
//#define MAP_X 32
//#define MAP_Y 32

//! Maps: empty50, spiral50
#define MAP_X 56
#define MAP_Y 56

//! Maps: labyrinth, height (53,60), wall
//#define MAP_X 104
//#define MAP_Y 104

//! Maps: empty200
//#define MAP_X 200
//#define MAP_Y 200

#define MAP_SIZE (MAP_X*MAP_Y)

#define GRID_DIVISION 1 // 1 grid's sub grid size = GRID_DIVISION^2  (OBS! minimum 1)
#define MAP_X_R (MAP_X*GRID_DIVISION)
#define MAP_Y_R (MAP_Y*GRID_DIVISION)
#define MAP_SIZE_R (MAP_SIZE*GRID_DIVISION*GRID_DIVISION)

typedef short integer;

typedef struct {	//same as short_coord
	integer x;
	integer y;
} IntPoint2D;

typedef struct {
	float x;
	float y;
} FloatPoint2D;

typedef struct {
	integer node;
	integer backtrack_iterator;
} list_double_entry;

typedef struct {
	integer pos;
	integer backtrack_it;
	integer steps_from_start;
	float est_dist_start_to_dest_via_pos;
} node;

typedef struct {	//same as IntPoint2D
	integer x;
	integer y;
} short_coord;

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

typedef enum {
	BUSY,
	OCCUPIED,
	DONE,
	EMPTY,
	UNKNOWN
} DeviceMemoryStatus;

typedef enum {
	OK,
	BAD_ARG,
	BAD_RES
} Result;

typedef struct {
	int owner_id;	//how map_storage identifies the map
	int queue_id;	//how the queue identifies the map (MUST BE SECOND VARIABLE)
	DeviceMemoryStatus status;
	bool initialized;
	cudaEvent_t begin, done;
	float* map;
	cudaPitchedPtr device_map_ptr;
} AttractingFieldMemory;

bool operator==(const AttractingFieldMemory& first, const AttractingFieldMemory& second);
bool operator!=(const AttractingFieldMemory& first, const AttractingFieldMemory& second);

typedef struct {
	IntPoint2D destination;	//how map_storage identifies the map
	int queue_id;	//how the queue identifies the map  (MUST BE SECOND VARIABLE)
	bool air_path;
	DeviceMemoryStatus status;
	bool initialized;
	cudaEvent_t begin, done;
	float* map;
	cudaPitchedPtr device_map_ptr;
} InfluenceMapMemory;

bool operator==(const InfluenceMapMemory& first, const InfluenceMapMemory& second);
bool operator!=(const InfluenceMapMemory& first, const InfluenceMapMemory& second);

//DEVICE FUNCTION
__global__ void DeviceAttractingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, int owner_type_id, cudaPitchedPtr device_map);
__global__ void DeviceRepellingPFGeneration(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map_ground, cudaPitchedPtr device_map_air);
__global__ void DeviceGroundIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map, cudaPitchedPtr dynamic_map_device_pointer);
__global__ void DeviceAirIMGeneration(IntPoint2D destination, cudaPitchedPtr device_map);
__global__ void DeviceUpdateDynamicMap(IntPoint2D top_left, IntPoint2D bottom_right, IntPoint2D center, float radius, int new_value, cudaPitchedPtr dynamic_map_device_pointer);

__global__ void TestDevice3DArrayUsage(Entity* device_unit_list_pointer, int nr_of_units, cudaPitchedPtr device_map);
__global__ void TestDeviceLookupUsage(float* result);

class CUDA {
	//friend class MapStorage;
	
public:
	__host__ CUDA();
	__host__ ~CUDA();
	
	//Initialization
	__host__ void InitializeCUDA(const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions, float ground_PF[][MAP_Y_R][1], float air_PF[][MAP_Y_R][1]);
	__host__ void PrintGenInfo();
	__host__ void CreateUnitLookupOnHost(std::string file);
	__host__ void TransferStaticMapToHost();
	__host__ void AllocateDeviceMemory();
	__host__ void SpecifyDeviceFunctionAttributes();
	__host__ void BindRepellingMapsToTransferParams();
	__host__ void TransferUnitLookupToDevice();
	__host__ void DeviceTransfer(bool dynamic_terrain[][MAP_Y_R][1]);
	__host__ void Tests(float ground_avoidance_PF[][MAP_Y_R][1], float air_avoidance_PF[][MAP_Y_R][1]);

	//Runtime functionality
	__host__ void TransferUnitsToDevice();
	__host__ void TransferDynamicMapToDevice(bool dynamic_terrain[][MAP_Y_R][1]);

	//Runtime jobs
	__host__ int QueueDeviceJob(int owner_id, float* map = nullptr);	//start PF generation job
	__host__ int QueueDeviceJob(IntPoint2D destination, bool air_path, float* map = nullptr);	//start IM generation job
	__host__ Result ExecuteDeviceJobs();
	__host__ Result TransferMapToHost(int id);	//start transfer from device to pre-defined host array
	__host__ DeviceMemoryStatus CheckJobStatus(int id);
	__host__ DeviceMemoryStatus CheckJobStatus(AttractingFieldMemory* mem);
	__host__ DeviceMemoryStatus CheckJobStatus(InfluenceMapMemory* mem);

	//Other functionality
	__host__ const sc2::ObservationInterface* GetObservation();
	__host__ sc2::DebugInterface* GetDebug();
	__host__ sc2::ActionInterface* GetAction();
	__host__ sc2::ActionFeatureLayerInterface* GetActionFeature();
	__host__ void PrintUnitInfoToFile(std::string filename);
	__host__ void ReadUnitInfoFromFile(std::string filename);
	__host__ std::vector<int> GetUnitsID();
	__host__ void SetRadiusForUnits(std::vector<float> radius);
	__host__ void SetIsFlyingForUnits(std::vector<bool> is_flying);
	__host__ int GetPosOFUnitInHostUnitVec(sc2::UNIT_TYPEID typeID);
	__host__ int GetUnitIDInHostUnitVec(sc2::UnitTypeID unit_id);
	__host__ int GetSizeOfUnitInfoList();
	__host__ int TranslateSC2IDToDeviceID(sc2::UnitTypeID sc2_id);
	__host__ void SetHostUnitList(std::vector<Entity>& host_unit_list);
	__host__ void PrintDeviceMemoryUsage(std::string location = "");
	__host__ void SyncDevice();
	
	//Kernel launches
	__host__ void RepellingPFGeneration();
	__host__ void AttractingPFGeneration(int owner_type_id, float map[][MAP_Y_R][1], cudaPitchedPtr device_map);
	__host__ void IMGeneration(IntPoint2D destination, float map[][MAP_Y_R][1], bool air_path, cudaPitchedPtr device_map);
	__host__ void UpdateDynamicMap(IntPoint2D center, float radius, int value);

	__host__ void Test3DArrayUsage();
	__host__ void TestLookupTable();
	
	//Error checking (should not be used in release)
	__host__ void Check(cudaError_t blob, std::string location = "unknown", bool print_res = false);
	__host__ void PopErrorsCheck(std::string location = "unknown");
	
private:
	//class pointers
	//MapStorage* map_storage;
	const sc2::ObservationInterface* observation;
	sc2::DebugInterface* debug;
	sc2::ActionInterface* actions;
	sc2::ActionFeatureLayerInterface* actions_feature_layer;

	//thread, block & grid sizes
	dim3 dim_block_high;
	dim3 dim_grid_high;
	dim3 dim_block_low;
	dim3 dim_grid_low;
	int threads_in_grid_high;
	int threads_in_grid_low;
	
	//device memory single map pointers
	cudaPitchedPtr dynamic_map_device_pointer;
	cudaPitchedPtr repelling_pf_ground_map_pointer;
	cudaPitchedPtr repelling_pf_air_map_pointer;

	//device memory array pointers
	UnitInfoDevice* unit_lookup_device_pointer;
	Entity* device_unit_list_pointer;
	int* device_IM_shared_array_size_per_block;
	int* device_IM_shared_array_size_per_thread;
	
	//device memory map lists & queues
	std::vector<AttractingFieldMemory> PF_mem;
	std::vector<InfluenceMapMemory> IM_mem;
	std::queue<int> PF_queue;
	std::queue<int> IM_queue;
	int next_id;
	cudaEvent_t repelling_PF_event_done;
	cudaMemcpy3DParms repelling_PF_memcpy_params_ground;
	cudaMemcpy3DParms repelling_PF_memcpy_params_air;

	//host data (holders for host->device transfers)
	float* ground_PF;
	float* air_PF;
	std::vector<UnitInfo> host_unit_info;
	std::vector<UnitInfoDevice> device_unit_lookup_on_host;
	std::unordered_map<sc2::UNIT_TYPEID, unsigned int> host_unit_transform;
	std::vector<Entity> host_unit_list;
	int unit_list_max_length;
};
#endif