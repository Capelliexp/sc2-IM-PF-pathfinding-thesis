#pragma once

#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <vector>
#include <unordered_map>
#include <list>

//#include "../include/sc2api/sc2_typeenums.h"
//#include "../include/sc2api/sc2_common.h"

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"

//#include "../examples/CUDA/cuda_header.cuh"	//do NOT include, causes shit 2 b strange

#define MAP_X 10 
#define MAP_Y 10 
#define MAP_SIZE MAP_X*MAP_Y
#define GRID_DIVISION 1 // 1 grid's sub grid size = GRID_DIVISION^2 

//read bookmarks

/*
Textures:
* destinations (global, ground or air)
* avoidance (global, ground and air)
* attraction (per unit type)

Data:
* static terrain
* dynamic terrain
* units
*/

struct Unit {
	sc2::UNIT_TYPEID id;
	sc2::Point2D position;
	bool enemy = false;
};

struct Destination_IM {
	sc2::Point2D destination;
	float map[MAP_X][MAP_Y];
};

//struct Attraction {
//	sc2::UNIT_TYPEID id = sc2::UNIT_TYPEID::INVALID;
//	float map[map_x][map_y];
//};

class MapStorage {
	friend class CUDA;	//might be wrong? used to access private maps & units

public:
	MapStorage();
	~MapStorage();

	void Initialize(const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions,
		sc2::ActionFeatureLayerInterface* actions_feature_layer);

	void Test();
	bool CheckIfFileExists(std::string filename);


	void PrintStatus(std::string msg);
	void PrintMap(sc2::ImageData map, std::string name);

	//! The bot is abdle to print its IM to a file.
	void PrintIM();

	std::list<Destination_IM> destinations_ground_IM;
	std::list<Destination_IM> destinations_air_IM;

	float ground_avoidance_PF[MAP_X][MAP_Y];
	float air_avoidance_PF[MAP_X][MAP_Y];

	//std::vector<Attraction> unit_attraction_PF;
	std::unordered_map<sc2::UNIT_TYPEID, float[MAP_X][MAP_Y]> unit_attraction_PF;

private:
	//! Craetes the influence map based on the size of the map.
	void CreateIM();

	//! Function that is used to add a list of units to the IM.
	//!< \param units The list of units to be added.
	void IMAddUnits(sc2::Units units);

	//! Function that is used to add an unit to the IM.
	//! Uses radius to indicate which tiles that can't be pathed.
	//!< \param unit The unit to be added.
	void IMAddUnit(const sc2::Unit* unit);

	//! Function that is used to remove an unit from the IM.
	//! We know that the tiles that the building occupied can be pathed now.
	//! No need to calculate the radius.
	//!< \param unit The unit to be removed.
	void IMRemoveUnit(const sc2::Unit* unit);

	//! Function that is used to check if a given unit is a structure.
	//!< \param unit The unit to be checked.
	//!< \return Returns true if the unit is a structure, false otherwise.
	bool IsStructure(const sc2::Unit* unit);

	void SpawnEveryUnit();

	void PrintUnits();

	void AddObjectiveToIM(sc2::Point2D objective);

	//CUDA* cuda;	//do NOT include 
	const sc2::ObservationInterface* observation;
	sc2::DebugInterface* debug;
	sc2::ActionInterface* actions;
	sc2::ActionFeatureLayerInterface* actions_feature_layer;

	bool static_terrain[MAP_X][MAP_Y];	//update at start
	bool dynamic_terrain[MAP_X][MAP_Y];	//update on-building-creation, on-building-destruction, on-building-vision
	std::vector<Unit> units;	//update per frame, includes movable units and hostile structures

	bool update_terrain;

	//---------------

	int lastSize;
	//! Width that is multiplied with pathingGridSize to get actual width of the pathfinding grid
	int width;
	//! Width that is multiplied with pathingGridSize to get actual height of the pathfinding grid
	int height;
	//! Pathing grid is 8 time larger than what is returned from API
	int pathingGridSize = 4;
	//! Vector representing the pathfinding grid for ground units.
	std::vector<float> InfluenceMap;
	std::vector<sc2::Point2D> objectives;
	std::vector<std::vector<float>> PotentialField;
	bool MapPrinted = false;
	
	int kFeatureLayerSize;
	int kPixelDrawSize;
	int kDrawSize;
};
#endif