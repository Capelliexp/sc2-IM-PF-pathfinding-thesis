#pragma once

#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <vector>
#include <unordered_map>
#include <list>

#include "../include/sc2api/sc2_typeenums.h"
#include "../include/sc2api/sc2_common.h"

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
public:
	MapStorage();
	~MapStorage();

	friend class CUDA;	//might be wrong? used to access private maps & units

	void Test();

	std::list<Destination_IM> destinations_ground_IM;
	std::list<Destination_IM> destinations_air_IM;

	float ground_avoidance_PF[MAP_X][MAP_Y];
	float air_avoidance_PF[MAP_X][MAP_Y];

	//std::vector<Attraction> unit_attraction_PF;
	std::unordered_map<sc2::UNIT_TYPEID, float[MAP_X][MAP_Y]> unit_attraction_PF;

private:
	bool static_terrain[MAP_X][MAP_Y];	//update at start
	bool dynamic_terrain[MAP_X][MAP_Y];	//update on-building-creation, on-building-destruction, on-building-vision
	std::vector<Unit> units;	//update per frame, includes movable units and hostile structures

	bool update_terrain;
};
#endif