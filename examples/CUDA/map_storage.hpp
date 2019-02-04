#pragma once

#ifndef MAP_STORAGE
#define MAP_STORAGE

#include <vector>
#include <unordered_map>
#include <list>

#include "../include/sc2api/sc2_typeenums.h"
#include "../include/sc2api/sc2_common.h"

#define map_x 10 
#define map_y 10 
#define bot_faction_unit_type_count 5

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
	bool enemy = false;
	bool ground_path = true;
	sc2::UNIT_TYPEID id;
	sc2::Point2D position;
};

struct Destination_IM {
	sc2::Point2D destination;
	float map[map_x][map_y];
};

//struct Attraction {
//	sc2::UNIT_TYPEID id = sc2::UNIT_TYPEID::INVALID;
//	float map[map_x][map_y];
//};

class MapStorage {
public:
	MapStorage();
	~MapStorage();

	void Test();

	std::list<Destination_IM> destinations_ground_IM;
	std::list<Destination_IM> destinations_air_IM;

	float ground_avoidance_PF[map_x][map_y];
	float air_avoidance_PF[map_x][map_y];

	//std::vector<Attraction> unit_attraction_PF;
	std::unordered_map<sc2::UNIT_TYPEID, float[map_x][map_y]> unit_attraction_PF;

	bool update_terrain;

private:
	bool static_terrain[map_x][map_y];	//update at start
	bool dynamic_terrain[map_x][map_y];	//update on-building-creation, on-building-destruction, on-building-vision
	std::vector<Unit> units;	//update per frame, includes movable units and hostile structures
};
#endif