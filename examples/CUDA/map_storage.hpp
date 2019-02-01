#pragma once

#include <vector>
//#include <unordered_map>
#include <list>

#include "../include/sc2api/sc2_typeenums.h"
#include "../include/sc2api/sc2_common.h"

#define map_x 10 
#define map_y 10 
#define bot_faction_unit_type_count 5

//read bookmarks

/*
Textures:
* destinations (global, ground and/or air)
* avoidance (global, ground and air)
* attraction (per unit type)

Data:
* static terrain
* dynamic terrain
* units
*/

struct unit {
	bool enemy = false;
	bool ground_path = true;
	sc2::UNIT_TYPEID id;
	sc2::Point2D position;
};

struct destination_IM {
	sc2::Point2D destination;
	float map[map_x][map_y];
};

struct attraction {
	sc2::UNIT_TYPEID id = sc2::UNIT_TYPEID::INVALID;
	float map[map_x][map_y];
};

//--------------------

class MapStorage {
public:
	MapStorage();
	~MapStorage();

	void Test();

	std::list<destination_IM> destinations_ground_IM;
	std::list<destination_IM> destinations_air_IM;

	float ground_avoidance_PF[map_x][map_y];
	float air_avoidance_PF[map_x][map_y];

	std::vector<attraction> unit_attraction_PF;

private:
	bool static_terrain[map_x][map_y];	//update at start
	bool dynamic_terrain[map_x][map_y];	//update on-building-creation, on-building-destruction, on-building-vision
	std::vector<unit> units;		//updated on-unit-creation, on-unit-death, on-unit-vision
};

//--------------------

MapStorage::MapStorage() {
	unit_attraction_PF.reserve(bot_faction_unit_type_count);
}

MapStorage::~MapStorage() {

}

void MapStorage::Test() {
	for (int x = 0; x < map_x; ++x)
		for (int y = 0; y < map_y; ++y) {
			static_terrain[x][y] = x-y;
			dynamic_terrain[x][y] = x-y;
		}

	units.push_back({ true, true, sc2::UNIT_TYPEID::ZERG_HYDRALISK, { 3, 3 } });
	units.push_back({ true, true, sc2::UNIT_TYPEID::ZERG_HYDRALISK, { 6, 6 } });
}
