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
#include "../LoadPNG/lodepng.h"
#include "../examples/CUDA/cuda_header.cuh"	//do NOT include, causes shit 2 b strange
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

struct Attracting_PF {
	int id;
	float map[MAP_X_R][MAP_Y_R][1];
};

struct Destination_IM {
	bool air_path = false;
	sc2::Point2D destination;
	float map[MAP_X_R][MAP_Y_R][1];
};

class MapStorage {
	//friend class CUDA;	//might be wrong? used to access private maps & units
public:
	enum colors
	{
		RED,
		GREEN,
		BLUE,
		YELLOW,
		PURPLE,
		WHITE,
		BLACK
	};
public:
	MapStorage();
	~MapStorage();

	void Initialize(const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions,
		sc2::ActionFeatureLayerInterface* actions_feature_layer);

	void Test();
	//! Function to check if the file exists
	//!< \param filename String of the filename to check for. Include relevant ending of file (.txt, .png, ...)
	//!< \return Return true if file found.
	bool CheckIfFileExists(std::string filename);

	void PrintStatus(std::string msg);
	void PrintMap(float map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file);
	void PrintMap(bool map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file);
	void PrintMap(int map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file);

	void Update(clock_t dt);
	std::vector<int> GetUnitsID();
	int GetSizeOfUnitInfoList();
	int GetPosOFUnitInHostUnitVec(sc2::UNIT_TYPEID typeID);
	void SetRadiusForUnits(std::vector<float> radius);
	//! The bot is abdle to print its IM to a file.
	//void PrintIM();

	//! Function to check if ground IM destination exists
	//!< \param pos sc2::Point2D indicating destination of the IM. Used to check with already existing IMs.
	//!< \return Returns a reference to the IM if it exists, otherwise nullptr.
	Destination_IM* CheckGroundDestination(sc2::Point2D pos);
	//! Function to check if air IM destination exists
	//!< \param pos sc2::Point2D indicating destination of the IM. Used to check with already existing IMs.
	//!< \return Returns a reference to the IM if it exists, otherwise nullptr.
	Destination_IM* CheckAirDestination(sc2::Point2D pos);
	//! Functions to get IM destination for ground.
	//! Will create IM if needed.
	//!< \param pos sc2::Point2D the position to create an IM to.
	//!< \return Returns a reference to the IM, will return nullptr if something went wrong.
	Destination_IM* GetGroundDestination(sc2::Point2D pos);
	//! Functions to get IM destination for air.
	//! Will create IM if needed.
	//!< \param pos sc2::Point2D the position to create an IM to.
	//!< \return Returns a reference to the IM, will return nullptr if something went wrong.
	Destination_IM* GetAirDestination(sc2::Point2D pos);

	std::list<Destination_IM> destinations_ground_IM;
	std::list<Destination_IM> destinations_air_IM;

	float ground_avoidance_PF[MAP_X_R][MAP_Y_R][1];
	float air_avoidance_PF[MAP_X_R][MAP_Y_R][1];

	//std::vector<Attraction> unit_attraction_PF;
	//std::unordered_map<sc2::UNIT_TYPEID, float[MAP_X_R][MAP_Y_R]> unit_attraction_PF;
	std::unordered_map<sc2::UNIT_TYPEID, float*> unit_attraction_PF;

private:
	//! Craetes the influence map based on the size of the map.
	void CreateIM();

	//! Function that is used to add a list of units to the IM.
	//!< \param units The list of units to be added.
	//void IMAddUnits(sc2::Units units);

	//! Function that is used to add an unit to the IM.
	//! Uses radius to indicate which tiles that can't be pathed.
	//!< \param unit The unit to be added.
	//void IMAddUnit(const sc2::Unit* unit);

	//! Function that is used to remove an unit from the IM.
	//! We know that the tiles that the building occupied can be pathed now.
	//! No need to calculate the radius.
	//!< \param unit The unit to be removed.
	//void IMRemoveUnit(const sc2::Unit* unit);

	//! Function that is used to check if a given unit is a structure.
	//!< \param unit The unit to be checked.
	//!< \return Returns true if the unit is a structure, false otherwise.
	bool IsStructure(const sc2::Unit* unit);

	//! Function to create an image.
	//! Function will reset the image variable.
	//!< \param map A 2D array of bools containing the areas on the map that is non-patheble.
	//!< \param width Integer representing the width of the map.
	//!< \param height Integer representing the height of the map.
	//!< \param file The name of the image that is to be created.
	void CreateImage(bool map[MAP_X_R][MAP_Y_R][1], int width, int height);
	void CreateImage(float map[MAP_X_R][MAP_Y_R][1], int width, int height, colors color);
	//! Function to add elements to the image.
	//!< \param map A 2D array of floats containing the elements to add to the map.
	//!< \param width Integer representing the width of the map.
	//!< \param height Integer representing the height of the map.
	//!< \param colors The color of the elements to be added to the map
	void AddToImage(float map[MAP_X_R][MAP_Y_R][1], int width, int height, colors color);
	//! Function to add elements to the image.
	//!< \param map A 2D array of bools containing the elements to add to the map.
	//!< \param width Integer representing the width of the map.
	//!< \param height Integer representing the height of the map.
	//!< \param colors The color of the elements to be added to the map
	void AddToImage(bool map[MAP_X_R][MAP_Y_R][1], int width, int height, colors color);
	//! Function to print the image.
	//!< \param filename The name that the image have when printed to disk.
	//!< \param width Integer representing the width of the map.
	//!< \param height Integer representing the height of the map.
	void PrintImage(std::string filename, int width, int height);

	std::vector<float> DetermineColor(colors color);

	void CreateUnitLookUpTable();

	//void SpawnEveryUnit();

	//void PrintUnits();

	//void AddObjectiveToIM(sc2::Point2D objective);

private:
	CUDA* cuda;	//do NOT include 
	const sc2::ObservationInterface* observation;
	sc2::DebugInterface* debug;
	sc2::ActionInterface* actions;
	sc2::ActionFeatureLayerInterface* actions_feature_layer;

	bool dynamic_terrain[MAP_X_R][MAP_Y_R][1];	//update on-building-creation, on-building-destruction, on-building-vision

	sc2::Units units;	//update per frame, includes movable units and hostile structures

	//! image is an vector that holds the values representing the map
	std::vector<float> image;
	//! max_value is an float holding the largest, non center value, in the map. Center value of units are usually > 1000, these values are outliers and can be clamped.
	float max_value;
	bool update_terrain;

	//---------------

	////! Width of map
	//int map_x;
	////! Height of map
	//int map_y;
	////! Pathing grid is 8 time larger than what is returned from API. Specifies how much we subdivide the grid.
	//int grid_division;
	////! Map width that is multiplied with pathingGridSize to get actual width of the pathfinding grid
	//int map_x_r;
	////! Map height that is multiplied with pathingGridSize to get actual height of the pathfinding grid
	//int map_y_r;
	
	int kFeatureLayerSize;
	int kPixelDrawSize;
	int kDrawSize;
};
#endif