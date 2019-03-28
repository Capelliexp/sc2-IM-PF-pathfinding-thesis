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
#include "../examples/CUDA/cuda_header.cuh"
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

struct Potential_Field {
	sc2::UnitTypeID sc2_id;
	float map[MAP_X_R][MAP_Y_R][1];
};

class MapStorage {
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
		sc2::ActionFeatureLayerInterface* actions_feature_layer, bool astar);

	void Test();

	std::vector<int> GetUnitsID();
	int GetUnitIDInHostUnitVec(sc2::UnitTypeID unit_id);
	int GetSizeOfUnitInfoList();
	int GetPosOFUnitInHostUnitVec(sc2::UNIT_TYPEID typeID);
	void SetRadiusForUnits(std::vector<float> radius);

	//! Functions to get IM destination for ground.
	//! Will request creation of IM if needed.
	//!< \param pos sc2::Point2D the position to create an IM to.
	//!< \return Returns a reference to the IM, will return nullptr if something went wrong.
	Destination_IM& RequestGroundDestination(sc2::Point2D pos);
	//! Functions to get IM destination for air.
	//! Will request creation of IM if needed.
	//!< \param pos sc2::Point2D the position to create an IM to.
	//!< \return Returns a reference to the IM, will return nullptr if something went wrong.
	Destination_IM& RequestAirDestination(sc2::Point2D pos);

	void UpdateEntityVector(std::vector<Entity>& host_unit_list);

	float GetGroundAvoidancePFValue(int x, int y);

	void CreateAttractingPF(sc2::UnitTypeID unit_id);
	
	float GetAttractingPF(sc2::UnitTypeID unit_id, int x, int y);

	bool GetDynamicMap(int x, int y);

	void TransferPFFromDevice();
	void TransferIMFromDevice();
	void ChangeDeviceDynamicMap(sc2::Point2D center, float radius, int value);
	void ExecuteDeviceJobs();

	void PrintMap(sc2::Point2D pos, int x, int y, std::string name);

	void UpdateIMAtsar();

private:
	//! Craetes the influence map based on the size of the map.
	void CreateIM();

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
	//! Function to check if the file exists
	//!< \param filename String of the filename to check for. Include relevant ending of file (.txt, .png, ...)
	//!< \return Return true if file found.
	bool CheckIfFileExists(std::string filename);
	//! Function to determine color
	//! The function returns a vector holding three floats in a vector. That can be between 0 and 1. It indicates how much of the color that is wanted.
	//!< \param color Enum indicating color.
	//!< \return Returns a vector that indicates if any or all RGB values are wanted.
	std::vector<float> DetermineColor(colors color);

	void CreateUnitLookUpTable();

	void PrintStatus(std::string msg);
	void PrintMap(float map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file);
	void PrintMap(bool map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file);
	void PrintMap(int map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file);

private:
	CUDA* cuda;
	const sc2::ObservationInterface* observation;
	sc2::DebugInterface* debug;
	sc2::ActionInterface* actions;
	sc2::ActionFeatureLayerInterface* actions_feature_layer;

	std::unordered_map<sc2::UNIT_TYPEID, float*> unit_attraction_PF;

	bool dynamic_terrain[MAP_X_R][MAP_Y_R][1];	//update on-building-creation, on-building-destruction, on-building-vision

	//! List of IMs for both ground and air.
	std::list<Destination_IM> destinations_ground_IM;
	std::list<Destination_IM> destinations_air_IM;

	//! PF that indicates what to avoid on ground or in air
	float ground_repelling_PF[MAP_X_R][MAP_Y_R][1]; //WILL BE REPLACED
	float air_repelling_PF[MAP_X_R][MAP_Y_R][1];    //WILL BE REPLACED

	//! List of attracting PFs
	std::list<Potential_Field> attracting_PFs;

	//! image is an vector that holds the values representing the map
	std::vector<float> image;
	//! max_value is an float holding the largest, non center value, in the map. Center value of units are usually > 1000, these values are outliers and can be clamped.
	float max_value;

	std::vector<int> requested_PF;
	std::vector<int> requested_IM;
};
#endif
