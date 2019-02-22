#include "../examples/CUDA/map_storage.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <iostream>

#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

MapStorage::MapStorage() {
    //unit_attraction_PF.reserve(bot_faction_unit_type_count);
    update_terrain = true;
}

MapStorage::~MapStorage() {

}

void MapStorage::Initialize(const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions,
    sc2::ActionFeatureLayerInterface* actions_feature_layer){
    this->observation = observations;
    this->debug = debug;
    this->actions = actions;
    this->actions_feature_layer = actions_feature_layer;
}

void MapStorage::Test() {
    for (int x = 0; x < MAP_X_R; ++x)
        for (int y = 0; y < MAP_Y_R; ++y) {
            static_terrain[x][y] = x - y;
            dynamic_terrain[x][y] = x - y;
        }

    units.push_back({ sc2::UNIT_TYPEID::TERRAN_HELLION, { 50, 50 }, true });
    //units.push_back({ sc2::UNIT_TYPEID::TERRAN_HELLION, { 6, 6 }, false});
}

//---------------------------------------------------------------

//! Prints the given string to the console.
//!< \param msg The message to be printed to the console.
void MapStorage::PrintStatus(std::string msg)
{
    //int64_t bot_identifier = int64_t(this) & 0xFFFLL;
    //std::cout << std::to_string(bot_identifier) << ": " << msg << std::endl;

    std::cout << "map_storage: " << msg << std::endl;
}

//void MapStorage::PrintMap(sc2::ImageData map, std::string file)
//{
//    PrintStatus("Map height: " + std::to_string(map.height));
//    PrintStatus("Map width: " + std::to_string(map.width));
//    std::ofstream out(file + ".txt");
//    int width = map.width;
//    for (int i = 0; i < map.height; i++)
//    {
//        for (int j = 0; j < width; j++)
//        {
//            if (map.data[j + i * width] == 0)
//                out << 0;
//            else
//                out << 1;
//        }
//        out << std::endl;
//    }
//    out.close();
//}

void MapStorage::PrintMap(float map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file)
{
    std::ofstream out(file + ".txt");
    int width = x;
    for (int i = 0; i < y; i++)
    {
        for (int j = 0; j < width; j++) out << map[i][j][0] << ", ";
        out << std::endl;
    }
    out.close();
}

void MapStorage::CreateImage(float map[MAP_X_R][MAP_Y_R][1], int width, int height, std::string file)
{
    //Encode the image
    unsigned error = lodepng::encode(file, image, width, height);

    //if there's an error, display it
    if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

//! The bot is abdle to print its IM to a file.
void MapStorage::PrintIM()
{
    std::stringstream str(std::stringstream::out | std::stringstream::binary);
    for (int y = height - 1; y >= 0; --y)
    {
        for (int x = 0; x < width; ++x)
        {
            str << InfluenceMap[x + y * width] << " ";
        }
        str << std::endl;
    }
    std::ofstream file;
    file.open("InfluenceMap.txt", std::ofstream::binary);
    PrintStatus("File open and printing to file");
    file.write(str.str().c_str(), str.str().length());
    file.close();
    PrintStatus("File closed");
}

//! Craetes the influence map based on the size of the map.
void MapStorage::CreateIM()
{
    std::string IM = observation->GetGameInfo().pathing_grid.data;    //here
    //The multiplication is done due to that the returned pathing grid is the wrong size. It is the same size as placement grid.
    width = observation->GetGameInfo().pathing_grid.width * pathingGridSize;
    height = observation->GetGameInfo().pathing_grid.height * pathingGridSize;

    InfluenceMap = std::vector<float>(width*height);

    //Fill a 8x8 cube with the same value
    for (int i = 0; i < height / pathingGridSize; ++i)
    {
        int iP = i * pathingGridSize;
        for (int j = 0; j < width / pathingGridSize; ++j)
        {
            int jP = j * pathingGridSize;
            for (int y = 0; y < pathingGridSize; ++y)
            {
                int yp = (y + iP) * width;
                for (int x = 0; x < pathingGridSize; ++x)
                {
                    int xp = x + jP;
                    InfluenceMap[xp + yp] = (IM[j + i * width / pathingGridSize] == -1) ? 0 : 1;
                }
            }
        }
    }
}

//! Function that is used to add a list of units to the IM.
//!< \param units The list of units to be added.
void MapStorage::IMAddUnits(sc2::Units units)
{
    for (const auto& unit : units)
    {
        IMAddUnit(unit);
    }
}

//! Function that is used to add an unit to the IM.
//! Uses radius to indicate which tiles that can't be pathed.
//!< \param unit The unit to be added.
void MapStorage::IMAddUnit(const sc2::Unit* unit)
{
    int d = int(unit->radius * 2) * pathingGridSize;  //How many squares does the building occupy in one direction?
    int rr = pow(unit->radius * pathingGridSize, 2);
    int xc = unit->pos.x * pathingGridSize;
    int yc = unit->pos.y * pathingGridSize;
    int startX = xc - d / 2;
    int startY = yc - d / 2;

    for (int y = 0; y < d; ++y)
    {
        int yp = pow(y - d / 2, 2);
        for (int x = 0; x < d; ++x)
        {
            float dd = pow(x - d / 2, 2) + yp;
            InfluenceMap[startX + x + (startY + y) * width] = (dd < rr) ? 0 : 1;
        }
    }
}

//! Function that is used to remove an unit from the IM.
//! We know that the tiles that the building occupied can be pathed now.
//! No need to calculate the radius.
//!< \param unit The unit to be removed.
void MapStorage::IMRemoveUnit(const sc2::Unit * unit)
{
    int d = int(unit->radius * 2) * pathingGridSize;  //How many squares does the building occupy in one direction?
    int xc = unit->pos.x * pathingGridSize;
    int yc = unit->pos.y * pathingGridSize;
    int startX = xc - d / 2;
    int startY = yc - d / 2;

    for (int y = 0; y < d; ++y)
    {
        int yp = (startY + y) * width;
        for (int x = 0; x < d; ++x)
        {
            InfluenceMap[startX + x + yp] = 0;
        }
    }
}

//! Function that is used to check if a given unit is a structure.
//!< \param unit The unit to be checked.
//!< \return Returns true if the unit is a structure, false otherwise.
bool MapStorage::IsStructure(const sc2::Unit * unit)
{
    auto& attributes = observation->GetUnitTypeData().at(unit->unit_type).attributes; //here
    bool is_structure = false;
    for (const auto& attribute : attributes) {
        if (attribute == sc2::Attribute::Structure) {
            is_structure = true;
        }
    }
    return is_structure;
}

void MapStorage::AddObjectiveToIM(sc2::Point2D objective)
{
    PrintStatus("Started adding objective.");
    int start = 2 * pathingGridSize;
    double start_s = clock();
    int ww = width * 1.5;
    for (int y = start; y < height - 2; ++y)
    {
        int yp = pow(y - objective.y * pathingGridSize, 2);
        for (int x = start; x < width - 2; ++x)
        {
            if (InfluenceMap[x + y * width] == 1)
            {
                int xp = pow(x - objective.x * pathingGridSize, 2);
                InfluenceMap[x + y * width] = ww - sqrt(xp + yp);
            }
        }
    }
    double stop_s = clock();
    PrintStatus("Objective added. Took " + std::to_string((stop_s - start_s) / 1000));
}

bool MapStorage::CheckIfFileExists(std::string filename)
{
    return std::filesystem::exists(filename);
}
