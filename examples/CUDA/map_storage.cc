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
    max_value = 0;
}

MapStorage::~MapStorage() {
    delete cuda;
}

void MapStorage::Initialize(const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions,
    sc2::ActionFeatureLayerInterface* actions_feature_layer) {
    this->observation = observations;
    this->debug = debug;
    this->actions = actions;
    this->actions_feature_layer = actions_feature_layer;

    CreateIM();
    //PrintIM();

    cuda = new CUDA();
    cuda->InitializeCUDA(observations, debug, actions);
    CreateUnitLookUpTable();
    cuda->HostTransfer(units);
    cuda->DeviceTransfer(dynamic_terrain);
    cuda->Tests(ground_avoidance_PF, air_avoidance_PF);

    cuda->UpdateDynamicMap({ 10, 10 }, 4, false);

    PrintMap(ground_avoidance_PF, MAP_X_R, MAP_Y_R, "ground");
    PrintMap(air_avoidance_PF, MAP_X_R, MAP_Y_R, "air");

    PrintMap(dynamic_terrain, MAP_X_R, MAP_Y_R, "dynamic");
    CreateImage(dynamic_terrain, MAP_X_R, MAP_Y_R);
    AddToImage(ground_avoidance_PF, MAP_X_R, MAP_Y_R, colors::BLUE);
    PrintImage("image.png", MAP_X_R, MAP_Y_R);
    //CreateImage2(dynamic_terrain, MAP_X_R, MAP_Y_R, "image.png");

    Destination_IM map;
    map.destination = {28,28};
    map.air_path = false;
    cuda->IMGeneration(IntPoint2D{ (integer)map.destination.x, (integer)map.destination.y }, map.map, false);
    //Add the map to list.
    PrintMap(map.map, MAP_X_R, MAP_Y_R, "IM");
    CreateImage(map.map, MAP_X_R, MAP_Y_R);
    PrintImage("res.png", MAP_X_R, MAP_Y_R);
}

void MapStorage::Test() {
    //units.push_back({ sc2::UNIT_TYPEID::TERRAN_HELLION, { 20, 20 }, true });
    //units.push_back({ sc2::UNIT_TYPEID::TERRAN_HELLION, { 6, 6 }, false});
}

//---------------------------------------------------------------

//! Prints the given string to the console.
//!< \param msg The message to be printed to the console.
void MapStorage::PrintStatus(std::string msg) {
    std::cout << "map_storage: " << msg << std::endl;
}

void MapStorage::PrintMap(float map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file) {
    std::ofstream out(file + ".txt");
    int width = x;
    for (int i = 0; i < y; i++)
    {
        for (int j = 0; j < width; j++) out << map[i][j][0] << ", ";
        out << std::endl;
    }
    out.close();
}

void MapStorage::PrintMap(bool map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file) {
    std::ofstream out(file + ".txt");
    int width = x;
    for (int i = 0; i < y; i++)
    {
        for (int j = 0; j < width; j++) out << map[i][j][0] << ", ";
        out << std::endl;
    }
    out.close();
}

void MapStorage::PrintMap(int map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file) {
    std::ofstream out(file + ".txt");
    int width = x;
    for (int i = 0; i < y; i++)
    {
        for (int j = 0; j < width; j++) out << map[i][j][0] << ", ";
        out << std::endl;
    }
    out.close();
}

void MapStorage::Update(clock_t dt) {
    cuda->Update(dt, units, ground_avoidance_PF, air_avoidance_PF);
}

std::vector<int> MapStorage::GetUnitsID() {
    return cuda->GetUnitsID();
}

int MapStorage::GetSizeOfUnitInfoList() {
    return cuda->GetSizeOfUnitInfoList();
}

int MapStorage::GetPosOFUnitInHostUnitVec(sc2::UNIT_TYPEID typeID) {
    return cuda->GetPosOFUnitInHostUnitVec(typeID);
}

void MapStorage::SetRadiusForUnits(std::vector<float> radius) {
    cuda->SetRadiusForUnits(radius);
}

//Encode from raw pixels to an in-memory PNG file first, then write it to disk
//The image argument has width * height RGBA pixels or width * height * 4 bytes
// std::vector<unsigned char>& image
void MapStorage::CreateImage(bool map[MAP_X_R][MAP_Y_R][1], int width, int height) {
    image.resize(width * height * 4);
    for (unsigned y = 0; y < height; y++)
        for (unsigned x = 0; x < width; x++) {
            float mapP = map[x][y][0];
            image[4 * width * y + 4 * x + 0] = mapP;
            image[4 * width * y + 4 * x + 1] = mapP;
            image[4 * width * y + 4 * x + 2] = mapP;
            image[4 * width * y + 4 * x + 3] = 255;
        }
}

void MapStorage::CreateImage(float map[MAP_X_R][MAP_Y_R][1], int width, int height) {
    image.resize(width * height * 4);
    for (unsigned y = 0; y < height; y++)
        for (unsigned x = 0; x < width; x++) {
            float mapP = map[x][y][0];
            if (mapP < 32767)
                max_value = max(max_value, mapP);
            image[4 * width * x + 4 * y + 0] = mapP;
            image[4 * width * x + 4 * y + 1] = mapP;
            image[4 * width * x + 4 * y + 2] = mapP;
            image[4 * width * x + 4 * y + 3] = 255;
        }
}

void MapStorage::AddToImage(float map[MAP_X_R][MAP_Y_R][1], int width, int height, colors color) {
    std::vector<float> selected_color = DetermineColor(color);
    //! Can be optimized to only loop over the area that is affected. Need a radius parameter
    for (unsigned y = 0; y < height; y++)
        for (unsigned x = 0; x < width; x++) {
            //If the tile is unpathebale don't write to it.
            if (image[4 * width * y + 4 * x + 0] == 0 && image[4 * width * y + 4 * x + 1] == 0 && image[4 * width * y + 4 * x + 2] == 0) continue;
            float mapP = map[x][y][0];
            //! The thing that is to be added, doesn't affect the tile. Continue to next iteration
            if (mapP == 0) continue;
            if (mapP < 200)
                max_value = max(max_value, mapP);
            image[4 * width * y + 4 * x + 0] += selected_color[0] * mapP;
            image[4 * width * y + 4 * x + 1] += selected_color[1] * mapP;
            image[4 * width * y + 4 * x + 2] += selected_color[2] * mapP;
            image[4 * width * y + 4 * x + 3] = 255;
        }
}

void MapStorage::AddToImage(bool map[MAP_X_R][MAP_Y_R][1], int width, int height, colors color) {
    for (unsigned y = 0; y < height; y++)
        for (unsigned x = 0; x < width; x++) {
            //If the pixel is unpathebale don't write to it.
            if (image[4 * width * y + 4 * x + 0] == 0 && image[4 * width * y + 4 * x + 1] == 0 && image[4 * width * y + 4 * x + 2] == 0) continue;
            float mapP = map[x][y][0];
            std::vector<float> selected_color = DetermineColor(color);
            image[4 * width * y + 4 * x + 0] = selected_color[0] * mapP;
            image[4 * width * y + 4 * x + 1] = selected_color[1] * mapP;
            image[4 * width * y + 4 * x + 2] = selected_color[2] * mapP;
            image[4 * width * y + 4 * x + 3] = 255;
        }
}

void MapStorage::PrintImage(std::string filename, int width, int height) {
    //Encode the image
    std::vector<unsigned char> printImage(width * height * 4);
    for (int i = 0; i < image.size(); i+=4) {
        float i0 = min(image[i + 0], max_value);
        float i1 = min(image[i + 1], max_value);
        float i2 = min(image[i + 2], max_value);

        if (i0 == 0 && i2 == 0 && i2 == 0)
            i0 = i1 = i2 = 0;
        else if (i0 == 1 && i2 == 1 && i2 == 1)
            i0 = i1 = i2 = 255;
        else {
            i0 = i0 <= 1 ? 0 : 255 * (1 - (max_value - i0) / max_value);
            i1 = i1 <= 1 ? 0 : 255 * (1 - (max_value - i1) / max_value);
            i2 = i2 <= 1 ? 0 : 255 * (1 - (max_value - i2) / max_value);
        }
        printImage[i + 0] = i0;
        printImage[i + 1] = i1;
        printImage[i + 2] = i2;
        printImage[i + 3] = image[i + 3];

    }
    unsigned error = lodepng::encode(filename, printImage, width, height);

    //if there's an error, display it
    if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

std::vector<float> MapStorage::DetermineColor(colors color) {
    std::vector<float> selected_color{0.0, 0.0, 0.0};
    switch (color)
    {
    case MapStorage::RED:
        selected_color[0] = 1.0;
        break;
    case MapStorage::GREEN:
        selected_color[1] = 1.0;
        break;
    case MapStorage::BLUE:
        selected_color[2] = 1.0;
        break;
    case MapStorage::YELLOW:
        selected_color[0] = 1.0;
        selected_color[1] = 1.0;
        break;
    case MapStorage::PURPLE:
        selected_color[0] = 1.0;
        selected_color[2] = 1.0;
        break;
    case BLACK:
        break;
    default:
        break;
    }
    return selected_color;
}

void MapStorage::CreateUnitLookUpTable() {
    std::string file = "unitInfo.txt";
    if (CheckIfFileExists(file))
        cuda->ReadUnitInfoFromFile(file);
    else {
        cuda->CreateUnitLookupOnHost(file);
    }
}

//! The bot is abdle to print its IM to a file.
//void MapStorage::PrintIM()
//{
//    std::stringstream str(std::stringstream::out | std::stringstream::binary);
//    for (int y = map_y_r - 1; y >= 0; --y)
//    {
//        for (int x = 0; x < map_x_r; ++x)
//        {
//            str << InfluenceMap[x + y * map_x_r] << " ";
//        }
//        str << std::endl;
//    }
//    std::ofstream file;
//    file.open("InfluenceMap.txt", std::ofstream::binary);
//    PrintStatus("File open and printing to file");
//    file.write(str.str().c_str(), str.str().length());
//    file.close();
//    PrintStatus("File closed");
//}

Destination_IM * MapStorage::CheckGroundDestination(sc2::Point2D pos) {
    Destination_IM* destination = nullptr;
    for (Destination_IM dest : destinations_ground_IM)
        if (dest.destination == pos) {
            destination = &dest;
            break;
        }
    //if (destination == nullptr)
        //Create IM
    return destination;
}

Destination_IM * MapStorage::CheckAirDestination(sc2::Point2D pos) {
    Destination_IM* destination = nullptr;
    for (Destination_IM dest : destinations_air_IM)
        if (dest.destination == pos) {
            destination = &dest;
            break;
        }
    //if (destination == nullptr)
        //Create IM
    return destination;
}

Destination_IM * MapStorage::GetGroundDestination(sc2::Point2D pos) {
    //Call cuda to create IM
    for (Destination_IM dest : destinations_ground_IM) {
        if (dest.destination == pos)
            return &dest;
    }
    Destination_IM map;
    map.destination = pos;
    map.air_path = false;
    cuda->IMGeneration(IntPoint2D{ (integer)pos.x, (integer)pos.y }, map.map, false);
    //Add the map to list.
    PrintMap(map.map, MAP_X_R, MAP_Y_R, "IM");
    CreateImage(map.map, MAP_X_R, MAP_Y_R);
    PrintImage("res.png", MAP_X_R, MAP_Y_R);
    return &map;
}

Destination_IM * MapStorage::GetAirDestination(sc2::Point2D pos) {
    //Call cuda to create IM

    //Add the map to list.
    return nullptr;
}

//! Craetes the influence map based on the size of the map.
void MapStorage::CreateIM() {
    std::string IM = observation->GetGameInfo().pathing_grid.data;    //here
    //The multiplication is done due to that the returned pathing grid is the wrong size. It is the same size as placement grid.
    /*width = observation->GetGameInfo().pathing_grid.width * GRID_DIVISION;
    height = observation->GetGameInfo().pathing_grid.height * GRID_DIVISION;*/

    //InfluenceMap = std::vector<float>(width*height);

    //Fill a 8x8 cube with the same value
    for (int i = 0; i < MAP_Y; ++i)
    {
        int iP = i * GRID_DIVISION;
        for (int j = 0; j < MAP_X; ++j)
        {
            int jP = j * GRID_DIVISION;
            for (int y = 0; y < GRID_DIVISION; ++y)
            {
                int yp = (y + iP);
                for (int x = 0; x < GRID_DIVISION; ++x)
                {
                    int xp = x + jP;
                    //InfluenceMap[xp + yp] = (IM[j + i * width / GRID_DIVISION] == -1) ? 0 : 1;
                    dynamic_terrain[yp][xp][0] = (IM[j + i * MAP_X] == -1) ? 0 : 1;
                    //dynamic_terrain[xp][MAP_Y_R - yp][0] = (IM[j + i * MAP_X] == -1) ? 0 : 1;
                    //dynamic_terrain[yp][xp][0] = xp + yp*10;
                    //Should maybe xp + yp * map_x
                }
            }
        }
    }
}

//! Function that is used to add a list of units to the IM.
//!< \param units The list of units to be added.
//void MapStorage::IMAddUnits(sc2::Units units)
//{
//    for (const auto& unit : units)
//    {
//        IMAddUnit(unit);
//    }
//}

//! Function that is used to add an unit to the IM.
//! Uses radius to indicate which tiles that can't be pathed.
//!< \param unit The unit to be added.
//void MapStorage::IMAddUnit(const sc2::Unit* unit)
//{
//    int d = int(unit->radius * 2) * grid_division;  //How many squares does the building occupy in one direction?
//    int rr = pow(unit->radius * grid_division, 2);
//    int xc = unit->pos.x * grid_division;
//    int yc = unit->pos.y * grid_division;
//    int startX = xc - d / 2;
//    int startY = yc - d / 2;
//
//    for (int y = 0; y < d; ++y)
//    {
//        int yp = pow(y - d / 2, 2);
//        for (int x = 0; x < d; ++x)
//        {
//            float dd = pow(x - d / 2, 2) + yp;
//            InfluenceMap[startX + x + (startY + y) * width] = (dd < rr) ? 0 : 1;
//        }
//    }
//}

//! Function that is used to remove an unit from the IM.
//! We know that the tiles that the building occupied can be pathed now.
//! No need to calculate the radius.
//!< \param unit The unit to be removed.
//void MapStorage::IMRemoveUnit(const sc2::Unit * unit)
//{
//    int d = int(unit->radius * 2) * pathingGridSize;  //How many squares does the building occupy in one direction?
//    int xc = unit->pos.x * pathingGridSize;
//    int yc = unit->pos.y * pathingGridSize;
//    int startX = xc - d / 2;
//    int startY = yc - d / 2;
//
//    for (int y = 0; y < d; ++y)
//    {
//        int yp = (startY + y) * width;
//        for (int x = 0; x < d; ++x)
//        {
//            InfluenceMap[startX + x + yp] = 0;
//        }
//    }
//}

//! Function that is used to check if a given unit is a structure.
//!< \param unit The unit to be checked.
//!< \return Returns true if the unit is a structure, false otherwise.
bool MapStorage::IsStructure(const sc2::Unit * unit) {
    auto& attributes = observation->GetUnitTypeData().at(unit->unit_type).attributes; //here
    bool is_structure = false;
    for (const auto& attribute : attributes) {
        if (attribute == sc2::Attribute::Structure) {
            is_structure = true;
        }
    }
    return is_structure;
}

//void MapStorage::AddObjectiveToIM(sc2::Point2D objective)
//{
//    PrintStatus("Started adding objective.");
//    int start = 2 * pathingGridSize;
//    double start_s = clock();
//    int ww = width * 1.5;
//    for (int y = start; y < height - 2; ++y)
//    {
//        int yp = pow(y - objective.y * pathingGridSize, 2);
//        for (int x = start; x < width - 2; ++x)
//        {
//            if (InfluenceMap[x + y * width] == 1)
//            {
//                int xp = pow(x - objective.x * pathingGridSize, 2);
//                InfluenceMap[x + y * width] = ww - sqrt(xp + yp);
//            }
//        }
//    }
//    double stop_s = clock();
//    PrintStatus("Objective added. Took " + std::to_string((stop_s - start_s) / 1000));
//}

bool MapStorage::CheckIfFileExists(std::string filename) {
    return std::filesystem::exists(filename);
}
