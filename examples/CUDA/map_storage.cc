#include "../examples/CUDA/map_storage.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <iostream>
#include <filesystem>
#include "../tools.h"

MapStorage::MapStorage() {
}

MapStorage::~MapStorage() {
    if (cuda != NULL)
        delete cuda;
}

void MapStorage::Initialize(const sc2::ObservationInterface* observations, sc2::DebugInterface* debug, sc2::ActionInterface* actions,
    sc2::ActionFeatureLayerInterface* actions_feature_layer, bool astar, bool astarPF) {
    this->observation = observations;
    this->debug = debug;
    this->actions = actions;
    this->actions_feature_layer = actions_feature_layer;
    this->max_value = 0;
    CreateIM();

    cuda = nullptr;

	if (!astar) {
        PrintMemoryUsage("CudaInit pre");
		cuda = new CUDA();
		cuda->InitializeCUDA(observations, debug, actions, ground_repelling_PF, air_repelling_PF);
		CreateUnitLookUpTable();
		cuda->AllocateDeviceMemory();
        cuda->SpecifyDeviceFunctionAttributes();
        if (!astarPF)
		    cuda->DeviceTransferDynamicMap(dynamic_terrain);
        cuda->DeviceTransferUnitLookup();
		cuda->BindRepellingMapsToTransferParams();
		cuda->Tests(ground_repelling_PF, air_repelling_PF);

        //PrintMap(ground_repelling_PF, MAP_X_R, MAP_Y_R, "ground");
        //PrintMap(air_repelling_PF, MAP_X_R, MAP_Y_R, "air");
        //PrintMap(dynamic_terrain, MAP_X_R, MAP_Y_R, "dynamic");

        //CreateImage(dynamic_terrain, MAP_X_R, MAP_Y_R);
        //max_value = 1;
        //PrintImage("dynamic.png", MAP_X_R, MAP_Y_R);

        PrintMemoryUsage("CudaInit post");
    }
}

void MapStorage::Reset() {
    this->max_value = 0;
    CreateIM();
    if (cuda != NULL)
        cuda->Reset();
    image.clear();
    requested_PF.clear();
    requested_IM.clear();
    attracting_PFs.resize(0);
    destinations_ground_IM.resize(0);
    destinations_air_IM.resize(0);
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
        for (int j = 0; j < width; j++) out << map[i][j][0] << ",";
        out << std::endl;
    }
    out.close();
}

void MapStorage::PrintMap(bool map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file) {
    std::ofstream out(file + ".txt");
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            out << map[i][j][0] << ",";
        }
        out << std::endl;
    }
    out.close();
}

void MapStorage::PrintMap(int map[MAP_X_R][MAP_Y_R][1], int x, int y, std::string file) {
    std::ofstream out(file + ".txt");
    int width = x;
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < width; j++) out << map[i][j][0] << ",";
        out << std::endl;
    }
    out.close();
}

void MapStorage::PrintCUDAMemoryUsage(std::string location){
    cuda->PrintDeviceMemoryUsage(location);
}

void MapStorage::PrintMap(sc2::Point2D pos, int x, int y, std::string name) {
    for (auto& d : destinations_ground_IM) {
        if (d.destination == pos) {
            PrintMap(d.map, x, y, name);
            CreateImage(d.map, x, y, colors::GREEN);
            PrintImage(name + ".png", x, y);
            return;
        }
    }
    for (auto& d : destinations_air_IM) {
        if (d.destination == pos) {
            PrintMap(d.map, x, y, name);
            CreateImage(d.map, x, y, colors::GREEN);
            PrintImage(name + ".png", x, y);
            return;
        }
    }
}

void MapStorage::PrintGroundPF(std::string name) {
    std::ofstream out(name + ".txt");
    for (int i = 0; i < MAP_Y_R; i++) {
        for (int j = 0; j < MAP_X_R; j++)
            out << ground_repelling_PF[i][j][0] << ",";
        out << std::endl;
    }
    out.close();
}

void MapStorage::CreateImage(sc2::Point2D pos, int x, int y, std::string name) {
    for (auto& d : destinations_ground_IM) {
        if (d.destination == pos) {
            PrintMap(d.map, x, y, name);
            CreateImage(d.map, x, y, colors::GREEN);
            return;
        }
    }
    for (auto& d : destinations_air_IM) {
        if (d.destination == pos) {
            PrintMap(d.map, x, y, name);
            CreateImage(d.map, x, y, colors::GREEN);
            return;
        }
    }
}

void MapStorage::CreateImageDynamic() {
    CreateImage(dynamic_terrain, MAP_X_R, MAP_Y_R);
    max_value = 255;
}

void MapStorage::AddPathToImage(std::vector<sc2::Point2D> path, colors color) {
    std::vector<float> selected_color = DetermineColor(color);
    for (int i = 0; i < path.size(); ++i) {
        image[4 * MAP_X_R * (int)path[i].y + 4 * (int)path[i].x + 0] = selected_color[0] * 255;
        image[4 * MAP_X_R * (int)path[i].y + 4 * (int)path[i].x + 1] = selected_color[1] * 255;
        image[4 * MAP_X_R * (int)path[i].y + 4 * (int)path[i].x + 2] = selected_color[2] * 255;
    }
}

void MapStorage::PrintImage(int x, int y, std::string name) {
    PrintImage(name + ".png", x, y);
}

std::vector<int> MapStorage::GetUnitsID() {
    return cuda->GetUnitsID();
}

int MapStorage::GetUnitIDInHostUnitVec(sc2::UnitTypeID unit_id) {
    return cuda->GetUnitIDInHostUnitVec(unit_id);
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
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            float mapP = map[y][x][0];
            image[4 * width * y + 4 * x + 0] = mapP;
            image[4 * width * y + 4 * x + 1] = mapP;
            image[4 * width * y + 4 * x + 2] = mapP;
            image[4 * width * y + 4 * x + 3] = 255;
        }
}

void MapStorage::CreateImage(float map[MAP_X_R][MAP_Y_R][1], int width, int height, colors color) {
    std::vector<float> selected_color = DetermineColor(color);
    image.resize(width * height * 4);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            float mapP = map[x][y][0];
            if (mapP < 32767)
                max_value = max(max_value, mapP);
            if (mapP == -2) {
                image[4 * width * x + 4 * y + 0] = mapP;
                image[4 * width * x + 4 * y + 1] = mapP;
                image[4 * width * x + 4 * y + 2] = mapP;
                image[4 * width * x + 4 * y + 3] = 255;
            }
            else if (mapP == 0) {
                image[4 * width * x + 4 * y + 0] = 255;
                image[4 * width * x + 4 * y + 1] = 0;
                image[4 * width * x + 4 * y + 2] = 0;
                image[4 * width * x + 4 * y + 3] = 255;
            }
            else {
                image[4 * width * x + 4 * y + 0] = selected_color[0] * mapP;
                image[4 * width * x + 4 * y + 1] = selected_color[1] * mapP;
                image[4 * width * x + 4 * y + 2] = selected_color[2] * mapP;
                image[4 * width * x + 4 * y + 3] = 255;
            }
        }
}

void MapStorage::AddToImage(float map[MAP_X_R][MAP_Y_R][1], int width, int height, colors color) {
    std::vector<float> selected_color = DetermineColor(color);
    //! Can be optimized to only loop over the area that is affected. Need a radius parameter
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
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
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
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
        float i0 = image[i + 0] == 255 ? 255 : min(image[i + 0], max_value);
        float i1 = image[i + 1] == 255 ? 255 : min(image[i + 1], max_value);
        float i2 = image[i + 2] == 255 ? 255 : min(image[i + 2], max_value);

        if (i0 == -2 && i1 == -2 && i2 == -2)   //Can't walk here
            i0 = i1 = i2 = 0;
        else if (i0 == -1 && i1 == -1 && i2 == -1) //Can walk here
            i0 = i1 = i2 = 255;
        else {  //Objective or unit
            if (i0 != 0 && i1 == 0 && i2 == 0) {    //Red pixels
                if (i0 != 255)
                    i1 = i2 = 255 * (1 - (max_value - i0) / max_value);
                i0 = 255;
            }
            else if (i0 == 0 && i1 != 0 && i2 == 0) {   //Green pixels
                if (i1 != 255)
                    i0 = i2 = 255 * (1 - (max_value - i1) / max_value);
                i1 = 255;
            }
            else if (i0 == 0 && i1 == 0 && i2 != 0) {   //Blue pixels
                if (i2 != 255)
                    i0 = i1 = 255 * (1 - (max_value - i2) / max_value);
                i2 = 255;
            }
            else if ((i0 != 0 && i1 != 0 && i2 != 0) || (i0 == 1 && i1 == 1 && i2 == 1)) {   //White pixels
                if (i0 == 255 || i0 == 1) {
                    i0 = 255;
                    i1 = 255;
                    i2 = 255;
                }
                else {
                    i0 = 255 * (1 - (max_value - i0) / max_value);
                    i1 = 255 * (1 - (max_value - i1) / max_value);
                    i2 = 255 * (1 - (max_value - i2) / max_value);
                }
            }
            else if (i0 != 0 && i1 != 0 && i2 == 0) {   //Red and Green pixels  Yellow
                if (i0 != 255)
                    i2 = 255 * (1 - (max_value - i0) / max_value);
                i0 = i1 = 255;
            }
            else if (i0 != 0 && i1 == 0 && i2 != 0) {   //Red and Blue pixels   Purple
                if (i0 != 255)
                    i1 = 255 * (1 - (max_value - i0) / max_value);
                i0 = i2 = 255;
            }
            else if (i0 == 0 && i1 != 0 && i2 != 0) {   //Green and Blue pixels Cyan
                if (i1 != 255)
                    i0 = 255 * (1 - (max_value - i1) / max_value);
                i1 = i2 = 255;
            }
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
    case WHITE:
        selected_color[0] = 1.0;
        selected_color[1] = 1.0;
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

Destination_IM & MapStorage::RequestGroundDestination(sc2::Point2D pos) {
    for (auto& dest : destinations_ground_IM) {
        if (dest.destination == pos)
            return dest;
    }
    Destination_IM map;
    destinations_ground_IM.push_back(map);
    destinations_ground_IM.back().destination = pos;
    destinations_ground_IM.back().air_path = false;
    requested_IM.push_back(cuda->QueueDeviceJob({ (integer)pos.x, (integer)pos.y }, false, (float*)destinations_ground_IM.back().map));
    return destinations_ground_IM.back();
}

Destination_IM & MapStorage::RequestAirDestination(sc2::Point2D pos) {
    for (Destination_IM dest : destinations_air_IM) {
        if (dest.destination == pos)
            return dest;
    }
    destinations_air_IM.push_back({});
    destinations_air_IM.back().destination = pos;
    destinations_air_IM.back().air_path = true;
    requested_IM.push_back(cuda->QueueDeviceJob({ (integer)pos.x, (integer)pos.y }, true, (float*)destinations_air_IM.back().map));
    return destinations_air_IM.back();
}

void MapStorage::UpdateEntityVector(std::vector<Entity>& host_unit_list) {
    cuda->SetHostUnitList(host_unit_list);
    cuda->TransferUnitsToDevice();
}

float MapStorage::GetGroundAvoidancePFValue(int x, int y) {
    if (ground_repelling_PF[x][y][0] < 0)
        return 0;
    return ground_repelling_PF[x][y][0];
}

void MapStorage::CreateAttractingPF(sc2::UnitTypeID unit_id) {
    std::list<Potential_Field>::iterator it = attracting_PFs.begin();
    for (; it != attracting_PFs.end(); it++) {
        if (it->sc2_id == unit_id) {
            requested_PF.push_back(cuda->QueueDeviceJob(cuda->GetUnitIDInHostUnitVec(unit_id), (float*)it->map));
            return;
        }
    }

    attracting_PFs.push_back({});
    attracting_PFs.back().sc2_id = unit_id;
    requested_PF.push_back(cuda->QueueDeviceJob(cuda->GetUnitIDInHostUnitVec(unit_id), (float*)attracting_PFs.back().map));
}

void MapStorage::ExecuteDeviceJobs(bool astarPF) {
    if (!astarPF)
        cuda->ExecuteDeviceJobs();
    else
        cuda->ExecuteDeviceJobs(PFType::Large);
    //PrintMap(ground_repelling_PF, MAP_X_R, MAP_Y_R, "PF");
}

void MapStorage::UpdateIMAtsar() {
    CreateIM();
}

float MapStorage::GetUnitGroundWeaponRange(sc2::UnitTypeID sc2_unit_id) {
    return cuda->GetUnitGroundWeaponRange(sc2_unit_id);
}

void MapStorage::TransferPFFromDevice() {
    for (int i = 0; i < requested_PF.size(); ++i) {
        Result res = cuda->TransferMapToHost(requested_PF[i]);
        if (res == Result::OK) {
            requested_PF.erase(requested_PF.begin() + i);
            //PrintMap(attracting_PFs.back().map, MAP_X_R, MAP_Y_R, "attarctingPF");
        }
    }
}

void MapStorage::TransferIMFromDevice() {
    for (int i = 0; i < requested_IM.size(); ++i) {
        Result res = cuda->TransferMapToHost(requested_IM[i]);
        if (res == Result::OK) {
            requested_IM.erase(requested_IM.begin() + i);
        }
    }
}

void MapStorage::ChangeDeviceDynamicMap(sc2::Point2D center, float radius, int value){
    cuda->UpdateDynamicMap({ (integer)center.x, (integer)center.y }, radius, value);
}

float MapStorage::GetAttractingPF(sc2::UnitTypeID unit_id, int x, int y) {
    for (auto& it : attracting_PFs) {
        if (it.sc2_id == unit_id)
            return it.map[x][y][0];
    }
    return 0;
}

bool MapStorage::GetDynamicMap(int x, int y) {
    bool ret = dynamic_terrain[x][y][0];
    return ret;
}

//! Creates the influence map based on the size of the map.
void MapStorage::CreateIM() {
    std::string IM = observation->GetGameInfo().pathing_grid.data;
    //Fill a 8x8 cube with the same value or what GRID_DIVISION have for value, max 8x8
    for (int i = 0; i < MAP_Y; ++i) {
        int iP = i * GRID_DIVISION;
        for (int j = 0; j < MAP_X; ++j) {
            int jP = j * GRID_DIVISION;
            for (int y = 0; y < GRID_DIVISION; ++y) {
                int yp = (y + iP);
                for (int x = 0; x < GRID_DIVISION; ++x) {
                    int xp = x + jP;
                    dynamic_terrain[xp][yp][0] = (IM[i + j * MAP_X] == -1) ? 0 : 1;
                }
            }
        }
    }
}

bool MapStorage::CheckIfFileExists(std::string filename) {
    return std::filesystem::exists(filename);
    //return false;
}
