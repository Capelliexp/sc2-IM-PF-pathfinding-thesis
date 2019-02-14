#include "../examples/CUDA/map_storage.hpp"

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

void MapStorage::Test() {
    for (int x = 0; x < MAP_X; ++x)
        for (int y = 0; y < MAP_Y; ++y) {
            static_terrain[x][y] = x - y;
            dynamic_terrain[x][y] = x - y;
        }

    units.push_back({ sc2::UNIT_TYPEID::ZERG_HYDRALISK, { 3, 3 }, true });
    units.push_back({ sc2::UNIT_TYPEID::ZERG_HYDRALISK, { 6, 6 }, false});
}

bool MapStorage::CheckIfFileExists(std::string filename)
{
    return std::filesystem::exists(filename);
}
