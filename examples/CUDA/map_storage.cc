#include "../examples/CUDA/map_storage.hpp"

MapStorage::MapStorage() {
    unit_attraction_PF.reserve(bot_faction_unit_type_count);
    update_terrain = true;
}

MapStorage::~MapStorage() {

}

void MapStorage::Test() {
    for (int x = 0; x < map_x; ++x)
        for (int y = 0; y < map_y; ++y) {
            static_terrain[x][y] = x - y;
            dynamic_terrain[x][y] = x - y;
        }

    units.push_back({ true, true, sc2::UNIT_TYPEID::ZERG_HYDRALISK, { 3, 3 } });
    units.push_back({ true, true, sc2::UNIT_TYPEID::ZERG_HYDRALISK, { 6, 6 } });
}