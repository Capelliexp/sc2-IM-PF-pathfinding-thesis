#pragma once

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"

#include "../ChatCommands/ChatCommands.h"
#include "../CUDA/cuda_header.cuh"
#include "../CUDA/map_storage.hpp"

class FooBot : public sc2::Agent {
public:
    FooBot(std::string map, bool spaw_alla_units = false);

    virtual void OnGameStart() final;
    virtual void OnStep() final;
    virtual void OnGameEnd() final;

private:
    //! Function to spawn all units that influence the PF
    void SpawnAllUnits();
    //! Function to get the radius and is_flying bool from all the spawned units.
    //! Will also send the info to CUDA.
    void GatherRadius();
    //! Function to execute the given command.
    void ExecuteCommand();
    //! Function to spawn a number of units at given position
    //!< param unit_id Unit id of unit(s) to spawn
    //!< param amount Number of units to spawn
    //!< param pos Position to spawn units at
    //!< param player ID of the player that the units belong to. If left blank the units are given to player one.
    void SpawnUnits(sc2::UNIT_TYPEID unit_id, int amount, sc2::Point2D pos, int player = 1);
    //! Function to set destination for the given units and the behavior.
    void SetDestination(sc2::Units units, sc2::Point2D pos, sc2::ABILITY_ID type_of_movement, sc2::Point2D start = { -1, -1 }, sc2::Point2D end = { -1, -1 });
    void SetBehavior(sc2::Units units, sc2::ABILITY_ID behavior);

    void CommandsOnEmpty50();
    void CommandsOnEmpty200();
    void CommandsOnHeight();
    void CommandsOnLabyrinth();
    void CommandsOnWall();

    bool CheckIfUnitsSpawned(int amount, std::vector<sc2::UnitTypeID> types);

private:
    MapStorage* map_storage;
    CUDA* cuda;
    clock_t step_clock;
    ChatCommands* chat_commands;
    //! Integer that represents the map.
    int map;
    //! Integer that represents the current command.
    int command;
    bool spawn_all_units;
    //! Bool indicating if the command can spawn units.
    bool spawn_units;
    bool get_radius = true;
    uint32_t restarts_;
};