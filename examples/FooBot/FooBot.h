#pragma once

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"

#include "../CUDA/map_storage.hpp"

enum behaviors
{
    ATTACK,
    DEFENCE,
    PASSIVE
};

struct Node {
    int x;
    int y;
    int parentX;
    int parentY;
    float euc_dist;
    float walk_dist;
};

//! struct holding a unit and info about its destination and the behavior
struct Unit {
    const sc2::Unit* unit;
    behaviors behavior;
    Destination_IM* destination;
    float dist_traveled;
    sc2::Point2D last_pos;
    sc2::Point2D next_pos;
};

//! Struct holding unit and the path to its destination. Used for A*
struct AstarUnit {
    const sc2::Unit* unit;
    behaviors behavior;
    std::vector<Node> path;
    float dist_traveled;
    sc2::Point2D last_pos;
};

class FooBot : public sc2::Agent {
public:
    FooBot(std::string map, bool spawn_all_units = false);

    virtual void OnGameStart() final;
    virtual void OnStep() final;
    virtual void OnGameEnd() final;
    virtual void OnUnitEnterVision(const sc2::Unit* unit) final;
    virtual void OnUnitDestroyed(const sc2::Unit* unit) final;
    virtual void OnUnitCreated(const sc2::Unit* unit) final;

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
    void SetDestination(std::vector<Unit>& units_vec, sc2::Point2D pos, behaviors type_of_movement, bool air_unit, sc2::Point2D start = { -1, -1 }, sc2::Point2D end = { -1, -1 });
    void SetDestination(std::vector<AstarUnit>& units_vec, sc2::Point2D pos, behaviors type_of_movement, bool air_unit, sc2::Point2D start = { -1, -1 }, sc2::Point2D end = { -1, -1 });

    //! Function to set the behaviour for units.
    //! Only used for setting behaviour for enemy units. But can be used for friendly units.
    void SetBehavior(std::vector<Unit>& units_vec, sc2::ABILITY_ID behavior);

    //! Function to update all units.
    void UpdateUnitsPaths();
    void UpdateAstarPath();
    void UpdateAstarPFPath();
    void RecreateAstarPaths();

    std::vector<Node> Astar(Node agent, sc2::Point2D destination);
    float CalculateEuclideanDistance(sc2::Point2D pos, sc2::Point2D dest);
    bool NodeExistsInList(sc2::Point2D pos, std::vector<Node> list);

    void CreateAttractingPFs();
    void UpdateHostUnitList();

    //! Functions to execute commands on different maps.
    void CommandsOnEmpty50();
    void CommandsOnEmpty200();
    void CommandsOnHeight();
    void CommandsOnLabyrinth();
    void CommandsOnWall();
    void CommandsOnEmpty20();
    void CommandsOnSpiral50();

    //! Function that is used to check if a given unit is a structure.
    //!< \param unit The unit to be checked.
    //!< \return Returns true if the unit is a structure, false otherwise.
    bool IsStructure(const sc2::Unit* unit);

    //! Function to print values in the game world.
    void printValues(int unit, sc2::Point2D pos);

private:
    MapStorage* map_storage;

    std::vector<Entity> host_unit_list;
    //! A vector of units
    std::vector<Unit> player_units;
    std::vector<Unit> enemy_units;

    //! A* Units
    std::vector<AstarUnit> astar_units;

    //! Bool indicating if any buildings have been created or destroyed since last frame.
    bool new_buildings;

    //! Bools representing what algorithm is used, if both are false IM+PF algorithm is used.
    bool astar;
    bool astarPF;

    //! Integer that represents the current command.
    int command;
    //! Integers and Bool to indicate and help with unit actions during commands
    int spawned_player_units;
    int spawned_enemy_units;
    bool destination_set;

    //! Integer that represents the current map;
    int map;


    bool spawn_all_units;
    bool get_radius = true;
    uint32_t restarts_;
};
