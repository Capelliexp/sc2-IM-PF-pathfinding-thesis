#pragma once

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"

#include "../ChatCommands/ChatCommands.h"
#include "../CUDA/cuda_header.cuh"
#include "../CUDA/map_storage.hpp"

class FooBot : public sc2::Agent {
public:
    FooBot(std::string map);

    virtual void OnGameStart() final;

    virtual void OnStep() final;

    virtual void OnGameEnd() final;

    void SpawnAllUnits();

    void GatherRadius();

    void ExecuteCommand(int command);

private:
    MapStorage* map_storage;
    CUDA* cuda;
    clock_t step_clock;
    ChatCommands* chat_commands;
    int map;
    bool get_radius = true;
    uint32_t restarts_;
};
