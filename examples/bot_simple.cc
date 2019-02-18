#pragma once

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"

#include "sc2utils/sc2_manage_process.h"

//#include "cuda_wrapper.hpp"
#include "CUDA/cuda_header.cuh"
#include "CUDA/map_storage.hpp"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <math.h>

class FooBot : public sc2::Agent {
public:
    uint32_t restarts_;
    bool get_radius = true;
    FooBot() :
        restarts_(0) {
    }

    virtual void OnGameStart() final {
        std::cout << "Starting a new game (" << restarts_ << " restarts)" << std::endl;

        map_storage = new MapStorage();
        cuda = new CUDA();
        map_storage->Initialize(Observation(), Debug(), Actions(), ActionsFeatureLayer());
        map_storage->Test();
        cuda->InitializeCUDA(map_storage, Observation(), Debug(), Actions(), ActionsFeatureLayer());
        step_clock = clock();
        //SpawnAllUnits();
    };

    virtual void OnStep() final {
        uint32_t game_loop = Observation()->GetGameLoop();

        /*if (game_loop % 100 == 0) {
            sc2::Units units = Observation()->GetUnits(sc2::Unit::Alliance::Self);
            for (auto& it_unit : units) {
                sc2::Point2D target = sc2::FindRandomLocation(Observation()->GetGameInfo());
                Actions()->UnitCommand(it_unit, sc2::ABILITY_ID::SMART, target);
            }
        }*/

        /*if (Observation()->GetUnits(sc2::Unit::Alliance::Self).size() > 95 && get_radius) {
            GatherRadius();
            get_radius = false;
        }*/

        cuda->Update(clock() - step_clock);

        step_clock = clock();
    };

    virtual void OnGameEnd() final {
        ++restarts_;
        std::cout << "Game ended after: " << Observation()->GetGameLoop() << " loops " << std::endl;

        delete cuda;
        delete map_storage;
    };

    void SpawnAllUnits() {
        std::vector<int> unit_IDs = cuda->GetUnitsID();
        for (int unit_ID : unit_IDs)
        {
            sc2::Point2D p = sc2::FindRandomLocation(Observation()->GetGameInfo());
            Debug()->DebugCreateUnit(sc2::UNIT_TYPEID(unit_ID), p, 1, 1);
            Debug()->SendDebug();
        }

    }

    void GatherRadius() {
        sc2::Units units = Observation()->GetUnits(sc2::Unit::Alliance::Self);
        std::vector<float> unitRadius(cuda->GetSizeOfUnitInfoList());
        for (auto& unit : units) {
            int pos = cuda->GetPosOFUnitInHostUnitVec(unit->unit_type);
            unitRadius[pos] = unit->radius;
        }
        cuda->SetRadiusForUnits(unitRadius);
    }

private:
    MapStorage* map_storage;
    CUDA* cuda;
    clock_t step_clock;

};

//*************************************************************************************************
int main(int argc, char* argv[]) {
    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    // Add the custom bot, it will control the players.
    FooBot bot;

    coordinator.SetParticipants({
        CreateParticipant(sc2::Race::Terran, &bot),
        CreateComputer(sc2::Race::Terran)
    });
    coordinator.SetRealtime(true);
    // Start the game.
    coordinator.LaunchStarcraft();

    // Step forward the game simulation.
    bool do_break = false;
    char* str = "Test/EditorTest.SC2Map";
    while (!do_break) {
        //coordinator.StartGame(sc2::kMapBelShirVestigeLE);
        coordinator.StartGame(str);
        while (coordinator.Update() && !do_break) {
            if (sc2::PollKeyPress()) {
                do_break = true;
            }
        }
    }

    return 0;
}
