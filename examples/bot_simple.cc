#pragma once

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"

#include "sc2utils/sc2_manage_process.h"

#include "FooBot/FooBot.h"
#include "tools.h"
#include "CUDA/cuda_header.cuh"

#include "Synchapi.h"

#include <time.h>
#include <windows.h>
#include <chrono>
#include <iostream>

typedef enum {
    CHRONO,
    CHRONO_SYNC_PRE_UPDATE,
    CHRONO_SYNC_POST_UPDATE
} MeasureType;

//*************************************************************************************************
int main(int argc, char* argv[]) {
    Sleep(1000);
    PrintMemoryUsage("start");

    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    Sleep(1000);
    PrintMemoryUsage("SC2 coordinator initialization");
    srand(time(NULL));
    
    coordinator.SetStepSize(1);

    //! Om du ändrar denna variable. Glöm inte att ändra #define MAP_X och #define MAP_Y i map_storage.hpp.
    std::string map = "empty50";

    //! Experiment/Labyrinth
    //std::string map = "easy";
    //std::string map = "medium";
    //std::string map = "hard_two";
    //std::string map = "hard_one";

    int command = 9;

    // Add the custom bot, it will control the players.
    FooBot bot(map, command);

    Sleep(1000);
    PrintMemoryUsage("FooBot initialization");

    coordinator.SetParticipants({
        CreateParticipant(sc2::Race::Terran, &bot),
        CreateComputer(sc2::Race::Terran)
    });
    coordinator.SetRealtime(false);
    // Start the game.
    coordinator.LaunchStarcraft();

    Sleep(1000);
    PrintMemoryUsage("SC2 launch");

    // Step forward the game simulation.
    map = std::string("Test/" + map + ".SC2Map");
    //map = std::string("Experiment/Labyrinth/" + map + ".SC2Map");
    char* str = new char[map.size()];
    std::strcpy(str, map.c_str());
    
    //coordinator.StartGame(sc2::kMapBelShirVestigeLE);
    coordinator.StartGame(str);

    MeasureType clock_type = MeasureType::CHRONO;
    bool active = true;
    long long int frame_nr = 0;
    float elapsed_frame_time = 0;
    uint32_t restarts_check = 0;

    std::vector<float> frame_storage;
    frame_storage.reserve(10000);

    std::chrono::steady_clock::time_point frame_start;
    std::chrono::steady_clock::time_point frame_end;

    //VRF BLIR PROGRAMMET LÅNGSAMMARE EFTER TESTENS SLUT?!?

    //--------
    switch(clock_type){
    case MeasureType::CHRONO:
        frame_start = std::chrono::steady_clock::now();

        while (active) {
            active = coordinator.Update();

            frame_end = std::chrono::steady_clock::now();
            elapsed_frame_time = ((float)std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count()) / 1000.f;
            frame_start = std::chrono::steady_clock::now();

            //save frame time data
            frame_storage.push_back(elapsed_frame_time);
            if (frame_storage.capacity() - frame_storage.size() < 10) frame_storage.reserve(frame_storage.capacity() + 10000);

            if (bot.print_frame_data) {
                bot.print_frame_data = false;
                PrintFrameTimesToFile(frame_storage.data(), frame_storage.size(), "chrono_no_sync");
                frame_storage.clear();
            }
            if (bot.restarts_ != restarts_check) {
                frame_storage.clear();
                restarts_check = bot.restarts_;
            }
        }
        break;

        //----------------------------------------------

    case MeasureType::CHRONO_SYNC_PRE_UPDATE:
        frame_start = std::chrono::steady_clock::now();

        while (active) {
            cudaDeviceSynchronize();
            active = coordinator.Update();

            frame_end = std::chrono::steady_clock::now();
            elapsed_frame_time = ((float)std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count()) / 1000.f;
            frame_start = std::chrono::steady_clock::now();


            //save frame time data
            frame_storage.push_back(elapsed_frame_time);
            if (frame_storage.capacity() - frame_storage.size() < 10) frame_storage.reserve(frame_storage.capacity() + 10000);

            if (bot.print_frame_data) {
                bot.print_frame_data = false;
                PrintFrameTimesToFile(frame_storage.data(), frame_storage.size(), "chrono_pre_sync");
                frame_storage.clear();
            }
            if (bot.restarts_ != restarts_check) {
                frame_storage.clear();
                restarts_check = bot.restarts_;
            }
        }
        break;

        //----------------------------------------------

    case MeasureType::CHRONO_SYNC_POST_UPDATE:
        frame_start = std::chrono::steady_clock::now();

        while (active) {
            active = coordinator.Update();
            cudaDeviceSynchronize();

            frame_end = std::chrono::steady_clock::now();
            elapsed_frame_time = ((float)std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count()) / 1000.f;
            frame_start = std::chrono::steady_clock::now();

            //save frame time data
            frame_storage.push_back(elapsed_frame_time);
            if (frame_storage.capacity() - frame_storage.size() < 10) frame_storage.reserve(frame_storage.capacity() + 10000);

            if (bot.print_frame_data) {
                bot.print_frame_data = false;
                PrintFrameTimesToFile(frame_storage.data(), frame_storage.size(), "chrono_post_sync");
                frame_storage.clear();
            }
            if (bot.restarts_ != restarts_check) {
                frame_storage.clear();
                restarts_check = bot.restarts_;
            }
        }
        break;
    }

    return 0;
}
