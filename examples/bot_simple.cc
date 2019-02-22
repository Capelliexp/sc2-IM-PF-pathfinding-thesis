#pragma once

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"

#include "sc2utils/sc2_manage_process.h"

#include "FooBot/FooBot.h"



//*************************************************************************************************
int main(int argc, char* argv[]) {
    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    std::string map = "labyrinth";
    // Add the custom bot, it will control the players.
    FooBot bot(map);

    coordinator.SetParticipants({
        CreateParticipant(sc2::Race::Terran, &bot),
        CreateComputer(sc2::Race::Terran)
    });
    coordinator.SetRealtime(false);
    // Start the game.
    coordinator.LaunchStarcraft();

    // Step forward the game simulation.
    bool do_break = false;
    map = std::string("Test/" + map + ".SC2Map");
    char* str = new char[map.size()];
    std::strcpy(str, map.c_str());
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
