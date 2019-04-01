#pragma once

#include "sc2api/sc2_api.h"
#include "sc2lib/sc2_lib.h"

#include "sc2utils/sc2_manage_process.h"

#include "FooBot/FooBot.h"
#include "tools.h"

#include "Synchapi.h"


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

    coordinator.SetStepSize(1);

    //! Om du �ndrar denna variable. Gl�m inte att �ndra #define MAP_X och #define MAP_Y i map_storage.hpp.
    std::string map = "empty50";
    // Add the custom bot, it will control the players.
    FooBot bot(map);

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
