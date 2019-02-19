#pragma once

#include <string>
#include <vector>

#include "sc2api/sc2_api.h"

//! Chat parser for taking in commands through the chat
class ChatCommands
{
public:
    ChatCommands(const sc2::ObservationInterface* observer, const sc2::DebugInterface* debug, std::string map);
    ~ChatCommands();

    //! Function that executes the given command if it is valid for the map.
    //!< \param messages A list of messages from the chat.
    //!< \return Returns an int specifying what test to execute.
    int AddCommandToList(std::vector<sc2::ChatMessage> messages);

private:
    //! Function that is called to execute the command.
    void ExecuteCommands();

private:
    //! Commands that can be performaed from the chat.
    std::vector<std::string> valid_commands;
    //! Integer representing the map
    int map;
    const sc2::ObservationInterface* observer;
    const sc2::DebugInterface* debug;
};
