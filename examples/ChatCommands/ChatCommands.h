#pragma once

#include <string>
#include <vector>

#include "sc2api/sc2_api.h"

//! Chat parser for taking in commands through the chat
class ChatCommands
{
public:
    ChatCommands(int map);
    ~ChatCommands();

    //! Function that executes the given command if it is valid for the map.
    //!< \param messages A list of messages from the chat.
    //!< \return Returns an int specifying what test to execute.
    int ParseCommands(std::string messages);

private:

private:
    //! Commands that can be performaed from the chat.
    std::vector<std::string> valid_commands;
    //! Integer representing the map
    int map;
};
