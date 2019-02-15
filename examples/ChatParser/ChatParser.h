#pragma once

#include <string>
#include <vector>

#include "sc2api/sc2_api.h"

//! Chat parser for taking in commands through the chat
class ChatParser
{
public:
    ChatParser();
    ~ChatParser();

    //! Function that adds the given messages to the commands, if they are valid.
    //!< \param messages A list of messages from the chat.
    std::vector<sc2::ChatMessage> AddCommandToList(std::vector<sc2::ChatMessage> messages);

private:
    

private:
    //! Commands from the chat.
    std::vector<std::string> messages;
    //! Commands that can be performaed from the chat.
    std::vector<std::string> valid_commands;
};
