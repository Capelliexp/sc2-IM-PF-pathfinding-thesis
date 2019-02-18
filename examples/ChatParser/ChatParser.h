#pragma once

#include <string>
#include <vector>

#include "sc2api/sc2_api.h"

//! Chat parser for taking in commands through the chat
class ChatParser
{
public:
    ChatParser(const sc2::DebugInterface* debug);
    ~ChatParser();

    //! Function that adds the given messages to the commands, if they are valid.
    //!< \param messages A list of messages from the chat.
    //!< \return Returns an empty vector of strings except for when the GetGroups command is called.
    std::vector<std::string> AddCommandToList(std::vector<sc2::ChatMessage> messages);

private:
    //! Function to find if there is a match in message and valid_commands/valid_sub_commands.
    //!< \param message The message to match
    //!< \return Returns the position of the matched command. -1 if no match or invalid match is made.
    int FindMatch(std::string message);

    //! Function that is called to execute the stored commands.
    void ExecuteCommands();

private:
    //! Commands from the chat.
    std::vector<std::string> commands;
    //! Commands that can be performaed from the chat.
    std::vector<std::string> valid_commands;
    //! Sub-commands that can be performaed from the chat.
    std::vector<std::string> valid_sub_commands;
    //! If we are supposed to take sub-commands
    bool taking_sub_commands;

    const sc2::DebugInterface* debug;
};
