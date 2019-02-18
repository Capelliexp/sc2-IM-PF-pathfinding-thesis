#include "ChatParser.h"

#include <algorithm>

ChatParser::ChatParser(const sc2::DebugInterface* debug)
{
	this->debug = debug;
	this->valid_commands = { "MOVE", "ADDTOGROUP", "CREATEGROUP", "REMOVEGROUP", "SPAWN", "SETBEHAVIORFORGROUP", "GETGROUPS"};
	this->valid_sub_commands = { "UNIT", "BEHAVIOR", "SENDCOMMAND"};
	this->taking_sub_commands = false;
}

ChatParser::~ChatParser()
{
}

std::vector<std::string> ChatParser::AddCommandToList(std::vector<sc2::ChatMessage> messages)
{
	std::vector<std::string> out_messages;
	for (sc2::ChatMessage message : messages) {
		std::string msg = message.message;
		std::transform(msg.begin(), msg.end(), msg.begin(), ::toupper);
		int command = FindMatch(msg);
		if (command == -1) {
			if (!taking_sub_commands) {
				commands.push_back(msg);
				switch (command)
				{
				case 0:	//Move
				case 1:	//AddToGroup
					taking_sub_commands = true;
					break;
				case 2:	//CreateGroup
				case 3:	//RemoveGroup
				case 4:	//Spwan
				case 5:	//SetBehaviorForGroup
					ExecuteCommands();
					break;
				case 6:	//GetGroups
					ExecuteCommands();
					break;
				default:
					break;
				}
			}
			else {
				switch (command)
				{
				case 0:	//Unit 
				case 1:	//Behavior 
					commands.push_back(msg);
				case 2:	//SendCommand
					taking_sub_commands = false;
					ExecuteCommands();
					break;
				default:
					break;
				}
			}
			
		}
			
	}
	return out_messages;
}

int ChatParser::FindMatch(std::string message)
{
	int pos;
	if (!taking_sub_commands) {
		for (int i = 0; i < valid_commands.size(); ++i) {
			if (message.find(valid_commands[i]) == 0) {
				
				return i;
			}
		}
	}
	else {
		for (int i = 0; i < valid_sub_commands.size(); ++i) {
			if (message.find(valid_sub_commands[i]) == 0) {
				if (i == 2)
					taking_sub_commands = false;
				return i;
			}
		}
	}
	return -1;
}

void ChatParser::ExecuteCommands()
{
}
