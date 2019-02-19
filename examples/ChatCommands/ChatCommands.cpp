#include "ChatCommands.h"

#include <algorithm>

ChatCommands::ChatCommands(const sc2::ObservationInterface* observer, const sc2::DebugInterface* debug, std::string map)
{
	this->observer = observer;
	this->debug = debug;

	if		(map == "empty50")		this->map = 1;
	else if (map == "empty200")		this->map = 2;
	else if (map == "height")		this->map = 3;
	else if (map == "labyrinth")	this->map = 4;
	else if (map == "wall")			this->map = 5;

	switch (this->map)
	{
	case 0:
		this->valid_commands = { "1", "2", "3", "4", "5", "6", "7" };
		break;
	case 1:
		this->valid_commands = { "1", "2", "3", "4", "5", "6", "7" };
		break;
	case 2:
		this->valid_commands = {};
		break;
	case 3:
		this->valid_commands = {};
		break;
	case 4:
		this->valid_commands = {};
		break;
	default:
		break;
	}
}

ChatCommands::~ChatCommands()
{

}

int ChatCommands::AddCommandToList(std::vector<sc2::ChatMessage> messages)
{
	for (int i = 0; i < valid_commands.size(); ++i) {

	}
	return 0;
}

void ChatCommands::ExecuteCommands()
{

}
