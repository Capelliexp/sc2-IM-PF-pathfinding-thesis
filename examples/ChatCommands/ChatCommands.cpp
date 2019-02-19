#include "ChatCommands.h"

#include <algorithm>

ChatCommands::ChatCommands(int map)
{
	this->map = map;

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

int ChatCommands::AddCommandToList(std::string message)
{
	for (int i = 0; i < valid_commands.size(); ++i) {
		if (message == valid_commands[i])
			return i;
	}
	return -1;
}
