#include "ChatCommands.h"

#include <algorithm>

ChatCommands::ChatCommands(int map)
{
	this->map = map;

	switch (this->map)
	{
	case 1:	//empty50
		this->valid_commands = { "1", "2", "3", "4", "5", "6", "7" };
		break;
	case 2:	//empty200
		this->valid_commands = { "1", "2", "3", "4", "5", "6", "7" };
		break;
	case 3:	//height
		this->valid_commands = {};
		break;
	case 4:	//labyrinth
		this->valid_commands = {"1"};
		break;
	case 5:	//wall
		this->valid_commands = {};
		break;
	default:
		break;
	}
}

ChatCommands::~ChatCommands()
{

}

int ChatCommands::ParseCommands(std::string message)
{
	for (int i = 0; i < valid_commands.size(); ++i) {
		if (message == valid_commands[i])
			return i+1;
	}
	return 0;
}
