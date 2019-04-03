#include "FooBot.h"
#include "../tools.h"

FooBot::FooBot(std::string map, bool spaw_alla_units) : 
	restarts_(0), 
	spawn_all_units(spaw_alla_units) {
	this->command = 0;
	this->spawned_player_units = 0;
	this->spawned_enemy_units = 0;
	this->astar = false;
	this->new_buildings = false;
	if (map == "empty50")			this->map = 1;
	else if (map == "empty200")		this->map = 2;
	else if (map == "height")		this->map = 3;
	else if (map == "labyrinth")	this->map = 4;
	else if (map == "wall")			this->map = 5;
	else if (map == "empty20")		this->map = 6;
	else if (map == "spiral50")		this->map = 7;
	else							this->map = 0;	//Not a valid testmap
}

void FooBot::OnGameStart() {
	std::cout << "Starting a new game (" << restarts_ << " restarts)" << std::endl;

	map_storage = new MapStorage();
	chat_commands = new ChatCommands(map);
	
	map_storage->Initialize(Observation(), Debug(), Actions(), ActionsFeatureLayer(), astar);
	map_storage->Test();

	Debug()->DebugFastBuild();
	Debug()->DebugGiveAllResources();

	/*Debug()->DebugCreateUnit(sc2::UNIT_TYPEID::ZERG_DRONE, sc2::Point2D(10, 10), 1, 1);
	Debug()->SendDebug();*/

	if (spawn_all_units) SpawnAllUnits();

	PrintMemoryUsage("OnGameStart");
}

void FooBot::OnStep() {
	uint32_t game_loop = Observation()->GetGameLoop();
	//Messages from chat
	/*std::vector<sc2::ChatMessage> in_messages = Observation()->GetChatMessages();
	if (in_messages.size() > 0 && command == 0)
		command = chat_commands->ParseCommands(in_messages[0].message);*/

	//RAM & VRAM stat prints
	if (GetKeyState('P') & 0x8000) PrintMemoryUsage("runtime");
	if (GetKeyState('L') & 0x8000) map_storage->PrintCUDAMemoryUsage("runtime");

	//commands
	if (command == 0) {
		if (GetKeyState('1') & 0x8000) command = 1;
	}

	if (new_buildings) {
		new_buildings = false;
		map_storage->UpdateIMAtsar();
	}

	if (!astar) {
		//Map transfer PF a/r
		map_storage->TransferPFFromDevice();
	}

	//Set destination
	ExecuteCommand();

	if (!astar) {
		//Unit transfer
		UpdateHostUnitList();
	}

	if (!astar) {
		//Pathfinding
		UpdateUnitsPaths();
	}
	else {
		UpdateAstarPath();
	}

	if (!astar) {
		//Map transfer IM
		map_storage->TransferIMFromDevice();

		//Start IM
		//Start PF a/r
		CreateAttractingPFs();
		map_storage->ExecuteDeviceJobs();
	}


	Actions()->SendActions();

	if (spawn_all_units)
		if (Observation()->GetUnits(sc2::Unit::Alliance::Self).size() > 95 && get_radius) {
			GatherRadius();
			get_radius = false;
		}
}

void FooBot::OnGameEnd() {
	++restarts_;
	std::cout << "Game ended after: " << Observation()->GetGameLoop() << " loops " << std::endl;

	delete map_storage;
	delete chat_commands;
}

void FooBot::OnUnitEnterVision(const sc2::Unit * unit) {
	if (!astar) {
		if (!IsStructure(unit) && unit->alliance == sc2::Unit::Alliance::Enemy) {
			Unit new_unit;
			new_unit.unit = unit;
			new_unit.behavior = behaviors::PASSIVE;
			this->enemy_units.push_back(new_unit);
		}
		else {
			//kernel launch
			map_storage->ChangeDeviceDynamicMap(unit->pos, unit->radius, -2);
		}
	}
	else {
		if (IsStructure(unit))
			new_buildings = true;
	}
}

void FooBot::OnUnitDestroyed(const sc2::Unit * unit) {
	if (!astar) {
		if (!IsStructure(unit)) {
			for (int i = 0; i < player_units.size(); ++i) {
				if (player_units[i].unit == unit) {
					player_units.erase(player_units.begin() + i);
					return;
				}
			}
		}
		else {
			//kernel launch
			map_storage->ChangeDeviceDynamicMap(unit->pos, unit->radius, 0);
		}
		for (int i = 0; i < enemy_units.size(); ++i) {
			if (enemy_units[i].unit == unit) {
				enemy_units.erase(enemy_units.begin() + i);
				return;
			}
		}
	}
}

void FooBot::OnUnitCreated(const sc2::Unit * unit) {
	if (!astar) {
		if (!IsStructure(unit) && unit->alliance == sc2::Unit::Alliance::Self) {
			Unit new_unit;
			new_unit.unit = unit;
			new_unit.behavior = behaviors::DEFENCE;
			new_unit.dist_traveled = 0;
			new_unit.last_pos = sc2::Point2D(-1, -1);
			new_unit.next_pos = sc2::Point2D(-1, -1);
			this->player_units.push_back(new_unit);
		}
		else {
			//kernel launch
			map_storage->ChangeDeviceDynamicMap(unit->pos, unit->radius, -2);
		}
	}
	else {
		if (!IsStructure(unit) && unit->alliance == sc2::Unit::Alliance::Self) {
			AstarUnit new_unit;
			new_unit.unit = unit;
			new_unit.last_pos = sc2::Point2D(-1, -1);
			new_unit.dist_traveled = 0;
			this->astar_units.push_back(new_unit);
		}
		else if (IsStructure(unit))
			new_buildings = true;
	}
}

void FooBot::SpawnAllUnits() {
	std::vector<int> unit_IDs = map_storage->GetUnitsID();
	for (int unit_ID : unit_IDs) {
		sc2::Point2D p = sc2::FindRandomLocation(Observation()->GetGameInfo());
		Debug()->DebugCreateUnit(sc2::UNIT_TYPEID(unit_ID), p, 1, 1);
		Debug()->SendDebug();
	}
}

void FooBot::GatherRadius() {
	sc2::Units units = Observation()->GetUnits(sc2::Unit::Alliance::Self);
	std::vector<float> unitRadius(map_storage->GetSizeOfUnitInfoList());
	for (auto& unit : units) {
		int pos = map_storage->GetPosOFUnitInHostUnitVec(unit->unit_type);
		unitRadius[pos] = unit->radius;
	}
	map_storage->SetRadiusForUnits(unitRadius);
}

void FooBot::ExecuteCommand() {
	switch (map) {
	case 1:
		CommandsOnEmpty50();
		break;
	case 2:
		CommandsOnEmpty200();
		break;
	case 3:
		CommandsOnHeight();
		break;
	case 4:
		CommandsOnLabyrinth();
		break;
	case 5:
		CommandsOnWall();
		break;
	case 6:
		CommandsOnEmpty20();
		break;
	case 7:
		CommandsOnSpiral50();
		break;
	default:
		command = 0;
		break;
	}
}

void FooBot::SpawnUnits(sc2::UNIT_TYPEID unit_id, int amount, sc2::Point2D pos, int player) {
	Debug()->DebugCreateUnit(unit_id, pos, player, amount);
	Debug()->SendDebug();
}

void FooBot::SetDestination(sc2::Units units, sc2::Point2D pos, sc2::ABILITY_ID type_of_movement, sc2::Point2D start, sc2::Point2D end) {
	//Start custom pathfinding here
	if (start.x == -1)
		Actions()->UnitCommand(units, type_of_movement, pos);
	else {
		sc2::Units subUnits;
		for (int i = 0; i < units.size(); ++i) {
			if (sc2::Point2D(units[i]->pos) >= start && sc2::Point2D(units[i]->pos) <= end)
				subUnits.push_back(units[i]);
		}
		Actions()->UnitCommand(subUnits, type_of_movement, pos);
	}
}

void FooBot::SetDestination(std::vector<Unit>& units_vec, sc2::Point2D pos, behaviors type_of_movement, bool air_unit, sc2::Point2D start, sc2::Point2D end) {
	pos.y = MAP_Y_R - 1 - pos.y;
	if (air_unit) {
		for (int i = 0; i < units_vec.size(); ++i) {
			if (start.x == -1) {
				units_vec[i].behavior = type_of_movement;
				units_vec[i].destination = &map_storage->RequestAirDestination(pos);
			}
			else if (sc2::Point2D(units_vec[i].unit->pos) >= start && sc2::Point2D(units_vec[i].unit->pos) <= end) {
				units_vec[i].behavior = type_of_movement;
				units_vec[i].destination = &map_storage->RequestAirDestination(pos);
			}
		}
	}
	else {
		for (int i = 0; i < units_vec.size(); ++i) {
			if (start.x == -1) {
				units_vec[i].behavior = type_of_movement;
				units_vec[i].destination = &map_storage->RequestGroundDestination(pos);
			}
			else if (sc2::Point2D(units_vec[i].unit->pos) >= start && sc2::Point2D(units_vec[i].unit->pos) <= end) {
				units_vec[i].behavior = type_of_movement;
				units_vec[i].destination = &map_storage->RequestGroundDestination(pos);
			}
		}
	}
}

void FooBot::SetDestination(std::vector<AstarUnit>& units_vec, sc2::Point2D pos, bool air_unit, sc2::Point2D start, sc2::Point2D end) {
	pos.y = MAP_Y_R - 1 - pos.y;
	Node agent;
	agent.euc_dist = 0;
	agent.parentX = -1;
	agent.parentY = -1;
	agent.walk_dist = 0;
	if (air_unit) {
	}
	else {
		for (int i = 0; i < units_vec.size(); ++i) {
			agent.x = units_vec[i].unit->pos.x;
			agent.y = MAP_Y_R - 1 - units_vec[i].unit->pos.y;
			if (start.x == -1) {
				units_vec[i].path = Astar(agent, pos);
			}
			else if (sc2::Point2D(units_vec[i].unit->pos) >= start && sc2::Point2D(units_vec[i].unit->pos) <= end) {
				units_vec[i].path = Astar(agent, pos);
			}
		}
	}
}

void FooBot::SetBehavior(sc2::Units units, sc2::ABILITY_ID behavior) {
	Actions()->UnitCommand(units, behavior);
}

void FooBot::SetBehavior(std::vector<Unit>& units_vec, sc2::ABILITY_ID behavior) {
	for (int i = 0; i < units_vec.size(); ++i) {
		Actions()->UnitCommand(units_vec[i].unit, behavior);
	}
}

//NOTE!!!! x- and y-coordinates are fliped.
void FooBot::UpdateUnitsPaths() {
	for (int i = 0; i < player_units.size(); ++i) {
		if (player_units[i].destination == nullptr) continue;		//No destination set
		if (player_units[i].destination->map[0][0][0] == -107374176) continue;	//No destination ready to be used
		//if (is not transfered... ) continue (new cuda event)


		

		sc2::Point2D current_pos = player_units[i].unit->pos;
		sc2::Point2D translated_pos = current_pos;
		
		translated_pos.x = translated_pos.x;
		translated_pos.y = MAP_Y_R - 1 - translated_pos.y;

		if (player_units[i].destination->destination == sc2::Point2D((int)translated_pos.x, (int)translated_pos.y)) {	//Found destination.
			sc2::Point2D new_pos = player_units[i].destination->destination;
			//Actions()->UnitCommand(player_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
			sc2::Point2D pos = sc2::Point2D(player_units[i].unit->pos.x, MAP_Y_R - 1 - player_units[i].unit->pos.y);
			player_units[i].dist_traveled += CalculateEuclideanDistance(player_units[i].last_pos, pos);
			player_units[i].last_pos = pos;
			std::cout << "Done: " << player_units[i].dist_traveled;
			player_units[i].destination = nullptr;
			continue;
		}

		if (player_units[i].last_pos.x == -1) {
			player_units[i].last_pos = sc2::Point2D(player_units[i].unit->pos.x, MAP_Y_R - 1 - player_units[i].unit->pos.y);
			map_storage->PrintMap(player_units[i].destination->destination, MAP_X_R, MAP_Y_R, "IM");
			map_storage->PrintMap(map_storage->GetGroundAvoidancePFValue((int)translated_pos.y, (int)translated_pos.x), MAP_X_R, MAP_Y_R, "PF");
		}
		else {
			sc2::Point2D pos = sc2::Point2D(player_units[i].unit->pos.x, MAP_Y_R - 1 - player_units[i].unit->pos.y);
			player_units[i].dist_traveled += CalculateEuclideanDistance(player_units[i].last_pos, pos);
			player_units[i].last_pos = pos;
		}

		//Get the value from the IM and PF to determine the total value of the tile.
		float current_value = player_units[i].destination->map[(int)translated_pos.y][(int)translated_pos.x][0];
		float current_pf = 0;
		if (player_units[i].behavior == behaviors::DEFENCE)
			current_pf = map_storage->GetGroundAvoidancePFValue((int)translated_pos.y, (int)translated_pos.x);
		else if (player_units[i].behavior == behaviors::ATTACK)
			current_pf = map_storage->GetAttractingPF(player_units[i].unit->unit_type, (int)translated_pos.y, (int)translated_pos.x);
		current_value += current_pf;

		sc2::Point2D p1 = sc2::Point2D(translated_pos.x, translated_pos.y);
		sc2::Point2D p2 = sc2::Point2D(player_units[i].next_pos.x, player_units[i].next_pos.y); 
		float unit_radius = player_units[i].unit->radius < 0.5 ? 0.5 : 1.5;
		if ((p1.x >= p2.x - unit_radius && p1.x <= p2.x + unit_radius && p1.y >= p2.y - unit_radius && p1.y <= p2.y + unit_radius) || p2.x == -1) {
			std::vector<sc2::Point2D> udlr;
			udlr.push_back(sc2::Point2D(p1.x + 0, p1.y + 1));
			udlr.push_back(sc2::Point2D(p1.x + 1, p1.y + 1));
			udlr.push_back(sc2::Point2D(p1.x + 1, p1.y + 0));
			udlr.push_back(sc2::Point2D(p1.x + 1, p1.y - 1));
			udlr.push_back(sc2::Point2D(p1.x + 0, p1.y - 1));
			udlr.push_back(sc2::Point2D(p1.x - 1, p1.y - 1));
			udlr.push_back(sc2::Point2D(p1.x - 1, p1.y + 0));
			udlr.push_back(sc2::Point2D(p1.x - 1, p1.y + 1));



			//std::vector<sc2::Point2D> udlr;				//y,  x
			//udlr.push_back(translated_pos + sc2::Point2D( 0,  1));	//Down
			//udlr.push_back(translated_pos + sc2::Point2D( 1,  1));	//Down, right
			//udlr.push_back(translated_pos + sc2::Point2D( 1,  0));	//Right
			//udlr.push_back(translated_pos + sc2::Point2D( 1, -1));	//Up, right
			//udlr.push_back(translated_pos + sc2::Point2D( 0, -1));	//Up
			//udlr.push_back(translated_pos + sc2::Point2D(-1, -1));	//Up, left
			//udlr.push_back(translated_pos + sc2::Point2D(-1,  0));	//Left
			//udlr.push_back(translated_pos + sc2::Point2D(-1,  1));	//Down, left

			//printValues(i, translated_pos);

			float min_value = 5000;
			int next_tile = 0;
			for (int j = 0; j < udlr.size(); ++j) {
				//Get the value from the IM and PF to determine the total value of the new tile.
				float new_value = player_units[i].destination->map[(int)udlr[j].y][(int)udlr[j].x][0];

				if (new_value < 0) continue;
				float pf_value = 0;
				if (player_units[i].behavior == behaviors::DEFENCE)
					pf_value = map_storage->GetGroundAvoidancePFValue((int)udlr[j].y, (int)udlr[j].x);
				else if (player_units[i].behavior == behaviors::ATTACK)
					pf_value = map_storage->GetAttractingPF(player_units[i].unit->unit_type, (int)udlr[j].y, (int)udlr[j].x);
				new_value += pf_value;

				//if (new_value < 0) continue;	//Unpathable terrain
				//This needs to be modified.
				//if ((current_value - new_value) > 2) continue;	//Invalid move
				if (min_value > new_value) {
					min_value = min(new_value, min_value);
					next_tile = j;
				}
			}

			sc2::Point2D new_pos = sc2::Point2D(udlr[next_tile].x, udlr[next_tile].y);
			player_units[i].next_pos = new_pos;
			new_pos.y = MAP_Y_R - 1 - new_pos.y;
			if (player_units[i].behavior == behaviors::DEFENCE || player_units[i].behavior == behaviors::PASSIVE)
				Actions()->UnitCommand(player_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
			else if (player_units[i].behavior == behaviors::ATTACK) {
				if (player_units[i].unit->weapon_cooldown == 0)
					Actions()->UnitCommand(player_units[i].unit, sc2::ABILITY_ID::ATTACK, new_pos);
				else
					Actions()->UnitCommand(player_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
			}
		}
	}
	Debug()->SendDebug();
}

void FooBot::UpdateAstarPath() {
	for (int i = 0; i < astar_units.size(); ++i) {
		if (astar_units[i].path.size() > 0) {
			sc2::Point2D p1 = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - 1 - astar_units[i].unit->pos.y);
			sc2::Point2D p2 = sc2::Point2D(astar_units[i].path.back().x, astar_units[i].path.back().y);
			
			if (astar_units[i].last_pos.x == -1)
				astar_units[i].last_pos = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - 1 - astar_units[i].unit->pos.y);
			else if (astar_units[i].path.size() > 0) {	//Calculate dist until it's one node left.
				sc2::Point2D pos = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - 1 - astar_units[i].unit->pos.y);
				astar_units[i].dist_traveled += CalculateEuclideanDistance(astar_units[i].last_pos, pos);
				astar_units[i].last_pos = pos;
			}
			//Gör som med IM+PF
			float unit_radius = astar_units[i].unit->radius < 0.5 ? 0.5 : 1.5;
			if (p1.x >= p2.x - unit_radius && p1.x <= p2.x + unit_radius && p1.y >= p2.y - unit_radius && p1.y <= p2.y + unit_radius) {
				sc2::Point2D last_path_pos = sc2::Point2D(astar_units[i].path.back().x, astar_units[i].path.back().y);
				astar_units[i].path.pop_back();
				//Detta känns inte helt rätt. Speciellt vid sista steget av pathen.
				if (astar_units[i].path.size() > 0) {
					sc2::Point2D new_pos = sc2::Point2D(astar_units[i].path.back().x, MAP_Y_R - 1 - astar_units[i].path.back().y);
					Actions()->UnitCommand(astar_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
				}
				else {	//Calculates dist for last node.
					sc2::Point2D current_pos = astar_units[i].unit->pos;
					current_pos.y = MAP_Y_R - 1 - current_pos.y;
					astar_units[i].dist_traveled += CalculateEuclideanDistance(current_pos, last_path_pos);
					std::cout << "Done: " << astar_units[i].dist_traveled;
				}
			}
		}
	}
}

void FooBot::RecreateAstarPaths() {
	for (int i = 0; i < astar_units.size(); ++i) {
		Node agent;
		agent.euc_dist = 0;
		agent.parentX = -1;
		agent.parentY = -1;
		agent.walk_dist = 0;
		agent.x = astar_units[i].unit->pos.x;
		agent.y = MAP_Y_R - 1 - astar_units[i].unit->pos.y;
		sc2::Point2D dest = sc2::Point2D(astar_units[i].path.front().x, astar_units[i].path.front().y);
		astar_units[i].path = Astar(agent, dest);
	}
}

std::vector<Node> FooBot::Astar(Node agent, sc2::Point2D destination) {
	std::vector<Node> empty;
	if (!map_storage->GetDynamicMap(agent.y, agent.x))
		return empty;
	if (sc2::Point2D(agent.x, agent.y) == destination)
		return empty;
	std::vector<Node> closed_list;
	std::vector<Node> open_list;

	Node start;
	start.euc_dist = CalculateEuclideanDistance(sc2::Point2D(agent.x, agent.y), destination);
	start.walk_dist = 0;
	start.parentX = -1;
	start.parentY = -1;
	start.x = agent.x;
	start.y = agent.y;
	/*open_list.emplace_back(start);*/

	bool destination_found = false;
	Node node = start;
	closed_list.push_back(start);
	//! Path to destination
	while (!destination_found) {
		if (node.x == destination.x && node.y == destination.y) {
			destination_found = true;
			continue;
		}

		Node a;
		a.parentX = node.x;
		a.parentY = node.y;
		a.walk_dist = node.walk_dist + 1;

		std::vector<sc2::Point2D> adjacadjacent_nodes;
		adjacadjacent_nodes.push_back(sc2::Point2D(node.x + 0, node.y + 1));	//Up
		adjacadjacent_nodes.push_back(sc2::Point2D(node.x + 1, node.y + 0));	//Right
		adjacadjacent_nodes.push_back(sc2::Point2D(node.x - 0, node.y - 1));	//Down
		adjacadjacent_nodes.push_back(sc2::Point2D(node.x - 1, node.y - 0));	//Left
		

		for (int i = 0; i < adjacadjacent_nodes.size(); ++i) {
			bool dynamic = map_storage->GetDynamicMap(adjacadjacent_nodes[i].y, adjacadjacent_nodes[i].x);
			bool open = NodeExistsInList(adjacadjacent_nodes[i], open_list);
			bool close = NodeExistsInList(adjacadjacent_nodes[i], closed_list);
			if (dynamic && !open && !close) {
				a.x = adjacadjacent_nodes[i].x;
				a.y = adjacadjacent_nodes[i].y;
				a.euc_dist = CalculateEuclideanDistance(sc2::Point2D(a.x, a.y), destination);
				open_list.push_back(a);
			}
		}

		int nearest_node = -1;
		float shortest_distance = FLT_MAX;
		for (int i = 0; i < open_list.size(); ++i) {
			if ((open_list[i].euc_dist + open_list[i].walk_dist) < shortest_distance) {
				shortest_distance = open_list[i].euc_dist + open_list[i].walk_dist;
				nearest_node = i;
			}
		}
		closed_list.push_back(open_list[nearest_node]);
		open_list.erase(open_list.begin() + nearest_node);
		node = closed_list.back();
	}

	std::vector<Node> path;
	path.push_back(closed_list.back());

	bool generating_shortest_path = true;
	//! Get shortest distance path
	while (generating_shortest_path) {
		for (int i = 0; i < closed_list.size(); ++i) {
			if (path.back().parentX == closed_list[i].x && path.back().parentY == closed_list[i].y) {
				int closer_node = -1;
				for (int j = 0; j < closed_list.size(); ++j) {
					float dist = 2;
					if (closed_list[i].parentX == closed_list[j].x && closed_list[i].parentY == closed_list[j].y) {
						float new_dist = CalculateEuclideanDistance(sc2::Point2D(path.back().x, path.back().y), sc2::Point2D(closed_list[j].x, closed_list[j].y));
						if (new_dist < dist) {
							closer_node = j;
							dist = new_dist;
							/*path.push_back(closed_list[j]);
							break;*/
						}
					}
				}
				if (closer_node != -1) {
					path.push_back(closed_list[closer_node]);
					break;
				}
				path.push_back(closed_list[i]);
				break;
			}
		}
		if (path.back().x == agent.x && path.back().y == agent.y)
			generating_shortest_path = false;
	}
	return path;
}

float FooBot::CalculateEuclideanDistance(sc2::Point2D pos, sc2::Point2D dest) {
	float H = (sqrt((pos.x - dest.x)*(pos.x - dest.x) + (pos.y - dest.y)*(pos.y - dest.y)));
	return H;
}

bool FooBot::NodeExistsInList(sc2::Point2D pos, std::vector<Node> list) {
	for (Node n : list) {
		if (n.x == pos.x && n.y == pos.y)
			return true;
	}
	return false;
}

void FooBot::printValues(int unit, sc2::Point2D pos) {
	sc2::Point3D pp = player_units[unit].unit->pos;
	pp.z += 0.1f;
	sc2::Point3D p = { pos.x, pos.y, player_units[unit].unit->pos.z };
	p.z += 0.1f;
	for (int i = -5; i <= 5; ++i) {
		for (int j = -5; j <= 5; ++j) {
			if (p.x < MAP_X_R && p.y < MAP_Y_R && p.x >= 0 && p.y >= 0) {
				float new_value = player_units[unit].destination->map[(int)p.y - i][(int)p.x + j][0];
				if (new_value >= 0)
					new_value += map_storage->GetGroundAvoidancePFValue((int)p.y - i, (int)p.x + j/* + 1*/);
				Debug()->DebugTextOut(std::to_string(new_value), { pp.x + j, pp.y + i, pp.z }, sc2::Colors::Green, 8);
			}
		}
	}
}

void FooBot::CreateAttractingPFs() {
	std::map<sc2::UnitTypeID, int> player_unit_types;
	for (int i = 0; i < player_units.size(); ++i) {
		if (player_units[i].behavior == behaviors::ATTACK) {
			auto& searche = player_unit_types.find(player_units[i].unit->unit_type);
			if (searche == player_unit_types.end())
				player_unit_types[player_units[i].unit->unit_type] = 1;
			else
				searche->second += 1;
		}
	}
	//s�g till map_storage att ett specifikt antal PFs ska g�ras. Anv�nd player_unit_types f�r detta.
	for (auto& unit : player_unit_types)
		map_storage->CreateAttractingPF(unit.first);

}

void FooBot::UpdateHostUnitList() {
	host_unit_list.clear();
	for (int i = 0; i < player_units.size(); ++i) {
		Entity ent;
		ent.id = map_storage->GetUnitIDInHostUnitVec(player_units[i].unit->unit_type);
		ent.pos = { player_units[i].unit->pos.x, MAP_Y_R - 1 - player_units[i].unit->pos.y };
		ent.enemy = false;
		host_unit_list.push_back(ent);
	}
	for (int i = 0; i < enemy_units.size(); ++i) {
		Entity ent;
		ent.id = map_storage->GetUnitIDInHostUnitVec(enemy_units[i].unit->unit_type);
		ent.pos = { enemy_units[i].unit->pos.x,  MAP_Y_R - 1 - enemy_units[i].unit->pos.y };
		ent.enemy = true;
		host_unit_list.push_back(ent);
	}
	map_storage->UpdateEntityVector(host_unit_list);
}

void FooBot::CommandsOnEmpty50() {
	switch (command) {
	case 1: {
		if (spawned_player_units == 0) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_SIEGETANK, 1, sc2::Point2D(5, 5));
			spawned_player_units = 1;
		}
		else if (!astar) {
			if (player_units.size() == spawned_player_units) {
				SetDestination(player_units, sc2::Point2D(25), behaviors::DEFENCE, false);
				spawned_player_units = 0;
				command = 0;
			}
		}
		else if (astar) {
			if (astar_units.size() == spawned_player_units) {
				SetDestination(astar_units, sc2::Point2D(25), false);
				spawned_player_units = 0;
				command = 0;
			}
		}
		break;
	}
	case 2: {
		if (spawned_player_units == 0) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_player_units = 1;
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(45), behaviors::PASSIVE, false);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	}
	case 3: {
		if (spawned_player_units == 0) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_player_units = 1;
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	}
	case 4: {
		if (spawned_player_units == 0) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_player_units = 1;
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	}
	case 5: {
		if (spawned_player_units == 0) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_player_units = 5;
		}
		else if (player_units.size() == 5) {
			SetDestination(player_units, sc2::Point2D(25), behaviors::ATTACK, false);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	}
	case 6: {
		if (spawned_player_units == 0) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_player_units = 5;
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	}
	case 7: {
		if (spawned_player_units == 0) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(45, 45));
			spawned_player_units = 10;
		}
		else if (player_units.size() == 10) {
			//This needs to be fixed. Converted to "new way"
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(45, 45), sc2::ABILITY_ID::MOVE, sc2::Point2D(3), sc2::Point2D(8));
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(5, 5), sc2::ABILITY_ID::MOVE, sc2::Point2D(43), sc2::Point2D(48));
			spawned_player_units = 0;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = 0;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnEmpty200() {
	switch (command) {
	case 1:
		if (spawned_player_units == 0) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(195), behaviors::PASSIVE, false);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	case 2:
		if (spawned_player_units == 0 && spawned_enemy_units == 0) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(195), behaviors::DEFENCE, false);
			spawned_player_units = 0;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = 0;
			command = 0;
		}
		break;
	case 3:
		if (spawned_player_units == 0 && spawned_enemy_units == 0) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(195), behaviors::ATTACK, false);
			spawned_player_units = 0;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = 0;
			command = 0;
		}
		break;
	case 4:
		if (spawned_player_units == 0) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(195), behaviors::ATTACK, false);
			spawned_player_units = 0;
			spawned_enemy_units = 0;
			command = 0;
		}
		break;
	case 5:
		if (spawned_player_units == 0 && spawned_enemy_units == 0) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 5;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(195), behaviors::ATTACK, false);
			spawned_player_units = 0;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = 0;
			command = 0;
		}
		break;
	case 6:
		if (spawned_player_units == 0) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 5;
			spawned_enemy_units = 1,
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(100), behaviors::ATTACK, false);
			spawned_player_units = 0;
			spawned_enemy_units = 0;
			command = 0;
		}
		break;
	case 7:
		if (spawned_player_units == 0) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 10;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units / 2, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units / 2, sc2::Point2D(195));
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(195), behaviors::DEFENCE, false, sc2::Point2D(3), sc2::Point2D(8));
			SetDestination(player_units, sc2::Point2D(5), behaviors::DEFENCE, false, sc2::Point2D(193), sc2::Point2D(198));
			spawned_player_units = 0;
			command = 0;
		}
		break;
	default:
		spawned_player_units = 0;
		spawned_enemy_units = 0;
		command = 0;
		break;
	}
}

void FooBot::CommandsOnHeight() {
	switch (command) {
	case 1:
		if (spawned_player_units == 0) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(53, 40), behaviors::PASSIVE, false);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	case 2:
		if (spawned_player_units == 0) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(95), behaviors::PASSIVE, false);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	case 3:
		if (spawned_player_units == 0 && spawned_enemy_units == 0) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(18, 40), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(28, 25), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 16), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 50), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(55, 60), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(20, 14), 2);
			spawned_player_units = 6;
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(95), behaviors::DEFENCE, false);
		}
		if (enemy_units.size() > 0) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
		}
		if (player_units.size() == spawned_player_units) {
			if (sc2::Point2D(player_units[0].unit->pos) == sc2::Point2D(95)) {
				spawned_player_units = 0;
				spawned_enemy_units = 0;
				command = 0;
			}
		}
		break;
	default:
		spawned_player_units = 0;
		spawned_enemy_units = 0;
		command = 0;
		break;
	}
}

void FooBot::CommandsOnLabyrinth() {
	switch (command) {
	case 1:
		if (spawned_player_units == 0) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(50));
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(95), behaviors::PASSIVE, false);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	case 2:
		if (spawned_player_units == 0) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(95));
			spawn_units = false;
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(50), behaviors::PASSIVE, false);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	case 3:
		if (spawned_player_units == 0 && spawned_enemy_units == 0) {
			spawned_player_units = 0;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(18, 40), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(28, 25), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 16), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 50), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(55, 60), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(20, 14), 2);
			spawned_enemy_units = 6;
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(95), behaviors::DEFENCE, false);
			spawned_player_units = 0;
		}
		if (enemy_units.size() > 0) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
		}
		if (sc2::Point2D(player_units[0].unit->pos) == sc2::Point2D(95)) {
			spawned_player_units = 0;
			spawned_enemy_units = 0;
			command = 0;
		}
		break;
	default:
		spawned_player_units = 0;
		spawned_enemy_units = 0;
		command = 0;
		break;
	}
}

void FooBot::CommandsOnWall() {
}
void FooBot::CommandsOnEmpty20() {
}
void FooBot::CommandsOnSpiral50() {
	switch (command) {
	case 1: {
		if (spawned_player_units == 0) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_SIEGETANK, spawned_player_units, sc2::Point2D(45));
			//SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(45));
		}
		else if (!astar) {
			if (player_units.size() == spawned_player_units) {
				SetDestination(player_units, sc2::Point2D(27), behaviors::DEFENCE, false);
				spawned_player_units = 0;
				command = 0;
			}
		}
		else if (astar) {
			if (astar_units.size() == spawned_player_units) {
				SetDestination(astar_units, sc2::Point2D(27), false);
				spawned_player_units = 0;
				command = 0;
			}
		}
		break;
	}
	case 2: {
		if (spawned_player_units == 0) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_VIKINGFIGHTER, spawned_player_units, sc2::Point2D(45));
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(5), behaviors::DEFENCE, true);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	}
	case 3: {
		if (spawned_player_units == 0 && spawned_enemy_units == 0) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_SIEGETANK, spawned_player_units, sc2::Point2D(45));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_enemy_units, sc2::Point2D(40, 18), 2);
		}
		else if (!astar) {
			if (player_units.size() == spawned_player_units) {
				SetDestination(player_units, sc2::Point2D(27), behaviors::ATTACK, false);
				spawned_player_units = 0;
			}
		}
		else if (astar) {
			if (astar_units.size() == spawned_player_units) {
				SetDestination(astar_units, sc2::Point2D(27), false);
				spawned_player_units = 0;
			}
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = 0;
			command = 0;
		}
		break;
	}
	case 4: {
		if (spawned_player_units == 0) {
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(45));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_enemy_units, sc2::Point2D(42, 18), 2);
		}
		else if (player_units.size() == spawned_player_units) {
			SetDestination(player_units, sc2::Point2D(27), behaviors::DEFENCE, false);
			spawned_player_units = 0;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = 0;
		spawned_enemy_units = 0;
		command = 0;
		break;
	}
	}
}

//! Function that is used to check if a given unit is a structure.
//!< \param unit The unit to be checked.
//!< \return Returns true if the unit is a structure, false otherwise.
bool FooBot::IsStructure(const sc2::Unit * unit) {
	auto& attributes = Observation()->GetUnitTypeData().at(unit->unit_type).attributes; //here
	bool is_structure = false;
	for (const auto& attribute : attributes) {
		if (attribute == sc2::Attribute::Structure) {
			is_structure = true;
		}
	}
	return is_structure;
}

bool FooBot::CheckIfUnitsSpawned(int amount, std::vector<sc2::UnitTypeID> types) {
	int counter = 0;
	for (sc2::UnitTypeID type : types)
		counter += Observation()->GetUnits(sc2::IsUnit(type)).size();
	if (counter == amount)
		return true;
	return false;
}

