#include "FooBot.h"
#include "../tools.h"

FooBot::FooBot(std::string map,  bool spawn_all_units) {
	this->restarts_ = 0;
	this->spawn_all_units = spawn_all_units;
	this->command = 0;
	this->spawned_player_units = -1;
	this->spawned_enemy_units = -1;
	this->destination_set = false;
	this->astar = false;
	this->astarPF = true;
	this->new_buildings = false;
	if		(map == "empty50")		this->map = 1;
	else if (map == "empty200")		this->map = 2;
	else if (map == "empty20")		this->map = 3;
	else if (map == "spiral50")		this->map = 4;
	else if (map == "easy")			this->map = 5;
	else if (map == "medium")		this->map = 6;
	else if (map == "hard_one")		this->map = 7;
	else if (map == "hard_two")		this->map = 8;
	else							this->map = 0;	//Not a valid test map
}

void FooBot::OnGameStart() {
	std::cout << "Starting a new game (" << restarts_ << " restarts)" << std::endl;

	map_storage = new MapStorage();
	
	map_storage->Initialize(Observation(), Debug(), Actions(), ActionsFeatureLayer(), astar, astarPF);
	map_storage->Test();

	Debug()->DebugFastBuild();
	Debug()->DebugGiveAllResources();

	if (spawn_all_units) SpawnAllUnits();

	PrintMemoryUsage("OnGameStart");
}

void FooBot::OnStep() {
	uint32_t game_loop = Observation()->GetGameLoop();

	//RAM & VRAM stat prints
	if (GetKeyState('P') & 0x8000) PrintMemoryUsage("runtime");
	if (GetKeyState('L') & 0x8000) map_storage->PrintCUDAMemoryUsage("runtime");

	//commands
	if (command == 0) {
		if (GetKeyState('1') & 0x8000) command = 1;
		if (GetKeyState('2') & 0x8000) command = 2; 
		if (GetKeyState('3') & 0x8000) command = 3; 
		if (GetKeyState('4') & 0x8000) command = 4;
		if (GetKeyState('5') & 0x8000) command = 5;
		if (GetKeyState('6') & 0x8000) command = 6;
		if (GetKeyState('7') & 0x8000) command = 7;
		if (GetKeyState('8') & 0x8000) command = 8;
	}

	if (new_buildings) {
		new_buildings = false;
		map_storage->UpdateIMAtsar();
	}

	if (!astar) {	//Same for IM+PF and A*+PF
		//Map transfer PF a/r
		map_storage->TransferPFFromDevice();
	}

	//Set destination
	ExecuteCommand();

	if (!astar) {	//Same for IM+PF and A*+PF
		//Unit transfer
		UpdateHostUnitList();
	}

	if (!astar && !astarPF) {
		//Pathfinding
		UpdateUnitsPaths();
	}
	else if (!astarPF) {
		UpdateAstarPath();
	}
	else {
		UpdateAstarPFPath();
	}

	if (!astar) {
		if (!astarPF) {
			//Map transfer IM
			map_storage->TransferIMFromDevice();
		}

		//Start IM
		//Start PF a/r
		CreateAttractingPFs();
		map_storage->ExecuteDeviceJobs(astarPF);
	}

	Actions()->SendActions();
	Debug()->SendDebug();

	if (spawn_all_units) {
		if (Observation()->GetUnits(sc2::Unit::Alliance::Self).size() > 95 && get_radius) {
			GatherRadius();
			get_radius = false;
		}
	}
}

void FooBot::OnGameEnd() {
	++restarts_;
	std::cout << "Game ended after: " << Observation()->GetGameLoop() << " loops " << std::endl;

	delete map_storage;
}

void FooBot::OnUnitEnterVision(const sc2::Unit * unit) {
	if (!IsStructure(unit) && unit->alliance == sc2::Unit::Alliance::Enemy) {
		Unit new_unit;
		new_unit.unit = unit;
		new_unit.behavior = behaviors::PASSIVE;
		this->enemy_units.push_back(new_unit);
	}
	if (!astar && !astarPF) {
		//kernel launch
		map_storage->ChangeDeviceDynamicMap(unit->pos, unit->radius, -2);
	}
	else {
		if (IsStructure(unit))
			new_buildings = true;
	}
}

void FooBot::OnUnitDestroyed(const sc2::Unit * unit) {
	if (IsStructure(unit)) {
		if (!astar && !astarPF)
			map_storage->ChangeDeviceDynamicMap(unit->pos, unit->radius, 0);
		else
			new_buildings = true;
		return;
	}
	else {
		if (!astar && !astarPF) {
			for (int i = 0; i < player_units.size(); ++i) {
				if (player_units[i].unit == unit) {
					player_units.erase(player_units.begin() + i);
					return;
				}
			}
		}
		else {
			for (int i = 0; i < astar_units.size(); ++i) {
				if (astar_units[i].unit == unit) {
					astar_units.erase(astar_units.begin() + i);
					return;
				}
			}
		}
	}
}

void FooBot::OnUnitCreated(const sc2::Unit * unit) {
	if (IsStructure(unit) && unit->alliance == sc2::Unit::Alliance::Self) {
		if (!astar && !astarPF)
			map_storage->ChangeDeviceDynamicMap(unit->pos, unit->radius, -2);
		else
			new_buildings = true;
	}
	else if (unit->alliance == sc2::Unit::Alliance::Self) {
		if (!astar && !astarPF) {
			Unit new_unit;
			new_unit.unit = unit;
			new_unit.behavior = behaviors::DEFENCE;
			new_unit.dist_traveled = 0;
			new_unit.last_pos = sc2::Point2D(-1, -1);
			new_unit.next_pos = sc2::Point2D(-1, -1);
			this->player_units.push_back(new_unit);
		}
		else {
			sc2::UnitTypes types = Observation()->GetUnitTypeData();
			AstarUnit new_unit;
			for (auto& type : types)
				if (type.unit_type_id == unit->unit_type) {
					new_unit.sight_range = type.sight_range;
					break;
				}
			new_unit.PF_mode = false;
			new_unit.unit = unit;
			new_unit.behavior = behaviors::DEFENCE;
			new_unit.last_pos = sc2::Point2D(-1, -1);
			new_unit.dist_traveled = 0;
			this->astar_units.push_back(new_unit);
		}
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
		CommandsOnEmpty20();
		break;
	case 4:
		CommandsOnSpiral50();
		break;
	case 5:
		CommandsOnEasy();
		break;
	case 6:
		CommandsOnMedium();
		break;
	case 7:
		CommandsOnHardOne();
		break;
	case 8:
		CommandsOnHardTwo();
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

void FooBot::SetDestination(std::vector<Unit>& units_vec, sc2::Point2D pos, behaviors type_of_movement, bool air_unit, sc2::Point2D start, sc2::Point2D end) {
	pos.y = MAP_Y_R - pos.y;
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

void FooBot::SetDestination(std::vector<AstarUnit>& units_vec, sc2::Point2D pos, behaviors type_of_movement, bool air_unit, sc2::Point2D start, sc2::Point2D end) {
	pos.y = MAP_Y_R - pos.y;
	Node agent;
	agent.euc_dist = 0;
	agent.parentX = -1;
	agent.parentY = -1;
	agent.walk_dist = 0;
	if (air_unit) {
	}
	else {
		for (int i = 0; i < units_vec.size(); ++i) {
			units_vec[i].behavior = type_of_movement;
			agent.x = units_vec[i].unit->pos.x;
			agent.y = MAP_Y_R - units_vec[i].unit->pos.y;
			if (start.x == -1) {
				units_vec[i].path = Astar(agent, pos);
			}
			else if (sc2::Point2D(units_vec[i].unit->pos) >= start && sc2::Point2D(units_vec[i].unit->pos) <= end) {
				units_vec[i].path = Astar(agent, pos);
			}
		}
	}
}

void FooBot::SetBehavior(std::vector<Unit>& units_vec, sc2::ABILITY_ID behavior, sc2::Point2D start, sc2::Point2D end, sc2::Point2D patrol_point) {
	for (int i = 0; i < units_vec.size(); ++i) {
		if (start.x == -1) {
			if (patrol_point.x == -1)
				Actions()->UnitCommand(units_vec[i].unit, behavior);
			else
				Actions()->UnitCommand(units_vec[i].unit, behavior, patrol_point);
		}
		else if (sc2::Point2D(units_vec[i].unit->pos) >= start && sc2::Point2D(units_vec[i].unit->pos) <= end) {
			if (patrol_point.x == -1)
				Actions()->UnitCommand(units_vec[i].unit, behavior);
			else
				Actions()->UnitCommand(units_vec[i].unit, behavior, patrol_point);
		}
	}
}

//NOTE!!!! x- and y-coordinates are flipped.
void FooBot::UpdateUnitsPaths() {
	for (int i = 0; i < player_units.size(); ++i) {
		if (player_units[i].destination == nullptr) continue;		//No destination set
		if (player_units[i].destination->map[0][0][0] == -107374176) continue;	//No destination ready to be used

		sc2::Point2D current_pos = player_units[i].unit->pos;
		sc2::Point2D translated_pos = current_pos;
		translated_pos.x = translated_pos.x;
		translated_pos.y = MAP_Y_R - translated_pos.y;

		PrintValues(i, translated_pos);

		if (player_units[i].destination->destination == sc2::Point2D((int)translated_pos.x, (int)translated_pos.y)) {	//Found destination.
			player_units[i].dist_traveled += CalculateEuclideanDistance(player_units[i].last_pos, player_units[i].next_pos);
			player_units[i].path_taken.push_back(player_units[i].next_pos);
			player_units[i].last_pos = translated_pos;
			std::cout << "Done: " << player_units[i].dist_traveled << std::endl;
			std::cout << "Damage taken:" << player_units[i].unit->health_max - player_units[i].unit->health << std::endl;

			map_storage->CreateImage(player_units[i].destination->destination, MAP_X_R, MAP_Y_R, "IM");
			map_storage->AddPathToImage(player_units[i].path_taken, map_storage->RED);
			map_storage->PrintImage(MAP_X_R, MAP_Y_R, "IM");

			player_units[i].destination = nullptr;
			continue;
		}

		if (player_units[i].last_pos.x == -1) {
			player_units[i].last_pos = sc2::Point2D(player_units[i].unit->pos.x, MAP_Y_R - player_units[i].unit->pos.y);
			player_units[i].path_taken.push_back(player_units[i].last_pos);

			map_storage->PrintMap(map_storage->GetGroundAvoidancePFValue((int)translated_pos.y, (int)translated_pos.x), MAP_X_R, MAP_Y_R, "PF");
		}
		else {
			sc2::Point2D pos = sc2::Point2D(player_units[i].unit->pos.x, MAP_Y_R - player_units[i].unit->pos.y);
			player_units[i].path_taken.push_back(pos);
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
		//float unit_radius = 0;
		//if ((p1.x >= p2.x - unit_radius && p1.x <= p2.x + unit_radius && p1.y >= p2.y - unit_radius && p1.y <= p2.y + unit_radius) || p2.x == -1) {
			std::vector<sc2::Point2D> udlr;
			udlr.push_back(sc2::Point2D(p1.x + 0, p1.y + 1));
			udlr.push_back(sc2::Point2D(p1.x + 1, p1.y + 1));
			udlr.push_back(sc2::Point2D(p1.x + 1, p1.y + 0));
			udlr.push_back(sc2::Point2D(p1.x + 1, p1.y - 1));
			udlr.push_back(sc2::Point2D(p1.x + 0, p1.y - 1));
			udlr.push_back(sc2::Point2D(p1.x - 1, p1.y - 1));
			udlr.push_back(sc2::Point2D(p1.x - 1, p1.y + 0));
			udlr.push_back(sc2::Point2D(p1.x - 1, p1.y + 1));

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

				if (min_value > new_value) {
					min_value = min(new_value, min_value);
					next_tile = j;
				}
				//Add check for min_value is better than current_value
			}

			sc2::Point2D new_pos = sc2::Point2D(udlr[next_tile].x, udlr[next_tile].y);
			player_units[i].next_pos = new_pos;
			new_pos.y = MAP_Y_R - new_pos.y;
			if (player_units[i].behavior == behaviors::DEFENCE || player_units[i].behavior == behaviors::PASSIVE)
				Actions()->UnitCommand(player_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
			else if (player_units[i].behavior == behaviors::ATTACK) {
				if (player_units[i].unit->weapon_cooldown == 0)
					Actions()->UnitCommand(player_units[i].unit, sc2::ABILITY_ID::ATTACK, new_pos);
				else
					Actions()->UnitCommand(player_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
			}
		//}
	}
	Debug()->SendDebug();
}

void FooBot::UpdateAstarPath() {
	for (int i = 0; i < astar_units.size(); ++i) {
		if (astar_units[i].path.size() > 0) {
			PrintPath(i);
			sc2::Point2D p1 = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - astar_units[i].unit->pos.y);
			sc2::Point2D p2 = sc2::Point2D(astar_units[i].path.back().x, astar_units[i].path.back().y);
			if (astar_units[i].last_pos.x == -1) {
				astar_units[i].last_pos = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - astar_units[i].unit->pos.y);
				astar_units[i].path_taken.push_back(astar_units[i].last_pos);
			}
			else if (astar_units[i].path.size() > 0) {	//Calculate dist until it's one node left.
				sc2::Point2D pos = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - astar_units[i].unit->pos.y);
				astar_units[i].path_taken.push_back(pos);
				astar_units[i].dist_traveled += CalculateEuclideanDistance(astar_units[i].last_pos, pos);
				astar_units[i].last_pos = pos;
			}
			float unit_radius = astar_units[i].unit->radius < 0.5 ? 0.5 : 1.5;
			if (p1.x >= p2.x - unit_radius && p1.x <= p2.x + unit_radius && p1.y >= p2.y - unit_radius && p1.y <= p2.y + unit_radius) {
				sc2::Point2D last_path_pos = sc2::Point2D(astar_units[i].path.back().x, astar_units[i].path.back().y);
				astar_units[i].path.pop_back();
				if (astar_units[i].path.size() > 0) {
					sc2::Point2D new_pos = sc2::Point2D(astar_units[i].path.back().x, MAP_Y_R - astar_units[i].path.back().y);
					Actions()->UnitCommand(astar_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
				}
				else {	//Calculates dist for last node.
					sc2::Point2D current_pos = astar_units[i].unit->pos;
					current_pos.y = MAP_Y_R - current_pos.y;
					astar_units[i].dist_traveled += CalculateEuclideanDistance(current_pos, last_path_pos);
					std::cout << "Done: " << astar_units[i].dist_traveled << std::endl;
					std::cout << "Damage taken:" << astar_units[i].unit->health_max - astar_units[i].unit->health << std::endl;

					astar_units[i].path_taken.push_back(last_path_pos);

					map_storage->CreateImageDynamic();
					map_storage->AddPathToImage(astar_units[i].path_taken, map_storage->GREEN);
					map_storage->PrintImage(MAP_X_R, MAP_Y_R, "IM_Astar");

				}
			}
		}
	}
}

void FooBot::UpdateAstarPFPath() {
	for (int i = 0; i < astar_units.size(); ++i) {
		if (astar_units[i].path.size() > 0) {
			sc2::Point2D p1 = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - astar_units[i].unit->pos.y);
			sc2::Point2D p2 = sc2::Point2D(astar_units[i].path.back().x, astar_units[i].path.back().y);
			bool astar = true;
			bool new_path = false;
			for (int j = 0; j < enemy_units.size(); ++j) {
				float dist = CalculateEuclideanDistance(astar_units[i].unit->pos, enemy_units[j].unit->pos);
				float enemy_weapon_range = map_storage->GetUnitGroundWeaponRange(enemy_units[i].unit->unit_type);
				enemy_weapon_range = max(enemy_weapon_range, 6.0);

				if (dist < enemy_weapon_range || (dist > enemy_weapon_range && astar_units[i].PF_mode) && dist < astar_units[i].sight_range) {
					astar_units[i].PF_mode = true;
					astar = false;
				}
				else if ((enemy_units[i].unit->display_type == sc2::Unit::Snapshot || enemy_units[i].unit->display_type == sc2::Unit::Visible) && !astar_units[i].PF_mode) {
					astar = true;
				}
				else if (dist > astar_units[i].sight_range && astar_units[i].PF_mode) {
					new_path = true;
				}
			}
			PrintValuesPF(i);
			map_storage->PrintGroundPF("PF");

			//PF
			// If unit is passive, it ignores enemies
			if (astar_units[i].PF_mode && !astar) {
				float current_pf = 0;
				if (astar_units[i].behavior == behaviors::DEFENCE)
					current_pf = map_storage->GetGroundAvoidancePFValue((int)p1.y, (int)p1.x);
				else if (astar_units[i].behavior == behaviors::ATTACK)
					current_pf = map_storage->GetAttractingPF(astar_units[i].unit->unit_type, (int)p1.y, (int)p1.x);

				
				std::vector<sc2::Point2D> udlr;
				udlr.push_back(sc2::Point2D(p1.x + 0, p1.y + 1));
				udlr.push_back(sc2::Point2D(p1.x + 1, p1.y + 1));
				udlr.push_back(sc2::Point2D(p1.x + 1, p1.y + 0));
				udlr.push_back(sc2::Point2D(p1.x + 1, p1.y - 1));
				udlr.push_back(sc2::Point2D(p1.x + 0, p1.y - 1));
				udlr.push_back(sc2::Point2D(p1.x - 1, p1.y - 1));
				udlr.push_back(sc2::Point2D(p1.x - 1, p1.y + 0));
				udlr.push_back(sc2::Point2D(p1.x - 1, p1.y + 1));

				float min_value = 5000;
				int next_tile = 0;
				for (int j = 0; j < udlr.size(); ++j) {
					float new_pf = 0;
					if (astar_units[i].behavior == behaviors::DEFENCE)
						new_pf = map_storage->GetGroundAvoidancePFValue((int)udlr[j].y, (int)udlr[j].x);
					else if (astar_units[i].behavior == behaviors::ATTACK)
						new_pf = map_storage->GetAttractingPF(astar_units[i].unit->unit_type, (int)udlr[j].y, (int)udlr[j].x);

					if (min_value > new_pf) {
						min_value = min(new_pf, min_value);
						next_tile = j;
					}
				}
				if (min_value < current_pf) {
					sc2::Point2D new_pos = sc2::Point2D(udlr[next_tile].x, MAP_Y_R - udlr[next_tile].y);
					if (astar_units[i].behavior == behaviors::DEFENCE)
						Actions()->UnitCommand(astar_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
					else if (astar_units[i].behavior == behaviors::ATTACK) {
						if (astar_units[i].unit->weapon_cooldown == 0)
							Actions()->UnitCommand(astar_units[i].unit, sc2::ABILITY_ID::ATTACK, new_pos);
						else
							Actions()->UnitCommand(astar_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
					}
				}
			}
			//A*
			else if (astar && !new_path) {
				if (astar_units[i].last_pos.x == -1) {
					astar_units[i].last_pos = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - astar_units[i].unit->pos.y);
				}
				else if (astar_units[i].path.size() > 0) {	//Calculate dist until it's one node left.
					sc2::Point2D pos = sc2::Point2D(astar_units[i].unit->pos.x, MAP_Y_R - astar_units[i].unit->pos.y);
					astar_units[i].dist_traveled += CalculateEuclideanDistance(astar_units[i].last_pos, pos);
					astar_units[i].last_pos = pos;
				}
				//if (/*Enemy units in range*/)
				float unit_radius = astar_units[i].unit->radius < 0.5 ? 0.5 : 1.5;
				if ((p1.x >= p2.x - unit_radius && p1.x <= p2.x + unit_radius && 
					p1.y >= p2.y - unit_radius && p1.y <= p2.y + unit_radius) ||
					astar_units[i].unit->orders.size() == 0) {
					sc2::Point2D last_path_pos = sc2::Point2D(astar_units[i].path.back().x, astar_units[i].path.back().y);
					astar_units[i].path.pop_back();
					if (astar_units[i].path.size() > 0) {
						sc2::Point2D new_pos = sc2::Point2D(astar_units[i].path.back().x, MAP_Y_R - astar_units[i].path.back().y);
						Actions()->UnitCommand(astar_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
					}
					else {	//Calculates dist for last node.
						sc2::Point2D current_pos = astar_units[i].unit->pos;
						current_pos.y = MAP_Y_R - current_pos.y;
						astar_units[i].dist_traveled += CalculateEuclideanDistance(current_pos, last_path_pos);
						std::cout << "Done: " << astar_units[i].dist_traveled << std::endl;
						std::cout << "Damage taken:" << astar_units[i].unit->health_max - astar_units[i].unit->health << std::endl;
					}
				}
			}
			//Redo A* path
			else if (new_path) {
				Node agent;
				agent.euc_dist = 0;
				agent.parentX = -1;
				agent.parentY = -1;
				agent.walk_dist = 0;
				agent.x = astar_units[i].unit->pos.x;
				agent.y = MAP_Y_R - astar_units[i].unit->pos.y;
				sc2::Point2D dest = sc2::Point2D(astar_units[i].path.front().x, astar_units[i].path.front().y);
				astar_units[i].path = Astar(agent, dest);
				astar_units[i].PF_mode = false;
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
		agent.y = MAP_Y_R - astar_units[i].unit->pos.y;
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
		adjacadjacent_nodes.push_back(sc2::Point2D(node.x + 0, node.y + 1));	//Down
		adjacadjacent_nodes.push_back(sc2::Point2D(node.x + 1, node.y + 0));	//Right
		adjacadjacent_nodes.push_back(sc2::Point2D(node.x - 0, node.y - 1));	//Up
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

void FooBot::PrintValues(int unit, sc2::Point2D pos) {
	sc2::Point3D pp = player_units[unit].unit->pos;
	pp.z += 0.1f;
	sc2::Point3D translated = pp;
	translated.y = MAP_Y_R - translated.y;
	for (int y = -5; y <= 5; ++y) {
		for (int x = -5; x <= 5; ++x) {
			sc2::Point3D p = sc2::Point3D(translated.x + x, translated.y - y, translated.z);
			if (translated.x < MAP_X_R && translated.y < MAP_Y_R && translated.x >= 0 && translated.y >= 0) {
				int value = player_units[unit].destination->map[(int)p.y][(int)p.x][0];
				Debug()->DebugTextOut(std::to_string(value), sc2::Point3D(int(pp.x + x) + 0.5, int(pp.y + y) + 0.5, pp.z), sc2::Colors::Green, 8);
			}
		}
	}
}

void FooBot::PrintValuesPF(int unit) {
	sc2::Point3D pp = astar_units[unit].unit->pos;
	pp.z += 0.1f;
	sc2::Point3D translated = pp;
	translated.y = MAP_Y_R - translated.y;
	for (int y = -5; y <= 5; ++y) {
		for (int x = -5; x <= 5; ++x) {
			sc2::Point3D p = sc2::Point3D(translated.x + x, translated.y - y, translated.z);
			if (translated.x < MAP_X_R && translated.y < MAP_Y_R && translated.x >= 0 && translated.y >= 0) {
				int pf = map_storage->GetGroundAvoidancePFValue((int)p.y, (int)p.x);
				Debug()->DebugTextOut(std::to_string(pf), sc2::Point3D(int(pp.x + x) + 0.5, int(pp.y + y) + 0.5, pp.z), sc2::Colors::Green, 8);
			}
		}
	}
}

void FooBot::PrintPath(int unit) {
	for (int i = 0; i < astar_units[unit].path.size(); ++i) {
		int value = astar_units[unit].path[i].walk_dist;
		sc2::Point3D pp = sc2::Point3D( astar_units[unit].path[i].x, MAP_Y_R - astar_units[unit].path[i].y, astar_units[unit].unit->pos.z);
		Debug()->DebugTextOut(std::to_string(value), sc2::Point3D(int(pp.x) + 0.5, int(pp.y) + 0.5, pp.z), sc2::Colors::Green, 8);
	}
}

void FooBot::CreateAttractingPFs() {
	std::map<sc2::UnitTypeID, int> player_unit_types;
	if (!astarPF) {
		for (int i = 0; i < player_units.size(); ++i) {
			if (player_units[i].behavior == behaviors::ATTACK) {
				auto& searche = player_unit_types.find(player_units[i].unit->unit_type);
				if (searche == player_unit_types.end())
					player_unit_types[player_units[i].unit->unit_type] = 1;
				else
					searche->second += 1;
			}
		}
	}
	else {
		for (int i = 0; i < astar_units.size(); ++i) {
			if (astar_units[i].behavior == behaviors::ATTACK) {
				auto& searche = player_unit_types.find(astar_units[i].unit->unit_type);
				if (searche == player_unit_types.end())
					player_unit_types[astar_units[i].unit->unit_type] = 1;
				else
					searche->second += 1;
			}
		}
	}
	//säg till map_storage att ett specifikt antal PFs ska göras. Använd player_unit_types för detta.
	for (auto& unit : player_unit_types)
		map_storage->CreateAttractingPF(unit.first);

}

void FooBot::UpdateHostUnitList() {
	host_unit_list.clear();
	if (!astarPF) {
		for (int i = 0; i < player_units.size(); ++i) {
			Entity ent;
			ent.id = map_storage->GetUnitIDInHostUnitVec(player_units[i].unit->unit_type);
			ent.pos = { player_units[i].unit->pos.x, MAP_Y_R - player_units[i].unit->pos.y };
			ent.enemy = false;
			host_unit_list.push_back(ent);
		}
	}
	else {
		for (int i = 0; i < astar_units.size(); ++i) {
			Entity ent;
			ent.id = map_storage->GetUnitIDInHostUnitVec(astar_units[i].unit->unit_type);
			ent.pos = { astar_units[i].unit->pos.x, MAP_Y_R - astar_units[i].unit->pos.y };
			ent.enemy = false;
			host_unit_list.push_back(ent);
		}
	}

	//Same for both IM+PF and A*+PF
	for (int i = 0; i < enemy_units.size(); ++i) {
		Entity ent;
		ent.id = map_storage->GetUnitIDInHostUnitVec(enemy_units[i].unit->unit_type);
		ent.pos = { enemy_units[i].unit->pos.x,  MAP_Y_R - enemy_units[i].unit->pos.y };
		ent.enemy = true;
		host_unit_list.push_back(ent);
	}
	map_storage->UpdateEntityVector(host_unit_list);
}

void FooBot::CommandsOnEmpty50() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			//SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_SIEGETANK, spawned_player_units, sc2::Point2D(5, 5));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(25), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(25), behaviors::DEFENCE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 2: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_enemy_units, sc2::Point2D(26, 21), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(45), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(45), behaviors::DEFENCE, false);
			spawned_player_units = -1;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 3: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(25, 25), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
			else SetDestination(astar_units, sc2::Point2D(45), behaviors::ATTACK, false);
			spawned_player_units = -1;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 4: {
		if (spawned_player_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(25, 25), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
			else SetDestination(astar_units, sc2::Point2D(45), behaviors::ATTACK, false);
			spawned_player_units = -1;
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 5: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 5;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(25, 25), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(25), behaviors::ATTACK, false);
			else SetDestination(astar_units, sc2::Point2D(25), behaviors::ATTACK, false);
			spawned_player_units = -1;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 6: {
		if (spawned_player_units == -1) {
			spawned_player_units = 5;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(25, 25), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
			else SetDestination(astar_units, sc2::Point2D(45), behaviors::ATTACK, false);
			spawned_player_units = -1;
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 7: {
		if (spawned_player_units == -1) {
			spawned_player_units = 10;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(45, 45));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) {
				SetDestination(player_units, sc2::Point2D(45), behaviors::DEFENCE, false, sc2::Point2D(3), sc2::Point2D(8));
				SetDestination(player_units, sc2::Point2D(5), behaviors::DEFENCE, false, sc2::Point2D(43), sc2::Point2D(48));
			}
			else {
				SetDestination(astar_units, sc2::Point2D(45), behaviors::DEFENCE, false, sc2::Point2D(3), sc2::Point2D(8));
				SetDestination(astar_units, sc2::Point2D(5), behaviors::DEFENCE, false, sc2::Point2D(43), sc2::Point2D(48));
			}
			spawned_player_units = -1;
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnEmpty200() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(195), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(195), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 2: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(195), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(195), behaviors::DEFENCE, false);
			spawned_player_units = -1;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 3: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(195), behaviors::ATTACK, false);
			else SetDestination(astar_units, sc2::Point2D(195), behaviors::ATTACK, false);
			spawned_player_units = -1;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 4: {
		if (spawned_player_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(195), behaviors::ATTACK, false);
			else SetDestination(astar_units, sc2::Point2D(195), behaviors::ATTACK, false);
			spawned_player_units = -1;
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 5: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 5;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(195), behaviors::ATTACK, false);
			else SetDestination(astar_units, sc2::Point2D(195), behaviors::ATTACK, false);
			spawned_player_units = -1;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 6: {
		if (spawned_player_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 5;
			spawned_enemy_units = 1,
				SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, spawned_enemy_units, sc2::Point2D(100), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(100), behaviors::ATTACK, false);
			else SetDestination(astar_units, sc2::Point2D(100), behaviors::ATTACK, false);
			spawned_player_units = -1;
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 7: {
		if (spawned_player_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 10;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units / 2, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units / 2, sc2::Point2D(195));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) {
				SetDestination(player_units, sc2::Point2D(45), behaviors::DEFENCE, false, sc2::Point2D(3), sc2::Point2D(8));
				SetDestination(player_units, sc2::Point2D(45), behaviors::DEFENCE, false, sc2::Point2D(193), sc2::Point2D(198));
			}
			else {
				SetDestination(astar_units, sc2::Point2D(45), behaviors::DEFENCE, false, sc2::Point2D(3), sc2::Point2D(8));
				SetDestination(astar_units, sc2::Point2D(45), behaviors::DEFENCE, false, sc2::Point2D(193), sc2::Point2D(198));
			}
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnHeight() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(53, 40), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(53, 40), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 2: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(95), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(95), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 3: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			spawned_player_units = 1;
			spawned_player_units = 6;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(18, 40), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(28, 25), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 16), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 50), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(55, 60), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(20, 14), 2);
		}
		else if ((player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) && !destination_set) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(95), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(95), behaviors::DEFENCE, false);
			destination_set = true;
		}
		if (enemy_units.size() > 0) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
		}
		if (player_units.size() == spawned_player_units) {
			if (sc2::Point2D(player_units[0].unit->pos) == sc2::Point2D(95)) {
				destination_set = false;
				spawned_player_units = -1;
				spawned_enemy_units = -1;
				command = 0;
			}
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnLabyrinth() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(50));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(95), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(95), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 2: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(95));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(50), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(50), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 3: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			spawned_player_units = -1;
			spawned_enemy_units = 6;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(18, 40), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(28, 25), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 16), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 50), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(55, 60), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(20, 14), 2);
		}
		else if ((player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) && !destination_set) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(95), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(95), behaviors::DEFENCE, false);
			destination_set = true;
		}
		if (enemy_units.size() > 0) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
		}
		if (sc2::Point2D(player_units[0].unit->pos) == sc2::Point2D(95)) {
			destination_set = false;
			spawned_player_units = -1;
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnWall() {
}

void FooBot::CommandsOnEmpty20() {
}

void FooBot::CommandsOnSpiral50() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_SIEGETANK, spawned_player_units, sc2::Point2D(45));
			//SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(45));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(27), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(27), behaviors::DEFENCE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 2: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_VIKINGFIGHTER, spawned_player_units, sc2::Point2D(45));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(5), behaviors::DEFENCE, true);
			else SetDestination(astar_units, sc2::Point2D(5), behaviors::DEFENCE, true);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 3: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			Debug()->DebugEnemyControl();
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			//SpawnUnits(sc2::UNIT_TYPEID::TERRAN_SIEGETANK, spawned_player_units, sc2::Point2D(45));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(45));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_enemy_units, sc2::Point2D(41, 18), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(27), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(27), behaviors::DEFENCE, false);
			spawned_player_units = -1;
		}
		if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::HOLDPOSITION);
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	case 4: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			spawned_player_units = 1;
			spawned_enemy_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(45));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_enemy_units, sc2::Point2D(41, 18), 2);
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(27), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(27), behaviors::DEFENCE, false);
			spawned_player_units = -1;
			spawned_enemy_units = -1;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnEasy() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(9, 6));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(47, 50), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(47, 50), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	case 2: {
		if (spawned_player_units == -1 && spawned_enemy_units == -1) {
			spawned_player_units = 1;
			spawned_enemy_units = 3;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(10, 7));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(40, 8), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(19, 18), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(42, 36), 2);
			Debug()->DebugEnemyControl();
			Debug()->DebugShowMap();
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(47, 50), behaviors::DEFENCE, false);
			else SetDestination(astar_units, sc2::Point2D(47, 50), behaviors::DEFENCE, false);
			spawned_player_units = -1;
		}
		else if (enemy_units.size() == spawned_enemy_units) {
			SetBehavior(enemy_units, sc2::ABILITY_ID::PATROL, { 39, 7 }, { 41, 9 }, {29, 8});
			SetBehavior(enemy_units, sc2::ABILITY_ID::PATROL, { 18, 17 }, { 20, 19 }, { 30, 18 });
			SetBehavior(enemy_units, sc2::ABILITY_ID::PATROL, { 41, 35 }, { 42, 37 }, { 32, 36 });

			spawned_enemy_units = -1;
			command = 0;
			//Debug()->DebugShowMap();
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnMedium() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(10, 7));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(47, 50), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(47, 50), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnHardOne() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(10, 7));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(47, 50), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(47, 50), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnHardTwo() {
	switch (command) {
	case 1: {
		if (spawned_player_units == -1) {
			spawned_player_units = 1;
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, spawned_player_units, sc2::Point2D(10, 7));
		}
		else if (player_units.size() == spawned_player_units || astar_units.size() == spawned_player_units) {
			if (!astar && !astarPF) SetDestination(player_units, sc2::Point2D(47, 50), behaviors::PASSIVE, false);
			else SetDestination(astar_units, sc2::Point2D(47, 50), behaviors::PASSIVE, false);
			spawned_player_units = -1;
			command = 0;
		}
		break;
	}
	default: {
		spawned_player_units = -1;
		spawned_enemy_units = -1;
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

