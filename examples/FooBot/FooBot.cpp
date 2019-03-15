#include "FooBot.h"

FooBot::FooBot(std::string map, bool spaw_alla_units) : 
	restarts_(0), 
	spawn_all_units(spaw_alla_units) {
	this->command = 0;
	this->spawned_units = 0;
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
	
	map_storage->Initialize(Observation(), Debug(), Actions(), ActionsFeatureLayer());
	map_storage->Test();

	step_clock = clock();

	Debug()->DebugFastBuild();
	Debug()->DebugGiveAllResources();

	/*Debug()->DebugCreateUnit(sc2::UNIT_TYPEID::ZERG_DRONE, sc2::Point2D(10, 10), 1, 1);
	Debug()->SendDebug();*/

	if (spawn_all_units) SpawnAllUnits();
}

void FooBot::OnStep() {
	uint32_t game_loop = Observation()->GetGameLoop();
	std::vector<sc2::ChatMessage> in_messages = Observation()->GetChatMessages();
	if (in_messages.size() > 0 && command == 0)
		command = chat_commands->ParseCommands(in_messages[0].message);
	ExecuteCommand();
	UpdateUnitsPaths();
	UpdateHostUnitList();
	map_storage->Update(clock() - step_clock);

	Actions()->SendActions();
	step_clock = clock();

	if (spawn_all_units)
		if (Observation()->GetUnits(sc2::Unit::Alliance::Self).size() > 95 && get_radius) {
			GatherRadius();
			get_radius = false;
		}
	CreatePFs();
}

void FooBot::OnGameEnd() {
	++restarts_;
	std::cout << "Game ended after: " << Observation()->GetGameLoop() << " loops " << std::endl;

	delete map_storage;
	delete chat_commands;
}

void FooBot::OnUnitEnterVision(const sc2::Unit * unit) {
	if (!IsStructure(unit) && unit->alliance == sc2::Unit::Alliance::Enemy) {
		FooBot::Unit new_unit;
		new_unit.unit = unit;
		new_unit.behavior = behaviors::PASSIVE;
		this->enemy_units.push_back(new_unit);
	}
}

void FooBot::OnUnitDestroyed(const sc2::Unit * unit) {
	//Delete unit if it died
	for (int i = 0; i < player_units.size(); ++i) {
		if (player_units[i].unit == unit) {
			player_units.erase(player_units.begin() + i);
			return;
		}
	}
	//Remove structure if destroyed
	//Remove structure from IMs
}

void FooBot::OnUnitCreated(const sc2::Unit * unit) {
	if (!IsStructure(unit) && unit->alliance == sc2::Unit::Alliance::Self) {
		FooBot::Unit new_unit;
		new_unit.unit = unit;
		new_unit.behavior = behaviors::DEFENCE;
		this->player_units.push_back(new_unit);
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

void FooBot::SetDestination(std::vector<FooBot::Unit>& units_vec, sc2::Point2D pos, behaviors type_of_movement, bool air_unit) {
	pos.y = MAP_Y_R - 1 - pos.y;
	if (air_unit) {
		for (int i = 0; i < units_vec.size(); ++i) {
			units_vec[i].behavior = type_of_movement;
			units_vec[i].destination = &map_storage->GetAirDestination(pos);
		}
	}
	else {
		for (int i = 0; i < units_vec.size(); ++i) {
			units_vec[i].behavior = type_of_movement;
			units_vec[i].destination = &map_storage->GetGroundDestination(pos);
		}
	}
	
}

void FooBot::SetBehavior(sc2::Units units, sc2::ABILITY_ID behavior) {
	Actions()->UnitCommand(units, behavior);
}

//NOTE!!!! x- and y-coordinates are fliped.
void FooBot::UpdateUnitsPaths() {
	float PF[MAP_X_R][MAP_Y_R][1];
	map_storage->GetGroundAvoidancePF(PF);
	for (int i = 0; i < player_units.size(); ++i) {
		if (player_units[i].destination == nullptr) continue;	//No destination to go to

		sc2::Point2D current_pos = player_units[i].unit->pos;
		sc2::Point2D translated_pos = current_pos;
		translated_pos.x = translated_pos.x;
		translated_pos.y = MAP_Y_R - 1 - translated_pos.y;

		if (player_units[i].destination->destination == sc2::Point2D((int)translated_pos.x, (int)translated_pos.y)) {	//Found destination.
			player_units[i].destination = nullptr;
			continue;
		}
		//Get the value from the IM and PF to determine the total value of the tile.
		int current_value = player_units[i].destination->map[(int)translated_pos.y][(int)translated_pos.x][0];
		//current_value += PF[(int)translated_pos.y][(int)translated_pos.x][0];

		std::vector<sc2::Point2D> udlr;				//y,  x
		udlr.push_back(translated_pos + sc2::Point2D( 0,  1));	//Down
		udlr.push_back(translated_pos + sc2::Point2D( 1,  1));	//Down, right
		udlr.push_back(translated_pos + sc2::Point2D( 1,  0));	//Right
		udlr.push_back(translated_pos + sc2::Point2D( 1, -1));	//Up, right
		udlr.push_back(translated_pos + sc2::Point2D( 0, -1));	//Up
		udlr.push_back(translated_pos + sc2::Point2D(-1, -1));	//Up, left
		udlr.push_back(translated_pos + sc2::Point2D(-1,  0));	//Left
		udlr.push_back(translated_pos + sc2::Point2D(-1,  1));	//Down, left

		float min_value = 5000;
		int next_tile = 0;
		for (int j = 0; j < udlr.size(); ++j) {
			//Get the value from the IM and PF to determine the total value of the new tile.
			int new_value = player_units[i].destination->map[(int)udlr[j].y][(int)udlr[j].x][0];
			//new_value += PF[(int)udlr[j].y][(int)udlr[j].x][0];

			if (new_value < 0) continue;	//Unpathable terrain
			//This needs to be modified.
			//if ((current_value - new_value) > 2) continue;	//Invalid move
			if (min_value > new_value) {
				min_value = min(new_value, min_value);
				next_tile = j;
			}
		}

		sc2::Point2D new_pos = udlr[next_tile];
		new_pos.y = MAP_Y_R - 1 - new_pos.y;
		Actions()->UnitCommand(player_units[i].unit, sc2::ABILITY_ID::MOVE, new_pos);
	}
}

void FooBot::CreatePFs() {
	host_unit_list.clear();
	std::map<int, int> player_unit_types;
	for (int i = 0; i < player_units.size(); ++i) {
		std::map<int, int>::iterator iter = player_unit_types.find(player_units[i].unit->unit_type);
		if (iter == player_unit_types.end())
			player_unit_types[map_storage->GetUnitIDInHostUnitVec(player_units[i].unit->unit_type)] = 1;
		else
			iter->second += 1;
	}
	//säg till map_storage att ett specifikt antal PFs ska göras. Använd player_unit_types för detta.
	for (auto& unit : player_unit_types)
		map_storage->CreateAttractingPF(unit.first);

}

void FooBot::UpdateHostUnitList() {
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
	map_storage->SetEntityVector(host_unit_list);
}

void FooBot::CommandsOnEmpty50() {
	switch (command) {
	case 1: {
		if (spawned_units == 0) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			spawned_units = 1;
		}
		else if (player_units.size() == spawned_units) {
			SetDestination(player_units, sc2::Point2D(25), behaviors::DEFENCE, false);
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	case 2: {
		if (spawned_units == 0) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_units = 1;
		}
		else if (player_units.size() == spawned_units) {
			SetDestination(player_units, sc2::Point2D(45), behaviors::PASSIVE, false);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	case 3: {
		if (spawned_units == 0) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_units = 1;
		}
		else if (player_units.size() == spawned_units) {
			SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	case 4: {
		if (spawned_units == 0) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_units = 1;
		}
		else if (player_units.size() == spawned_units) {
			SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	case 5: {
		if (spawned_units == 0) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_units = 5;
		}
		else if (player_units.size() == 5) {
			SetDestination(player_units, sc2::Point2D(25), behaviors::ATTACK, false);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	case 6: {
		if (spawned_units == 0) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawned_units = 5;
		}
		else if (player_units.size() == spawned_units) {
			SetDestination(player_units, sc2::Point2D(45), behaviors::ATTACK, false);
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	case 7: {
		if (spawned_units == 0) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(45, 45));
			spawned_units = 10;
		}
		else if (player_units.size() == 10) {
			//This needs to be fixed. Converted to "new way"
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(45, 45), sc2::ABILITY_ID::MOVE, sc2::Point2D(3), sc2::Point2D(8));
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(5, 5), sc2::ABILITY_ID::MOVE, sc2::Point2D(43), sc2::Point2D(48));
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	default: {
		spawned_units = 0;
		command = 0;
		break;
	}
	}
}

void FooBot::CommandsOnEmpty200() {
	switch (command) {
	case 1:
		if (spawn_units) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5));
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(195), sc2::ABILITY_ID::MOVE);
			spawn_units = true;
			command = 0;
		}
		break;
	case 2:
		if (spawn_units) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(100), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(195), sc2::ABILITY_ID::MOVE);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawn_units = true;
			command = 0;
		}
		break;
	case 3:
		if (spawn_units) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(100), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(195), sc2::ABILITY_ID::ATTACK);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawn_units = true;
			command = 0;
		}
		break;
	case 4:
		if (spawn_units) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(100), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(195), sc2::ABILITY_ID::ATTACK);
			spawn_units = true;
			command = 0;
		}
		break;
	case 5:
		if (spawn_units) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(100), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(5, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(100), sc2::ABILITY_ID::ATTACK);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawn_units = true;
			command = 0;
		}
		break;
	case 6:
		if (spawn_units) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(100), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(5, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(100), sc2::ABILITY_ID::ATTACK);
			spawn_units = true;
			command = 0;
		}
		break;
	case 7:
		if (spawn_units) {
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 5, sc2::Point2D(195));
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(10, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(195), sc2::ABILITY_ID::MOVE, sc2::Point2D(3), sc2::Point2D(8));
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(5), sc2::ABILITY_ID::MOVE, sc2::Point2D(193), sc2::Point2D(198));
			spawn_units = true;
			command = 0;
		}
		break;
	default:
		spawn_units = true;
		command = 0;
		break;
	}
}

void FooBot::CommandsOnHeight() {
	switch (command) {
	case 1:
		if (spawn_units) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5));
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(50, 60), sc2::ABILITY_ID::MOVE);
			spawn_units = true;
			command = 0;
		}
		break;
	case 2:
		if (spawn_units) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5));
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(95), sc2::ABILITY_ID::MOVE);
			spawn_units = true;
			command = 0;
		}
		break;
	case 3:
		if (spawn_units) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(18, 40), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(28, 25), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 16), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 50), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(55, 60), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(20, 14), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(95), sc2::ABILITY_ID::MOVE);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetBehavior(Observation()->GetUnits(sc2::Unit::Alliance::Enemy, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::ABILITY_ID::HOLDPOSITION);
		}
		if (Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)).size() == 1) {
			if (sc2::Point2D(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE))[0]->pos) == sc2::Point2D(95)) {
				spawn_units = true;
				command = 0;
			}
		}
		break;
	default:
		command = 0;
		spawn_units = true;
		break;
	}
}

void FooBot::CommandsOnLabyrinth() {
	switch (command) {
	case 1:
		if (spawn_units) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(50));
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(95), sc2::ABILITY_ID::MOVE);
			spawn_units = true;
			command = 0;
		}
		break;
	case 2:
		if (spawn_units) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(95));
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(50), sc2::ABILITY_ID::MOVE);
			spawn_units = true;
			command = 0;
		}
		break;
	case 3:
		if (spawn_units) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5));
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(18, 40), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(28, 25), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 16), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(75, 50), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(55, 60), 2);
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(20, 14), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(95), sc2::ABILITY_ID::MOVE);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE })) {
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::ABILITY_ID::HOLDPOSITION);
		}
		if (sc2::Point2D(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE))[0]->pos) == sc2::Point2D(95)) {
			spawn_units = true;
			command = 0;
		}
		break;
	default:
		command = 0;
		spawn_units = true;
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
		if (spawned_units == 0) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(45));
			spawned_units = 1;
		}
		else if (player_units.size() == spawned_units) {
			SetDestination(player_units, sc2::Point2D(7), behaviors::DEFENCE, false);
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	case 2: {
		if (spawned_units == 0) {
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_VIKINGFIGHTER, 1, sc2::Point2D(45));
			spawned_units = 1;
		}
		else if (player_units.size() == spawned_units) {
			SetDestination(player_units, sc2::Point2D(5), behaviors::DEFENCE, true);
			spawned_units = 0;
			command = 0;
		}
		break;
	}
	default: {
		spawned_units = 0;
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

