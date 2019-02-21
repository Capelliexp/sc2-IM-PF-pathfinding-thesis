#include "FooBot.h"

FooBot::FooBot(std::string map, bool spaw_alla_units) : 
	restarts_(0), 
	spawn_all_units(spaw_alla_units) {
	this->command = 0;
	if (map == "empty50")			this->map = 1;
	else if (map == "empty200")		this->map = 2;
	else if (map == "height")		this->map = 3;
	else if (map == "labyrinth")	this->map = 4;
	else if (map == "wall")			this->map = 5;
	else							this->map = 0;	//Not a valid testmap
}

void FooBot::OnGameStart() {
	std::cout << "Starting a new game (" << restarts_ << " restarts)" << std::endl;

	map_storage = new MapStorage();
	chat_commands = new ChatCommands(map);
	cuda = new CUDA();
	cuda->InitializeCUDA(map_storage, Observation(), Debug(), Actions(), ActionsFeatureLayer());
	map_storage->Initialize(Observation(), Debug(), Actions(), ActionsFeatureLayer());
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

	/*if (game_loop % 100 == 0) {
		sc2::Units units = Observation()->GetUnits(sc2::Unit::Alliance::Self);
		for (auto& it_unit : units) {
			sc2::Point2D target = sc2::FindRandomLocation(Observation()->GetGameInfo());
			Actions()->UnitCommand(it_unit, sc2::ABILITY_ID::SMART, target);
		}
	}*/

	if (spawn_all_units) {
		if (Observation()->GetUnits(sc2::Unit::Alliance::Self).size() > 95 && get_radius) {
			GatherRadius();
			get_radius = false;
		}
	}
	cuda->Update(clock() - step_clock);

	step_clock = clock();
}

void FooBot::OnGameEnd() {
	++restarts_;
	std::cout << "Game ended after: " << Observation()->GetGameLoop() << " loops " << std::endl;

	delete cuda;
	delete map_storage;
	delete chat_commands;
}

void FooBot::SpawnAllUnits() {
	std::vector<int> unit_IDs = cuda->GetUnitsID();
	for (int unit_ID : unit_IDs) {
		sc2::Point2D p = sc2::FindRandomLocation(Observation()->GetGameInfo());
		Debug()->DebugCreateUnit(sc2::UNIT_TYPEID(unit_ID), p, 1, 1);
		Debug()->SendDebug();
	}
}

void FooBot::GatherRadius() {
	sc2::Units units = Observation()->GetUnits(sc2::Unit::Alliance::Self);
	std::vector<float> unitRadius(cuda->GetSizeOfUnitInfoList());
	for (auto& unit : units) {
		int pos = cuda->GetPosOFUnitInHostUnitVec(unit->unit_type);
		unitRadius[pos] = unit->radius;
	}
	cuda->SetRadiusForUnits(unitRadius);
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
	default:
		command = 0;
		break;
	}
}

void FooBot::SpawnUnits(sc2::UNIT_TYPEID unit_id, int amount, sc2::Point2D pos, int player) {
	Debug()->DebugCreateUnit(unit_id, pos, player, amount);
	Debug()->SendDebug();
}

void FooBot::SetDestination(sc2::Units units, sc2::Point2D pos, sc2::ABILITY_ID type_of_movement) {
	//Start custom pathfinding here
	Actions()->UnitCommand(units, type_of_movement, pos);
}

void FooBot::SetBehavior(sc2::Units units, sc2::ABILITY_ID behavior)
{
	Actions()->UnitCommand(units, behavior);
}

void FooBot::CommandsOnEmpty50() {
	switch (command) {
	case 1:
		if (spawn_units)
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE }))
		{
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(45, 45), sc2::ABILITY_ID::MOVE);
			command = 0;
			spawn_units = true;
		}
		break;
	case 2:
		if (spawn_units)
		{
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE }))
		{
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(45, 45), sc2::ABILITY_ID::MOVE);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT }))
		{
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawn_units = true;
			command = 0;
		}
		break;
	case 3:
		if (spawn_units)
		{
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE }))
		{
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(45, 45), sc2::ABILITY_ID::ATTACK);
		}
		if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::PROTOSS_ZEALOT }))
		{
			SetBehavior(Observation()->GetUnits(sc2::IsUnit(sc2::UNIT_TYPEID::PROTOSS_ZEALOT)), sc2::ABILITY_ID::HOLDPOSITION);
			spawn_units = true;
			command = 0;
		}
		break;
	case 4:
		if (spawn_units)
		{
			Debug()->DebugEnemyControl();
			SpawnUnits(sc2::UNIT_TYPEID::TERRAN_MARINE, 1, sc2::Point2D(5, 5));
			SpawnUnits(sc2::UNIT_TYPEID::PROTOSS_ZEALOT, 1, sc2::Point2D(25, 25), 2);
			spawn_units = false;
		}
		else if (CheckIfUnitsSpawned(1, { sc2::UNIT_TYPEID::TERRAN_MARINE }))
		{
			SetDestination(Observation()->GetUnits(sc2::Unit::Alliance::Self, sc2::IsUnit(sc2::UNIT_TYPEID::TERRAN_MARINE)), sc2::Point2D(45, 45), sc2::ABILITY_ID::MOVE);
		}
		break;
	case 5:
		break;
	case 6:
		break;
	case 7:
		break;
	default:
		command = 0;
		break;
	}
}

void FooBot::CommandsOnEmpty200() {
}

void FooBot::CommandsOnHeight() {
}

void FooBot::CommandsOnLabyrinth() {
}

void FooBot::CommandsOnWall() {
}

bool FooBot::CheckIfUnitsSpawned(int amount, std::vector<sc2::UnitTypeID> types) {
	int counter = 0;
	for (sc2::UnitTypeID type : types)
		counter += Observation()->GetUnits(sc2::IsUnit(type)).size();
	if (counter == amount)
		return true;
	return false;
}

