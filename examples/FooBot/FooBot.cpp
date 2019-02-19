#include "FooBot.h"

FooBot::FooBot(std::string map) : restarts_(0) {
	if (map == "empty50")		this->map = 1;
	else if (map == "empty200")		this->map = 2;
	else if (map == "height")		this->map = 3;
	else if (map == "labyrinth")	this->map = 4;
	else if (map == "wall")			this->map = 5;
}

void FooBot::OnGameStart() {
	std::cout << "Starting a new game (" << restarts_ << " restarts)" << std::endl;

	map_storage = new MapStorage();
	chat_commands = new ChatCommands(map);
	cuda = new CUDA();
	cuda->InitializeCUDA(map_storage, Observation(), Debug(), Actions(), ActionsFeatureLayer());
	map_storage->Initialize(Observation(), Debug(), Actions(), ActionsFeatureLayer());
	step_clock = clock();

	Debug()->DebugCreateUnit(sc2::UNIT_TYPEID::TERRAN_MARINE, sc2::Point2D(10, 10), 1, 1);
	Debug()->SendDebug();

	//SpawnAllUnits();
}

void FooBot::OnStep() {
	uint32_t game_loop = Observation()->GetGameLoop();
	std::vector<sc2::ChatMessage> in_messages = Observation()->GetChatMessages();
	int command;
	if (in_messages.size() > 0)
	{
		command = chat_commands->AddCommandToList(in_messages[0].message);
	}
	/*if (game_loop % 100 == 0) {
		sc2::Units units = Observation()->GetUnits(sc2::Unit::Alliance::Self);
		for (auto& it_unit : units) {
			sc2::Point2D target = sc2::FindRandomLocation(Observation()->GetGameInfo());
			Actions()->UnitCommand(it_unit, sc2::ABILITY_ID::SMART, target);
		}
	}*/

	/*if (Observation()->GetUnits(sc2::Unit::Alliance::Self).size() > 95 && get_radius) {
		GatherRadius();
		get_radius = false;
	}*/

	cuda->Update(clock() - step_clock);

	step_clock = clock();
}

void FooBot::OnGameEnd() {
	++restarts_;
	std::cout << "Game ended after: " << Observation()->GetGameLoop() << " loops " << std::endl;

	delete cuda;
	delete map_storage;
}

void FooBot::SpawnAllUnits() {
	std::vector<int> unit_IDs = cuda->GetUnitsID();
	for (int unit_ID : unit_IDs)
	{
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

void FooBot::ExecuteCommand(int command) {
	switch (map)
	{
	default:
		break;
	}
}

