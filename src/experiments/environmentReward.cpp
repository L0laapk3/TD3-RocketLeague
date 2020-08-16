
#include "environment.h"
#include "math/math.h"
#include <random>
#include "GameData.h"


// this function uses raw input data and/or Observation data
// it computes the reward aswell as whether the episode is done or not
void Environment::computeReward(const BotInputData& input) {

	float distance = norm(target - input.car.pos);
	//SuperSonicML::Share::cvarManager->log(std::string("dist ")+std::to_string(distance)+std::string(" ")+std::to_string(target[0])+std::string(" ")+std::to_string(car.pos[0]));
	
	reward = -distance / 10000.f;
	//SuperSonicML::Share::cvarManager->log(std::to_string(distance));
	done = distance < 100.f;// || input.car.pos[2] > 100.f;
}



void Environment::reset() {
	
	//SuperSonicML::Share::cvarManager->log(std::string("NEW TARGET ")+std::to_string(target[0])+"\t, "+std::to_string(target[1]));

	// important to call step at the end of reset
	step();
	resetFlag = true;
}