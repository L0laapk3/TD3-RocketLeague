
#include "environment.h"
#include "math/math.h"
#include <random>
#include "GameData.h"


// this function uses raw input data and/or Observation data
// it computes the reward aswell as whether the episode is done or not
void Environment::computeReward(const BotInputData& input) {
	auto car = input.car;

	float distance = norm(target - car.pos);
	//SuperSonicML::Share::cvarManager->log(std::string("dist ")+std::to_string(distance)+std::string(" ")+std::to_string(target[0])+std::string(" ")+std::to_string(car.pos[0]));
	
	reward = -distance;
	//SuperSonicML::Share::cvarManager->log(std::to_string(distance));
	done = distance < 100.f;
}



void Environment::reset() {
	static std::random_device rd;
	static std::mt19937 e2(rd());
	static std::uniform_real_distribution<float> xRand(-4000.f, 4000.f);
	static std::uniform_real_distribution<float> yRand(-5000.f, 5000.f);
	target = vec3c{xRand(e2), yRand(e2), 17.f};
	
	//SuperSonicML::Share::cvarManager->log(std::string("NEW TARGET ")+std::to_string(target[0])+"\t, "+std::to_string(target[1]));

	// important to call step at the end of reset
	step();
}