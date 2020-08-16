#pragma once

#include <bots/BotClass.h>
#include "GameData.h"

class Experiment {
public:
	Experiment() { }
	virtual ~Experiment() { };
	virtual void process(const BotInputData& input, ControllerInput& output, CarWrapper* car, BallWrapper* ball) { };
};
