#pragma once

#include <bots/BotClass.h>

class Experiment {
public:
	Experiment() { }
	virtual ~Experiment() { };
	virtual void process(const BotInputData& input, ControllerInput& output) { };
};
