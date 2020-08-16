#ifndef PROJECT_ENVIRONMENT_H
#define PROJECT_ENVIRONMENT_H

#include <array>
#include <thread>
#include <mutex>
#include "Experiment.h"
#include <bots/BotClass.h>

#include "observation.h"
#include "GameData.h"
#include "action.h"


// responsible for starting mainML.cpp

class Environment : public Experiment {
private:
	std::thread mainMLThread;
	std::mutex mAction;
	std::mutex mObserve;
	std::condition_variable cvAction;
	std::condition_variable cvObserve;
	bool observationAvailable = false;
	bool actionAvailable = false;

	bool resetFlag = false;

	void computeReward(const BotInputData& input);

public:
	Environment();
	virtual ~Environment();
	virtual void process(const BotInputData& input, ControllerInput& output, CarWrapper* car);

	// above this line: internal and SuperSonicML definitions
	// ----------------------------------------------------------- //
	// below this line: definitions for usage in mainML

	bool stopThread = false;

	// observe() must be called before reading observations everytime after step() or reset().

	// todo: replace with something more general
	vec3c target;

	void observe();

	Observation observation;
	Action action;
	float reward;
	bool done;

	void step();
	void reset();
};


void mainML(Environment*);


#endif