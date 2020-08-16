
#include "environment.h"
#include "GameData.h"

#include "chrono"
#include <random>

Environment::Environment() {
	mainMLThread = std::thread(mainML, this);
};

Environment::~Environment() {
	auto observeLk = std::unique_lock<std::mutex>(mObserve);
	observationAvailable = true;
	stopThread = true;
	observeLk.unlock();
	cvObserve.notify_one();
	mainMLThread.join();
}

void Environment::process(const BotInputData& input, ControllerInput& output, CarWrapper* car, BallWrapper* ball) {
	auto observeLk = std::unique_lock<std::mutex>(mObserve);
	observation.readBotInput(input, target);
	computeReward(input);
	observationAvailable = true;
	observeLk.unlock();
	cvObserve.notify_one();
	// everything from observe() until step() happens now
	auto actionLk = std::unique_lock<std::mutex>(mAction);
	cvAction.wait(actionLk, [&](){ return actionAvailable; });
	action.writeControllerOutput(output);
	actionAvailable = false;
	actionLk.unlock();
	
	if (resetFlag) {
		resetFlag = false;
		static std::random_device rd;
		static std::mt19937 e2(rd());
		static std::uniform_real_distribution<float> xRand(-3000.f, 3000.f);
		static std::uniform_real_distribution<float> yRand(-4000.f, 4000.f);
		static std::uniform_real_distribution<float> yawRand(0.f, 6.28318f);
		car->SetLocation(Vector(xRand(e2), yRand(e2), 17.f));
		car->SetRotation(Rotator(0, yawRand(e2), 0));
		car->SetVelocity(Vector(0, 0, 0));
		car->SetAngularVelocity(Vector(0, 0, 0), false);
		target = vec3c{xRand(e2), yRand(e2), 17.f};
		ball->SetLocation(Vector(target[0], target[1], target[2]));
		ball->SetVelocity(Vector(0, 0, 0));
		ball->SetAngularVelocity(Vector(0, 0, 0), false);
	}
}


void Environment::step() {
	auto actionLk = std::unique_lock<std::mutex>(mAction);
	actionAvailable = true;
	actionLk.unlock();
	cvAction.notify_one();
}

void Environment::observe() {
	auto observeLk = std::unique_lock<std::mutex>(mObserve);
	cvObserve.wait(observeLk, [&](){ return observationAvailable; });
	observationAvailable = false;
	observeLk.unlock();
}