
#include "environment.h"
#include "agent.h"
#include "GameData.h"
#include <thread>


void trainEnvironment(Environment* env, Agent& agent) {
	env->reset();
	env->observe();

	float total_reward = 0;
	int total_steps = 0;
	for (; total_steps < 1000; total_steps++) {
		total_steps++;
		agent.act(env->observation, env->action);
		auto oldObservation = env->observation;
		env->step();
		env->observe();
		total_reward += env->reward;
		agent.addExperienceState(oldObservation, env->action, env->reward, env->observation, env->done);
		if (env->done || env->stopThread)
			break;
	}

	SuperSonicML::Share::cvarManager->log(std::string("total reward: ")+std::to_string(total_reward)+"\tsteps: "+std::to_string(total_steps));
}


void learnLoop(Environment* env, Agent* agent) {
	while (!env->stopThread)
		if (*SuperSonicML::Share::cvarEnableTraining)
			agent->learn();
		else
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

}


std::thread learnThread;
void mainML(Environment* env) {
	Agent agent = Agent(Observation::size, Action::size, 1);
	learnThread = std::thread(learnLoop, env, &agent);

	env->observe();
	while (!env->stopThread)
		trainEnvironment(env, agent);
	
	learnThread.join();
}

