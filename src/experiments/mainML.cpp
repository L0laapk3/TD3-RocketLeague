
#include "environment.h"
#include "agent.h"
#include "GameData.h"
#include <thread>
#include <chrono>


auto actCount = 0ULL;
auto learnCount = 0ULL;
float totalReward = 0.f;
auto totalSteps = 0ULL;
float avgLastDistance = 0.f;
void trainEnvironment(Environment* env, Agent& agent) {
	env->reset();
	env->observe();

	for (auto i = 0ULL; i < 1000; i++) {
		agent.act(env->observation, env->action);
		actCount++;
		auto oldObservation = env->observation;
		env->step();
		env->observe();
		totalReward += env->reward;
		totalSteps++;
		agent.addExperienceState(oldObservation, env->action, env->reward, env->observation, env->done);
		if (env->done || env->stopThread)
			break;
	}
	float distance = env->reward * 10000.f;
	if (avgLastDistance == 0.f)
		avgLastDistance = distance;
	else
		avgLastDistance = .99f * avgLastDistance + .01f * distance;

	
    static auto lastMsg = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastMsg).count() / 1000000.f;

    if (elapsed >= 2.f) {
        char buf[200];
        sprintf_s(buf, "%d tps | %d lps | %.2f avgReward | %d avgLastDistance", (int)(actCount / elapsed), (int)(learnCount / elapsed), (totalReward / totalSteps), (int)avgLastDistance);
        SuperSonicML::Share::cvarManager->log(buf);
        lastMsg = now;
        actCount = 0;
        learnCount = 0;
    }

	// if (*SuperSonicML::Share::cvarEnableTraining)
	// 	SuperSonicML::Share::cvarManager->log(std::string("total reward: ")+std::to_string(total_reward)+"\tsteps: "+std::to_string(total_steps));
}


void learnLoop(Environment* env, Agent* agent) {
	while (!env->stopThread)
		if (!*SuperSonicML::Share::cvarEnableTraining)
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		else {
			agent->learn();
			learnCount += 512;
		}

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

