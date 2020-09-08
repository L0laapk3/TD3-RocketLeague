
#include "environment.h"
#include "agent.h"
#include "GameData.h"
#include <thread>
#include <chrono>



auto actCount = 0ULL;
auto learnCount = 0ULL;
float totalReward = 0.f;
auto totalSteps = 0ULL;
float lastDist = 0;
void trainEnvironment(Environment* env, Agent& agent) {
	env->reset();
	env->observe();

	for (auto i = 0ULL; i < 600; i++) {
		agent.act(env->observation, env->action);
		actCount++;
		auto oldObservation = env->observation;
		env->step();
		// if (i % 16 == 0) {
		// 	agent.learn();
		// 	learnCount += 64;
		// }
		env->observe();
		totalReward += env->reward;
		totalSteps++;
		agent.addExperienceState(oldObservation, env->action, env->reward, env->observation, env->done);
		if (env->done || env->stopThread)
			break;
	}
	lastDist = .9f * lastDist + .1f * env->reward * -10000.f;
	
    static auto lastMsg = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastMsg).count() / 1000000.f;

    if (elapsed >= 2.f) {
        char buf[200];
		
        sprintf_s(buf, "%d tps (%dx) | %d lps | %.2f avgReward | %d avgLastDist | %s %s", (int)(actCount / elapsed), (int)(actCount / elapsed / 120.f), (int)(learnCount / elapsed), (totalReward / totalSteps), (int)lastDist, agent.actorLocal->toString().c_str(), agent.criticLocal->toString().c_str());
        SuperSonicML::Share::cvarManager->log(buf);
        lastMsg = now;
        actCount = 0;
        learnCount = 0;
		totalReward /= 2;
		totalSteps /= 2;
    }

	// if (*SuperSonicML::Share::cvarEnableTraining)
	// 	SuperSonicML::Share::cvarManager->log(std::string("total reward: ")+std::to_string(total_reward)+"\tsteps: "+std::to_string(total_steps));
}


void learnLoop(Environment* env, Agent* agent) {
	while (!env->stopThread) {
		if (!*SuperSonicML::Share::cvarEnableTraining)
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		else {
			//std::this_thread::sleep_for(std::chrono::milliseconds(100));
			agent->learn();
			learnCount += 512;
		}
	}
}


std::thread learnThread;
void mainML(Environment* env) {
	Agent agent = Agent();
	env->observe();
	learnThread = std::thread(learnLoop, env, &agent);

	while (!env->stopThread)
		trainEnvironment(env, agent);
	
	learnThread.join();
}

