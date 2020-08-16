#pragma once

#include <torch/torch.h>
#include "ddpgModel.h"
#include "ReplayBuffer.h"
#include "environment.h"
#include "OUNoise.h"
#include <mutex>

class Agent {
private:
    //std::string getExecutablePath();

    void softUpdate(torch::nn::Module& local, torch::nn::Module& target, double tau);

    OUNoise noise;
    torch::Device device; 
    ReplayBuffer memory;


public:
    int numOfThisAgent;
    static int totalNumberOfAgents; 

    Agent();

    void addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done);
    void act(const Observation& state, Action& action);
    void reset();
    void learn();

    void saveCheckPoints(int e);
    void loadCheckPoints(int e);

	std::mutex mActor;
    Actor actorLocal;
    Actor actorTarget;
    torch::optim::Adam actorOptimizer;

	std::mutex mCritic;
    Critic criticLocal;
    Critic criticTarget;
    torch::optim::Adam criticOptimizer;

};


