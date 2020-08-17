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

    void softUpdate(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target, float tau);

    OUNoise noise;
    torch::Device learnDevice;
    torch::Device evalDevice;
    ReplayBuffer memory;


public:
    int numOfThisAgent;
    static int totalNumberOfAgents; 

    Agent();

    void addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done);
    void act(const Observation& state, Action& action);
    void reset();
    void learn();

    void save();
    void load();

	std::mutex mActor;
	std::mutex mCritic;
    std::shared_ptr<Actor> actorLocal;
    std::shared_ptr<Critic> criticLocal;

    std::shared_ptr<Actor> actorLocalCPU;
    std::shared_ptr<Actor> actorTarget;

    std::shared_ptr<Critic> criticTarget;

    torch::optim::Adam actorOptimizer;
    torch::optim::Adam criticOptimizer;

};


