
#include "agent.h"
#include "ReplayBuffer.h"
#include "environment.h"

#include "GameData.h"


constexpr bool ALLOW_GPU = false;

constexpr size_t BATCH_SIZE = 64;          // minibatch size
constexpr double GAMMA = 0.99;             // discount factor
constexpr double TAU = 1e-3;               // for soft update of target parameters
constexpr double LR_CRITIC = 5e-3; //1e-3  // learning rate of the critic
constexpr double LR_ACTOR =  5e-4; //1e-4  // learning rate of the actor
constexpr double WEIGHT_DECAY = 0;         // L2 weight decay
constexpr bool   ADD_NOISE = true;

int Agent::totalNumberOfAgents = 0;

Agent::Agent() :
    device(ALLOW_GPU && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),

    actorLocal(device),
    actorLocalAct(actorLocal, device),
    //actor_local_cpu(actorLocal, torch::Device(torch::kCPU)),
    actorTarget(actorLocal, device),
    actorOptimizer(actorLocal.parameters(), /*lr=*/LR_ACTOR),

    criticLocal(device),
    criticTarget(criticLocal, device),
    criticOptimizer(criticLocal.parameters(), /*lr=*/LR_CRITIC),

    noise()
    {
        actorLocal.train();
        actorLocalAct.eval();
    }


void Agent::act(const Observation& state, Action& actionOutput) {


	
    // char buf[200];
    // sprintf_s(buf, "observe %.5f %.5f %.5f", state.array[0], state.array[1], state.array[2]);
    // SuperSonicML::Share::cvarManager->log(buf);

    torch::Tensor torchState = torch::from_blob((void*)state.array.data(), {state.size}, at::kFloat).to(torch::kCPU, false, true);

    // sprintf_s(buf, "act %d %d %.5f %.5f %.5f", torchState.dim(), torchState.sizes()[0], torchState.accessor<float,1>()[0], torchState.accessor<float,1>()[1], torchState.accessor<float,1>()[2]);
    // SuperSonicML::Share::cvarManager->log(buf);
	
    auto actorLk = std::unique_lock(mActor);
    torch::NoGradGuard guard;
    auto action = actorLocalAct.forward(torchState);
    actorLk.unlock();
    if (ADD_NOISE) {
        std::vector<float> v(action.data<float>(), action.data<float>() + action.numel());
        noise.sample(v);
        for (size_t i =0; i < v.size(); i++)
            actionOutput[i] = std::fmin(std::fmax(v[i],-1.f), 1.f);
    }
    
}

void Agent::reset() {
    noise.reset();
}

void Agent::addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done) {
    if (*SuperSonicML::Share::cvarEnableTraining)
        memory.addExperienceState(state, action, reward, nextState, done);
    // Learn, if enough samples are available in memory
}

void Agent::learn() {
    memory.flushBuffer();
    if (memory.getLength() < (std::min)(BATCH_SIZE, memory.maxSize)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }
//    Update policy and value parameters using given batch of experience tuples.
//    QTargets = r + γ * criticTarget(next_state, actorTarget(next_state))

    auto& [state, action, reward, nextState] = memory.sample(BATCH_SIZE, device);


    // char buf[200];
    // sprintf_s(buf, "learn %d %d %d %f.5 %f.5 %f.5", state.dim(), state.sizes()[0], state.sizes()[1], state.accessor<float,2>()[0][0], state.accessor<float,2>()[0][1], state.accessor<float,2>()[0][2]);//, action.accessor<float,2>()[0][2]);
    // SuperSonicML::Share::cvarManager->log(buf);

    //SuperSonicML::Share::cvarManager->log(actorLocal.toString().c_str());
// ---------------------------- update critic ---------------------------- #

    auto actionsNext = actorTarget.forward(nextState);

    auto QTargetsNext = criticTarget.forward(nextState, actionsNext);


    auto QTargets = reward + (GAMMA * QTargetsNext);
    auto criticLk = std::unique_lock(mCritic);
    auto QExpected = criticLocal.forward(state, action); 
    criticLk.unlock();

    torch::Tensor criticLoss = torch::mse_loss(QExpected, QTargets.detach());
    
    // char buf[200];
    // sprintf_s(buf, "cLoss %d %d %.5f  %d %d %d  %d %d %d", criticLoss.dim(), criticLoss.sizes()[0], criticLoss.item<float>(), QExpected.dim(), QExpected.sizes()[0], QExpected.sizes()[1], QTargets.dim(), QTargets.sizes()[0], QTargets.sizes()[1]);
    // SuperSonicML::Share::cvarManager->log(buf);
    //SuperSonicML::Share::cvarManager->log("critic loss "+std::to_string(criticLoss.item<float>()));
    criticOptimizer.zero_grad();
    criticLoss.backward();
    //criticLk.lock();
    //SuperSonicML::Share::cvarManager->log("critic "+criticLocal.toString());
    criticLk.lock();
    criticOptimizer.step();
    criticLk.unlock();
    //SuperSonicML::Share::cvarManager->log("critic "+criticLocal.toString());
    softUpdate(criticLocal, criticTarget, TAU); // still belongs to update critic but moved to fall in lock

// ---------------------------- update actor ---------------------------- #

    auto actionsPred = actorLocal.forward(state);

    auto actorLoss = -criticLocal.forward(state, actionsPred).mean();
    //criticLk.unlock();

    actorOptimizer.zero_grad();
    actorLoss.backward();
    //SuperSonicML::Share::cvarManager->log("actor "+actorLocal.toString());
    actorOptimizer.step();
   // SuperSonicML::Share::cvarManager->log("actor "+actorLocal.toString());

// ----------------------- update target networks ----------------------- #
    softUpdate(actorLocal, actorTarget, TAU);
    
    auto actorLk = std::unique_lock(mActor);
    actorLocalAct.copy_(actorLocal, torch::kCPU);
    actorLk.unlock();

    //actor_local_cpu.copy_(actorLocal);
    //actorLk.unlock();
    
    //SuperSonicML::Share::cvarManager->log(actorLocal.toString().c_str());
    //SuperSonicML::Share::cvarManager->log(actor_local_cpu.toString().c_str());

    static int learnCount = 0;
    if (++learnCount % 1000 == 0)
        saveCheckPoints(learnCount);
}

void Agent::softUpdate(torch::nn::Module& local, torch::nn::Module& target, double tau)
{
//    Soft update model parameters.
//    θ_target = τ*θ_local + (1 - τ)*θ_target
    torch::NoGradGuard guard;
    for (size_t i = 0; i < target.parameters().size(); i++)
        target.parameters()[i].copy_(tau * local.parameters()[i] + (1.0 - tau) * target.parameters()[i]);
}

static const std::string basePath = "C:/Users/Kris/Documents/coding/RLBot/ML/SuperSonicML/";
void Agent::saveCheckPoints(int eps)
{
    return;
    auto fileActor (basePath + "checkpoints/ckp_actor_agent" + std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto fileCritic(basePath + "checkpoints/ckp_critic_agent"+ std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    SuperSonicML::Share::cvarManager->log(fileActor);
    auto actorLk = std::unique_lock(mActor);
    auto criticLk = std::unique_lock(mCritic);
    //actor_local_cpu.save(fileActor);
    //criticLocal.save(fileCritic);
}

void Agent::loadCheckPoints(int eps)
{
    auto fileActor (basePath + "checkpoints/ckp_actor_agent" + std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto fileCritic(basePath + "checkpoints/ckp_critic_agent"+ std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto actorLk = std::unique_lock(mActor);
    auto criticLk = std::unique_lock(mCritic);
    //torch::load(actor_local_cpu, fileActor);
    //torch::load(criticLocal, fileCritic);
}

// std::string Agent::getExecutablePath() 
// {
//     char buff[PATH_MAX];
//     ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff) - 1);
//     if (len != -1) {
//         buff[len] = '\0';
//         std::string path = std::string(buff);
//         std::size_t found = path.find_last_of("/");
//         path = path.substr(0,found)+"/";
//         return path;
//     }
//     std::cout << "Could not determine path of executable" << std::endl;
//     return "";
// }
