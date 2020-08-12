
#include "agent.h"
#include "ReplayBuffer.h"
#include "environment.h"

#include "GameData.h"


constexpr bool ALLOW_GPU = false;

constexpr size_t BATCH_SIZE = 512;        // minibatch size
constexpr double GAMMA = 0.99;            // discount factor
constexpr double TAU = 1e-3;              // for soft update of target parameters
constexpr double LR_ACTOR = 1e-4;         // learning rate of the actor
constexpr double LR_CRITIC = 1e-4;//1e-3  // learning rate of the critic
constexpr double WEIGHT_DECAY = 0;        // L2 weight decay
constexpr bool   ADD_NOISE = true;

int Agent::totalNumberOfAgents = 0;

Agent::Agent() :
    device(ALLOW_GPU && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),

    actor_local(device),
    actor_local_cpu(actor_local, torch::Device(torch::kCPU)),
    actor_target(actor_local, device),
    actor_optimizer(actor_local.parameters(), /*lr=*/LR_ACTOR),

    critic_local(device),
    critic_target(critic_local, device),
    critic_optimizer(critic_local.parameters(), /*lr=*/LR_CRITIC),

    noise()
    { }


void Agent::act(const Observation& state, Action& actionOutput) {
    torch::Tensor torchState = torch::tensor(torch::ArrayRef(state.array));

    auto actorLk = std::unique_lock(mActor);
    actor_local_cpu.eval();
    torch::NoGradGuard guard;
    auto action = actor_local_cpu.forward(torchState);
    actor_local_cpu.train();
    actorLk.unlock();
    std::vector<float> v(action.data<float>(), action.data<float>() + action.numel());
    if (ADD_NOISE)
        noise.sample(v);
    for (size_t i =0; i < v.size(); i++)
        actionOutput[i] = std::fmin(std::fmax(v[i],-1.f), 1.f);
    
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
    if (memory.getLength() < std::min(BATCH_SIZE, memory.maxSize)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }
//    Update policy and value parameters using given batch of experience tuples.
//    Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

    SuperSonicML::Share::cvarManager->log(actor_local.toString().c_str());
    auto& [state, action, reward, nextState, done] = memory.sample(BATCH_SIZE, device);
// ---------------------------- update critic ---------------------------- #

    auto actions_next = actor_target.forward(nextState);
    auto Q_targets_next = critic_target.forward(nextState, actions_next);
    auto Q_targets = reward + (GAMMA * Q_targets_next * (1 - done));
    auto criticLk = std::unique_lock(mCritic);
    auto Q_expected = critic_local.forward(state, action); 
    criticLk.unlock();

    torch::Tensor critic_loss = torch::mse_loss(Q_expected, Q_targets.detach());
    critic_optimizer.zero_grad();
    critic_loss.backward();
    critic_optimizer.step();

// ---------------------------- update actor ---------------------------- #

    auto actions_pred = actor_local.forward(state);
    criticLk.lock();
    soft_update(critic_local, critic_target, TAU); // still belongs to update critic but moved to fall in lock
    auto actor_loss = -critic_local.forward(state, actions_pred).mean();
    SuperSonicML::Share::cvarManager->log("loss: " + std::to_string(actor_loss.item<float>()));
    criticLk.unlock();

    actor_optimizer.zero_grad();
    actor_loss.backward();
    actor_optimizer.step();
    SuperSonicML::Share::cvarManager->log(actor_local.toString().c_str());

// ----------------------- update target networks ----------------------- #
    soft_update(actor_local, actor_target, TAU);
    actor_target.to(torch::kCPU);
    auto actorLk = std::unique_lock(mActor);
    soft_update(actor_local_cpu, actor_target, TAU);
    actorLk.unlock();
    actor_target.to(device);

    static int learnCount = 0;
    if (++learnCount % 1000 == 0)
        saveCheckPoints(learnCount);
}

void Agent::soft_update(torch::nn::Module& local, torch::nn::Module& target, double tau)
{
//    Soft update model parameters.
//    θ_target = τ*θ_local + (1 - τ)*θ_target
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < target.parameters().size(); i++)
        target.parameters()[i].copy_(tau * local.parameters()[i] + (1.0 - tau) * target.parameters()[i]);
}

static const std::string basePath = "C:/Users/Kris/Documents/coding/RLBot/ML/SuperSonicML/";
void Agent::saveCheckPoints(int eps)
{
    auto fileActor (basePath + "checkpoints/ckp_actor_agent" + std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto fileCritic(basePath + "checkpoints/ckp_critic_agent"+ std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    SuperSonicML::Share::cvarManager->log(fileActor);
    auto actorLk = std::unique_lock(mActor);
    auto criticLk = std::unique_lock(mCritic);
    //actor_local_cpu.save(fileActor);
    //critic_local.save(fileCritic);
}

void Agent::loadCheckPoints(int eps)
{
    auto fileActor (basePath + "checkpoints/ckp_actor_agent" + std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto fileCritic(basePath + "checkpoints/ckp_critic_agent"+ std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto actorLk = std::unique_lock(mActor);
    auto criticLk = std::unique_lock(mCritic);
    //torch::load(actor_local_cpu, fileActor);
    //torch::load(critic_local, fileCritic);
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
