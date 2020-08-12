
#include "agent.h"
#include "ReplayBuffer.h"
#include "environment.h"

#include "GameData.h"


constexpr bool ALLOW_GPU = true;

size_t BATCH_SIZE = 256;        // minibatch size
double GAMMA = 0.99;            // discount factor
double TAU = 1e-3;              // for soft update of target parameters
double LR_ACTOR = 1e-4;         // learning rate of the actor
double LR_CRITIC = 1e-3;        // learning rate of the critic
double WEIGHT_DECAY = 0;        // L2 weight decay

int Agent::totalNumberOfAgents = 0;
auto learnCount = 0ULL;
auto actCount = 0ULL;

Agent::Agent(int state_size, int action_size, int random_seed )
: actor_local(std::make_shared<Actor>(state_size, action_size, random_seed)),
actor_target(std::make_shared<Actor>(state_size, action_size, random_seed)),
actor_optimizer(actor_local->parameters(), /*lr=*/LR_ACTOR),
critic_local(std::make_shared<Critic>(state_size, action_size, random_seed)),
critic_target(std::make_shared<Critic>(state_size, action_size, random_seed)),
critic_optimizer(critic_local->parameters(), /*lr=*/LR_CRITIC),
device(torch::kCPU)
{
    numOfThisAgent = ++totalNumberOfAgents;

    torch::DeviceType device_type;
    if (torch::cuda::is_available() && ALLOW_GPU) {
        device_type = torch::kCUDA;
	    SuperSonicML::Share::cvarManager->log(std::string("CUDA available"));
    } else {
        device_type = torch::kCPU;
	    SuperSonicML::Share::cvarManager->log(std::string("CPU only"));

    }
    device = torch::Device(device_type);

    stateSize = state_size;
    actionSize = action_size;
    seed = random_seed;

//  Actor Network (w/ Target Network)
    auto actorLk = std::unique_lock(mActor);
    actor_local->to(device);
    actor_target->to(device);
    hard_copy_weights(actor_target, actor_local);
    actorLk.unlock();

//  Critic Network (w/ Target Network)
    auto criticLk = std::unique_lock(mCritic);
    critic_local->to(device);
    critic_target->to(device);
    hard_copy_weights(critic_target, critic_local);
    criticLk.unlock();

    //critic_optimizer.options.weight_decay_ = WEIGHT_DECAY;

    noise = new OUNoise(static_cast<size_t>(action_size));
}

void Agent::hard_copy_weights( std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target)
{
    
    for (size_t i = 0; i < target->parameters().size(); i++) {
        target->parameters()[i] = local->parameters()[i];
    }
}

void Agent::act(const Observation& state, Action& actionOutput) {
    torch::Tensor torchState = torch::tensor(torch::ArrayRef(state.array)).to(device);

    auto actorLk = std::unique_lock(mActor);
    actor_local->eval();
    torch::NoGradGuard guard;
    auto action = actor_local->forward(torchState).to(torch::kCPU);
    actor_local->train();
    actorLk.unlock();
    std::vector<float> v(action.data<float>(), action.data<float>() + action.numel());
    if (true)
        noise->sample(v);
    for (size_t i =0; i < v.size(); i++)
        actionOutput[i] = std::fmin(std::fmax(v[i],-1.f), 1.f);

    actCount++;

    
    static auto lastMsg = std::chrono::system_clock::now();
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastMsg).count();

    if (elapsed >= 2) {
        char buf[200];
        sprintf_s(buf, "%d tps | %d lps", (int)(actCount / elapsed), (int)(learnCount / elapsed));
        SuperSonicML::Share::cvarManager->log(buf);
        lastMsg = now;
        actCount = 0;
        learnCount = 0;
    }
}

void Agent::reset() {
    noise->reset();
}

void Agent::addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done) {

    memory.addExperienceState(state, action, reward, nextState, done);
    // Learn, if enough samples are available in memory
}

void Agent::learn() {
    if (memory.getLength() < std::min(BATCH_SIZE, memory.maxSize)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return;
    }
//    Update policy and value parameters using given batch of experience tuples.
//    Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

    auto& [state, action, reward, nextState, done] = memory.sample(BATCH_SIZE, device);

// ---------------------------- update critic ---------------------------- #

    auto actions_next = actor_target->forward(nextState);
    auto Q_targets_next = critic_target->forward(nextState, actions_next);
    auto Q_targets = reward + (GAMMA * Q_targets_next * (1 - done));
    auto criticLk = std::unique_lock(mCritic);
    auto Q_expected = critic_local->forward(state, action); 
    criticLk.unlock();

    torch::Tensor critic_loss = torch::mse_loss(Q_expected, Q_targets.detach());
    critic_optimizer.zero_grad();
    critic_loss.backward();

    critic_optimizer.step();

    auto actorLk = std::unique_lock(mActor);
    auto actions_pred = actor_local->forward(state);
    actorLk.unlock();

    criticLk.lock();
    soft_update(critic_local, critic_target, TAU);

// ---------------------------- update actor ---------------------------- #

    auto actor_loss = -critic_local->forward(state, actions_pred).mean();
    criticLk.unlock();

    actor_optimizer.zero_grad();
    actor_loss.backward();
    actor_optimizer.step();

// ----------------------- update target networks ----------------------- #
    actorLk.lock();
    soft_update(actor_local, actor_target, TAU);
    actorLk.unlock();

    learnCount += BATCH_SIZE;
}

void Agent::soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target, double tau)
{
//    Soft update model parameters.
//    θ_target = τ*θ_local + (1 - τ)*θ_target
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < target->parameters().size(); i++) {
        target->parameters()[i].copy_(tau * local->parameters()[i] + (1.0 - tau) * target->parameters()[i]);
    }
}

void Agent::saveCheckPoints(int eps)
{
    std::string path = "";//getExecutablePathCopy();
    auto fileActor (path + "checkpoints/ckp_actor_agent" + std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto fileCritic(path + "checkpoints/ckp_critic_agent"+ std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto actorLk = std::unique_lock(mActor);
    auto criticLk = std::unique_lock(mCritic);
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(actor_local) , fileActor);
    torch::save(std::dynamic_pointer_cast<torch::nn::Module>(critic_local) , fileCritic);
}

void Agent::loadCheckPoints(int eps)
{
    std::string path = "";//getExecutablePathCopy();
    auto fileActor (path + "checkpoints/ckp_actor_agent" + std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto fileCritic(path + "checkpoints/ckp_critic_agent"+ std::to_string(numOfThisAgent) +"_" + std::to_string(eps) + ".pt");
    auto actorLk = std::unique_lock(mActor);
    auto criticLk = std::unique_lock(mCritic);
    torch::load(actor_local, fileActor);
    torch::load(critic_local, fileCritic);
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
