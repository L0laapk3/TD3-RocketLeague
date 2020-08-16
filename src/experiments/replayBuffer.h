#pragma once

#include <array>
#include "action.h"
#include "observation.h"
#include "torch/torch.h"
#include <mutex>
#include <stack>

struct Experience {
    std::array<float, Observation::size> state;
    std::array<float, Action::size> action;
    float reward;
    std::array<float, Observation::size> nextState;
};
struct Batch {
    torch::Tensor state;
    torch::Tensor action;
    torch::Tensor reward;
    torch::Tensor nextState;
};

class ReplayBuffer {
public:
    ReplayBuffer();
    ~ReplayBuffer();

    static const size_t maxSize = 1 << 20; //1M
    size_t getLength();

    void addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done);
    void addExperienceState(Experience experience);
    void flushBuffer();

    Batch sample(int batchSize, torch::Device& device);

private:
    std::mutex mBacklog;
    std::stack<Experience> experienceBacklog;
    std::array<Experience, maxSize>* circularBuffer;
    size_t index = 0;
    bool full = false;
};

