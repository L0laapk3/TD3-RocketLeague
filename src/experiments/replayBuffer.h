#pragma once

#include <array>
#include <algorithm>
#include "action.h"
#include "observation.h"
#include "torch/torch.h"

struct Experience {
    Observation state;
    Action action;
    float reward;
    Observation nextState;
    float done;
};
struct Batch {
    torch::Tensor state;
    torch::Tensor action;
    torch::Tensor reward;
    torch::Tensor nextState;
    torch::Tensor done;
};

class ReplayBuffer {
public:

    static const size_t maxSize = 1 << 16;
    size_t getLength() const;

    void addExperienceState(Experience experience);

    Batch sample(size_t batchSize);

private:
    std::array<Experience, maxSize> circularBuffer;
    size_t index = 0;
    bool full = false;
};

