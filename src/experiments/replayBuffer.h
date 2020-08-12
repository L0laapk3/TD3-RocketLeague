#pragma once

#include <array>
#include "action.h"
#include "observation.h"
#include "torch/torch.h"

struct Experience {
    std::array<float, Observation::size> state;
    std::array<float, Action::size> action;
    float reward;
    std::array<float, Observation::size> nextState;
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

    static const size_t maxSize = 1 << 12;
    size_t getLength() {
        return full ? circularBuffer.size() : index;
    }

    void addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done);
    void addExperienceState(Experience experience);

    Batch sample(int batchSize, torch::Device& device);

private:
    std::array<Experience, maxSize> circularBuffer;
    size_t index = 0;
    bool full = false;
};

