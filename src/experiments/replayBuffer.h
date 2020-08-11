#ifndef PROJECT_REPLAYBUFFER_H
#define PROJECT_REPLAYBUFFER_H

#include <array>
#include "torch/torch.h"

typedef std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Experience;

class ReplayBuffer {
public:
    ReplayBuffer() {}

    void addExperienceState(torch::Tensor state, torch::Tensor action, torch::Tensor reward, torch::Tensor next_state, torch::Tensor done)
    {
        addExperienceState(std::make_tuple(state, action, reward, next_state, done));
    }

    void addExperienceState(Experience experience) {
        circular_buffer[index++] = experience;
        if (index == circular_buffer.size()) {
            full = true;
            index = 0;
        }
    }

    std::vector<Experience> sample(int num_agent) {
        std::vector<Experience> experiences;
        for (int i = 0; i < num_agent; i++) {
            experiences.push_back(sample());
        }
        return experiences;
    }

    Experience sample() {
            return circular_buffer[static_cast<size_t>(rand() % static_cast<int>(full ? circular_buffer.size() : index))];
    }

    size_t getLength() {
        return circular_buffer.size();
    }



private:
    std::array<Experience, 1 << 14> circular_buffer;
    size_t index = 0;
    bool full = false;
};

#endif //PROJECT_REPLAYBUFFER_H

