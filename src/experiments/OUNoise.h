#pragma once

#include <vector>
#include "action.h"

class OUNoise {
//"""Ornstein-Uhlenbeck process."""
private:
    size_t size;
    std::vector<float> mu;
    std::vector<float> state;
    
    // aggressive
    float theta=0.15;
    float sigma=0.1;

    // conservative/default
    // float theta=0.01;
    // float sigma=0.5;

public:
    OUNoise () {
        size = Action::size;
        mu = std::vector<float>(size, 0);
        reset();
    }

    void reset() {
        state = mu;
    }

    std::vector<float> sample(std::vector<float> action) {
    //"""Update internal state and return it as a noise sample."""
        for (size_t i = 0; i < state.size(); i++) {
            auto random = ((float) rand() / (RAND_MAX));
            float dx = theta * (mu[i] - state[i]) + sigma * random;
            state[i] = state[i] + dx;
            action[i] += state[i];
        }
        return action;
    }
};

