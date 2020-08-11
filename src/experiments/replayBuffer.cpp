
#include "replayBuffer.h"



void ReplayBuffer::addExperienceState(Experience experience) {
	circularBuffer[index++] = experience;
	if (index == circularBuffer.size()) {
		full = true;
		index = 0;
	}
}

void ReplayBuffer::getLength() {
	return full ? circularBuffer.size() : index;
}


Batch sample(size_t batchSize) {
	auto sample = std::sample(circularBuffer.begin(), full ? circularBuffer.end() : circularBuffer.begin() + index, batchSize);
	std::array<Observation> states(batchSize);
    std::vector<Action> actions(batchSize);
    std::vector<float> rewards(batchSize);
    std::vector<Observation> nextStates(batchSize);
    std::vector<float> dones(batchSize);
	for (size_t i = 0; i < batchSize; i++) {
		states[i] = sample[i].state;
		actions[i] = sample[i].action;
		rewards[i] = sample[i].reward;
		nextStates[i] = sample[i].nextStates;
		dones[i] = sample[i].done;
	}
	return {
		torch::tensor(torch::ArrayRef(states)).to(device),
		torch::tensor(torch::ArrayRef(actions)).to(device),
		torch::tensor(torch::ArrayRef(rewards)).to(device),
		torch::tensor(torch::ArrayRef(nextStates)).to(device),
		torch::tensor(torch::ArrayRef(dones)).to(device),
	};
}