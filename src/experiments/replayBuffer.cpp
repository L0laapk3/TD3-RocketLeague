
#include "replayBuffer.h"
#include <algorithm>
#include <random>
#include "GameData.h"



void ReplayBuffer::addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done) {
	addExperienceState(Experience{state.array, action.array, reward, nextState.array, done ? 1.f : 0.f});
}

void ReplayBuffer::addExperienceState(Experience experience) {
	circularBuffer[index++] = experience;
	if (index == circularBuffer.size()) {
		full = true;
		index = 0;
	}
}

// void ReplayBuffer::getLength() {
// 	return full ? circularBuffer.size() : index;
// }


Batch ReplayBuffer::sample(int batchSize, torch::Device& device) {
	auto states = torch::empty({batchSize, Observation::size}, torch::kFloat);
	auto actions = torch::empty({batchSize, Action::size}, torch::kFloat);
	auto rewards = torch::empty({batchSize}, torch::kFloat);
	auto nextStates = torch::empty({batchSize, Observation::size}, torch::kFloat);
	auto dones = torch::empty({batchSize}, torch::kFloat);

	static auto rand = std::mt19937{std::random_device{}()};
    static std::uniform_int_distribution<size_t> dist(0, getLength() - 1);
	for (size_t i = 0; i < batchSize; i++) {
		Experience& sample = circularBuffer[dist(rand)];
		for (int j = 0; j < Observation::size; j++) {
			float s = sample.state[j];
			states[i][j] = s;
			nextStates[i][j] = sample.nextState[j];
		}
		for (int j = 0; j < Action::size; j++)
			actions[i][j] = sample.action[j];
		rewards[i] = torch::tensor(sample.reward, torch::dtype(torch::kFloat));
		dones[i] = sample.done;
	}

	return {
		states.to(device),
		actions.to(device),
		rewards.to(device),
		nextStates.to(device),
		dones.to(device),
	};
}