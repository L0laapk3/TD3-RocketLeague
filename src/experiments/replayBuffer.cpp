
#include "replayBuffer.h"
#include <algorithm>
#include <random>
#include <stack>
#include "GameData.h"



void ReplayBuffer::addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done) {
	addExperienceState(Experience{state.array, action.array, reward, nextState.array, done ? 1.f : 0.f});
}

std::stack<Experience> experienceBacklog;
void ReplayBuffer::addExperienceState(Experience experience) {
	experienceBacklog.push(experience);
	auto bufferLk = std::unique_lock(mBuffer, std::defer_lock);
	if (bufferLk.try_lock()) {
		while (!experienceBacklog.empty()) {
			circularBuffer[index++] = experienceBacklog.top();
			experienceBacklog.pop();
		}

		if (index == circularBuffer.size()) {
			full = true;
			index = 0;
		}
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
	size_t i = 0;
	while(i < batchSize) {
		auto bufferLk = std::unique_lock(mBuffer);
		Experience sample = circularBuffer[dist(rand)];
		bufferLk.unlock();
		for (int j = 0; j < Observation::size; j++) {
			float s = sample.state[j];
			states[i][j] = s;
			nextStates[i][j] = sample.nextState[j];
		}
		for (int j = 0; j < Action::size; j++)
			actions[i][j] = sample.action[j];
		rewards[i] = torch::tensor(sample.reward, torch::dtype(torch::kFloat));
		dones[i] = sample.done;
		i++;
	}

	return {
		states.to(device),
		actions.to(device),
		rewards.to(device),
		nextStates.to(device),
		dones.to(device),
	};
}