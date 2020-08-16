
#include "replayBuffer.h"
#include <algorithm>
#include <random>
#include "GameData.h"


ReplayBuffer::ReplayBuffer() {
	circularBuffer = new std::array<Experience, maxSize>();
}
ReplayBuffer::~ReplayBuffer() {
	delete circularBuffer;
}



void ReplayBuffer::addExperienceState(Observation& state, Action& action, float reward, Observation& nextState, bool done) {
	addExperienceState(Experience{state.array, action.array, reward, nextState.array, done ? 1.f : 0.f});
}

void ReplayBuffer::addExperienceState(Experience experience) {
	auto backlogLk = std::unique_lock(mBacklog);
	experienceBacklog.push(experience);
}

size_t ReplayBuffer::getLength() {
	return full ? maxSize : index;
}

void ReplayBuffer::flushBuffer() {
	while (!experienceBacklog.empty()) {
		auto backlogLk = std::unique_lock(mBacklog);
		(*circularBuffer)[index] = experienceBacklog.top();
		experienceBacklog.pop();
		backlogLk.unlock();
		if (++index > maxSize) {
			full = true;
			index = 0;
		}
	}
}


Batch ReplayBuffer::sample(int batchSize, torch::Device& device) {
	auto states = torch::empty({batchSize, Observation::size}, torch::kFloat);
	auto actions = torch::empty({batchSize, Action::size}, torch::kFloat);
	auto rewards = torch::empty({batchSize}, torch::kFloat);
	auto nextStates = torch::empty({batchSize, Observation::size}, torch::kFloat);
	auto dones = torch::empty({batchSize}, torch::kFloat);

	static auto rand = std::mt19937{std::random_device{}()};
	std::uniform_int_distribution<size_t> halfDist(0, index);
    static std::uniform_int_distribution<size_t> dist(0, maxSize);
	for(size_t i = 0; i < batchSize; i++) {
		Experience sample = (*circularBuffer)[full ? dist(rand) : halfDist(rand)];
    	states[i] = torch::from_blob((void*)sample.state.data(), {Observation::size}, at::kFloat);
    	actions[i] = torch::from_blob((void*)sample.action.data(), {Action::size}, at::kFloat);
		rewards[i] = sample.reward;
    	nextStates[i] = torch::from_blob((void*)sample.nextState.data(), {Observation::size}, at::kFloat);
		dones[i] = sample.done;
	}

	return {
		states.to(device, false, true),
		actions.to(device, false, true),
		rewards.to(device),
		nextStates.to(device, false, true),
		dones.to(device),
	};
}