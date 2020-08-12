#pragma once

#include <array>

#include "math/math.h"
#include "bots/BotClass.h"

class Action {
public:
	static const int size = 1;
	union {
		std::array<float, size> array;
		struct {
			float steer;
			//float throttle;
			//float boost;
			//float handbrake;
		} named;
	};
	float& operator[] (const size_t i) { return array[i]; }

	void writeControllerOutput(ControllerInput& output);
};
