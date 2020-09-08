#pragma once

#include <array>

#include "math/math.h"
#include "bots/BotClass.h"


class Observation {
public:
	Observation() {};
	Observation(const Observation& o) : array(o.array) {};
	~Observation() {};

	static const int size = 8;
	union {
		std::array<float, size> array;
		struct {
			vec2c carPos;
			// vec3c carAng;
			vec2c carVel;
			vec2c carForward;
			// vec3c carUp;
			vec2c targetPos;
			// vec2c targetLocalPos;
		} named;
	};
	float& operator[] (const size_t i) { return array[i]; }

	void readBotInput(const BotInputData& input, const vec3c& target);
};
