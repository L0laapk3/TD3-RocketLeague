#pragma once

#include <array>

#include "math/math.h"
#include "bots/BotClass.h"


class Observation {
public:
	Observation() {};
	Observation(const Observation& o) : array(o.array) {};
	~Observation() {};

	static const int size = 3;
	union {
		std::array<float, size> array;
		struct {
			//vec3c carPos;
			//vec3c carAng;
			//vec3c carVel;
			//vec3c carForward;
			//vec3c carUp;
			//vec3c targetPos;
			vec3c targetLocalPos;
		} named;
	};
	float& operator[] (const size_t i) { return array[i]; }

	void readBotInput(const BotInputData& input, const vec3c& target);
};
