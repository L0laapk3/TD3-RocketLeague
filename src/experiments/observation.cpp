
#include "observation.h"

#include "GameData.h"
#include <algorithm>

static auto scalePosition = [](vec3c pos) {
	return vec3c{pos[0] / 4150.f, pos[1] / 6000.f, pos[2] / 2100.f};
};

void Observation::readBotInput(const BotInputData& input, const vec3c& target) {
	auto ball = input.ball;
	auto car = input.car;

	named.carPos = (vec2c)scalePosition(car.pos);
	// named.carAng = car.ang / 5.5f;
	// named.carAng /= fmaxf(1.f, norm(named.carAng) / 1.f);
	named.carVel = (vec2c)dot(car.vel, car.orientation) / 13000.f;
	named.carForward = (vec2c)car.forward();
	// named.carUp = car.up();
	named.targetPos = (vec2c)target;
	// named.targetLocalPos = (vec2c)dot(target - car.pos, car.orientation) / 13000.f;
}