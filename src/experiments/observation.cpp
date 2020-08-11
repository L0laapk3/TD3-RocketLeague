
#include "observation.h"

#include "GameData.h"

static auto scalePosition = [](vec3c pos) {
	vec3c normed = {pos[0] / 4150, pos[1] / 6000, pos[2] / 2100};
	return clip(normed, -1, 1);
};

void Observation::readBotInput(const BotInputData& input, const vec3c& target) {
	auto ball = input.ball;
	auto car = input.car;

	named.carPos = scalePosition(car.pos);
	named.carAng = car.ang / 5.5f;
	named.carAng /= fmaxf(1.f, norm(named.carAng) / 1.f);
	named.carVel = scalePosition(car.vel);
	named.carForward = car.forward();
	named.carUp = car.up();
	named.targetPos = target;
	named.targetLocalPos = dot(target - car.pos, car.orientation) / 13000.f;
}