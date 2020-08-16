
#include "action.h"
#include <algorithm>




void Action::writeControllerOutput(ControllerInput& output) {
	output.Steer = std::clamp(named.steer, -1.f, 1.f);
	output.Throttle = std::clamp(named.throttle, -1.f, 1.f);
	// output.HoldingBoost = output.ActivateBoost = named.boost > 0.f;
	// output.Handbrake = named.handbrake > 0.f;
}