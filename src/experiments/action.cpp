
#include "action.h"




void Action::writeControllerOutput(ControllerInput& output) {
	output.Steer = clip(named.steer, -1.f, 1.f);
	output.Throttle = clip(named.throttle, -1.f, 1.f);
}