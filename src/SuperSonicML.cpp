#include "SuperSonicML.h"

namespace SuperSonicML::Plugin {

	BAKKESMOD_PLUGIN(SuperSonicMLPlugin, SuperSonicML::Constants::pluginName, SuperSonicML::Constants::versionString, 0)

	void SuperSonicMLPlugin::onLoad() {
		pluginInstance = this;

		SuperSonicML::Share::cvarManager = this->cvarManager;
		SuperSonicML::Share::gameWrapper = this->gameWrapper;

		this->cvarManager->registerCvar("ml_enabled", "0", "Enable/Disable the plugin", true, true, 0.f, true, 1.f).bindTo(SuperSonicML::Share::cvarEnabled);
		this->cvarManager->registerCvar("ml_render", "0", "Enable/disable rendering", true, true, 0.f, true, 1.f).bindTo(SuperSonicML::Share::cvarRender);
		if(this->gameWrapper->IsInFreeplay()) // Hot reloading
			this->cvarManager->executeCommand("ml_enabled 1", true);
		this->cvarManager->registerCvar("ml_slide_enabled", "1", "Enable/Disable slide", true, true, 0.f, true, 1.f).bindTo(SuperSonicML::Share::cvarEnableSlide);
		this->cvarManager->registerCvar("ml_batch_size", "4", "", true, true, 1, true, 16).bindTo(SuperSonicML::Share::cvarBatchSize);
		this->cvarManager->registerCvar("ml_training_enabled", "1", "Enable/Disable training", true, true, 0.f, true, 1.f).bindTo(SuperSonicML::Share::cvarEnableTraining);
		this->cvarManager->registerCvar("ml_user_is_teacher", "0", "Use your inputs as teacher input", true, true, 0.f, true, 1.f).bindTo(SuperSonicML::Share::cvarEnableUserAsTeacher);

		this->gameWrapper->HookEventWithCaller<CarWrapper>("Function TAGame.Car_TA.SetVehicleInput", std::bind(&SuperSonicML::Hooks::UpdateData, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
	}

	void SuperSonicMLPlugin::onUnload() {
		SuperSonicML::Hooks::unload();
		pluginInstance = nullptr;
	}
}