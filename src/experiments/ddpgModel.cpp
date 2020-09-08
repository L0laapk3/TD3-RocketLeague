#include <math.h>
#include "ddpgModel.h"
#include "GameData.h"
#include <string>
#include "action.h"
#include "observation.h"



void initLayer(torch::nn::Linear& fc) {
    // https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init
    float r = sqrtf(6.f / (fc->weight.sizes()[0] + fc->weight.sizes()[1]));
    torch::nn::init::uniform_(fc->weight, -r, r);
}


/******************* ACTOR *******************/

Actor::Actor() : torch::nn::Module() {
    //torch::manual_seed(1);
    fc1 = register_module("fc1", torch::nn::Linear(Observation::size, fc1_units));//fc1_units));
    fc2 = register_module("fc2", torch::nn::Linear(fc1_units, fc2_units));
    fc3 = register_module("fc3", torch::nn::Linear(fc2_units, Action::size));
//  bn1 = register_module("bn1", torch::nn::BatchNorm(fc1_units));
}

Actor::Actor(torch::Device device) : Actor() {
    reset_parameters();
    this->to(device);
}
Actor::Actor(std::shared_ptr<Actor> actor, torch::Device device) : Actor() {
    this->to(device);
    this->copy_(actor, device);
}

void Actor::copy_(std::shared_ptr<Actor> actor, torch::Device device) {
    for (size_t i = 0; i < parameters().size(); i++)
        parameters()[i].data().copy_(actor->parameters()[i].to(device, false, true).data());
}

void Actor::reset_parameters() {
    initLayer(fc1);
    initLayer(fc2);
    initLayer(fc3);
}

torch::Tensor Actor::forward(torch::Tensor x)
{
    x = fc1->forward(x);
    x = torch::leaky_relu(x);
    x = fc2->forward(x);
    x = torch::leaky_relu(x);
    x = fc3->forward(x);
    x = torch::tanh(x);
    return x;

}

std::string Actor::toString() {
    char buf[200];
	sprintf_s(buf, "%.5f %.5f %.5f", fc1->weight[0][0].item<float>(), fc1->weight[1][0].item<float>(), fc1->weight[2][0].item<float>());//, fc1->weight[0][3].item<float>(), fc1->weight[0][4].item<float>(), fc1->weight[0][5].item<float>());
    //sprintf_s(buf, "%.5f %.5f", fc1->weight[0][0].item<float>(), fc1->bias[0].item<float>());//, fc2->weight[0][0].item<float>(), fc2->bias[0].item<float>(), fc3->weight[0][0].item<float>(), fc3->bias[0].item<float>());
    return std::string(buf);  
}

// torch::nn::BatchNormOptions Actor::bn_options(int64_t features){
//     torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
//     bn_options.affine_ = true;
//     bn_options.stateful_ = true;
//     return bn_options;
// }

std::shared_ptr<Actor> Actor::load(std::string path, torch::Device device) {
    struct privateConstructorPassTrough : public Actor {};
    auto actor = (std::shared_ptr<Actor>)std::make_shared<privateConstructorPassTrough>();
    torch::load(actor, path);
    actor->to(device);
    return actor;
}



/******************* Critic *****************/


Critic::Critic() : torch::nn::Module() {
    //torch::manual_seed(1);
    fcs1 = register_module("fcs1", torch::nn::Linear(Observation::size, fcs1_units));//fcs1_units));
    fc2 = register_module("fc2", torch::nn::Linear(fcs1_units + Action::size, fc2_units));
    fc3 = register_module("fc3", torch::nn::Linear(fc2_units, 1));
//    bn1 = register_module("bn1", torch::nn::BatchNorm(fcs1_units));
}

Critic::Critic(torch::Device device) : Critic() {
    reset_parameters();
    this->to(device);
}
Critic::Critic(std::shared_ptr<Critic> critic, torch::Device device) : Critic() {
    this->to(device);
    this->copy_(critic, device);
}

void Critic::copy_(std::shared_ptr<Critic> critic, torch::Device device) {
    for (size_t i = 0; i < parameters().size(); i++)
        parameters()[i].data().copy_(critic->parameters()[i].to(device, false, true).data());
}

void Critic::reset_parameters() {
    initLayer(fcs1);
    initLayer(fc2);
    initLayer(fc3);
}


std::string Critic::toString() {
    char buf[200];
	sprintf_s(buf, "%.5f %.5f %.5f", fcs1->weight[0][0].item<float>(), fcs1->weight[1][0].item<float>(), fcs1->weight[2][0].item<float>());//, fcs1->weight[0][3].item<float>(), fcs1->weight[0][4].item<float>(), fcs1->weight[0][5].item<float>());
    return std::string(buf);  
}

torch::Tensor Critic::forward(torch::Tensor x, torch::Tensor action)
{

    if (x.dim() == 1)
        x = torch::unsqueeze(x, 0);

    if (action.dim() == 1)
        action = torch::unsqueeze(action,0);

    auto xs = torch::leaky_relu(fcs1->forward(x));
//    xs = bn1->forward(xs);
    x = torch::cat({xs,action}, /*dim=*/1);
    x = torch::leaky_relu(fc2->forward(x));
    return fc3->forward(x);
}

std::shared_ptr<Critic> Critic::load(std::string path, torch::Device device) {
    struct privateConstructorPassTrough : public Critic {};
    auto critic = (std::shared_ptr<Critic>)std::make_shared<privateConstructorPassTrough>();
    torch::load(critic, path);
    critic->to(device);
    return critic;
}
