#include <math.h>
#include "ddpgModel.h"
#include "GameData.h"





void initLayer(torch::nn::Linear& fc) {
    // https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init
    float r = sqrtf(6.f / (fc->weight.sizes()[0] + fc->weight.sizes()[1]));
    torch::nn::init::uniform_(fc->weight, -r, r);
}


/******************* ACTOR *******************/

Actor::Actor(int64_t state_size, int64_t action_size, int64_t seed, int64_t fc1_units, int64_t fc2_units) : torch::nn::Module()
{
  torch::manual_seed(seed);
  fc1 = register_module("fc1", torch::nn::Linear(state_size, fc1_units));
  fc2 = register_module("fc2", torch::nn::Linear(fc1_units, fc2_units));
  fc3 = register_module("fc3", torch::nn::Linear(fc2_units, action_size));
//  bn1 = register_module("bn1", torch::nn::BatchNorm(fc1_units));
  reset_parameters();
}

void Actor::reset_parameters()
{
    initLayer(fc1);
    initLayer(fc2);
    initLayer(fc3);
}

torch::Tensor Actor::forward(torch::Tensor x)
{
    x = torch::relu(fc1->forward(x));
//    bn1->forward(x);
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    x = torch::tanh(x);
    return x;

}

// torch::nn::BatchNormOptions Actor::bn_options(int64_t features){
//     torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
//     bn_options.affine_ = true;
//     bn_options.stateful_ = true;
//     return bn_options;
// }



/******************* Critic *****************/


Critic::Critic(int64_t state_size, int64_t action_size, int64_t seed, int64_t fcs1_units, int64_t fc2_units) : torch::nn::Module()
{
    torch::manual_seed(seed);
    fcs1 = register_module("fcs1", torch::nn::Linear(state_size, fcs1_units));
    fc2 = register_module("fc2", torch::nn::Linear(fcs1_units + action_size, fc2_units));
    fc3 = register_module("fc3", torch::nn::Linear(fc2_units, 1));
//    bn1 = register_module("bn1", torch::nn::BatchNorm(fcs1_units));
    reset_parameters();
}

void Critic::reset_parameters()
{
    initLayer(fcs1);
    initLayer(fc2);
    initLayer(fc3);
}

torch::Tensor Critic::forward(torch::Tensor x, torch::Tensor action)
{
    if (x.dim() == 1)
        x = torch::unsqueeze(x, 0);

    if (action.dim() == 1)
        action = torch::unsqueeze(action,0);

    auto xs = torch::relu(fcs1->forward(x));
//    xs = bn1->forward(xs);
    x = torch::cat({xs,action}, /*dim=*/1);
    x = torch::relu(fc2->forward(x));
    return fc3->forward(x);
}

