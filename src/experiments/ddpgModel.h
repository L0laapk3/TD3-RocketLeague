#ifndef PROJECT_DDPG_MODEL_H
#define PROJECT_DDPG_MODEL_H

#include <torch/torch.h>

class Actor : public torch::nn::Module {
private:
    Actor();

public:
    static const int fc1_units = 40;
    static const int fc2_units = 30;

    Actor(torch::Device device);
    Actor(const Actor& actor, torch::Device device);
    void reset_parameters();

    torch::Tensor forward(torch::Tensor state);
    //torch::nn::BatchNormOptions bn_options(int64_t features);

    void copy_(const Actor& actor);
    std::string toString();

    torch::nn::Linear fc1{nullptr};//, fc2{nullptr}, fc3{nullptr};
    //torch::nn::BatchNorm bn1{nullptr};
};


/******************* Critic *****************/

class Critic : public torch::nn::Module {
private:
    Critic();
public:
    static const int fcs1_units = 40;
    static const int fc2_units = 30;

    Critic(torch::Device device);
    Critic(const Critic& critic, torch::Device device);

    void reset_parameters();

    torch::Tensor forward(torch::Tensor x, torch::Tensor action);


    torch::nn::Linear fcs1{nullptr}, fc2{nullptr}, fc3{nullptr};
    //torch::nn::BatchNorm bn1{nullptr};
};

#endif //PROJECT_DDPG_MODEL_H
