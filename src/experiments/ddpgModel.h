#ifndef PROJECT_DDPG_MODEL_H
#define PROJECT_DDPG_MODEL_H

#include <torch/torch.h>
#include <memory>
#include <string>

class Actor : public torch::nn::Module {
private:
    Actor();

public:
    static const int fc1_units = 100;
    static const int fc2_units = 100;

    Actor(torch::Device device);
    Actor(std::shared_ptr<Actor> actor, torch::Device device);
    void copy_(std::shared_ptr<Actor> actor, torch::Device device);

    void reset_parameters();

    torch::Tensor forward(torch::Tensor state);
    //torch::nn::BatchNormOptions bn_options(int64_t features);

    std::string toString();

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    static std::shared_ptr<Actor> load(std::string path, torch::Device device);
    //torch::nn::BatchNorm bn1{nullptr};
};


/******************* Critic *****************/

class Critic : public torch::nn::Module {
private:
    Critic();
public:
    static const int fcs1_units = 100;
    static const int fc2_units = 100;

    Critic(torch::Device device);
    Critic(std::shared_ptr<Critic> critic, torch::Device device);
    void copy_(std::shared_ptr<Critic> critic, torch::Device device);

    void reset_parameters();

    torch::Tensor forward(torch::Tensor x, torch::Tensor action);

    std::string toString();

    torch::nn::Linear fcs1{nullptr}, fc2{nullptr}, fc3{nullptr};
    //torch::nn::BatchNorm bn1{nullptr};

    static std::shared_ptr<Critic> load(std::string path, torch::Device device);
};

#endif //PROJECT_DDPG_MODEL_H
