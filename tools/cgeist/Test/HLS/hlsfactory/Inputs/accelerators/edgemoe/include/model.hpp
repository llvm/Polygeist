#ifndef __MODEL_HPP__
#define __MODEL_HPP__

constexpr unsigned int INPUT_CHANNELS = 3;
constexpr unsigned int INPUT_WIDTH = 256;
constexpr unsigned int INPUT_HEIGHT = 128;

constexpr unsigned int FEATURE_DIM = 192;
constexpr unsigned int VIT_HIDDEN_DIM = 768;
constexpr unsigned int EXPERT_HIDDEN_DIM = 384;

constexpr unsigned int PATCH_WIDTH = 16;
constexpr unsigned int PATCH_HEIGHT = 16;
constexpr unsigned int NUM_PATCHES = (INPUT_WIDTH / PATCH_WIDTH) * (INPUT_HEIGHT / PATCH_HEIGHT) + 1;

constexpr unsigned int NUM_LAYERS = 12;
constexpr unsigned int NUM_HEADS = 3;

constexpr unsigned int NUM_EXPERTS = 16;
constexpr unsigned int NUM_SELECTED_EXPERTS = 2;

#endif
