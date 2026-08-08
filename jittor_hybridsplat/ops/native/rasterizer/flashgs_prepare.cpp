#include "hybridsplat_rasterizer_c_api.h"

#include "model_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>

namespace {

void write_error(char* destination, uint64_t capacity, const char* text) noexcept
{
    if (destination == nullptr || capacity == 0) return;
    const char* source = text == nullptr ? "" : text;
    const size_t length = std::min(std::strlen(source), static_cast<size_t>(capacity - 1));
    if (length != 0) std::memcpy(destination, source, length);
    destination[length] = '\0';
}

int32_t fail(const char* text, char* error_text, uint64_t error_capacity) noexcept
{
    write_error(error_text, error_capacity, text);
    return static_cast<int32_t>(HS_STATUS_INVALID_ARGUMENT);
}

}  // namespace

extern "C" int32_t hs_flashgs_prepare_npz_host_f32_v1(
    uint64_t point_count,
    const float* opacity_raw1,
    const float* scaling_raw2,
    const float* rotation_raw4,
    const float* specular_raw1,
    const float* roughness_raw1,
    float* opacity1,
    float* scaling2,
    float* rotation4,
    float* covariance6,
    float* specular1,
    float* roughness1,
    char* error_text,
    uint64_t error_capacity) noexcept
{
    write_error(error_text, error_capacity, "");
    if (error_capacity != 0 && error_text == nullptr) {
        return static_cast<int32_t>(HS_STATUS_INVALID_ARGUMENT);
    }
    if (point_count == 0) return static_cast<int32_t>(HS_STATUS_OK);
    if (opacity_raw1 == nullptr || scaling_raw2 == nullptr || rotation_raw4 == nullptr ||
        specular_raw1 == nullptr || roughness_raw1 == nullptr || opacity1 == nullptr ||
        scaling2 == nullptr || rotation4 == nullptr || covariance6 == nullptr ||
        specular1 == nullptr || roughness1 == nullptr) {
        return fail("FlashGS NPZ preparation received a null host buffer", error_text, error_capacity);
    }
    try {
        const auto* scale_input = reinterpret_cast<const glm::vec2*>(scaling_raw2);
        const auto* rotation_input = reinterpret_cast<const glm::vec4*>(rotation_raw4);
        auto* scale_output = reinterpret_cast<glm::vec2*>(scaling2);
        auto* rotation_output = reinterpret_cast<glm::vec4*>(rotation4);
        for (uint64_t index = 0; index < point_count; ++index) {
            opacity1[index] = flashgs::sigmoid_f32(opacity_raw1[index]);
            scale_output[index] = glm::vec2(
                std::exp(scale_input[index].x),
                std::exp(scale_input[index].y));
            rotation_output[index] = glm::normalize(rotation_input[index]);
            flashgs::computeCov3D(
                scale_output[index],
                1.0f,
                rotation_output[index],
                covariance6 + index * 6);
            specular1[index] = flashgs::sigmoid_f32(specular_raw1[index]);
            roughness1[index] = flashgs::sigmoid_f32(roughness_raw1[index]);
        }
        return static_cast<int32_t>(HS_STATUS_OK);
    } catch (const std::exception& exception) {
        write_error(error_text, error_capacity, exception.what());
        return static_cast<int32_t>(HS_STATUS_INTERNAL_ERROR);
    } catch (...) {
        write_error(error_text, error_capacity, "unknown FlashGS NPZ preparation error");
        return static_cast<int32_t>(HS_STATUS_INTERNAL_ERROR);
    }
}
