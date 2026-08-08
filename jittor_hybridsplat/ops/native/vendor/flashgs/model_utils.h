#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

#include <cmath>

namespace flashgs {

inline float sigmoid_f32(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

inline void computeCov3D(const glm::vec2 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = 0.000001f;

    glm::vec4 q = rot;
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;
    glm::mat3 Sigma = glm::transpose(M) * M;

    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

} // namespace flashgs
