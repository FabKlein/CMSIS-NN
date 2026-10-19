/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv2d_3x3_packed_f32.c
 * Description:  Support: direct NHWC 3x3 convolution with packed f32 weights
 *
 * $Date:        19 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/* Reuse each packed output-channel vector across four adjacent interior pixels.
 * Border pixels and output-width tails use the single-output path with explicit zero padding.
 */
__STATIC_FORCEINLINE void conv3x3_packed_f32(const float32_t *input,
                                             int32_t in_c,
                                             int32_t in_h,
                                             int32_t in_w,
                                             const float32_t *weights,
                                             const float32_t *bias,
                                             float32_t *output,
                                             int32_t out_c,
                                             int32_t out_h,
                                             int32_t out_w,
                                             int32_t stride,
                                             int32_t padding)
{
    const size_t plane = (size_t)9 * in_c * 4;
    for (int32_t oy = 0; oy < out_h; ++oy)
    {
        const int32_t iy = oy * stride - padding;
        int32_t ox = 0;
        while (ox < out_w)
        {
            const int32_t ix = ox * stride - padding;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            if (ox + 3 < out_w && iy >= 0 && iy + 2 < in_h && ix >= 0 && ix + 3 * stride + 2 < in_w)
            {
                const int32_t row_len = 3 * in_c;
                for (int32_t oc = 0; oc < out_c; oc += 4)
                {
                    const mve_pred16_t pred = vctp32q((uint32_t)(out_c - oc));
                    float32x4_t a0 = bias ? vld1q_z(bias + oc, pred) : vdupq_n_f32(0.0f);
                    float32x4_t a1 = a0;
                    float32x4_t a2 = a0;
                    float32x4_t a3 = a0;
                    const float32_t *w = weights + (size_t)(oc / 4) * plane;
                    for (int32_t ky = 0; ky < 3; ++ky)
                    {
                        const float32_t *x0 = input + ((size_t)(iy + ky) * in_w + ix) * in_c;
                        const float32_t *x1 = x0 + (size_t)stride * in_c;
                        const float32_t *x2 = x1 + (size_t)stride * in_c;
                        const float32_t *x3 = x2 + (size_t)stride * in_c;
                        for (int32_t k = 0; k < row_len; ++k)
                        {
                            const float32x4_t v = vld1q(w);
                            w += 4;
                            a0 = vfmaq_n_f32(a0, v, x0[k]);
                            a1 = vfmaq_n_f32(a1, v, x1[k]);
                            a2 = vfmaq_n_f32(a2, v, x2[k]);
                            a3 = vfmaq_n_f32(a3, v, x3[k]);
                        }
                    }
                    float32_t *dst = output + ((size_t)oy * out_w + ox) * out_c + oc;
                    vst1q_p(dst, a0, pred);
                    vst1q_p(dst + out_c, a1, pred);
                    vst1q_p(dst + 2 * out_c, a2, pred);
                    vst1q_p(dst + 3 * out_c, a3, pred);
                }
                ox += 4;
                continue;
            }
    #endif
            for (int32_t oc = 0; oc < out_c; oc += 4)
            {
                const float32_t *w = weights + (size_t)(oc / 4) * plane;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                const mve_pred16_t pred = vctp32q((uint32_t)(out_c - oc));
                float32x4_t acc = bias ? vld1q_z(bias + oc, pred) : vdupq_n_f32(0.0f);
                for (int32_t ky = 0; ky < 3; ++ky)
                {
                    for (int32_t kx = 0; kx < 3; ++kx)
                    {
                        const int32_t y = iy + ky;
                        const int32_t x = ix + kx;
                        const bool valid = (uint32_t)y < (uint32_t)in_h && (uint32_t)x < (uint32_t)in_w;
                        const float32_t *src = valid ? input + ((size_t)y * in_w + x) * in_c : NULL;
                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            acc = vfmaq_n_f32(acc, vld1q(w), valid ? src[ic] : 0.0f);
                            w += 4;
                        }
                    }
                }
                vst1q_p(output + ((size_t)oy * out_w + ox) * out_c + oc, acc, pred);
    #else
                const int32_t lanes = MIN(4, out_c - oc);
                for (int32_t lane = 0; lane < lanes; ++lane)
                {
                    float32_t acc = bias ? bias[oc + lane] : 0.0f;
                    int32_t k = 0;
                    for (int32_t ky = 0; ky < 3; ++ky)
                    {
                        for (int32_t kx = 0; kx < 3; ++kx)
                        {
                            const int32_t y = iy + ky;
                            const int32_t x = ix + kx;
                            const bool valid = (uint32_t)y < (uint32_t)in_h && (uint32_t)x < (uint32_t)in_w;
                            for (int32_t ic = 0; ic < in_c; ++ic, ++k)
                            {
                                const float32_t value = valid ? input[((size_t)y * in_w + x) * in_c + ic] : 0.0f;
                                acc += value * w[(size_t)k * 4 + lane];
                            }
                        }
                    }
                    output[((size_t)oy * out_w + ox) * out_c + oc + lane] = acc;
                }
    #endif
            }
            ++ox;
        }
    }
}

void arm_nn_conv2d_3x3_s1_p0_packed_f32(const float32_t *__RESTRICT input,
                                        int32_t in_c,
                                        int32_t in_h,
                                        int32_t in_w,
                                        const float32_t *__RESTRICT weights,
                                        const float32_t *__RESTRICT bias,
                                        float32_t *__RESTRICT output,
                                        int32_t out_c,
                                        int32_t out_h,
                                        int32_t out_w)
{
    conv3x3_packed_f32(input, in_c, in_h, in_w, weights, bias, output, out_c, out_h, out_w, 1, 0);
}

void arm_nn_conv2d_3x3_s1_p1_packed_f32(const float32_t *__RESTRICT input,
                                        int32_t in_c,
                                        int32_t in_h,
                                        int32_t in_w,
                                        const float32_t *__RESTRICT weights,
                                        const float32_t *__RESTRICT bias,
                                        float32_t *__RESTRICT output,
                                        int32_t out_c,
                                        int32_t out_h,
                                        int32_t out_w)
{
    conv3x3_packed_f32(input, in_c, in_h, in_w, weights, bias, output, out_c, out_h, out_w, 1, 1);
}

void arm_nn_conv2d_3x3_s2_p0_packed_f32(const float32_t *__RESTRICT input,
                                        int32_t in_c,
                                        int32_t in_h,
                                        int32_t in_w,
                                        const float32_t *__RESTRICT weights,
                                        const float32_t *__RESTRICT bias,
                                        float32_t *__RESTRICT output,
                                        int32_t out_c,
                                        int32_t out_h,
                                        int32_t out_w)
{
    conv3x3_packed_f32(input, in_c, in_h, in_w, weights, bias, output, out_c, out_h, out_w, 2, 0);
}

void arm_nn_conv2d_3x3_s2_p1_packed_f32(const float32_t *__RESTRICT input,
                                        int32_t in_c,
                                        int32_t in_h,
                                        int32_t in_w,
                                        const float32_t *__RESTRICT weights,
                                        const float32_t *__RESTRICT bias,
                                        float32_t *__RESTRICT output,
                                        int32_t out_c,
                                        int32_t out_h,
                                        int32_t out_w)
{
    conv3x3_packed_f32(input, in_c, in_h, in_w, weights, bias, output, out_c, out_h, out_w, 2, 1);
}

#endif /* ARM_NN_ENABLE_F32 */
