/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv2d_3x3_packed_f16.c
 * Description:  Support: direct NHWC 3x3 convolution with packed f16 weights
 *
 * $Date:        19 October 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

/* Four horizontal outputs reuse each eight-output-channel weight vector.
 * Interior rows are contiguous: KH/KW/IC reduction order matches packed GEMM.
 * Borders and output-width tails use one output, with explicit zero padding.
 * No scratch allocation or im2row copy is needed. The dispatcher checks geometry.
 */
__STATIC_FORCEINLINE void conv3x3_packed_f16(const float16_t *input,
                                             int32_t in_c,
                                             int32_t in_h,
                                             int32_t in_w,
                                             const float16_t *weights,
                                             const float16_t *bias,
                                             float16_t *output,
                                             int32_t out_c,
                                             int32_t out_h,
                                             int32_t out_w,
                                             int32_t stride,
                                             int32_t padding)
{
    const size_t plane = (size_t)9 * in_c * 8;
    for (int32_t oy = 0; oy < out_h; ++oy)
    {
        const int32_t iy = oy * stride - padding;
        int32_t ox = 0;
        while (ox < out_w)
        {
            const int32_t ix = ox * stride - padding;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
            if (ox + 3 < out_w && iy >= 0 && iy + 2 < in_h && ix >= 0 && ix + 3 * stride + 2 < in_w)
            {
                const int32_t row_len = 3 * in_c;
                for (int32_t oc = 0; oc < out_c; oc += 8)
                {
                    const mve_pred16_t pred = vctp16q((uint32_t)(out_c - oc));
                    float16x8_t a0 = bias ? vld1q_z(bias + oc, pred) : vdupq_n_f16(0);
                    float16x8_t a1 = a0;
                    float16x8_t a2 = a0;
                    float16x8_t a3 = a0;
                    const float16_t *w = weights + (size_t)(oc / 8) * plane;
                    for (int32_t ky = 0; ky < 3; ++ky)
                    {
                        const float16_t *x0 = input + ((size_t)(iy + ky) * in_w + ix) * in_c;
                        const float16_t *x1 = x0 + stride * in_c;
                        const float16_t *x2 = x1 + stride * in_c;
                        const float16_t *x3 = x2 + stride * in_c;
                        for (int32_t k = 0; k < row_len; ++k)
                        {
                            /* Packed blocks include all eight lanes, even for a channel tail. */
                            const float16x8_t v = vld1q(w);
                            w += 8;
                            a0 = vfmaq(a0, v, x0[k]);
                            a1 = vfmaq(a1, v, x1[k]);
                            a2 = vfmaq(a2, v, x2[k]);
                            a3 = vfmaq(a3, v, x3[k]);
                        }
                    }
                    float16_t *dst = output + ((size_t)oy * out_w + ox) * out_c + oc;
                    vst1q_p(dst, a0, pred);
                    vst1q_p(dst + out_c, a1, pred);
                    vst1q_p(dst + 2 * out_c, a2, pred);
                    vst1q_p(dst + 3 * out_c, a3, pred);
                }
                ox += 4;
                continue;
            }
    #endif
            for (int32_t oc = 0; oc < out_c; oc += 8)
            {
                const float16_t *w = weights + (size_t)(oc / 8) * plane;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                const mve_pred16_t pred = vctp16q((uint32_t)(out_c - oc));
                float16x8_t acc = bias ? vld1q_z(bias + oc, pred) : vdupq_n_f16(0);
                for (int32_t ky = 0; ky < 3; ++ky)
                {
                    for (int32_t kx = 0; kx < 3; ++kx)
                    {
                        const int32_t y = iy + ky;
                        const int32_t x = ix + kx;
                        const bool valid = (uint32_t)y < (uint32_t)in_h && (uint32_t)x < (uint32_t)in_w;
                        const float16_t *src = valid ? input + ((size_t)y * in_w + x) * in_c : NULL;
                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16_t value = valid ? src[ic] : (float16_t)0;
                            acc = vfmaq(acc, vld1q(w), value);
                            w += 8;
                        }
                    }
                }
                vst1q_p(output + ((size_t)oy * out_w + ox) * out_c + oc, acc, pred);
    #else
                const int32_t lanes = MIN(8, out_c - oc);
                for (int32_t lane = 0; lane < lanes; ++lane)
                {
                    _Float16 acc = bias ? (_Float16)bias[oc + lane] : (_Float16)0;
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
                                const _Float16 value = valid ? input[((size_t)y * in_w + x) * in_c + ic] : 0;
                                acc += value * (_Float16)w[(size_t)k * 8 + lane];
                            }
                        }
                    }
                    output[((size_t)oy * out_w + ox) * out_c + oc + lane] = (float16_t)acc;
                }
    #endif
            }
            ++ox;
        }
    }
}

void arm_nn_conv2d_3x3_s1_p0_packed_f16(const float16_t *__RESTRICT input,
                                        int32_t in_c,
                                        int32_t in_h,
                                        int32_t in_w,
                                        const float16_t *__RESTRICT weights,
                                        const float16_t *__RESTRICT bias,
                                        float16_t *__RESTRICT output,
                                        int32_t out_c,
                                        int32_t out_h,
                                        int32_t out_w)
{
    conv3x3_packed_f16(input, in_c, in_h, in_w, weights, bias, output, out_c, out_h, out_w, 1, 0);
}

void arm_nn_conv2d_3x3_s1_p1_packed_f16(const float16_t *__RESTRICT input,
                                        int32_t in_c,
                                        int32_t in_h,
                                        int32_t in_w,
                                        const float16_t *__RESTRICT weights,
                                        const float16_t *__RESTRICT bias,
                                        float16_t *__RESTRICT output,
                                        int32_t out_c,
                                        int32_t out_h,
                                        int32_t out_w)
{
    conv3x3_packed_f16(input, in_c, in_h, in_w, weights, bias, output, out_c, out_h, out_w, 1, 1);
}

void arm_nn_conv2d_3x3_s2_p0_packed_f16(const float16_t *__RESTRICT input,
                                        int32_t in_c,
                                        int32_t in_h,
                                        int32_t in_w,
                                        const float16_t *__RESTRICT weights,
                                        const float16_t *__RESTRICT bias,
                                        float16_t *__RESTRICT output,
                                        int32_t out_c,
                                        int32_t out_h,
                                        int32_t out_w)
{
    conv3x3_packed_f16(input, in_c, in_h, in_w, weights, bias, output, out_c, out_h, out_w, 2, 0);
}

void arm_nn_conv2d_3x3_s2_p1_packed_f16(const float16_t *__RESTRICT input,
                                        int32_t in_c,
                                        int32_t in_h,
                                        int32_t in_w,
                                        const float16_t *__RESTRICT weights,
                                        const float16_t *__RESTRICT bias,
                                        float16_t *__RESTRICT output,
                                        int32_t out_c,
                                        int32_t out_h,
                                        int32_t out_w)
{
    conv3x3_packed_f16(input, in_c, in_h, in_w, weights, bias, output, out_c, out_h, out_w, 2, 1);
}

#endif /* ARM_NN_ENABLE_F16 */
