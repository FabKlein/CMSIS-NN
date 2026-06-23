/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_depthwise_conv1d_k9_f32.c
 * Description:  Support: NHWC depthwise 1D convolution kernel size 9 for f32
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_depthwise_conv_small_mult_f32.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

void arm_nn_depthwise_conv1d_k9_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                         int32_t in_c,
                                         int32_t in_w,
                                         int32_t ch_mult,
                                         const float32_t *__RESTRICT kernel,
                                         const float32_t *__RESTRICT b,
                                         float32_t *__RESTRICT out,
                                         int32_t out_w)
{
    ARM_NN_ASSERT(out_w == in_w - 9 + 1);
    (void)in_w;
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* Fill vector lanes across input channels for small channel multipliers. */
    if (in_c > 1 && (ch_mult == 1 || ch_mult == 2))
    {
        arm_nn_depthwise_conv_small_mult_f32(
            x_nhwc, 1, in_c, in_w, ch_mult, kernel, b, out, out_w, 1, 9, false, false, 0, 0);
        return;
    }
    #endif

    const int32_t out_c = in_c * ch_mult;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    int32_t ow = 0;
    for (; ow + 3 < out_w; ow += 4)
    {
        float32_t *dst0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float32_t *dst1 = out + (size_t)(ow + 1) * (size_t)out_c;
        float32_t *dst2 = out + (size_t)(ow + 2) * (size_t)out_c;
        float32_t *dst3 = out + (size_t)(ow + 3) * (size_t)out_c;

        for (int32_t c = 0; c < in_c; ++c)
        {
            for (int32_t m = 0; m < ch_mult; m += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(ch_mult - m));
                const int32_t oc = c * ch_mult + m;
                const float32x4_t bias = b ? vld1q_z(b + oc, p) : vdupq_n_f32(0.0f);
                float32x4_t acc0 = bias;
                float32x4_t acc1 = bias;
                float32x4_t acc2 = bias;
                float32x4_t acc3 = bias;

                for (int32_t k = 0; k < 9; ++k)
                {
                    const float32x4_t weight = vld1q_z(kernel + (size_t)k * out_c + oc, p);
                    acc0 = vfmaq_n_f32(acc0, weight, x_nhwc[((size_t)ow + k) * in_c + c]);
                    acc1 = vfmaq_n_f32(acc1, weight, x_nhwc[((size_t)ow + k + 1) * in_c + c]);
                    acc2 = vfmaq_n_f32(acc2, weight, x_nhwc[((size_t)ow + k + 2) * in_c + c]);
                    acc3 = vfmaq_n_f32(acc3, weight, x_nhwc[((size_t)ow + k + 3) * in_c + c]);
                }
                vst1q_p(dst0 + oc, acc0, p);
                vst1q_p(dst1 + oc, acc1, p);
                vst1q_p(dst2 + oc, acc2, p);
                vst1q_p(dst3 + oc, acc3, p);
            }
        }
    }

    for (; ow + 1 < out_w; ow += 2)
    {
        float32_t *dst0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float32_t *dst1 = out + (size_t)(ow + 1) * (size_t)out_c;

        for (int32_t c = 0; c < in_c; ++c)
        {
            for (int32_t m = 0; m < ch_mult; m += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(ch_mult - m));
                const int32_t oc = c * ch_mult + m;
                const float32x4_t bias = b ? vld1q_z(b + oc, p) : vdupq_n_f32(0.0f);
                float32x4_t acc0 = bias;
                float32x4_t acc1 = bias;

                for (int32_t k = 0; k < 9; ++k)
                {
                    const float32x4_t weight = vld1q_z(kernel + (size_t)k * out_c + oc, p);
                    acc0 = vfmaq_n_f32(acc0, weight, x_nhwc[((size_t)ow + k) * in_c + c]);
                    acc1 = vfmaq_n_f32(acc1, weight, x_nhwc[((size_t)ow + k + 1) * in_c + c]);
                }
                vst1q_p(dst0 + oc, acc0, p);
                vst1q_p(dst1 + oc, acc1, p);
            }
        }
    }

    if (ow < out_w)
    {
        float32_t *dst = out + (size_t)ow * (size_t)out_c;
        for (int32_t c = 0; c < in_c; ++c)
        {
            for (int32_t m = 0; m < ch_mult; m += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(ch_mult - m));
                const int32_t oc = c * ch_mult + m;
                float32x4_t acc = b ? vld1q_z(b + oc, p) : vdupq_n_f32(0.0f);

                for (int32_t k = 0; k < 9; ++k)
                {
                    acc = vfmaq_n_f32(
                        acc, vld1q_z(kernel + (size_t)k * out_c + oc, p), x_nhwc[((size_t)ow + k) * in_c + c]);
                }
                vst1q_p(dst + oc, acc, p);
            }
        }
    }
    #else
    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        float32_t *dst = out + (size_t)ow * (size_t)out_c;
        for (int32_t c = 0; c < in_c; ++c)
        {
            for (int32_t m = 0; m < ch_mult; ++m)
            {
                const int32_t oc = c * ch_mult + m;
                float32_t acc = b ? b[oc] : 0.0f;
                for (int32_t k = 0; k < 9; ++k)
                {
                    acc += x_nhwc[((size_t)ow + k) * in_c + c] * kernel[(size_t)k * out_c + oc];
                }
                dst[oc] = acc;
            }
        }
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F32 */
