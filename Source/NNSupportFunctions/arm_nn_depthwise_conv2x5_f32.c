/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_depthwise_conv2x5_f32.c
 * Description:  Support: NHWC depthwise 2x5 convolution kernel for f32
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

void arm_nn_depthwise_conv2x5_nhwc_f32(const float32_t *__RESTRICT x_nhwc,
                                       int32_t batches,
                                       int32_t in_c,
                                       int32_t in_w,
                                       int32_t ch_mult,
                                       const float32_t *__RESTRICT kernel,
                                       const float32_t *__RESTRICT b,
                                       float32_t *__RESTRICT out,
                                       int32_t out_w,
                                       float32_t act_min,
                                       float32_t act_max)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    /* Fill vector lanes across input channels for small channel multipliers. */
    if (in_c > 1 && (ch_mult == 1 || ch_mult == 2))
    {
        arm_nn_depthwise_conv_small_mult_f32(
            x_nhwc, batches, in_c, in_w, ch_mult, kernel, b, out, out_w, 2, 5, true, true, act_min, act_max);
        return;
    }
    #endif

    const int32_t out_c = in_c * ch_mult;
    const size_t in_batch_stride = (size_t)2U * (size_t)in_w * (size_t)in_c;
    const size_t out_batch_stride = (size_t)out_w * (size_t)out_c;

    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    const float32x4_t v_act_min = vdupq_n_f32(act_min);
    const float32x4_t v_act_max = vdupq_n_f32(act_max);
    #endif

    for (int32_t batch = 0; batch < batches; ++batch)
    {
        const float32_t *in_batch = x_nhwc + (size_t)batch * in_batch_stride;
        float32_t *out_batch = out + (size_t)batch * out_batch_stride;

        for (int32_t ow = 0; ow < out_w; ++ow)
        {
            const float32_t *row0 = in_batch + (size_t)ow * (size_t)in_c;
            const float32_t *row1 = in_batch + ((size_t)in_w + (size_t)ow) * (size_t)in_c;

            for (int32_t c = 0; c < in_c; ++c)
            {
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
                for (int32_t m = 0; m < ch_mult; m += 4)
                {
                    const mve_pred16_t p = vctp32q((uint32_t)(ch_mult - m));
                    const int32_t oc = c * ch_mult + m;
                    float32x4_t acc = b ? vld1q_z(b + oc, p) : vdupq_n_f32(0.0f);

                    for (int32_t k = 0; k < 5; ++k)
                    {
                        acc = vfmaq_n_f32(acc, vld1q_z(kernel + (size_t)k * out_c + oc, p), row0[(size_t)k * in_c + c]);
                        acc = vfmaq_n_f32(
                            acc, vld1q_z(kernel + ((size_t)5 + k) * out_c + oc, p), row1[(size_t)k * in_c + c]);
                    }
                    acc = vmaxnmq(acc, v_act_min);
                    acc = vminnmq(acc, v_act_max);
                    vst1q_p(out_batch + (size_t)ow * out_c + oc, acc, p);
                }
    #else
                for (int32_t m = 0; m < ch_mult; ++m)
                {
                    const int32_t oc = c * ch_mult + m;
                    float32_t acc = b ? b[oc] : 0.0f;
                    for (int32_t k = 0; k < 5; ++k)
                    {
                        acc += row0[(size_t)k * in_c + c] * kernel[(size_t)k * out_c + oc];
                        acc += row1[(size_t)k * in_c + c] * kernel[((size_t)5 + k) * out_c + oc];
                    }
                    acc = MAX(acc, act_min);
                    acc = MIN(acc, act_max);
                    out_batch[(size_t)ow * out_c + oc] = acc;
                }
    #endif
            }
        }
    }
}

#endif /* ARM_NN_ENABLE_F32 */
