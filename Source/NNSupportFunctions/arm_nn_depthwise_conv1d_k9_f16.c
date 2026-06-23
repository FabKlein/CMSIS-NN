/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_depthwise_conv1d_k9_f16.c
 * Description:  Support: NHWC depthwise 1D convolution kernel size 9 for f16
 *
 * $Date:        19 Jun 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

void arm_nn_depthwise_conv1d_k9_nhwc_f16(const float16_t *__RESTRICT x_nhwc,
                                         int32_t in_c,
                                         int32_t in_w,
                                         int32_t ch_mult,
                                         const float16_t *__RESTRICT kernel,
                                         const float16_t *__RESTRICT b,
                                         float16_t *__RESTRICT out,
                                         int32_t out_w)
{
    ARM_NN_ASSERT(out_w == in_w - 9 + 1);

    const int32_t out_c = in_c * ch_mult;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    int32_t ow = 0;
    /* Keep this path in MVE intrinsics for portability across supported toolchains. */
    for (; ow + 3 < out_w; ow += 4)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        const float16_t *x5 = x_nhwc + (size_t)(ow + 5) * (size_t)in_c;
        const float16_t *x6 = x_nhwc + (size_t)(ow + 6) * (size_t)in_c;
        const float16_t *x7 = x_nhwc + (size_t)(ow + 7) * (size_t)in_c;
        const float16_t *x8 = x_nhwc + (size_t)(ow + 8) * (size_t)in_c;
        const float16_t *x9 = x_nhwc + (size_t)(ow + 9) * (size_t)in_c;
        const float16_t *x10 = x_nhwc + (size_t)(ow + 10) * (size_t)in_c;
        const float16_t *x11 = x_nhwc + (size_t)(ow + 11) * (size_t)in_c;
        float16_t *dst0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float16_t *dst1 = out + (size_t)(ow + 1) * (size_t)out_c;
        float16_t *dst2 = out + (size_t)(ow + 2) * (size_t)out_c;
        float16_t *dst3 = out + (size_t)(ow + 3) * (size_t)out_c;

        for (int32_t c = 0; c < in_c; ++c)
        {
            for (int32_t m = 0; m < ch_mult; m += 8)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(ch_mult - m));
                const int32_t oc = c * ch_mult + m;
                const float16x8_t bias = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc0 = bias;
                float16x8_t acc1 = bias;
                float16x8_t acc2 = bias;
                float16x8_t acc3 = bias;
                const float16x8_t w0 = vld1q_z(kernel + (size_t)0 * out_c + oc, p);
                const float16x8_t w1 = vld1q_z(kernel + (size_t)1 * out_c + oc, p);
                const float16x8_t w2 = vld1q_z(kernel + (size_t)2 * out_c + oc, p);
                const float16x8_t w3 = vld1q_z(kernel + (size_t)3 * out_c + oc, p);
                const float16x8_t w4 = vld1q_z(kernel + (size_t)4 * out_c + oc, p);
                const float16x8_t w5 = vld1q_z(kernel + (size_t)5 * out_c + oc, p);
                const float16x8_t w6 = vld1q_z(kernel + (size_t)6 * out_c + oc, p);
                const float16x8_t w7 = vld1q_z(kernel + (size_t)7 * out_c + oc, p);
                const float16x8_t w8 = vld1q_z(kernel + (size_t)8 * out_c + oc, p);

                acc0 = vfmaq_n_f16(acc0, w0, x0[c]);
                acc1 = vfmaq_n_f16(acc1, w0, x1[c]);
                acc2 = vfmaq_n_f16(acc2, w0, x2[c]);
                acc3 = vfmaq_n_f16(acc3, w0, x3[c]);
                acc0 = vfmaq_n_f16(acc0, w1, x1[c]);
                acc1 = vfmaq_n_f16(acc1, w1, x2[c]);
                acc2 = vfmaq_n_f16(acc2, w1, x3[c]);
                acc3 = vfmaq_n_f16(acc3, w1, x4[c]);
                acc0 = vfmaq_n_f16(acc0, w2, x2[c]);
                acc1 = vfmaq_n_f16(acc1, w2, x3[c]);
                acc2 = vfmaq_n_f16(acc2, w2, x4[c]);
                acc3 = vfmaq_n_f16(acc3, w2, x5[c]);
                acc0 = vfmaq_n_f16(acc0, w3, x3[c]);
                acc1 = vfmaq_n_f16(acc1, w3, x4[c]);
                acc2 = vfmaq_n_f16(acc2, w3, x5[c]);
                acc3 = vfmaq_n_f16(acc3, w3, x6[c]);
                acc0 = vfmaq_n_f16(acc0, w4, x4[c]);
                acc1 = vfmaq_n_f16(acc1, w4, x5[c]);
                acc2 = vfmaq_n_f16(acc2, w4, x6[c]);
                acc3 = vfmaq_n_f16(acc3, w4, x7[c]);
                acc0 = vfmaq_n_f16(acc0, w5, x5[c]);
                acc1 = vfmaq_n_f16(acc1, w5, x6[c]);
                acc2 = vfmaq_n_f16(acc2, w5, x7[c]);
                acc3 = vfmaq_n_f16(acc3, w5, x8[c]);
                acc0 = vfmaq_n_f16(acc0, w6, x6[c]);
                acc1 = vfmaq_n_f16(acc1, w6, x7[c]);
                acc2 = vfmaq_n_f16(acc2, w6, x8[c]);
                acc3 = vfmaq_n_f16(acc3, w6, x9[c]);
                acc0 = vfmaq_n_f16(acc0, w7, x7[c]);
                acc1 = vfmaq_n_f16(acc1, w7, x8[c]);
                acc2 = vfmaq_n_f16(acc2, w7, x9[c]);
                acc3 = vfmaq_n_f16(acc3, w7, x10[c]);
                acc0 = vfmaq_n_f16(acc0, w8, x8[c]);
                acc1 = vfmaq_n_f16(acc1, w8, x9[c]);
                acc2 = vfmaq_n_f16(acc2, w8, x10[c]);
                acc3 = vfmaq_n_f16(acc3, w8, x11[c]);
                vst1q_p(dst0 + oc, acc0, p);
                vst1q_p(dst1 + oc, acc1, p);
                vst1q_p(dst2 + oc, acc2, p);
                vst1q_p(dst3 + oc, acc3, p);
            }
        }
    }

    for (; ow + 1 < out_w; ow += 2)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        const float16_t *x5 = x_nhwc + (size_t)(ow + 5) * (size_t)in_c;
        const float16_t *x6 = x_nhwc + (size_t)(ow + 6) * (size_t)in_c;
        const float16_t *x7 = x_nhwc + (size_t)(ow + 7) * (size_t)in_c;
        const float16_t *x8 = x_nhwc + (size_t)(ow + 8) * (size_t)in_c;
        const float16_t *x9 = x_nhwc + (size_t)(ow + 9) * (size_t)in_c;
        float16_t *dst0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float16_t *dst1 = out + (size_t)(ow + 1) * (size_t)out_c;

        for (int32_t c = 0; c < in_c; ++c)
        {
            for (int32_t m = 0; m < ch_mult; m += 8)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(ch_mult - m));
                const int32_t oc = c * ch_mult + m;
                const float16x8_t bias = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc0 = bias;
                float16x8_t acc1 = bias;
                const float16x8_t w0 = vld1q_z(kernel + (size_t)0 * out_c + oc, p);
                const float16x8_t w1 = vld1q_z(kernel + (size_t)1 * out_c + oc, p);
                const float16x8_t w2 = vld1q_z(kernel + (size_t)2 * out_c + oc, p);
                const float16x8_t w3 = vld1q_z(kernel + (size_t)3 * out_c + oc, p);
                const float16x8_t w4 = vld1q_z(kernel + (size_t)4 * out_c + oc, p);
                const float16x8_t w5 = vld1q_z(kernel + (size_t)5 * out_c + oc, p);
                const float16x8_t w6 = vld1q_z(kernel + (size_t)6 * out_c + oc, p);
                const float16x8_t w7 = vld1q_z(kernel + (size_t)7 * out_c + oc, p);
                const float16x8_t w8 = vld1q_z(kernel + (size_t)8 * out_c + oc, p);

                acc0 = vfmaq_n_f16(acc0, w0, x0[c]);
                acc1 = vfmaq_n_f16(acc1, w0, x1[c]);
                acc0 = vfmaq_n_f16(acc0, w1, x1[c]);
                acc1 = vfmaq_n_f16(acc1, w1, x2[c]);
                acc0 = vfmaq_n_f16(acc0, w2, x2[c]);
                acc1 = vfmaq_n_f16(acc1, w2, x3[c]);
                acc0 = vfmaq_n_f16(acc0, w3, x3[c]);
                acc1 = vfmaq_n_f16(acc1, w3, x4[c]);
                acc0 = vfmaq_n_f16(acc0, w4, x4[c]);
                acc1 = vfmaq_n_f16(acc1, w4, x5[c]);
                acc0 = vfmaq_n_f16(acc0, w5, x5[c]);
                acc1 = vfmaq_n_f16(acc1, w5, x6[c]);
                acc0 = vfmaq_n_f16(acc0, w6, x6[c]);
                acc1 = vfmaq_n_f16(acc1, w6, x7[c]);
                acc0 = vfmaq_n_f16(acc0, w7, x7[c]);
                acc1 = vfmaq_n_f16(acc1, w7, x8[c]);
                acc0 = vfmaq_n_f16(acc0, w8, x8[c]);
                acc1 = vfmaq_n_f16(acc1, w8, x9[c]);
                vst1q_p(dst0 + oc, acc0, p);
                vst1q_p(dst1 + oc, acc1, p);
            }
        }
    }

    if (ow < out_w)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        const float16_t *x5 = x_nhwc + (size_t)(ow + 5) * (size_t)in_c;
        const float16_t *x6 = x_nhwc + (size_t)(ow + 6) * (size_t)in_c;
        const float16_t *x7 = x_nhwc + (size_t)(ow + 7) * (size_t)in_c;
        const float16_t *x8 = x_nhwc + (size_t)(ow + 8) * (size_t)in_c;
        float16_t *dst = out + (size_t)ow * (size_t)out_c;

        for (int32_t c = 0; c < in_c; ++c)
        {
            for (int32_t m = 0; m < ch_mult; m += 8)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(ch_mult - m));
                const int32_t oc = c * ch_mult + m;
                const float16x8_t bias = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc = bias;

                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)0 * out_c + oc, p), x0[c]);
                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)1 * out_c + oc, p), x1[c]);
                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)2 * out_c + oc, p), x2[c]);
                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)3 * out_c + oc, p), x3[c]);
                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)4 * out_c + oc, p), x4[c]);
                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)5 * out_c + oc, p), x5[c]);
                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)6 * out_c + oc, p), x6[c]);
                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)7 * out_c + oc, p), x7[c]);
                acc = vfmaq_n_f16(acc, vld1q_z(kernel + (size_t)8 * out_c + oc, p), x8[c]);
                vst1q_p(dst + oc, acc, p);
            }
        }
    }
    #else
    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        const float16_t *x5 = x_nhwc + (size_t)(ow + 5) * (size_t)in_c;
        const float16_t *x6 = x_nhwc + (size_t)(ow + 6) * (size_t)in_c;
        const float16_t *x7 = x_nhwc + (size_t)(ow + 7) * (size_t)in_c;
        const float16_t *x8 = x_nhwc + (size_t)(ow + 8) * (size_t)in_c;
        float16_t *dst = out + (size_t)ow * (size_t)out_c;

        for (int32_t c = 0; c < in_c; ++c)
        {
            for (int32_t m = 0; m < ch_mult; ++m)
            {
                const int32_t oc = c * ch_mult + m;
                float32_t acc0 = b ? (float32_t)b[oc] : 0.0f;
                acc0 += (float32_t)x0[c] * (float32_t)kernel[(size_t)0 * out_c + oc];
                acc0 += (float32_t)x1[c] * (float32_t)kernel[(size_t)1 * out_c + oc];
                acc0 += (float32_t)x2[c] * (float32_t)kernel[(size_t)2 * out_c + oc];
                acc0 += (float32_t)x3[c] * (float32_t)kernel[(size_t)3 * out_c + oc];
                acc0 += (float32_t)x4[c] * (float32_t)kernel[(size_t)4 * out_c + oc];
                acc0 += (float32_t)x5[c] * (float32_t)kernel[(size_t)5 * out_c + oc];
                acc0 += (float32_t)x6[c] * (float32_t)kernel[(size_t)6 * out_c + oc];
                acc0 += (float32_t)x7[c] * (float32_t)kernel[(size_t)7 * out_c + oc];
                acc0 += (float32_t)x8[c] * (float32_t)kernel[(size_t)8 * out_c + oc];
                dst[oc] = (float16_t)acc0;
            }
        }
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F16 */
