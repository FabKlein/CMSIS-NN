/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv1d_k3_packed_f16.c
 * Description:  Support: NHWC 1D convolution kernel size 3 for packed f16 weights
 *
 * $Date:        29 Apr 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

void arm_nn_conv1d_k3_packed_f16(const float16_t *__RESTRICT x_nhwc,
                                 int32_t in_c,
                                 int32_t in_w,
                                 const float16_t *__RESTRICT kernel_packed,
                                 const float16_t *__RESTRICT b,
                                 float16_t *__RESTRICT out,
                                 int32_t out_c,
                                 int32_t out_w)
{
    ARM_NN_ASSERT(out_w == in_w - 3 + 1);

    const int32_t block_cols = 8;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    int32_t ow = 0;
    for (; ow + 5 < out_w; ow += 6)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        const float16_t *x5 = x_nhwc + (size_t)(ow + 5) * (size_t)in_c;
        const float16_t *x6 = x_nhwc + (size_t)(ow + 6) * (size_t)in_c;
        const float16_t *x7 = x_nhwc + (size_t)(ow + 7) * (size_t)in_c;
        float16_t *y0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float16_t *y1 = out + (size_t)(ow + 1) * (size_t)out_c;
        float16_t *y2 = out + (size_t)(ow + 2) * (size_t)out_c;
        float16_t *y3 = out + (size_t)(ow + 3) * (size_t)out_c;
        float16_t *y4 = out + (size_t)(ow + 4) * (size_t)out_c;
        float16_t *y5 = out + (size_t)(ow + 5) * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;
            float16x8_t vacc2 = vacc0;
            float16x8_t vacc3 = vacc0;
            float16x8_t vacc4 = vacc0;
            float16x8_t vacc5 = vacc0;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                const float16_t sx0 = x0[ic];
                const float16_t sx1 = x1[ic];
                const float16_t sx2 = x2[ic];
                const float16_t sx3 = x3[ic];
                const float16_t sx4 = x4[ic];
                const float16_t sx5 = x5[ic];
                const float16_t sx6 = x6[ic];
                const float16_t sx7 = x7[ic];

                float16x8_t vw = vld1q(w_ic + 0U * in_c * block_cols);
                vacc0 = vfmaq(vacc0, vw, sx0);
                vacc1 = vfmaq(vacc1, vw, sx1);
                vacc2 = vfmaq(vacc2, vw, sx2);
                vacc3 = vfmaq(vacc3, vw, sx3);
                vacc4 = vfmaq(vacc4, vw, sx4);
                vacc5 = vfmaq(vacc5, vw, sx5);
                vw = vld1q(w_ic + 1U * in_c * block_cols);
                vacc0 = vfmaq(vacc0, vw, sx1);
                vacc1 = vfmaq(vacc1, vw, sx2);
                vacc2 = vfmaq(vacc2, vw, sx3);
                vacc3 = vfmaq(vacc3, vw, sx4);
                vacc4 = vfmaq(vacc4, vw, sx5);
                vacc5 = vfmaq(vacc5, vw, sx6);
                vw = vld1q(w_ic + 2U * in_c * block_cols);
                vacc0 = vfmaq(vacc0, vw, sx2);
                vacc1 = vfmaq(vacc1, vw, sx3);
                vacc2 = vfmaq(vacc2, vw, sx4);
                vacc3 = vfmaq(vacc3, vw, sx5);
                vacc4 = vfmaq(vacc4, vw, sx6);
                vacc5 = vfmaq(vacc5, vw, sx7);
            }

            vst1q(y0 + oc, vacc0);
            vst1q(y1 + oc, vacc1);
            vst1q(y2 + oc, vacc2);
            vst1q(y3 + oc, vacc3);
            vst1q(y4 + oc, vacc4);
            vst1q(y5 + oc, vacc5);
        }

        if (oc < out_c)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(out_c - oc));
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;
            float16x8_t vacc2 = vacc0;
            float16x8_t vacc3 = vacc0;
            float16x8_t vacc4 = vacc0;
            float16x8_t vacc5 = vacc0;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                const float16_t sx0 = x0[ic];
                const float16_t sx1 = x1[ic];
                const float16_t sx2 = x2[ic];
                const float16_t sx3 = x3[ic];
                const float16_t sx4 = x4[ic];
                const float16_t sx5 = x5[ic];
                const float16_t sx6 = x6[ic];
                const float16_t sx7 = x7[ic];

                float16x8_t vw = vld1q_z(w_ic + 0U * in_c * block_cols, p);
                vacc0 = vfmaq(vacc0, vw, sx0);
                vacc1 = vfmaq(vacc1, vw, sx1);
                vacc2 = vfmaq(vacc2, vw, sx2);
                vacc3 = vfmaq(vacc3, vw, sx3);
                vacc4 = vfmaq(vacc4, vw, sx4);
                vacc5 = vfmaq(vacc5, vw, sx5);
                vw = vld1q_z(w_ic + 1U * in_c * block_cols, p);
                vacc0 = vfmaq(vacc0, vw, sx1);
                vacc1 = vfmaq(vacc1, vw, sx2);
                vacc2 = vfmaq(vacc2, vw, sx3);
                vacc3 = vfmaq(vacc3, vw, sx4);
                vacc4 = vfmaq(vacc4, vw, sx5);
                vacc5 = vfmaq(vacc5, vw, sx6);
                vw = vld1q_z(w_ic + 2U * in_c * block_cols, p);
                vacc0 = vfmaq(vacc0, vw, sx2);
                vacc1 = vfmaq(vacc1, vw, sx3);
                vacc2 = vfmaq(vacc2, vw, sx4);
                vacc3 = vfmaq(vacc3, vw, sx5);
                vacc4 = vfmaq(vacc4, vw, sx6);
                vacc5 = vfmaq(vacc5, vw, sx7);
            }

            vst1q_p(y0 + oc, vacc0, p);
            vst1q_p(y1 + oc, vacc1, p);
            vst1q_p(y2 + oc, vacc2, p);
            vst1q_p(y3 + oc, vacc3, p);
            vst1q_p(y4 + oc, vacc4, p);
            vst1q_p(y5 + oc, vacc5, p);
        }
    }
    for (; ow + 3 < out_w; ow += 4)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        const float16_t *x4 = x_nhwc + (size_t)(ow + 4) * (size_t)in_c;
        const float16_t *x5 = x_nhwc + (size_t)(ow + 5) * (size_t)in_c;
        float16_t *y0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float16_t *y1 = out + (size_t)(ow + 1) * (size_t)out_c;
        float16_t *y2 = out + (size_t)(ow + 2) * (size_t)out_c;
        float16_t *y3 = out + (size_t)(ow + 3) * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;
            float16x8_t vacc2 = vacc0;
            float16x8_t vacc3 = vacc0;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                const float16x8_t vw0 = vld1q(w_ic + 0U * in_c * block_cols);
                const float16x8_t vw1 = vld1q(w_ic + 1U * in_c * block_cols);
                const float16x8_t vw2 = vld1q(w_ic + 2U * in_c * block_cols);

                vacc0 = vfmaq(vacc0, vw0, x0[ic]);
                vacc1 = vfmaq(vacc1, vw0, x1[ic]);
                vacc2 = vfmaq(vacc2, vw0, x2[ic]);
                vacc3 = vfmaq(vacc3, vw0, x3[ic]);
                vacc0 = vfmaq(vacc0, vw1, x1[ic]);
                vacc1 = vfmaq(vacc1, vw1, x2[ic]);
                vacc2 = vfmaq(vacc2, vw1, x3[ic]);
                vacc3 = vfmaq(vacc3, vw1, x4[ic]);
                vacc0 = vfmaq(vacc0, vw2, x2[ic]);
                vacc1 = vfmaq(vacc1, vw2, x3[ic]);
                vacc2 = vfmaq(vacc2, vw2, x4[ic]);
                vacc3 = vfmaq(vacc3, vw2, x5[ic]);
            }

            vst1q(y0 + oc, vacc0);
            vst1q(y1 + oc, vacc1);
            vst1q(y2 + oc, vacc2);
            vst1q(y3 + oc, vacc3);
        }

        if (oc < out_c)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(out_c - oc));
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;
            float16x8_t vacc2 = vacc0;
            float16x8_t vacc3 = vacc0;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                const float16x8_t vw0 = vld1q_z(w_ic + 0U * in_c * block_cols, p);
                const float16x8_t vw1 = vld1q_z(w_ic + 1U * in_c * block_cols, p);
                const float16x8_t vw2 = vld1q_z(w_ic + 2U * in_c * block_cols, p);

                vacc0 = vfmaq(vacc0, vw0, x0[ic]);
                vacc1 = vfmaq(vacc1, vw0, x1[ic]);
                vacc2 = vfmaq(vacc2, vw0, x2[ic]);
                vacc3 = vfmaq(vacc3, vw0, x3[ic]);
                vacc0 = vfmaq(vacc0, vw1, x1[ic]);
                vacc1 = vfmaq(vacc1, vw1, x2[ic]);
                vacc2 = vfmaq(vacc2, vw1, x3[ic]);
                vacc3 = vfmaq(vacc3, vw1, x4[ic]);
                vacc0 = vfmaq(vacc0, vw2, x2[ic]);
                vacc1 = vfmaq(vacc1, vw2, x3[ic]);
                vacc2 = vfmaq(vacc2, vw2, x4[ic]);
                vacc3 = vfmaq(vacc3, vw2, x5[ic]);
            }

            vst1q_p(y0 + oc, vacc0, p);
            vst1q_p(y1 + oc, vacc1, p);
            vst1q_p(y2 + oc, vacc2, p);
            vst1q_p(y3 + oc, vacc3, p);
        }
    }

    for (; ow + 1 < out_w; ow += 2)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        const float16_t *x3 = x_nhwc + (size_t)(ow + 3) * (size_t)in_c;
        float16_t *y0 = out + (size_t)(ow + 0) * (size_t)out_c;
        float16_t *y1 = out + (size_t)(ow + 1) * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                const float16x8_t vw0 = vld1q(w_ic + 0U * in_c * block_cols);
                const float16x8_t vw1 = vld1q(w_ic + 1U * in_c * block_cols);
                const float16x8_t vw2 = vld1q(w_ic + 2U * in_c * block_cols);

                vacc0 = vfmaq(vacc0, vw0, x0[ic]);
                vacc1 = vfmaq(vacc1, vw0, x1[ic]);
                vacc0 = vfmaq(vacc0, vw1, x1[ic]);
                vacc1 = vfmaq(vacc1, vw1, x2[ic]);
                vacc0 = vfmaq(vacc0, vw2, x2[ic]);
                vacc1 = vfmaq(vacc1, vw2, x3[ic]);
            }

            vst1q(y0 + oc, vacc0);
            vst1q(y1 + oc, vacc1);
        }

        if (oc < out_c)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(out_c - oc));
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc0 = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
            float16x8_t vacc1 = vacc0;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                const float16x8_t vw0 = vld1q_z(w_ic + 0U * in_c * block_cols, p);
                const float16x8_t vw1 = vld1q_z(w_ic + 1U * in_c * block_cols, p);
                const float16x8_t vw2 = vld1q_z(w_ic + 2U * in_c * block_cols, p);

                vacc0 = vfmaq(vacc0, vw0, x0[ic]);
                vacc1 = vfmaq(vacc1, vw0, x1[ic]);
                vacc0 = vfmaq(vacc0, vw1, x1[ic]);
                vacc1 = vfmaq(vacc1, vw1, x2[ic]);
                vacc0 = vfmaq(vacc0, vw2, x2[ic]);
                vacc1 = vfmaq(vacc1, vw2, x3[ic]);
            }

            vst1q_p(y0 + oc, vacc0, p);
            vst1q_p(y1 + oc, vacc1, p);
        }
    }

    if (ow < out_w)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                vacc = vfmaq(vacc, vld1q(w_ic + 0U * in_c * block_cols), x0[ic]);
                vacc = vfmaq(vacc, vld1q(w_ic + 1U * in_c * block_cols), x1[ic]);
                vacc = vfmaq(vacc, vld1q(w_ic + 2U * in_c * block_cols), x2[ic]);
            }

            vst1q(y + oc, vacc);
        }

        if (oc < out_c)
        {
            const mve_pred16_t p = vctp16q((uint32_t)(out_c - oc));
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            float16x8_t vacc = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                vacc = vfmaq(vacc, vld1q_z(w_ic + 0U * in_c * block_cols, p), x0[ic]);
                vacc = vfmaq(vacc, vld1q_z(w_ic + 1U * in_c * block_cols, p), x1[ic]);
                vacc = vfmaq(vacc, vld1q_z(w_ic + 2U * in_c * block_cols, p), x2[ic]);
            }

            vst1q_p(y + oc, vacc, p);
        }
    }

    #else
    for (int32_t ow = 0; ow < out_w; ++ow)
    {
        const float16_t *x0 = x_nhwc + (size_t)(ow + 0) * (size_t)in_c;
        const float16_t *x1 = x_nhwc + (size_t)(ow + 1) * (size_t)in_c;
        const float16_t *x2 = x_nhwc + (size_t)(ow + 2) * (size_t)in_c;
        float16_t *y = out + (size_t)ow * (size_t)out_c;

        for (int32_t oc = 0; oc < out_c; ++oc)
        {
            const int32_t lane = oc % block_cols;
            const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * 3U * (size_t)in_c * block_cols;
            _Float16 acc = b ? (_Float16)b[oc] : (_Float16)0.0f;

            for (int32_t ic = 0; ic < in_c; ++ic)
            {
                const float16_t *w_ic = w_base + (size_t)ic * block_cols;
                acc += (_Float16)x0[ic] * (_Float16)w_ic[(size_t)0 * in_c * block_cols + lane];
                acc += (_Float16)x1[ic] * (_Float16)w_ic[(size_t)1 * in_c * block_cols + lane];
                acc += (_Float16)x2[ic] * (_Float16)w_ic[(size_t)2 * in_c * block_cols + lane];
            }

            y[oc] = (float16_t)acc;
        }
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F16 */
