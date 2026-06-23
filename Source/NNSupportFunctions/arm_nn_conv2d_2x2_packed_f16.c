/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv2d_2x2_packed_f16.c
 * Description:  Support: NHWC direct convolution kernel size 2x2 for packed f16 weights
 *
 * $Date:        23 Jun 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

void arm_nn_conv2d_2x2_packed_f16(const float16_t *__RESTRICT x_nhwc,
                                  int32_t in_c,
                                  int32_t in_h,
                                  int32_t in_w,
                                  const float16_t *__RESTRICT kernel_packed,
                                  const float16_t *__RESTRICT b,
                                  float16_t *__RESTRICT out,
                                  int32_t out_c,
                                  int32_t out_h,
                                  int32_t out_w)
{
    const int32_t block_cols = 8;
    const int32_t kernel_h = 2;
    const int32_t kernel_w = 2;
    ARM_NN_ASSERT(out_h == in_h - kernel_h + 1);
    ARM_NN_ASSERT(out_w == in_w - kernel_w + 1);
    const size_t kernel_plane = (size_t)kernel_h * (size_t)kernel_w * (size_t)in_c;

    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    for (int32_t oy = 0; oy < out_h; ++oy)
    {
        int32_t ox = 0;

        for (; ox + 5 < out_w; ox += 6)
        {
            float16_t *dst0 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 0)) * (size_t)out_c;
            float16_t *dst1 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 1)) * (size_t)out_c;
            float16_t *dst2 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 2)) * (size_t)out_c;
            float16_t *dst3 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 3)) * (size_t)out_c;
            float16_t *dst4 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 4)) * (size_t)out_c;
            float16_t *dst5 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 5)) * (size_t)out_c;

            int32_t oc = 0;
            for (; oc + block_cols <= out_c; oc += block_cols)
            {
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float16x8_t acc0 = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc1 = acc0;
                float16x8_t acc2 = acc0;
                float16x8_t acc3 = acc0;
                float16x8_t acc4 = acc0;
                float16x8_t acc5 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row0 =
                        x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)(ox + 0)) * (size_t)in_c;
                    const float16_t *row1 = row0 + (size_t)in_c;
                    const float16_t *row2 = row1 + (size_t)in_c;
                    const float16_t *row3 = row2 + (size_t)in_c;
                    const float16_t *row4 = row3 + (size_t)in_c;
                    const float16_t *row5 = row4 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const float16_t *x1 = row1 + (size_t)kx * (size_t)in_c;
                        const float16_t *x2 = row2 + (size_t)kx * (size_t)in_c;
                        const float16_t *x3 = row3 + (size_t)kx * (size_t)in_c;
                        const float16_t *x4 = row4 + (size_t)kx * (size_t)in_c;
                        const float16_t *x5 = row5 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16x8_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
                            acc0 = vfmaq(acc0, w, x0[ic]);
                            acc1 = vfmaq(acc1, w, x1[ic]);
                            acc2 = vfmaq(acc2, w, x2[ic]);
                            acc3 = vfmaq(acc3, w, x3[ic]);
                            acc4 = vfmaq(acc4, w, x4[ic]);
                            acc5 = vfmaq(acc5, w, x5[ic]);
                        }
                    }
                }

                vst1q(dst0 + oc, acc0);
                vst1q(dst1 + oc, acc1);
                vst1q(dst2 + oc, acc2);
                vst1q(dst3 + oc, acc3);
                vst1q(dst4 + oc, acc4);
                vst1q(dst5 + oc, acc5);
            }

            if (oc < out_c)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(out_c - oc));
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float16x8_t acc0 = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc1 = acc0;
                float16x8_t acc2 = acc0;
                float16x8_t acc3 = acc0;
                float16x8_t acc4 = acc0;
                float16x8_t acc5 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row0 =
                        x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)(ox + 0)) * (size_t)in_c;
                    const float16_t *row1 = row0 + (size_t)in_c;
                    const float16_t *row2 = row1 + (size_t)in_c;
                    const float16_t *row3 = row2 + (size_t)in_c;
                    const float16_t *row4 = row3 + (size_t)in_c;
                    const float16_t *row5 = row4 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const float16_t *x1 = row1 + (size_t)kx * (size_t)in_c;
                        const float16_t *x2 = row2 + (size_t)kx * (size_t)in_c;
                        const float16_t *x3 = row3 + (size_t)kx * (size_t)in_c;
                        const float16_t *x4 = row4 + (size_t)kx * (size_t)in_c;
                        const float16_t *x5 = row5 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16x8_t w =
                                vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
                            acc0 = vfmaq(acc0, w, x0[ic]);
                            acc1 = vfmaq(acc1, w, x1[ic]);
                            acc2 = vfmaq(acc2, w, x2[ic]);
                            acc3 = vfmaq(acc3, w, x3[ic]);
                            acc4 = vfmaq(acc4, w, x4[ic]);
                            acc5 = vfmaq(acc5, w, x5[ic]);
                        }
                    }
                }

                vst1q_p(dst0 + oc, acc0, p);
                vst1q_p(dst1 + oc, acc1, p);
                vst1q_p(dst2 + oc, acc2, p);
                vst1q_p(dst3 + oc, acc3, p);
                vst1q_p(dst4 + oc, acc4, p);
                vst1q_p(dst5 + oc, acc5, p);
            }
        }

        for (; ox + 3 < out_w; ox += 4)
        {
            float16_t *dst0 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 0)) * (size_t)out_c;
            float16_t *dst1 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 1)) * (size_t)out_c;
            float16_t *dst2 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 2)) * (size_t)out_c;
            float16_t *dst3 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 3)) * (size_t)out_c;

            int32_t oc = 0;
            for (; oc + block_cols <= out_c; oc += block_cols)
            {
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float16x8_t acc0 = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc1 = acc0;
                float16x8_t acc2 = acc0;
                float16x8_t acc3 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row0 =
                        x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)(ox + 0)) * (size_t)in_c;
                    const float16_t *row1 = row0 + (size_t)in_c;
                    const float16_t *row2 = row1 + (size_t)in_c;
                    const float16_t *row3 = row2 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const float16_t *x1 = row1 + (size_t)kx * (size_t)in_c;
                        const float16_t *x2 = row2 + (size_t)kx * (size_t)in_c;
                        const float16_t *x3 = row3 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16x8_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
                            acc0 = vfmaq(acc0, w, x0[ic]);
                            acc1 = vfmaq(acc1, w, x1[ic]);
                            acc2 = vfmaq(acc2, w, x2[ic]);
                            acc3 = vfmaq(acc3, w, x3[ic]);
                        }
                    }
                }

                vst1q(dst0 + oc, acc0);
                vst1q(dst1 + oc, acc1);
                vst1q(dst2 + oc, acc2);
                vst1q(dst3 + oc, acc3);
            }

            if (oc < out_c)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(out_c - oc));
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float16x8_t acc0 = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc1 = acc0;
                float16x8_t acc2 = acc0;
                float16x8_t acc3 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row0 =
                        x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)(ox + 0)) * (size_t)in_c;
                    const float16_t *row1 = row0 + (size_t)in_c;
                    const float16_t *row2 = row1 + (size_t)in_c;
                    const float16_t *row3 = row2 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const float16_t *x1 = row1 + (size_t)kx * (size_t)in_c;
                        const float16_t *x2 = row2 + (size_t)kx * (size_t)in_c;
                        const float16_t *x3 = row3 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16x8_t w =
                                vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
                            acc0 = vfmaq(acc0, w, x0[ic]);
                            acc1 = vfmaq(acc1, w, x1[ic]);
                            acc2 = vfmaq(acc2, w, x2[ic]);
                            acc3 = vfmaq(acc3, w, x3[ic]);
                        }
                    }
                }

                vst1q_p(dst0 + oc, acc0, p);
                vst1q_p(dst1 + oc, acc1, p);
                vst1q_p(dst2 + oc, acc2, p);
                vst1q_p(dst3 + oc, acc3, p);
            }
        }

        for (; ox + 1 < out_w; ox += 2)
        {
            float16_t *dst0 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 0)) * (size_t)out_c;
            float16_t *dst1 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 1)) * (size_t)out_c;

            int32_t oc = 0;
            for (; oc + block_cols <= out_c; oc += block_cols)
            {
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float16x8_t acc0 = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc1 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row0 =
                        x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)(ox + 0)) * (size_t)in_c;
                    const float16_t *row1 = row0 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const float16_t *x1 = row1 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16x8_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
                            acc0 = vfmaq(acc0, w, x0[ic]);
                            acc1 = vfmaq(acc1, w, x1[ic]);
                        }
                    }
                }

                vst1q(dst0 + oc, acc0);
                vst1q(dst1 + oc, acc1);
            }

            if (oc < out_c)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(out_c - oc));
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float16x8_t acc0 = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);
                float16x8_t acc1 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row0 =
                        x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)(ox + 0)) * (size_t)in_c;
                    const float16_t *row1 = row0 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const float16_t *x1 = row1 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16x8_t w =
                                vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
                            acc0 = vfmaq(acc0, w, x0[ic]);
                            acc1 = vfmaq(acc1, w, x1[ic]);
                        }
                    }
                }

                vst1q_p(dst0 + oc, acc0, p);
                vst1q_p(dst1 + oc, acc1, p);
            }
        }

        for (; ox < out_w; ++ox)
        {
            float16_t *dst = out + ((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c;
            int32_t oc = 0;
            for (; oc + block_cols <= out_c; oc += block_cols)
            {
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float16x8_t acc = b ? vld1q(b + oc) : vdupq_n_f16((float16_t)0.0f);

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x = row + (size_t)kx * (size_t)in_c;
                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16x8_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
                            acc = vfmaq(acc, w, x[ic]);
                        }
                    }
                }

                vst1q(dst + oc, acc);
            }

            if (oc < out_c)
            {
                const mve_pred16_t p = vctp16q((uint32_t)(out_c - oc));
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float16x8_t acc = b ? vld1q_z(b + oc, p) : vdupq_n_f16((float16_t)0.0f);

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x = row + (size_t)kx * (size_t)in_c;
                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16x8_t w =
                                vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
                            acc = vfmaq(acc, w, x[ic]);
                        }
                    }
                }

                vst1q_p(dst + oc, acc, p);
            }
        }
    }
    #else
    for (int32_t oy = 0; oy < out_h; ++oy)
    {
        for (int32_t ox = 0; ox < out_w; ++ox)
        {
            float16_t *dst = out + ((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c;
            for (int32_t oc = 0; oc < out_c; ++oc)
            {
                const int32_t lane = oc % block_cols;
                const float16_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float32_t acc = b ? (float32_t)b[oc] : 0.0f;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const float16_t *row = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const float16_t *x = row + (size_t)kx * (size_t)in_c;
                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const float16_t *w = w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols;
                            acc += (float32_t)x[ic] * (float32_t)w[lane];
                        }
                    }
                }

                dst[oc] = (float16_t)acc;
            }
        }
    }
    #endif
}

#endif /* ARM_NN_ENABLE_F16 */
