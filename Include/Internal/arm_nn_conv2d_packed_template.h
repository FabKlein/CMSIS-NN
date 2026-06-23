/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv2d_packed_template.h
 * Description:  Shared packed float 2D convolution template
 *
 * $Date:        10 September 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

/*
 * Shared compile-time body for packed F16/F32 2D convolution.
 * Intentionally no include guard: each inclusion emits one named function.
 * Include through arm_nn_conv2d_packed_f16_template.h or its F32 counterpart;
 * those adapters supply the scalar/vector types, packed block width and MVE operations.
 *
 * The caller supplies ARM_NN_CONV2D_PACKED_NAME, ARM_NN_CONV2D_PACKED_KERNEL_H
 * and ARM_NN_CONV2D_PACKED_KERNEL_W, then undefines them after inclusion.
 * Filter dimensions are positive compile-time constants. The generated kernel
 * requires unit stride/dilation and zero padding; the dispatcher validates geometry.
 */
#if !defined(ARM_NN_CONV2D_PACKED_NAME) || !defined(ARM_NN_CONV2D_PACKED_KERNEL_H) ||                                  \
    !defined(ARM_NN_CONV2D_PACKED_KERNEL_W)
    #error "Define the function name, kernel height and kernel width before including this template"
#endif

#if ARM_NN_CONV2D_PACKED_KERNEL_H <= 0 || ARM_NN_CONV2D_PACKED_KERNEL_W <= 0
    #error "Kernel height and width must be positive integer constants"
#endif

#define ARM_NN_CONV2D_PACKED_EXPAND6                                                                                   \
    (ARM_NN_CONV2D_PACKED_KERNEL_H == 2 &&                                                                             \
     (ARM_NN_CONV2D_PACKED_KERNEL_W == 2 || ARM_NN_CONV2D_PACKED_KERNEL_W == 3 || ARM_NN_CONV2D_PACKED_KERNEL_W == 5))

/* Expand the 2x2/2x3/2x5 six-output tiles one kernel row at a time.
 * Each input row supplies kernel_w + 5 scalar values shared by overlapping taps.
 * The input-channel loop surrounds the taps, reusing those scalar loads.
 * Other filter shapes retain the looped body below. Keep the smaller output
 * tiles looped; expansion can change their code generation.
 * LOAD selects full or predicated weight-vector loads.
 */
#if ARM_NN_CONV2D_PACKED_USE_MVE && ARM_NN_CONV2D_PACKED_EXPAND6
    #define ARM_NN_CONV2D_PACKED_TAP6(LOAD, K, A, B, C, D, E, F)                                                       \
        do                                                                                                             \
        {                                                                                                              \
            const vector_t w = LOAD(w_row + (size_t)(K) * (size_t)in_c * (size_t)block_cols);                          \
            acc0 = vfmaq(acc0, w, A);                                                                                  \
            acc1 = vfmaq(acc1, w, B);                                                                                  \
            acc2 = vfmaq(acc2, w, C);                                                                                  \
            acc3 = vfmaq(acc3, w, D);                                                                                  \
            acc4 = vfmaq(acc4, w, E);                                                                                  \
            acc5 = vfmaq(acc5, w, F);                                                                                  \
        } while (0)
    #define ARM_NN_CONV2D_PACKED_INPUTS_7(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6)
    #define ARM_NN_CONV2D_PACKED_TAPS_2(LOAD)                                                                          \
        ARM_NN_CONV2D_PACKED_TAP6(LOAD, 0, sx0, sx1, sx2, sx3, sx4, sx5);                                              \
        ARM_NN_CONV2D_PACKED_TAP6(LOAD, 1, sx1, sx2, sx3, sx4, sx5, sx6);
    #if ARM_NN_CONV2D_PACKED_KERNEL_W == 2
        #define ARM_NN_CONV2D_PACKED_INPUTS(OP) ARM_NN_CONV2D_PACKED_INPUTS_7(OP)
        #define ARM_NN_CONV2D_PACKED_TAPS(LOAD) ARM_NN_CONV2D_PACKED_TAPS_2(LOAD)
    #else
        #define ARM_NN_CONV2D_PACKED_TAPS_3(LOAD)                                                                      \
            ARM_NN_CONV2D_PACKED_TAPS_2(LOAD)                                                                          \
            ARM_NN_CONV2D_PACKED_TAP6(LOAD, 2, sx2, sx3, sx4, sx5, sx6, sx7);
        #if ARM_NN_CONV2D_PACKED_KERNEL_W == 3
            #define ARM_NN_CONV2D_PACKED_INPUTS(OP) ARM_NN_CONV2D_PACKED_INPUTS_7(OP) OP(7)
            #define ARM_NN_CONV2D_PACKED_TAPS(LOAD) ARM_NN_CONV2D_PACKED_TAPS_3(LOAD)
        #else
            #define ARM_NN_CONV2D_PACKED_INPUTS(OP) ARM_NN_CONV2D_PACKED_INPUTS_7(OP) OP(7) OP(8) OP(9)
            #define ARM_NN_CONV2D_PACKED_TAPS(LOAD)                                                                    \
                ARM_NN_CONV2D_PACKED_TAPS_3(LOAD)                                                                      \
                ARM_NN_CONV2D_PACKED_TAP6(LOAD, 3, sx3, sx4, sx5, sx6, sx7, sx8);                                      \
                ARM_NN_CONV2D_PACKED_TAP6(LOAD, 4, sx4, sx5, sx6, sx7, sx8, sx9);
        #endif
    #endif
    #define ARM_NN_CONV2D_PACKED_LOAD_INPUT(I)                                                                         \
        const scalar_t sx##I = input_row[(size_t)(I) * (size_t)in_c + (size_t)ic];
    #define ARM_NN_CONV2D_PACKED_ROW6(LOAD, KY)                                                                        \
        do                                                                                                             \
        {                                                                                                              \
            const scalar_t *input_row = input_base + (size_t)(KY) * (size_t)in_w * (size_t)in_c;                       \
            for (int32_t ic = 0; ic < in_c; ++ic)                                                                      \
            {                                                                                                          \
                const scalar_t *w_row =                                                                                \
                    w_base + ((size_t)(KY) * (size_t)kernel_w * (size_t)in_c + (size_t)ic) * (size_t)block_cols;       \
                ARM_NN_CONV2D_PACKED_INPUTS(ARM_NN_CONV2D_PACKED_LOAD_INPUT)                                           \
                ARM_NN_CONV2D_PACKED_TAPS(LOAD)                                                                        \
            }                                                                                                          \
        } while (0)
    #define ARM_NN_CONV2D_PACKED_REDUCE6(LOAD)                                                                         \
        do                                                                                                             \
        {                                                                                                              \
            const scalar_t *input_base = x_nhwc + ((size_t)oy * (size_t)in_w + (size_t)ox) * (size_t)in_c;             \
            ARM_NN_CONV2D_PACKED_ROW6(LOAD, 0);                                                                        \
            ARM_NN_CONV2D_PACKED_ROW6(LOAD, 1);                                                                        \
        } while (0)
    #define ARM_NN_CONV2D_PACKED_LOAD_P(PTR) vld1q_z((PTR), p)
#endif

void ARM_NN_CONV2D_PACKED_NAME(const ARM_NN_CONV2D_PACKED_SCALAR_T *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_h,
                               int32_t in_w,
                               const ARM_NN_CONV2D_PACKED_SCALAR_T *__RESTRICT kernel_packed,
                               const ARM_NN_CONV2D_PACKED_SCALAR_T *__RESTRICT b,
                               ARM_NN_CONV2D_PACKED_SCALAR_T *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_h,
                               int32_t out_w)
{
    typedef ARM_NN_CONV2D_PACKED_SCALAR_T scalar_t;

    /* The expanded six-output MVE tiles reduce in row/channel/column order.
     * Other tiles and the scalar fallback retain row/column/channel order.
     * Rounding can therefore differ between tiles and generic GEMM.
     * Scalar F16 and F32 accumulate in float32; MVE uses native precision.
     */
    const int32_t block_cols = ARM_NN_CONV2D_PACKED_LANES;
    const int32_t kernel_h = ARM_NN_CONV2D_PACKED_KERNEL_H;
    const int32_t kernel_w = ARM_NN_CONV2D_PACKED_KERNEL_W;
    ARM_NN_ASSERT(out_h == in_h - kernel_h + 1);
    ARM_NN_ASSERT(out_w == in_w - kernel_w + 1);
    (void)in_h;
    const size_t kernel_plane = (size_t)kernel_h * (size_t)kernel_w * (size_t)in_c;

#if ARM_NN_CONV2D_PACKED_USE_MVE
    typedef ARM_NN_CONV2D_PACKED_VECTOR_T vector_t;

    for (int32_t oy = 0; oy < out_h; ++oy)
    {
        int32_t ox = 0;

        for (; ox + 5 < out_w; ox += 6)
        {
            scalar_t *dst0 = out + ((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c;
            scalar_t *dst1 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 1)) * (size_t)out_c;
            scalar_t *dst2 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 2)) * (size_t)out_c;
            scalar_t *dst3 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 3)) * (size_t)out_c;
            scalar_t *dst4 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 4)) * (size_t)out_c;
            scalar_t *dst5 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 5)) * (size_t)out_c;

            int32_t oc = 0;
            for (; oc + block_cols <= out_c; oc += block_cols)
            {
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                vector_t acc0 = b ? vld1q(b + oc) : ARM_NN_CONV2D_PACKED_ZERO();
                vector_t acc1 = acc0;
                vector_t acc2 = acc0;
                vector_t acc3 = acc0;
                vector_t acc4 = acc0;
                vector_t acc5 = acc0;

    #if ARM_NN_CONV2D_PACKED_EXPAND6
                ARM_NN_CONV2D_PACKED_REDUCE6(vld1q);
    #else
                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row0 = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    const scalar_t *row1 = row0 + (size_t)in_c;
                    const scalar_t *row2 = row1 + (size_t)in_c;
                    const scalar_t *row3 = row2 + (size_t)in_c;
                    const scalar_t *row4 = row3 + (size_t)in_c;
                    const scalar_t *row5 = row4 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x1 = row1 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x2 = row2 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x3 = row3 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x4 = row4 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x5 = row5 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const vector_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
                            acc0 = vfmaq(acc0, w, x0[ic]);
                            acc1 = vfmaq(acc1, w, x1[ic]);
                            acc2 = vfmaq(acc2, w, x2[ic]);
                            acc3 = vfmaq(acc3, w, x3[ic]);
                            acc4 = vfmaq(acc4, w, x4[ic]);
                            acc5 = vfmaq(acc5, w, x5[ic]);
                        }
                    }
                }

    #endif

                vst1q(dst0 + oc, acc0);
                vst1q(dst1 + oc, acc1);
                vst1q(dst2 + oc, acc2);
                vst1q(dst3 + oc, acc3);
                vst1q(dst4 + oc, acc4);
                vst1q(dst5 + oc, acc5);
            }

            if (oc < out_c)
            {
                const mve_pred16_t p = ARM_NN_CONV2D_PACKED_PREDICATE((uint32_t)(out_c - oc));
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                vector_t acc0 = b ? vld1q_z(b + oc, p) : ARM_NN_CONV2D_PACKED_ZERO();
                vector_t acc1 = acc0;
                vector_t acc2 = acc0;
                vector_t acc3 = acc0;
                vector_t acc4 = acc0;
                vector_t acc5 = acc0;

    #if ARM_NN_CONV2D_PACKED_EXPAND6
                ARM_NN_CONV2D_PACKED_REDUCE6(ARM_NN_CONV2D_PACKED_LOAD_P);
    #else
                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row0 = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    const scalar_t *row1 = row0 + (size_t)in_c;
                    const scalar_t *row2 = row1 + (size_t)in_c;
                    const scalar_t *row3 = row2 + (size_t)in_c;
                    const scalar_t *row4 = row3 + (size_t)in_c;
                    const scalar_t *row5 = row4 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x1 = row1 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x2 = row2 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x3 = row3 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x4 = row4 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x5 = row5 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const vector_t w = vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
                            acc0 = vfmaq(acc0, w, x0[ic]);
                            acc1 = vfmaq(acc1, w, x1[ic]);
                            acc2 = vfmaq(acc2, w, x2[ic]);
                            acc3 = vfmaq(acc3, w, x3[ic]);
                            acc4 = vfmaq(acc4, w, x4[ic]);
                            acc5 = vfmaq(acc5, w, x5[ic]);
                        }
                    }
                }

    #endif

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
            scalar_t *dst0 = out + ((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c;
            scalar_t *dst1 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 1)) * (size_t)out_c;
            scalar_t *dst2 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 2)) * (size_t)out_c;
            scalar_t *dst3 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 3)) * (size_t)out_c;

            int32_t oc = 0;
            for (; oc + block_cols <= out_c; oc += block_cols)
            {
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                vector_t acc0 = b ? vld1q(b + oc) : ARM_NN_CONV2D_PACKED_ZERO();
                vector_t acc1 = acc0;
                vector_t acc2 = acc0;
                vector_t acc3 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row0 = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    const scalar_t *row1 = row0 + (size_t)in_c;
                    const scalar_t *row2 = row1 + (size_t)in_c;
                    const scalar_t *row3 = row2 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x1 = row1 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x2 = row2 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x3 = row3 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const vector_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
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
                const mve_pred16_t p = ARM_NN_CONV2D_PACKED_PREDICATE((uint32_t)(out_c - oc));
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                vector_t acc0 = b ? vld1q_z(b + oc, p) : ARM_NN_CONV2D_PACKED_ZERO();
                vector_t acc1 = acc0;
                vector_t acc2 = acc0;
                vector_t acc3 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row0 = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    const scalar_t *row1 = row0 + (size_t)in_c;
                    const scalar_t *row2 = row1 + (size_t)in_c;
                    const scalar_t *row3 = row2 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x1 = row1 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x2 = row2 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x3 = row3 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const vector_t w = vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
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
            scalar_t *dst0 = out + ((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c;
            scalar_t *dst1 = out + ((size_t)oy * (size_t)out_w + (size_t)(ox + 1)) * (size_t)out_c;

            int32_t oc = 0;
            for (; oc + block_cols <= out_c; oc += block_cols)
            {
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                vector_t acc0 = b ? vld1q(b + oc) : ARM_NN_CONV2D_PACKED_ZERO();
                vector_t acc1 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row0 = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    const scalar_t *row1 = row0 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x1 = row1 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const vector_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
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
                const mve_pred16_t p = ARM_NN_CONV2D_PACKED_PREDICATE((uint32_t)(out_c - oc));
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                vector_t acc0 = b ? vld1q_z(b + oc, p) : ARM_NN_CONV2D_PACKED_ZERO();
                vector_t acc1 = acc0;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row0 = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    const scalar_t *row1 = row0 + (size_t)in_c;

                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                        const scalar_t *x1 = row1 + (size_t)kx * (size_t)in_c;

                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const vector_t w = vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
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
            scalar_t *dst = out + ((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c;
            int32_t oc = 0;
            for (; oc + block_cols <= out_c; oc += block_cols)
            {
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                vector_t acc = b ? vld1q(b + oc) : ARM_NN_CONV2D_PACKED_ZERO();

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x = row + (size_t)kx * (size_t)in_c;
                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const vector_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
                            acc = vfmaq(acc, w, x[ic]);
                        }
                    }
                }

                vst1q(dst + oc, acc);
            }

            if (oc < out_c)
            {
                const mve_pred16_t p = ARM_NN_CONV2D_PACKED_PREDICATE((uint32_t)(out_c - oc));
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                vector_t acc = b ? vld1q_z(b + oc, p) : ARM_NN_CONV2D_PACKED_ZERO();

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x = row + (size_t)kx * (size_t)in_c;
                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const vector_t w = vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
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
            scalar_t *dst = out + ((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c;
            for (int32_t oc = 0; oc < out_c; ++oc)
            {
                const int32_t lane = oc % block_cols;
                const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_plane * block_cols;
                float32_t acc = b ? (float32_t)b[oc] : 0.0f;

                for (int32_t ky = 0; ky < kernel_h; ++ky)
                {
                    const scalar_t *row = x_nhwc + ((size_t)(oy + ky) * (size_t)in_w + (size_t)ox) * (size_t)in_c;
                    for (int32_t kx = 0; kx < kernel_w; ++kx)
                    {
                        const size_t kernel_col_base = ((size_t)ky * (size_t)kernel_w + (size_t)kx) * (size_t)in_c;
                        const scalar_t *x = row + (size_t)kx * (size_t)in_c;
                        for (int32_t ic = 0; ic < in_c; ++ic)
                        {
                            const scalar_t *w = w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols;
                            acc += (float32_t)x[ic] * (float32_t)w[lane];
                        }
                    }
                }

                dst[oc] = (scalar_t)acc;
            }
        }
    }
#endif
}

#undef ARM_NN_CONV2D_PACKED_LOAD_P
#undef ARM_NN_CONV2D_PACKED_REDUCE6
#undef ARM_NN_CONV2D_PACKED_ROW6
#undef ARM_NN_CONV2D_PACKED_TAP6
#undef ARM_NN_CONV2D_PACKED_LOAD_INPUT
#undef ARM_NN_CONV2D_PACKED_INPUTS
#undef ARM_NN_CONV2D_PACKED_INPUTS_7
#undef ARM_NN_CONV2D_PACKED_TAPS
#undef ARM_NN_CONV2D_PACKED_TAPS_2
#undef ARM_NN_CONV2D_PACKED_TAPS_3
#undef ARM_NN_CONV2D_PACKED_EXPAND6
