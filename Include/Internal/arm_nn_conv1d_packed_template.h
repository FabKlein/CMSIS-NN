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
 * Title:        arm_nn_conv1d_packed_template.h
 * Description:  Shared packed float 1D convolution template
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

/*
 * Shared compile-time body for packed F16/F32 1D convolution.
 * Intentionally no include guard: each inclusion emits one named function.
 * Include through arm_nn_conv1d_packed_f16_template.h or its F32 counterpart;
 * those adapters supply the precision-specific types, packed block width and
 * MVE operations. Loads, stores and FMA use the existing overloaded intrinsics.
 *
 * The caller supplies ARM_NN_CONV1D_PACKED_NAME and ARM_NN_CONV1D_PACKED_KERNEL_W
 * and undefines them after inclusion. Kernel width must be 7 or 9; there is no runtime width argument.
 */
#if !defined(ARM_NN_CONV1D_PACKED_NAME) || !defined(ARM_NN_CONV1D_PACKED_KERNEL_W)
    #error "Define the function name and kernel width before including this template"
#endif

#if ARM_NN_CONV1D_PACKED_KERNEL_W != 7 && ARM_NN_CONV1D_PACKED_KERNEL_W != 9
    #error "Packed 1D template supports kernel widths 7 and 9"
#endif

/* Expand the six-output MVE tile explicitly. Its channel-first loop reuses
 * scalar inputs across overlapping windows and matches the handwritten k7/k9
 * kernels. Smaller tiles retain their existing loops: forcing expansion there
 * can regress short-output cases and affect code generation for this tile.
 * INPUTS enumerates the 12 (k7) or 14 (k9) values needed for six outputs.
 * LOAD selects full or predicated weight-vector loads.
 */
#if ARM_NN_CONV1D_PACKED_USE_MVE
    #define ARM_NN_CONV1D_PACKED_INPUTS_12(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7) OP(8) OP(9) OP(10) OP(11)
    #if ARM_NN_CONV1D_PACKED_KERNEL_W == 7
        #define ARM_NN_CONV1D_PACKED_INPUTS(OP) ARM_NN_CONV1D_PACKED_INPUTS_12(OP)
    #else
        #define ARM_NN_CONV1D_PACKED_INPUTS(OP) ARM_NN_CONV1D_PACKED_INPUTS_12(OP) OP(12) OP(13)
    #endif

    #define ARM_NN_CONV1D_PACKED_TAP6(LOAD, K, A, B, C, D, E, F)                                                       \
        do                                                                                                             \
        {                                                                                                              \
            const vector_t w = LOAD(w_ic + (size_t)(K) * (size_t)in_c * (size_t)block_cols);                           \
            acc0 = vfmaq(acc0, w, A);                                                                                  \
            acc1 = vfmaq(acc1, w, B);                                                                                  \
            acc2 = vfmaq(acc2, w, C);                                                                                  \
            acc3 = vfmaq(acc3, w, D);                                                                                  \
            acc4 = vfmaq(acc4, w, E);                                                                                  \
            acc5 = vfmaq(acc5, w, F);                                                                                  \
        } while (0)
    #define ARM_NN_CONV1D_PACKED_TAPS_7(LOAD)                                                                          \
        ARM_NN_CONV1D_PACKED_TAP6(LOAD, 0, sx0, sx1, sx2, sx3, sx4, sx5);                                              \
        ARM_NN_CONV1D_PACKED_TAP6(LOAD, 1, sx1, sx2, sx3, sx4, sx5, sx6);                                              \
        ARM_NN_CONV1D_PACKED_TAP6(LOAD, 2, sx2, sx3, sx4, sx5, sx6, sx7);                                              \
        ARM_NN_CONV1D_PACKED_TAP6(LOAD, 3, sx3, sx4, sx5, sx6, sx7, sx8);                                              \
        ARM_NN_CONV1D_PACKED_TAP6(LOAD, 4, sx4, sx5, sx6, sx7, sx8, sx9);                                              \
        ARM_NN_CONV1D_PACKED_TAP6(LOAD, 5, sx5, sx6, sx7, sx8, sx9, sx10);                                             \
        ARM_NN_CONV1D_PACKED_TAP6(LOAD, 6, sx6, sx7, sx8, sx9, sx10, sx11);
    #if ARM_NN_CONV1D_PACKED_KERNEL_W == 7
        #define ARM_NN_CONV1D_PACKED_TAPS(LOAD) ARM_NN_CONV1D_PACKED_TAPS_7(LOAD)
    #else
        #define ARM_NN_CONV1D_PACKED_TAPS(LOAD)                                                                        \
            ARM_NN_CONV1D_PACKED_TAPS_7(LOAD)                                                                          \
            ARM_NN_CONV1D_PACKED_TAP6(LOAD, 7, sx7, sx8, sx9, sx10, sx11, sx12);                                       \
            ARM_NN_CONV1D_PACKED_TAP6(LOAD, 8, sx8, sx9, sx10, sx11, sx12, sx13);
    #endif

    #define ARM_NN_CONV1D_PACKED_DECLARE_INPUT(I) const scalar_t *x##I = input_base + (size_t)(I) * (size_t)in_c;
    #define ARM_NN_CONV1D_PACKED_LOAD_INPUT(I) const scalar_t sx##I = x##I[ic];
    #define ARM_NN_CONV1D_PACKED_LOAD_P(PTR) vld1q_z((PTR), p)
    #define ARM_NN_CONV1D_PACKED_REDUCE6(LOAD)                                                                         \
        do                                                                                                             \
        {                                                                                                              \
            const scalar_t *input_base = x_nhwc + (size_t)ox * (size_t)in_c;                                           \
            ARM_NN_CONV1D_PACKED_INPUTS(ARM_NN_CONV1D_PACKED_DECLARE_INPUT)                                            \
            for (int32_t ic = 0; ic < in_c; ++ic)                                                                      \
            {                                                                                                          \
                const scalar_t *w_ic = w_base + (size_t)ic * (size_t)block_cols;                                       \
                ARM_NN_CONV1D_PACKED_INPUTS(ARM_NN_CONV1D_PACKED_LOAD_INPUT)                                           \
                ARM_NN_CONV1D_PACKED_TAPS(LOAD)                                                                        \
            }                                                                                                          \
        } while (0)
#endif

void ARM_NN_CONV1D_PACKED_NAME(const ARM_NN_CONV1D_PACKED_SCALAR_T *__RESTRICT x_nhwc,
                               int32_t in_c,
                               int32_t in_w,
                               const ARM_NN_CONV1D_PACKED_SCALAR_T *__RESTRICT kernel_packed,
                               const ARM_NN_CONV1D_PACKED_SCALAR_T *__RESTRICT b,
                               ARM_NN_CONV1D_PACKED_SCALAR_T *__RESTRICT out,
                               int32_t out_c,
                               int32_t out_w)
{
    typedef ARM_NN_CONV1D_PACKED_SCALAR_T scalar_t;

    /*
     * The six-output MVE tile uses the original kernels' channel-first reduction.
     * Smaller tiles and the scalar fallback remain tap-first. Floating-point
     * rounding can differ between tiles and generic GEMM for cancelling inputs.
     */
    const int32_t block_cols = ARM_NN_CONV1D_PACKED_LANES;
    const int32_t kernel_w = ARM_NN_CONV1D_PACKED_KERNEL_W;
    ARM_NN_ASSERT(out_w == in_w - kernel_w + 1);
    (void)in_w;
    const size_t kernel_elems = (size_t)kernel_w * (size_t)in_c;

#if ARM_NN_CONV1D_PACKED_USE_MVE
    typedef ARM_NN_CONV1D_PACKED_VECTOR_T vector_t;

    int32_t ox = 0;

    for (; ox + 5 < out_w; ox += 6)
    {
        scalar_t *dst0 = out + (size_t)ox * (size_t)out_c;
        scalar_t *dst1 = out + (size_t)(ox + 1) * (size_t)out_c;
        scalar_t *dst2 = out + (size_t)(ox + 2) * (size_t)out_c;
        scalar_t *dst3 = out + (size_t)(ox + 3) * (size_t)out_c;
        scalar_t *dst4 = out + (size_t)(ox + 4) * (size_t)out_c;
        scalar_t *dst5 = out + (size_t)(ox + 5) * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            vector_t acc0 = b ? vld1q(b + oc) : ARM_NN_CONV1D_PACKED_ZERO();
            vector_t acc1 = acc0;
            vector_t acc2 = acc0;
            vector_t acc3 = acc0;
            vector_t acc4 = acc0;
            vector_t acc5 = acc0;

            ARM_NN_CONV1D_PACKED_REDUCE6(vld1q);

            vst1q(dst0 + oc, acc0);
            vst1q(dst1 + oc, acc1);
            vst1q(dst2 + oc, acc2);
            vst1q(dst3 + oc, acc3);
            vst1q(dst4 + oc, acc4);
            vst1q(dst5 + oc, acc5);
        }

        if (oc < out_c)
        {
            const mve_pred16_t p = ARM_NN_CONV1D_PACKED_PREDICATE((uint32_t)(out_c - oc));
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            vector_t acc0 = b ? vld1q_z(b + oc, p) : ARM_NN_CONV1D_PACKED_ZERO();
            vector_t acc1 = acc0;
            vector_t acc2 = acc0;
            vector_t acc3 = acc0;
            vector_t acc4 = acc0;
            vector_t acc5 = acc0;

            ARM_NN_CONV1D_PACKED_REDUCE6(ARM_NN_CONV1D_PACKED_LOAD_P);

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
        scalar_t *dst0 = out + (size_t)ox * (size_t)out_c;
        scalar_t *dst1 = out + (size_t)(ox + 1) * (size_t)out_c;
        scalar_t *dst2 = out + (size_t)(ox + 2) * (size_t)out_c;
        scalar_t *dst3 = out + (size_t)(ox + 3) * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            vector_t acc0 = b ? vld1q(b + oc) : ARM_NN_CONV1D_PACKED_ZERO();
            vector_t acc1 = acc0;
            vector_t acc2 = acc0;
            vector_t acc3 = acc0;

            const scalar_t *row0 = x_nhwc + (size_t)ox * (size_t)in_c;
            const scalar_t *row1 = row0 + (size_t)in_c;
            const scalar_t *row2 = row1 + (size_t)in_c;
            const scalar_t *row3 = row2 + (size_t)in_c;

            for (int32_t kx = 0; kx < kernel_w; ++kx)
            {
                const size_t kernel_col_base = (size_t)kx * (size_t)in_c;
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

            vst1q(dst0 + oc, acc0);
            vst1q(dst1 + oc, acc1);
            vst1q(dst2 + oc, acc2);
            vst1q(dst3 + oc, acc3);
        }

        if (oc < out_c)
        {
            const mve_pred16_t p = ARM_NN_CONV1D_PACKED_PREDICATE((uint32_t)(out_c - oc));
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            vector_t acc0 = b ? vld1q_z(b + oc, p) : ARM_NN_CONV1D_PACKED_ZERO();
            vector_t acc1 = acc0;
            vector_t acc2 = acc0;
            vector_t acc3 = acc0;

            const scalar_t *row0 = x_nhwc + (size_t)ox * (size_t)in_c;
            const scalar_t *row1 = row0 + (size_t)in_c;
            const scalar_t *row2 = row1 + (size_t)in_c;
            const scalar_t *row3 = row2 + (size_t)in_c;

            for (int32_t kx = 0; kx < kernel_w; ++kx)
            {
                const size_t kernel_col_base = (size_t)kx * (size_t)in_c;
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

            vst1q_p(dst0 + oc, acc0, p);
            vst1q_p(dst1 + oc, acc1, p);
            vst1q_p(dst2 + oc, acc2, p);
            vst1q_p(dst3 + oc, acc3, p);
        }
    }

    for (; ox + 1 < out_w; ox += 2)
    {
        scalar_t *dst0 = out + (size_t)ox * (size_t)out_c;
        scalar_t *dst1 = out + (size_t)(ox + 1) * (size_t)out_c;

        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            vector_t acc0 = b ? vld1q(b + oc) : ARM_NN_CONV1D_PACKED_ZERO();
            vector_t acc1 = acc0;

            const scalar_t *row0 = x_nhwc + (size_t)ox * (size_t)in_c;
            const scalar_t *row1 = row0 + (size_t)in_c;

            for (int32_t kx = 0; kx < kernel_w; ++kx)
            {
                const size_t kernel_col_base = (size_t)kx * (size_t)in_c;
                const scalar_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                const scalar_t *x1 = row1 + (size_t)kx * (size_t)in_c;

                for (int32_t ic = 0; ic < in_c; ++ic)
                {
                    const vector_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
                    acc0 = vfmaq(acc0, w, x0[ic]);
                    acc1 = vfmaq(acc1, w, x1[ic]);
                }
            }

            vst1q(dst0 + oc, acc0);
            vst1q(dst1 + oc, acc1);
        }

        if (oc < out_c)
        {
            const mve_pred16_t p = ARM_NN_CONV1D_PACKED_PREDICATE((uint32_t)(out_c - oc));
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            vector_t acc0 = b ? vld1q_z(b + oc, p) : ARM_NN_CONV1D_PACKED_ZERO();
            vector_t acc1 = acc0;

            const scalar_t *row0 = x_nhwc + (size_t)ox * (size_t)in_c;
            const scalar_t *row1 = row0 + (size_t)in_c;

            for (int32_t kx = 0; kx < kernel_w; ++kx)
            {
                const size_t kernel_col_base = (size_t)kx * (size_t)in_c;
                const scalar_t *x0 = row0 + (size_t)kx * (size_t)in_c;
                const scalar_t *x1 = row1 + (size_t)kx * (size_t)in_c;

                for (int32_t ic = 0; ic < in_c; ++ic)
                {
                    const vector_t w = vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
                    acc0 = vfmaq(acc0, w, x0[ic]);
                    acc1 = vfmaq(acc1, w, x1[ic]);
                }
            }

            vst1q_p(dst0 + oc, acc0, p);
            vst1q_p(dst1 + oc, acc1, p);
        }
    }

    for (; ox < out_w; ++ox)
    {
        scalar_t *dst = out + (size_t)ox * (size_t)out_c;
        int32_t oc = 0;
        for (; oc + block_cols <= out_c; oc += block_cols)
        {
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            vector_t acc = b ? vld1q(b + oc) : ARM_NN_CONV1D_PACKED_ZERO();

            const scalar_t *row = x_nhwc + (size_t)ox * (size_t)in_c;
            for (int32_t kx = 0; kx < kernel_w; ++kx)
            {
                const size_t kernel_col_base = (size_t)kx * (size_t)in_c;
                const scalar_t *x = row + (size_t)kx * (size_t)in_c;
                for (int32_t ic = 0; ic < in_c; ++ic)
                {
                    const vector_t w = vld1q(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols);
                    acc = vfmaq(acc, w, x[ic]);
                }
            }

            vst1q(dst + oc, acc);
        }

        if (oc < out_c)
        {
            const mve_pred16_t p = ARM_NN_CONV1D_PACKED_PREDICATE((uint32_t)(out_c - oc));
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            vector_t acc = b ? vld1q_z(b + oc, p) : ARM_NN_CONV1D_PACKED_ZERO();

            const scalar_t *row = x_nhwc + (size_t)ox * (size_t)in_c;
            for (int32_t kx = 0; kx < kernel_w; ++kx)
            {
                const size_t kernel_col_base = (size_t)kx * (size_t)in_c;
                const scalar_t *x = row + (size_t)kx * (size_t)in_c;
                for (int32_t ic = 0; ic < in_c; ++ic)
                {
                    const vector_t w = vld1q_z(w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols, p);
                    acc = vfmaq(acc, w, x[ic]);
                }
            }

            vst1q_p(dst + oc, acc, p);
        }
    }
#else
    typedef ARM_NN_CONV1D_PACKED_ACCUMULATOR_T accumulator_t;

    for (int32_t ox = 0; ox < out_w; ++ox)
    {
        scalar_t *dst = out + (size_t)ox * (size_t)out_c;
        for (int32_t oc = 0; oc < out_c; ++oc)
        {
            const int32_t lane = oc % block_cols;
            const scalar_t *w_base = kernel_packed + ((size_t)oc / block_cols) * kernel_elems * block_cols;
            accumulator_t acc = b ? (accumulator_t)b[oc] : (accumulator_t)0;

            const scalar_t *row = x_nhwc + (size_t)ox * (size_t)in_c;
            for (int32_t kx = 0; kx < kernel_w; ++kx)
            {
                const size_t kernel_col_base = (size_t)kx * (size_t)in_c;
                const scalar_t *x = row + (size_t)kx * (size_t)in_c;
                for (int32_t ic = 0; ic < in_c; ++ic)
                {
                    const scalar_t *w = w_base + (kernel_col_base + (size_t)ic) * (size_t)block_cols;
                    acc += (accumulator_t)x[ic] * (accumulator_t)w[lane];
                }
            }

            dst[oc] = (scalar_t)acc;
        }
    }
#endif
}

#if ARM_NN_CONV1D_PACKED_USE_MVE
    #undef ARM_NN_CONV1D_PACKED_INPUTS_12
    #undef ARM_NN_CONV1D_PACKED_INPUTS
    #undef ARM_NN_CONV1D_PACKED_TAP6
    #undef ARM_NN_CONV1D_PACKED_TAPS_7
    #undef ARM_NN_CONV1D_PACKED_TAPS
    #undef ARM_NN_CONV1D_PACKED_DECLARE_INPUT
    #undef ARM_NN_CONV1D_PACKED_LOAD_INPUT
    #undef ARM_NN_CONV1D_PACKED_LOAD_P
    #undef ARM_NN_CONV1D_PACKED_REDUCE6
#endif
