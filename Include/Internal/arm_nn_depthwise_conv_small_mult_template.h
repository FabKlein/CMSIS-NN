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
 * Title:        arm_nn_depthwise_conv_small_mult_template.h
 * Description:  Shared MVE depthwise convolution loops for small multipliers
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

/*
 * Shared MVE loop for depthwise multipliers that divide the vector width.
 * Intentionally no include guard: the F16/F32 adapters emit separate helpers.
 * Each vector spans several input channels and stores consecutive output channels.
 * The caller supplies fixed filter dimensions, one input row per filter row,
 * and a multiplier of 1, 2, or (for F16 only) 4. No scratch buffer is required.
 *
 * interleave_rows preserves the calling kernel's existing reduction order.
 * Clamp only for support kernels whose API includes activation bounds.
 */
__STATIC_FORCEINLINE void ARM_NN_DW_SMALL_MULT_NAME(const ARM_NN_DW_SMALL_MULT_SCALAR_T *input,
                                                    int32_t batches,
                                                    int32_t in_c,
                                                    int32_t in_w,
                                                    int32_t ch_mult,
                                                    const ARM_NN_DW_SMALL_MULT_SCALAR_T *kernel,
                                                    const ARM_NN_DW_SMALL_MULT_SCALAR_T *bias,
                                                    ARM_NN_DW_SMALL_MULT_SCALAR_T *output,
                                                    int32_t out_w,
                                                    int32_t kernel_h,
                                                    int32_t kernel_w,
                                                    bool interleave_rows,
                                                    bool clamp,
                                                    ARM_NN_DW_SMALL_MULT_SCALAR_T act_min,
                                                    ARM_NN_DW_SMALL_MULT_SCALAR_T act_max)
{
    typedef ARM_NN_DW_SMALL_MULT_SCALAR_T scalar_t;
    typedef ARM_NN_DW_SMALL_MULT_VECTOR_T vector_t;
    const int32_t lanes = ARM_NN_DW_SMALL_MULT_LANES;
    const int32_t out_c = in_c * ch_mult;
    const int32_t input_channels_per_vector = lanes / ch_mult;
    const size_t input_batch_stride = (size_t)kernel_h * in_w * in_c;
    const size_t output_batch_stride = (size_t)out_w * out_c;
    ARM_NN_DW_SMALL_MULT_OFFSETS_T offsets = ARM_NN_DW_SMALL_MULT_INDICES();
    if (ch_mult == 2)
    {
        offsets = ARM_NN_DW_SMALL_MULT_SHIFT(offsets, 1);
    }
    else if (ch_mult == 4)
    {
        offsets = ARM_NN_DW_SMALL_MULT_SHIFT(offsets, 2);
    }

    for (int32_t batch = 0; batch < batches; ++batch)
    {
        const scalar_t *input_b = input + (size_t)batch * input_batch_stride;
        scalar_t *output_b = output + (size_t)batch * output_batch_stride;
        for (int32_t ic = 0; ic < in_c; ic += input_channels_per_vector)
        {
            const int32_t oc = ic * ch_mult;
            const mve_pred16_t p = ARM_NN_DW_SMALL_MULT_PREDICATE((uint32_t)(out_c - oc));
            const vector_t initial = bias ? vld1q_z(bias + oc, p) : ARM_NN_DW_SMALL_MULT_DUP(0);
            int32_t ow = 0;
            for (; ow + 3 < out_w; ow += 4)
            {
                vector_t acc0 = initial;
                vector_t acc1 = initial;
                vector_t acc2 = initial;
                vector_t acc3 = initial;
                for (int32_t k = 0; k < kernel_h * kernel_w; ++k)
                {
                    const int32_t ky = interleave_rows ? k % kernel_h : k / kernel_w;
                    const int32_t kx = interleave_rows ? k / kernel_h : k % kernel_w;
                    const vector_t weight = vld1q_z(kernel + ((size_t)ky * kernel_w + kx) * out_c + oc, p);
                    const scalar_t *src = input_b + ((size_t)ky * in_w + ow + kx) * in_c + ic;
                    acc0 = vfmaq(acc0, weight, ARM_NN_DW_SMALL_MULT_LOAD(src, offsets, p, ch_mult == 1));
                    acc1 = vfmaq(
                        acc1, weight, ARM_NN_DW_SMALL_MULT_LOAD(src + (size_t)1 * in_c, offsets, p, ch_mult == 1));
                    acc2 = vfmaq(
                        acc2, weight, ARM_NN_DW_SMALL_MULT_LOAD(src + (size_t)2 * in_c, offsets, p, ch_mult == 1));
                    acc3 = vfmaq(
                        acc3, weight, ARM_NN_DW_SMALL_MULT_LOAD(src + (size_t)3 * in_c, offsets, p, ch_mult == 1));
                }
                if (clamp)
                {
                    acc0 = vminnmq(vmaxnmq(acc0, ARM_NN_DW_SMALL_MULT_DUP(act_min)), ARM_NN_DW_SMALL_MULT_DUP(act_max));
                    acc1 = vminnmq(vmaxnmq(acc1, ARM_NN_DW_SMALL_MULT_DUP(act_min)), ARM_NN_DW_SMALL_MULT_DUP(act_max));
                    acc2 = vminnmq(vmaxnmq(acc2, ARM_NN_DW_SMALL_MULT_DUP(act_min)), ARM_NN_DW_SMALL_MULT_DUP(act_max));
                    acc3 = vminnmq(vmaxnmq(acc3, ARM_NN_DW_SMALL_MULT_DUP(act_min)), ARM_NN_DW_SMALL_MULT_DUP(act_max));
                }
                vst1q_p(output_b + (size_t)ow * out_c + oc, acc0, p);
                vst1q_p(output_b + ((size_t)ow + 1) * out_c + oc, acc1, p);
                vst1q_p(output_b + ((size_t)ow + 2) * out_c + oc, acc2, p);
                vst1q_p(output_b + ((size_t)ow + 3) * out_c + oc, acc3, p);
            }
            for (; ow < out_w; ++ow)
            {
                vector_t acc0 = initial;
                for (int32_t k = 0; k < kernel_h * kernel_w; ++k)
                {
                    const int32_t ky = interleave_rows ? k % kernel_h : k / kernel_w;
                    const int32_t kx = interleave_rows ? k / kernel_h : k % kernel_w;
                    const vector_t weight = vld1q_z(kernel + ((size_t)ky * kernel_w + kx) * out_c + oc, p);
                    const scalar_t *src = input_b + ((size_t)ky * in_w + ow + kx) * in_c + ic;
                    acc0 = vfmaq(acc0, weight, ARM_NN_DW_SMALL_MULT_LOAD(src, offsets, p, ch_mult == 1));
                }
                if (clamp)
                {
                    acc0 = vminnmq(vmaxnmq(acc0, ARM_NN_DW_SMALL_MULT_DUP(act_min)), ARM_NN_DW_SMALL_MULT_DUP(act_max));
                }
                vst1q_p(output_b + (size_t)ow * out_c + oc, acc0, p);
            }
        }
    }
}
