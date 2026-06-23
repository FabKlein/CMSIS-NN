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
 * Title:        arm_nn_depthwise_conv_small_mult_f16.h
 * Description:  F16 adapter for MVE depthwise convolution with small multipliers
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_DEPTHWISE_CONV_SMALL_MULT_F16_H
#define ARM_NN_DEPTHWISE_CONV_SMALL_MULT_F16_H

#include "arm_nnsupportfunctions.h"
#include <stdbool.h>

#if ARM_NN_ENABLE_F16 && defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)

/* Multipliers 2/4 duplicate input channels through fixed gather offsets.
 * Multiplier 1 uses contiguous loads. The output predicate also masks input tails.
 */
__STATIC_FORCEINLINE float16x8_t arm_nn_depthwise_small_mult_load_f16(const float16_t *input,
                                                                      uint16x8_t offsets,
                                                                      mve_pred16_t p,
                                                                      bool contiguous)
{
    return contiguous ? vld1q_z(input, p) : vldrhq_gather_shifted_offset_z_f16(input, offsets, p);
}

    #define ARM_NN_DW_SMALL_MULT_NAME arm_nn_depthwise_conv_small_mult_f16
    #define ARM_NN_DW_SMALL_MULT_SCALAR_T float16_t
    #define ARM_NN_DW_SMALL_MULT_VECTOR_T float16x8_t
    #define ARM_NN_DW_SMALL_MULT_OFFSETS_T uint16x8_t
    #define ARM_NN_DW_SMALL_MULT_LANES 8
    #define ARM_NN_DW_SMALL_MULT_INDICES() vidupq_n_u16(0, 1)
    #define ARM_NN_DW_SMALL_MULT_SHIFT(offsets, shift) vshrq_n_u16(offsets, shift)
    #define ARM_NN_DW_SMALL_MULT_PREDICATE vctp16q
    #define ARM_NN_DW_SMALL_MULT_DUP(value) vdupq_n_f16((float16_t)(value))
    #define ARM_NN_DW_SMALL_MULT_LOAD arm_nn_depthwise_small_mult_load_f16
    #include "arm_nn_depthwise_conv_small_mult_template.h"
    #undef ARM_NN_DW_SMALL_MULT_LOAD
    #undef ARM_NN_DW_SMALL_MULT_DUP
    #undef ARM_NN_DW_SMALL_MULT_PREDICATE
    #undef ARM_NN_DW_SMALL_MULT_SHIFT
    #undef ARM_NN_DW_SMALL_MULT_INDICES
    #undef ARM_NN_DW_SMALL_MULT_LANES
    #undef ARM_NN_DW_SMALL_MULT_OFFSETS_T
    #undef ARM_NN_DW_SMALL_MULT_VECTOR_T
    #undef ARM_NN_DW_SMALL_MULT_SCALAR_T
    #undef ARM_NN_DW_SMALL_MULT_NAME

#endif
#endif /* ARM_NN_DEPTHWISE_CONV_SMALL_MULT_F16_H */
