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
 * Title:        arm_nn_conv1d_packed_f16_template.h
 * Description:  Float16 adapter for packed 1D convolution template
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

/*
 * F16 adapter for the shared packed 1D convolution template.
 * Intentionally no include guard: each inclusion emits one named function.
 * Include after arm_nnsupportfunctions.h, inside ARM_NN_ENABLE_F16, with:
 *   ARM_NN_CONV1D_PACKED_NAME      public support-kernel function name
 *   ARM_NN_CONV1D_PACKED_KERNEL_W  kernel width (7 or 9)
 * The caller undefines these two parameters; this adapter cleans up its own.
 */
#define ARM_NN_CONV1D_PACKED_SCALAR_T float16_t
/* Preserve half-precision scalar accumulation for the k7/k9 kernels. */
#define ARM_NN_CONV1D_PACKED_ACCUMULATOR_T _Float16
#define ARM_NN_CONV1D_PACKED_LANES 8

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
    #define ARM_NN_CONV1D_PACKED_USE_MVE 1
    #define ARM_NN_CONV1D_PACKED_VECTOR_T float16x8_t
    #define ARM_NN_CONV1D_PACKED_ZERO() vdupq_n_f16((float16_t)0.0f)
    #define ARM_NN_CONV1D_PACKED_PREDICATE vctp16q
#else
    #define ARM_NN_CONV1D_PACKED_USE_MVE 0
#endif

#include "arm_nn_conv1d_packed_template.h"

#undef ARM_NN_CONV1D_PACKED_PREDICATE
#undef ARM_NN_CONV1D_PACKED_ZERO
#undef ARM_NN_CONV1D_PACKED_VECTOR_T
#undef ARM_NN_CONV1D_PACKED_USE_MVE
#undef ARM_NN_CONV1D_PACKED_LANES
#undef ARM_NN_CONV1D_PACKED_ACCUMULATOR_T
#undef ARM_NN_CONV1D_PACKED_SCALAR_T
