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
 * Title:        arm_conv_select_f16.h
 * Description:  Internal F16 specialized convolution selection
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONV_SELECT_F16_H
#define ARM_CONV_SELECT_F16_H

#include "arm_nn_types.h"
#include <stdbool.h>
#include <stddef.h>

#if ARM_NN_ENABLE_F16 && !defined(NN_DISABLE_SPECIALIZATION)

typedef void (*arm_conv_1d_kernel_f16)(const float16_t *,
                                       int32_t,
                                       int32_t,
                                       const float16_t *,
                                       const float16_t *,
                                       float16_t *,
                                       int32_t,
                                       int32_t);

typedef void (*arm_conv_2d_kernel_f16)(const float16_t *,
                                       int32_t,
                                       int32_t,
                                       int32_t,
                                       const float16_t *,
                                       const float16_t *,
                                       float16_t *,
                                       int32_t,
                                       int32_t,
                                       int32_t);

/* Internal selection token. Only resolve tokens returned by a successful
 * selection, using the resolver for the same precision. Failure clears the token.
 */
typedef struct
{
    size_t index;
    bool is_1d;
    bool packed;
} arm_conv_selection_f16;

/* Exactly one function is set when resolving a successful selection. */
typedef struct
{
    arm_conv_1d_kernel_f16 kernel_1d;
    arm_conv_2d_kernel_f16 kernel_2d;
} arm_conv_kernels_f16;

/*
 * Check shared specialized NHWC geometry, then match a registered filter shape, exact
 * stride/padding/dilation, and weight format. Convolution dispatch and scratch sizing use this same decision so they
 * agree on whether a zero-scratch specialization is available. A false result lets
 * the caller consider other convolution paths; it is not a public API error status.
 *
 * Pass selection == NULL to query availability without requesting a token. This
 * function does not reference kernel code. All registered kernels use zero scratch.
 */
bool arm_conv_select_specialized_f16(const cmsis_nn_conv_params_f16 *params,
                                     const cmsis_nn_dims *input_dims,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     arm_conv_selection_f16 *selection);

/* Resolve a selected index without another table search. Kept in a separate
 * object so buffer-size-only callers do not link specialized kernels.
 */
arm_conv_kernels_f16 arm_conv_resolve_specialized_f16(const arm_conv_selection_f16 *selection);

#endif

#endif /* ARM_CONV_SELECT_F16_H */
