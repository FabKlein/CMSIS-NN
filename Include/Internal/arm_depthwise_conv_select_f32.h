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
 * Title:        arm_depthwise_conv_select_f32.h
 * Description:  F32 depthwise specialization selection interface
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_DEPTHWISE_CONV_SELECT_F32_H
#define ARM_DEPTHWISE_CONV_SELECT_F32_H

#include "arm_nn_types.h"
#include <stdbool.h>
#include <stddef.h>

#if ARM_NN_ENABLE_F32 && !defined(NN_DISABLE_SPECIALIZATION)

/*
 * Match the depthwise registry after arm_depthwise_conv_geometry_is_valid succeeds.
 * Parameters and dimension pointers must be non-NULL. Execution validates once at
 * its public entry point; scratch-size queries perform the same checks before selection.
 *
 * Pass index == NULL to query availability. Otherwise a successful selection writes
 * the registry index; failure clears it. The execution table uses the same order.
 * This function references no kernel code. All registered kernels use zero scratch.
 */
bool arm_depthwise_conv_select_specialized_f32(const cmsis_nn_dw_conv_params_f32 *params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_dw_kernel_layout_f32 kernel_layout,
                                               size_t *index);

#endif
#endif /* ARM_DEPTHWISE_CONV_SELECT_F32_H */
