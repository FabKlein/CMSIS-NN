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
 * Title:        arm_conv_specialized_registry_f16.h
 * Description:  F16 specialized convolution registry
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

/*
 * Registry of specialized NHWC convolution kernels requiring no scratch buffer.
 * Each entry declares its exact stride, padding, and dilation using named fields
 * of arm_conv_specialized_requirements. These values must describe the implementation;
 * the fixed-purpose kernel signatures do not receive them as runtime arguments.
 * NULL means that weight format has no implementation for these requirements.
 *
 * The 1D call adapter handles vertical stride/padding for height-1 filters. Its
 * row kernel implements the horizontal requirements. A 2D kernel handles both axes.
 *
 * Intentionally no include guard: consumers define ARM_CONV_1D_SPEC and
 * ARM_CONV_2D_SPEC to emit availability metadata or typed function tables.
 * Both consumers preserve entry order, so selection resolves by index.
 */
#if !defined(ARM_CONV_1D_SPEC) || !defined(ARM_CONV_2D_SPEC)
    #error "Define both registry entry macros before including this file"
#endif

/* Width, packed kernel, standard-weight kernel, exact geometry requirements. */
ARM_CONV_1D_SPEC(9,
                 arm_nn_conv1d_k9_packed_f16,
                 NULL,
                 .stride = {.h = 1, .w = 1},
                 .padding = {.h = 0, .w = 0},
                 .dilation = {.h = 1, .w = 1})
ARM_CONV_1D_SPEC(7,
                 arm_nn_conv1d_k7_packed_f16,
                 NULL,
                 .stride = {.h = 1, .w = 1},
                 .padding = {.h = 0, .w = 0},
                 .dilation = {.h = 1, .w = 1})
ARM_CONV_1D_SPEC(5,
                 arm_nn_conv1d_k5_packed_f16,
                 arm_nn_conv1d_k5_nhwc_f16,
                 .stride = {.h = 1, .w = 1},
                 .padding = {.h = 0, .w = 0},
                 .dilation = {.h = 1, .w = 1})
ARM_CONV_1D_SPEC(3,
                 arm_nn_conv1d_k3_packed_f16,
                 arm_nn_conv1d_k3_nhwc_f16,
                 .stride = {.h = 1, .w = 1},
                 .padding = {.h = 0, .w = 0},
                 .dilation = {.h = 1, .w = 1})
ARM_CONV_1D_SPEC(2,
                 arm_nn_conv1d_k2_packed_f16,
                 NULL,
                 .stride = {.h = 1, .w = 1},
                 .padding = {.h = 0, .w = 0},
                 .dilation = {.h = 1, .w = 1})

/* Height, width, packed kernel, exact geometry requirements. */
ARM_CONV_2D_SPEC(2,
                 2,
                 arm_nn_conv2d_2x2_packed_f16,
                 .stride = {.h = 1, .w = 1},
                 .padding = {.h = 0, .w = 0},
                 .dilation = {.h = 1, .w = 1})
ARM_CONV_2D_SPEC(2,
                 3,
                 arm_nn_conv2d_2x3_packed_f16,
                 .stride = {.h = 1, .w = 1},
                 .padding = {.h = 0, .w = 0},
                 .dilation = {.h = 1, .w = 1})
ARM_CONV_2D_SPEC(2,
                 5,
                 arm_nn_conv2d_2x5_packed_f16,
                 .stride = {.h = 1, .w = 1},
                 .padding = {.h = 0, .w = 0},
                 .dilation = {.h = 1, .w = 1})
