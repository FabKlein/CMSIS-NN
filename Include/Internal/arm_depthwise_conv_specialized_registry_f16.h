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
 * Title:        arm_depthwise_conv_specialized_registry_f16.h
 * Description:  F16 specialized depthwise convolution registry
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

/*
 * Registry of specialized NHWC depthwise kernels using KC weights and no scratch buffer.
 * Append ARM_DW_SPEC(height, width, call_adapter, ...requirements) to register a kernel.
 * The named requirements are fields of arm_dw_requirements and must describe the
 * implementation. Entry order determines selection priority.
 *
 * All call adapters share one signature, so 1D and 2D kernels use the same entry macro.
 * Adapters are defined in the corresponding arm_depthwise_conv_specialized_f16/f32.h.
 *
 * Intentionally no include guard: consumers define ARM_DW_SPEC to generate
 * selection metadata or the function-pointer table, then undefine it. Both
 * consumers preserve entry order so the selected index resolves to the right adapter.
 */
#if !defined(ARM_DW_SPEC)
    #error "Define ARM_DW_SPEC before including this file"
#endif

/* k3/3x3 require multiplier one; k9/2x5 support any valid multiplier. */
ARM_DW_SPEC(1,
            9,
            arm_dw_spec_k9_1d_nhwc_f16_call,
            .multiplier = ARM_DW_MULTIPLIER_ANY,
            .stride_policy = ARM_DW_STRIDE_EXACT,
            .stride = {.h = 1, .w = 1},
            .padding_policy = ARM_DW_PADDING_EXACT,
            .padding = {.h = 0, .w = 0},
            .dilation = {.h = 1, .w = 1},
            .require_input_height_equals_filter = false)

ARM_DW_SPEC(1,
            3,
            arm_dw_spec_k3_1d_nhwc_f16_call,
            .multiplier = ARM_DW_MULTIPLIER_ONE,
            .stride_policy = ARM_DW_STRIDE_EXACT,
            .stride = {.h = 1, .w = 1},
            .padding_policy = ARM_DW_PADDING_EXACT,
            .padding = {.h = 0, .w = 0},
            .dilation = {.h = 1, .w = 1},
            .require_input_height_equals_filter = false)

ARM_DW_SPEC(2,
            5,
            arm_dw_spec_2x5_nhwc_f16_call,
            .multiplier = ARM_DW_MULTIPLIER_ANY,
            .stride_policy = ARM_DW_STRIDE_EXACT,
            .stride = {.h = 1, .w = 1},
            .padding_policy = ARM_DW_PADDING_EXACT,
            .padding = {.h = 0, .w = 0},
            .dilation = {.h = 1, .w = 1},
            .require_input_height_equals_filter = true)

ARM_DW_SPEC(3,
            3,
            arm_dw_spec_3x3_nhwc_f16_call,
            .multiplier = ARM_DW_MULTIPLIER_ONE,
            .stride_policy = ARM_DW_STRIDE_ANY,
            .padding_policy = ARM_DW_PADDING_ANY,
            .dilation = {.h = 1, .w = 1},
            .require_input_height_equals_filter = false)
