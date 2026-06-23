/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_convolve_f32.c"
#include "unity.h"

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

void setUp(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
}

void tearDown(void) {}
void test_conv_basic_f32_arm_convolve_f32(void) { conv_basic_f32_arm_convolve_f32(); }
void test_conv_basic_nhwc_f32_arm_convolve_f32(void) { conv_basic_nhwc_f32_arm_convolve_f32(); }
void test_conv_1x1_stride2_nhwc_f32_arm_convolve_f32(void) { conv_1x1_stride2_nhwc_f32_arm_convolve_f32(); }
void test_conv_k3_opt_f32_arm_convolve_f32(void) { conv_k3_opt_f32_arm_convolve_f32(); }
void test_conv_k5_opt_f32_arm_convolve_f32(void) { conv_k5_opt_f32_arm_convolve_f32(); }
void test_conv_k3_opt_nhwc_tuned_f32_arm_convolve_f32(void) { conv_k3_opt_nhwc_tuned_f32_arm_convolve_f32(); }
void test_conv_k5_opt_nhwc_tuned_f32_arm_convolve_f32(void) { conv_k5_opt_nhwc_tuned_f32_arm_convolve_f32(); }
void test_conv_kernel_2x2_f32_arm_convolve_f32(void) { conv_kernel_2x2_f32_arm_convolve_f32(); }
void test_conv_kernel_3x3_pad1_f32_arm_convolve_f32(void) { conv_kernel_3x3_pad1_f32_arm_convolve_f32(); }
void test_conv_match_basic_f32_arm_convolve_f32(void) { conv_match_basic_f32_arm_convolve_f32(); }
void test_conv_match_stride2pad1_f32_arm_convolve_f32(void) { conv_match_stride2pad1_f32_arm_convolve_f32(); }
void test_conv_match_conv_2_f32_arm_convolve_f32(void) { conv_match_conv_2_f32_arm_convolve_f32(); }
void test_conv_match_conv_3_f32_arm_convolve_f32(void) { conv_match_conv_3_f32_arm_convolve_f32(); }
void test_conv_match_conv_4_f32_arm_convolve_f32(void) { conv_match_conv_4_f32_arm_convolve_f32(); }
void test_conv_match_conv_5_f32_arm_convolve_f32(void)
{
#if defined(USING_FVP_CORSTONE_300)
    TEST_IGNORE_MESSAGE("Host-only due to FVP image size");
#else
    conv_match_conv_5_f32_arm_convolve_f32();
#endif
}
void test_conv_match_out_activation_f32_arm_convolve_f32(void) { conv_match_out_activation_f32_arm_convolve_f32(); }
void test_conv_match_dilation_golden_f32_arm_convolve_f32(void) { conv_match_dilation_golden_f32_arm_convolve_f32(); }
void test_conv_match_2x2_dilation_f32_arm_convolve_f32(void) { conv_match_2x2_dilation_f32_arm_convolve_f32(); }
void test_conv_match_2x3_dilation_f32_arm_convolve_f32(void) { conv_match_2x3_dilation_f32_arm_convolve_f32(); }
void test_conv_match_1x1_basic_f32_arm_convolve_f32(void) { conv_match_1x1_basic_f32_arm_convolve_f32(); }
void test_conv_match_1x1_stride_x_f32_arm_convolve_f32(void) { conv_match_1x1_stride_x_f32_arm_convolve_f32(); }
void test_conv_match_1x1_stride_x_y_f32_arm_convolve_f32(void) { conv_match_1x1_stride_x_y_f32_arm_convolve_f32(); }
void test_conv_match_1x1_stride_x_y_1_f32_arm_convolve_f32(void) { conv_match_1x1_stride_x_y_1_f32_arm_convolve_f32(); }
void test_conv_match_1x1_stride_x_y_2_f32_arm_convolve_f32(void) { conv_match_1x1_stride_x_y_2_f32_arm_convolve_f32(); }
void test_conv_match_1xn_1_f32_arm_convolve_f32(void) { conv_match_1xn_1_f32_arm_convolve_f32(); }
void test_conv_match_1xn_2_f32_arm_convolve_f32(void) { conv_match_1xn_2_f32_arm_convolve_f32(); }
void test_conv_match_1xn_3_f32_arm_convolve_f32(void) { conv_match_1xn_3_f32_arm_convolve_f32(); }
void test_conv_match_1xn_4_f32_arm_convolve_f32(void) { conv_match_1xn_4_f32_arm_convolve_f32(); }
void test_conv_match_1xn_5_f32_arm_convolve_f32(void) { conv_match_1xn_5_f32_arm_convolve_f32(); }
void test_conv_match_1xn_6_generic_f32_arm_convolve_f32(void) { conv_match_1xn_6_generic_f32_arm_convolve_f32(); }
void test_conv_match_1xn_7_f32_arm_convolve_f32(void) { conv_match_1xn_7_f32_arm_convolve_f32(); }
void test_conv_match_1xn_8_f32_arm_convolve_f32(void) { conv_match_1xn_8_f32_arm_convolve_f32(); }
void test_conv_match_3x2_dilation_f32_arm_convolve_f32(void) { conv_match_3x2_dilation_f32_arm_convolve_f32(); }
void test_conv_match_3x3_dilation_5x5_input_f32_arm_convolve_f32(void)
{
    conv_match_3x3_dilation_5x5_input_f32_arm_convolve_f32();
}
void test_conv_match_2x2_dilation_5x5_input_f32_arm_convolve_f32(void)
{
    conv_match_2x2_dilation_5x5_input_f32_arm_convolve_f32();
}
void test_conv_match_1x1_basic_f32_arm_convolve_f32_packed(void) { conv_match_1x1_basic_f32_arm_convolve_f32_packed(); }
void test_conv_basic_f32_arm_convolve_f32_packed(void) { conv_basic_f32_arm_convolve_f32_packed(); }
void test_conv_k3_opt_f32_arm_convolve_f32_packed(void) { conv_k3_opt_f32_arm_convolve_f32_packed(); }
void test_conv_k5_opt_f32_arm_convolve_f32_packed(void) { conv_k5_opt_f32_arm_convolve_f32_packed(); }
void test_conv_k5_output_width_boundaries_f32(void) { conv_k5_output_width_boundaries_f32(); }
void test_conv_k5_batch2_rows2_tails_null_bias_f32_arm_convolve_f32_packed(void)
{
    conv_k5_batch2_rows2_tails_null_bias_f32_arm_convolve_f32_packed();
}
void test_conv_k7_opt_f32_arm_convolve_f32_packed(void) { conv_k7_opt_f32_arm_convolve_f32_packed(); }
void test_conv_k9_exact_six_f32_arm_convolve_f32_packed(void) { conv_k9_exact_six_f32_arm_convolve_f32_packed(); }
void test_conv_k9_opt_f32_arm_convolve_f32_packed(void) { conv_k9_opt_f32_arm_convolve_f32_packed(); }
void test_conv_k2_packed_valid_f32_arm_convolve_f32_packed(void) { conv_k2_packed_valid_f32_arm_convolve_f32_packed(); }
void test_conv_kernel_2x2_f32_arm_convolve_f32_packed(void) { conv_kernel_2x2_f32_arm_convolve_f32_packed(); }
void test_conv_2x3_packed_valid_f32_arm_convolve_f32_packed(void)
{
    conv_2x3_packed_valid_f32_arm_convolve_f32_packed();
}
void test_conv_2x5_packed_valid_f32_arm_convolve_f32_packed(void)
{
    conv_2x5_packed_valid_f32_arm_convolve_f32_packed();
}

void test_conv_2x2_packed_tiles_f32(void) { conv_2x2_packed_tiles_f32(); }
void test_conv_2x3_packed_tiles_f32(void) { conv_2x3_packed_tiles_f32(); }
void test_conv_2x5_packed_tiles_f32(void) { conv_2x5_packed_tiles_f32(); }

void test_conv_k7_packed_cancellation_f32(void) { conv_k7_packed_cancellation_f32(); }
void test_conv_k9_packed_cancellation_f32(void) { conv_k9_packed_cancellation_f32(); }

void test_conv_specialized_registry_f32(void) { conv_specialized_registry_f32(); }

void test_conv_specialized_geometry_f32(void) { conv_specialized_geometry_f32(); }

void test_conv_specialized_stride2_adapter_f32(void) { conv_specialized_stride2_adapter_f32(); }
