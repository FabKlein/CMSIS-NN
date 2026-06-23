/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include <arm_nnsupportfunctions.h>
#include <unity.h>

#include "Internal/arm_conv_specialized_common.h"
#include "Internal/arm_conv_specialized_f32.h"
#include "../Common/float_packed_test_utils.h"
#include "../TestData/conv_1x1_stride2_nhwc_f32/test_data.h"
#include "../TestData/conv_2x3_packed_valid_f32/test_data.h"
#include "../TestData/conv_2x5_packed_valid_f32/test_data.h"
#include "../TestData/conv_basic_f32/test_data.h"
#include "../TestData/conv_basic_nhwc_f32/test_data.h"
#include "../TestData/conv_k2_packed_valid_f32/test_data.h"
#include "../TestData/conv_k3_opt_f32/test_data.h"
#include "../TestData/conv_k3_opt_nhwc_tuned_f32/test_data.h"
#include "../TestData/conv_k5_batch2_rows2_tails_null_bias_f32/test_data.h"
#include "../TestData/conv_k5_opt_f32/test_data.h"
#include "../TestData/conv_k5_opt_nhwc_tuned_f32/test_data.h"
#include "../TestData/conv_k7_opt_f32/test_data.h"
#include "../TestData/conv_k9_exact_six_f32/test_data.h"
#include "../TestData/conv_k9_opt_f32/test_data.h"
#include "../TestData/conv_kernel_2x2_f32/test_data.h"
#include "../TestData/conv_kernel_3x3_pad1_f32/test_data.h"
#include "../TestData/conv_match_1x1_basic_f32/test_data.h"
#include "../TestData/conv_match_1x1_stride_x_f32/test_data.h"
#include "../TestData/conv_match_1x1_stride_x_y_1_f32/test_data.h"
#include "../TestData/conv_match_1x1_stride_x_y_2_f32/test_data.h"
#include "../TestData/conv_match_1x1_stride_x_y_f32/test_data.h"
#include "../TestData/conv_match_1xn_1_f32/test_data.h"
#include "../TestData/conv_match_1xn_2_f32/test_data.h"
#include "../TestData/conv_match_1xn_3_f32/test_data.h"
#include "../TestData/conv_match_1xn_4_f32/test_data.h"
#include "../TestData/conv_match_1xn_5_f32/test_data.h"
#include "../TestData/conv_match_1xn_6_generic_f32/test_data.h"
#include "../TestData/conv_match_1xn_7_f32/test_data.h"
#include "../TestData/conv_match_1xn_8_f32/test_data.h"
#include "../TestData/conv_match_2x2_dilation_5x5_input_f32/test_data.h"
#include "../TestData/conv_match_2x2_dilation_f32/test_data.h"
#include "../TestData/conv_match_2x3_dilation_f32/test_data.h"
#include "../TestData/conv_match_3x2_dilation_f32/test_data.h"
#include "../TestData/conv_match_3x3_dilation_5x5_input_f32/test_data.h"
#include "../TestData/conv_match_basic_f32/test_data.h"
#include "../TestData/conv_match_conv_2_f32/test_data.h"
#include "../TestData/conv_match_conv_3_f32/test_data.h"
#include "../TestData/conv_match_conv_4_f32/test_data.h"
#if !defined(USING_FVP_CORSTONE_300)
    #include "../TestData/conv_match_conv_5_f32/test_data.h"
#endif
#include "../TestData/conv_match_dilation_golden_f32/test_data.h"
#include "../TestData/conv_match_out_activation_f32/test_data.h"
#include "../TestData/conv_match_stride2pad1_f32/test_data.h"

#define RUN_CONV_F32_CASE(CASE_PREFIX, case_name, tolerance)                                                           \
    void case_name##_arm_convolve_f32(void)                                                                            \
    {                                                                                                                  \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        cmsis_nn_context ctx = {0};                                                                                    \
        const cmsis_nn_conv_params_f32 conv_params = {                                                                 \
            .padding = {.w = CASE_PREFIX##_PADDING_W, .h = CASE_PREFIX##_PADDING_H},                                   \
            .stride = {.w = CASE_PREFIX##_STRIDE_W, .h = CASE_PREFIX##_STRIDE_H},                                      \
            .dilation = {.w = CASE_PREFIX##_DILATION_W, .h = CASE_PREFIX##_DILATION_H},                                \
            .activation = {.min = CASE_PREFIX##_OUT_ACTIVATION_MIN, .max = CASE_PREFIX##_OUT_ACTIVATION_MAX}};         \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .c = CASE_PREFIX##_IN_CH};                                                   \
        const cmsis_nn_dims filter_dims = {.n = CASE_PREFIX##_OUT_CH,                                                  \
                                           .w = CASE_PREFIX##_FILTER_W,                                                \
                                           .h = CASE_PREFIX##_FILTER_H,                                                \
                                           .c = CASE_PREFIX##_IN_CH};                                                  \
        const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CASE_PREFIX##_OUT_CH};                           \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                           \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
        const int32_t buf_size = arm_convolve_f32_get_buffer_size(                                                     \
            &conv_params, &input_dims, &filter_dims, &output_dims, CASE_PREFIX##_LAYOUT);                              \
                                                                                                                       \
        if (buf_size > 0)                                                                                              \
        {                                                                                                              \
            ctx.buf = malloc((size_t)buf_size);                                                                        \
            ctx.size = buf_size;                                                                                       \
            TEST_ASSERT_NOT_NULL(ctx.buf);                                                                             \
        }                                                                                                              \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_convolve_f32(&ctx,                                                                       \
                                           &conv_params,                                                               \
                                           &input_dims,                                                                \
                                           case_name##_input_data,                                                     \
                                           &filter_dims,                                                               \
                                           case_name##_weights_data,                                                   \
                                           &bias_dims,                                                                 \
                                           case_name##_biases_data,                                                    \
                                           &output_dims,                                                               \
                                           output,                                                                     \
                                           CASE_PREFIX##_LAYOUT));                                                     \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref_data[i], output[i]);                          \
            output[i] = 0.0f;                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        if (ctx.buf != NULL)                                                                                           \
        {                                                                                                              \
            memset(ctx.buf, 0, (size_t)buf_size);                                                                      \
            free(ctx.buf);                                                                                             \
            ctx.buf = NULL;                                                                                            \
            ctx.size = 0;                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        if (CASE_PREFIX##_USE_WRAPPER)                                                                                 \
        {                                                                                                              \
            const int32_t wrapper_buf_size =                                                                           \
                arm_convolve_wrapper_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);       \
            if (wrapper_buf_size > 0)                                                                                  \
            {                                                                                                          \
                ctx.buf = malloc((size_t)wrapper_buf_size);                                                            \
                ctx.size = wrapper_buf_size;                                                                           \
                TEST_ASSERT_NOT_NULL(ctx.buf);                                                                         \
            }                                                                                                          \
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                    \
                              arm_convolve_wrapper_f32(&ctx,                                                           \
                                                       &conv_params,                                                   \
                                                       &input_dims,                                                    \
                                                       case_name##_input_data,                                         \
                                                       &filter_dims,                                                   \
                                                       case_name##_weights_data,                                       \
                                                       &bias_dims,                                                     \
                                                       case_name##_biases_data,                                        \
                                                       &output_dims,                                                   \
                                                       output));                                                       \
            for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                           \
            {                                                                                                          \
                TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref_data[i], output[i]);                      \
            }                                                                                                          \
            if (ctx.buf != NULL)                                                                                       \
            {                                                                                                          \
                memset(ctx.buf, 0, (size_t)wrapper_buf_size);                                                          \
                free(ctx.buf);                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
    }

RUN_CONV_F32_CASE(CONV_BASIC_F32, conv_basic_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_BASIC_NHWC_F32, conv_basic_nhwc_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_1X1_STRIDE2_NHWC_F32, conv_1x1_stride2_nhwc_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_KERNEL_2X2_F32, conv_kernel_2x2_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_KERNEL_3X3_PAD1_F32, conv_kernel_3x3_pad1_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_K3_OPT_F32, conv_k3_opt_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_K5_OPT_F32, conv_k5_opt_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_K3_OPT_NHWC_TUNED_F32, conv_k3_opt_nhwc_tuned_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_K5_OPT_NHWC_TUNED_F32, conv_k5_opt_nhwc_tuned_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_BASIC_F32, conv_match_basic_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_STRIDE2PAD1_F32, conv_match_stride2pad1_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_CONV_2_F32, conv_match_conv_2_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_CONV_3_F32, conv_match_conv_3_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_CONV_4_F32, conv_match_conv_4_f32, 5.0e-4f)
#if !defined(USING_FVP_CORSTONE_300)
RUN_CONV_F32_CASE(CONV_MATCH_CONV_5_F32, conv_match_conv_5_f32, 5.0e-4f)
#endif
RUN_CONV_F32_CASE(CONV_MATCH_OUT_ACTIVATION_F32, conv_match_out_activation_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_DILATION_GOLDEN_F32, conv_match_dilation_golden_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_2X2_DILATION_F32, conv_match_2x2_dilation_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_2X3_DILATION_F32, conv_match_2x3_dilation_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_BASIC_F32, conv_match_1x1_basic_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_STRIDE_X_F32, conv_match_1x1_stride_x_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_STRIDE_X_Y_F32, conv_match_1x1_stride_x_y_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_STRIDE_X_Y_1_F32, conv_match_1x1_stride_x_y_1_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1X1_STRIDE_X_Y_2_F32, conv_match_1x1_stride_x_y_2_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_1_F32, conv_match_1xn_1_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_2_F32, conv_match_1xn_2_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_3_F32, conv_match_1xn_3_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_4_F32, conv_match_1xn_4_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_5_F32, conv_match_1xn_5_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_6_GENERIC_F32, conv_match_1xn_6_generic_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_7_F32, conv_match_1xn_7_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_1XN_8_F32, conv_match_1xn_8_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_3X2_DILATION_F32, conv_match_3x2_dilation_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_3X3_DILATION_5X5_INPUT_F32, conv_match_3x3_dilation_5x5_input_f32, 5.0e-4f)
RUN_CONV_F32_CASE(CONV_MATCH_2X2_DILATION_5X5_INPUT_F32, conv_match_2x2_dilation_5x5_input_f32, 5.0e-4f)

/*
 * The packed-convolution coverage allocates a temporary repacked filter and,
 * for generic conv, may also allocate scratch, so it needs explicit test
 * bodies instead of the standard RUN_CONV_F32_CASE macro.
 */
void conv_match_1x1_basic_f32_arm_convolve_f32_packed(void)
{
    float32_t output[CONV_MATCH_1X1_BASIC_F32_DST_SIZE] = {0};
    cmsis_nn_context ctx = {0};
    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = CONV_MATCH_1X1_BASIC_F32_PADDING_W, .h = CONV_MATCH_1X1_BASIC_F32_PADDING_H},
        .stride = {.w = CONV_MATCH_1X1_BASIC_F32_STRIDE_W, .h = CONV_MATCH_1X1_BASIC_F32_STRIDE_H},
        .dilation = {.w = CONV_MATCH_1X1_BASIC_F32_DILATION_W, .h = CONV_MATCH_1X1_BASIC_F32_DILATION_H},
        .activation = {.min = CONV_MATCH_1X1_BASIC_F32_OUT_ACTIVATION_MIN,
                       .max = CONV_MATCH_1X1_BASIC_F32_OUT_ACTIVATION_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = CONV_MATCH_1X1_BASIC_F32_INPUT_BATCHES,
                                      .w = CONV_MATCH_1X1_BASIC_F32_INPUT_W,
                                      .h = CONV_MATCH_1X1_BASIC_F32_INPUT_H,
                                      .c = CONV_MATCH_1X1_BASIC_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = CONV_MATCH_1X1_BASIC_F32_OUT_CH,
                                       .w = CONV_MATCH_1X1_BASIC_F32_FILTER_W,
                                       .h = CONV_MATCH_1X1_BASIC_F32_FILTER_H,
                                       .c = CONV_MATCH_1X1_BASIC_F32_IN_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CONV_MATCH_1X1_BASIC_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = CONV_MATCH_1X1_BASIC_F32_INPUT_BATCHES,
                                       .w = CONV_MATCH_1X1_BASIC_F32_OUTPUT_W,
                                       .h = CONV_MATCH_1X1_BASIC_F32_OUTPUT_H,
                                       .c = CONV_MATCH_1X1_BASIC_F32_OUTPUT_C};
    const int32_t buf_size = arm_convolve_f32_get_buffer_size(
        &conv_params, &input_dims, &filter_dims, &output_dims, CONV_MATCH_1X1_BASIC_F32_LAYOUT);
    float32_t *packed_weights = pack_rhs_nt_n_from_nt_t_f32(
        conv_match_1x1_basic_f32_weights_data,
        CONV_MATCH_1X1_BASIC_F32_OUT_CH,
        CONV_MATCH_1X1_BASIC_F32_FILTER_W * CONV_MATCH_1X1_BASIC_F32_FILTER_H * CONV_MATCH_1X1_BASIC_F32_IN_CH);

    if (buf_size > 0)
    {
        ctx.buf = malloc((size_t)buf_size);
        ctx.size = buf_size;
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f32(&ctx,
                                       &conv_params,
                                       &input_dims,
                                       conv_match_1x1_basic_f32_input_data,
                                       &filter_dims,
                                       packed_weights,
                                       &bias_dims,
                                       conv_match_1x1_basic_f32_biases_data,
                                       &output_dims,
                                       output,
                                       CONV_MATCH_1X1_BASIC_F32_LAYOUT));

    for (int i = 0; i < CONV_MATCH_1X1_BASIC_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, conv_match_1x1_basic_f32_output_ref_data[i], output[i]);
    }

    free(packed_weights);
    if (ctx.buf != NULL)
    {
        memset(ctx.buf, 0, (size_t)buf_size);
        free(ctx.buf);
    }
}

void conv_basic_f32_arm_convolve_f32_packed(void)
{
    float32_t output[CONV_BASIC_F32_DST_SIZE] = {0};
    cmsis_nn_context ctx = {0};
    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = CONV_BASIC_F32_PADDING_W, .h = CONV_BASIC_F32_PADDING_H},
        .stride = {.w = CONV_BASIC_F32_STRIDE_W, .h = CONV_BASIC_F32_STRIDE_H},
        .dilation = {.w = CONV_BASIC_F32_DILATION_W, .h = CONV_BASIC_F32_DILATION_H},
        .activation = {.min = CONV_BASIC_F32_OUT_ACTIVATION_MIN, .max = CONV_BASIC_F32_OUT_ACTIVATION_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = CONV_BASIC_F32_INPUT_BATCHES,
                                      .w = CONV_BASIC_F32_INPUT_W,
                                      .h = CONV_BASIC_F32_INPUT_H,
                                      .c = CONV_BASIC_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = CONV_BASIC_F32_OUT_CH,
                                       .w = CONV_BASIC_F32_FILTER_W,
                                       .h = CONV_BASIC_F32_FILTER_H,
                                       .c = CONV_BASIC_F32_IN_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CONV_BASIC_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = CONV_BASIC_F32_INPUT_BATCHES,
                                       .w = CONV_BASIC_F32_OUTPUT_W,
                                       .h = CONV_BASIC_F32_OUTPUT_H,
                                       .c = CONV_BASIC_F32_OUTPUT_C};
    const int32_t buf_size =
        arm_convolve_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims, CONV_BASIC_F32_LAYOUT);
    float32_t *packed_weights =
        pack_rhs_nt_n_from_nt_t_f32(conv_basic_f32_weights_data,
                                    CONV_BASIC_F32_OUT_CH,
                                    CONV_BASIC_F32_FILTER_W * CONV_BASIC_F32_FILTER_H * CONV_BASIC_F32_IN_CH);

    if (buf_size > 0)
    {
        ctx.buf = malloc((size_t)buf_size);
        ctx.size = buf_size;
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f32(&ctx,
                                       &conv_params,
                                       &input_dims,
                                       conv_basic_f32_input_data,
                                       &filter_dims,
                                       packed_weights,
                                       &bias_dims,
                                       conv_basic_f32_biases_data,
                                       &output_dims,
                                       output,
                                       CONV_BASIC_F32_LAYOUT));

    for (int i = 0; i < CONV_BASIC_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, conv_basic_f32_output_ref_data[i], output[i]);
    }

    free(packed_weights);
    if (ctx.buf != NULL)
    {
        memset(ctx.buf, 0, (size_t)buf_size);
        free(ctx.buf);
    }
}

void conv_k3_opt_f32_arm_convolve_f32_packed(void)
{
    float32_t output[CONV_K3_OPT_F32_DST_SIZE] = {0};
    cmsis_nn_context ctx = {0};
    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = CONV_K3_OPT_F32_PADDING_W, .h = CONV_K3_OPT_F32_PADDING_H},
        .stride = {.w = CONV_K3_OPT_F32_STRIDE_W, .h = CONV_K3_OPT_F32_STRIDE_H},
        .dilation = {.w = CONV_K3_OPT_F32_DILATION_W, .h = CONV_K3_OPT_F32_DILATION_H},
        .activation = {.min = CONV_K3_OPT_F32_OUT_ACTIVATION_MIN, .max = CONV_K3_OPT_F32_OUT_ACTIVATION_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = CONV_K3_OPT_F32_INPUT_BATCHES,
                                      .w = CONV_K3_OPT_F32_INPUT_W,
                                      .h = CONV_K3_OPT_F32_INPUT_H,
                                      .c = CONV_K3_OPT_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = CONV_K3_OPT_F32_OUT_CH,
                                       .w = CONV_K3_OPT_F32_FILTER_W,
                                       .h = CONV_K3_OPT_F32_FILTER_H,
                                       .c = CONV_K3_OPT_F32_IN_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CONV_K3_OPT_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = CONV_K3_OPT_F32_INPUT_BATCHES,
                                       .w = CONV_K3_OPT_F32_OUTPUT_W,
                                       .h = CONV_K3_OPT_F32_OUTPUT_H,
                                       .c = CONV_K3_OPT_F32_OUTPUT_C};
    const int32_t buf_size =
        arm_convolve_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims, CONV_K3_OPT_F32_LAYOUT);
    if (buf_size > 0)
    {
        ctx.buf = malloc((size_t)buf_size);
        ctx.size = buf_size;
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }
    float32_t *packed_weights =
        pack_rhs_nt_n_from_nt_t_f32(conv_k3_opt_f32_weights_data,
                                    CONV_K3_OPT_F32_OUT_CH,
                                    CONV_K3_OPT_F32_FILTER_H * CONV_K3_OPT_F32_FILTER_W * CONV_K3_OPT_F32_IN_CH);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f32(&ctx,
                                       &conv_params,
                                       &input_dims,
                                       conv_k3_opt_f32_input_data,
                                       &filter_dims,
                                       packed_weights,
                                       &bias_dims,
                                       conv_k3_opt_f32_biases_data,
                                       &output_dims,
                                       output,
                                       CONV_K3_OPT_F32_LAYOUT));

    for (int i = 0; i < CONV_K3_OPT_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, conv_k3_opt_f32_output_ref_data[i], output[i]);
    }

    free(ctx.buf);
    free(packed_weights);
}

void conv_k5_opt_f32_arm_convolve_f32_packed(void)
{
    float32_t output[CONV_K5_OPT_F32_DST_SIZE] = {0};
    cmsis_nn_context ctx = {0};
    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = CONV_K5_OPT_F32_PADDING_W, .h = CONV_K5_OPT_F32_PADDING_H},
        .stride = {.w = CONV_K5_OPT_F32_STRIDE_W, .h = CONV_K5_OPT_F32_STRIDE_H},
        .dilation = {.w = CONV_K5_OPT_F32_DILATION_W, .h = CONV_K5_OPT_F32_DILATION_H},
        .activation = {.min = CONV_K5_OPT_F32_OUT_ACTIVATION_MIN, .max = CONV_K5_OPT_F32_OUT_ACTIVATION_MAX},
        .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = CONV_K5_OPT_F32_INPUT_BATCHES,
                                      .w = CONV_K5_OPT_F32_INPUT_W,
                                      .h = CONV_K5_OPT_F32_INPUT_H,
                                      .c = CONV_K5_OPT_F32_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = CONV_K5_OPT_F32_OUT_CH,
                                       .w = CONV_K5_OPT_F32_FILTER_W,
                                       .h = CONV_K5_OPT_F32_FILTER_H,
                                       .c = CONV_K5_OPT_F32_IN_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CONV_K5_OPT_F32_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = CONV_K5_OPT_F32_INPUT_BATCHES,
                                       .w = CONV_K5_OPT_F32_OUTPUT_W,
                                       .h = CONV_K5_OPT_F32_OUTPUT_H,
                                       .c = CONV_K5_OPT_F32_OUTPUT_C};
    const int32_t buf_size =
        arm_convolve_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims, CONV_K5_OPT_F32_LAYOUT);
    if (buf_size > 0)
    {
        ctx.buf = malloc((size_t)buf_size);
        ctx.size = buf_size;
        TEST_ASSERT_NOT_NULL(ctx.buf);
    }
    float32_t *packed_weights =
        pack_rhs_nt_n_from_nt_t_f32(conv_k5_opt_f32_weights_data,
                                    CONV_K5_OPT_F32_OUT_CH,
                                    CONV_K5_OPT_F32_FILTER_H * CONV_K5_OPT_F32_FILTER_W * CONV_K5_OPT_F32_IN_CH);

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_convolve_f32(&ctx,
                                       &conv_params,
                                       &input_dims,
                                       conv_k5_opt_f32_input_data,
                                       &filter_dims,
                                       packed_weights,
                                       &bias_dims,
                                       conv_k5_opt_f32_biases_data,
                                       &output_dims,
                                       output,
                                       CONV_K5_OPT_F32_LAYOUT));

    for (int i = 0; i < CONV_K5_OPT_F32_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, conv_k5_opt_f32_output_ref_data[i], output[i]);
    }

    free(ctx.buf);
    free(packed_weights);
}

void conv_k5_output_width_boundaries_f32(void)
{
    enum
    {
        BATCHES = 2,
        ROWS = 2,
        IN_CH = 3,
        OUT_CH = 5,
        KERNEL_W = 5,
        MAX_OUT_W = 7,
        MAX_IN_W = MAX_OUT_W + KERNEL_W - 1,
        MAX_INPUT_SIZE = BATCHES * ROWS * MAX_IN_W * IN_CH,
        WEIGHT_SIZE = OUT_CH * KERNEL_W * IN_CH,
        MAX_OUTPUT_SIZE = BATCHES * ROWS * MAX_OUT_W * OUT_CH
    };
    static float32_t input[MAX_INPUT_SIZE];
    static float32_t weights[WEIGHT_SIZE];
    static float32_t output_standard[MAX_OUTPUT_SIZE];
    static float32_t output_packed[MAX_OUTPUT_SIZE];
    static float32_t reference[MAX_OUTPUT_SIZE];

    for (int32_t i = 0; i < WEIGHT_SIZE; ++i)
    {
        weights[i] = (float32_t)((i % 7) - 3) * 0.0625f;
    }
    float32_t *packed_weights = pack_rhs_nt_n_from_nt_t_f32(weights, OUT_CH, KERNEL_W * IN_CH);

    for (int32_t out_w = 1; out_w <= MAX_OUT_W; ++out_w)
    {
        const int32_t in_w = out_w + KERNEL_W - 1;
        const int32_t input_size = BATCHES * ROWS * in_w * IN_CH;
        const int32_t output_size = BATCHES * ROWS * out_w * OUT_CH;
        for (int32_t i = 0; i < input_size; ++i)
        {
            input[i] = (float32_t)((i % 11) - 5) * 0.03125f;
        }

        for (int32_t n = 0; n < BATCHES; ++n)
        {
            for (int32_t h = 0; h < ROWS; ++h)
            {
                for (int32_t ow = 0; ow < out_w; ++ow)
                {
                    for (int32_t oc = 0; oc < OUT_CH; ++oc)
                    {
                        float32_t sum = 0.0f;
                        for (int32_t kw = 0; kw < KERNEL_W; ++kw)
                        {
                            for (int32_t ic = 0; ic < IN_CH; ++ic)
                            {
                                const int32_t input_index = (((n * ROWS + h) * in_w + ow + kw) * IN_CH) + ic;
                                const int32_t weight_index = ((oc * KERNEL_W + kw) * IN_CH) + ic;
                                sum += input[input_index] * weights[weight_index];
                            }
                        }
                        sum = sum > 0.125f ? 0.125f : sum;
                        sum = sum < -0.125f ? -0.125f : sum;
                        reference[(((n * ROWS + h) * out_w + ow) * OUT_CH) + oc] = sum;
                    }
                }
            }
        }

        const cmsis_nn_dims input_dims = {.n = BATCHES, .h = ROWS, .w = in_w, .c = IN_CH};
        const cmsis_nn_dims filter_dims = {.n = OUT_CH, .h = 1, .w = KERNEL_W, .c = IN_CH};
        const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = OUT_CH};
        const cmsis_nn_dims output_dims = {.n = BATCHES, .h = ROWS, .w = out_w, .c = OUT_CH};
        cmsis_nn_conv_params_f32 params = {.padding = {0, 0},
                                           .stride = {1, 1},
                                           .dilation = {1, 1},
                                           .activation = {-0.125f, 0.125f},
                                           .weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD};

        int32_t buf_size =
            arm_convolve_f32_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
        cmsis_nn_context ctx = {.buf = buf_size > 0 ? malloc((size_t)buf_size) : NULL, .size = buf_size};
        arm_cmsis_nn_status status = arm_convolve_f32(&ctx,
                                                      &params,
                                                      &input_dims,
                                                      input,
                                                      &filter_dims,
                                                      weights,
                                                      &bias_dims,
                                                      NULL,
                                                      &output_dims,
                                                      output_standard,
                                                      ARM_NN_LAYOUT_NHWC);
        free(ctx.buf);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);

        params.weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED;
        buf_size =
            arm_convolve_f32_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC);
        ctx.buf = buf_size > 0 ? malloc((size_t)buf_size) : NULL;
        ctx.size = buf_size;
        status = arm_convolve_f32(&ctx,
                                  &params,
                                  &input_dims,
                                  input,
                                  &filter_dims,
                                  packed_weights,
                                  &bias_dims,
                                  NULL,
                                  &output_dims,
                                  output_packed,
                                  ARM_NN_LAYOUT_NHWC);
        free(ctx.buf);
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);

        for (int32_t i = 0; i < output_size; ++i)
        {
            TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, reference[i], output_standard[i]);
            TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, reference[i], output_packed[i]);
        }
    }

    free(packed_weights);
}

#if defined(NN_DISABLE_SPECIALIZATION)
    #define TEST_ASSERT_CONV_SPECIALIZATION_BUFFER_SIZE(size) TEST_ASSERT_TRUE((size) > 0)
#else
    #define TEST_ASSERT_CONV_SPECIALIZATION_BUFFER_SIZE(size) TEST_ASSERT_EQUAL_INT32(0, (size))
#endif

#define RUN_CONV_F32_PACKED_CASE(CASE_PREFIX, case_name, bias_data, tolerance)                                         \
    void case_name##_arm_convolve_f32_packed(void)                                                                     \
    {                                                                                                                  \
        float32_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        cmsis_nn_context ctx = {0};                                                                                    \
        const cmsis_nn_conv_params_f32 conv_params = {                                                                 \
            .padding = {.w = CASE_PREFIX##_PADDING_W, .h = CASE_PREFIX##_PADDING_H},                                   \
            .stride = {.w = CASE_PREFIX##_STRIDE_W, .h = CASE_PREFIX##_STRIDE_H},                                      \
            .dilation = {.w = CASE_PREFIX##_DILATION_W, .h = CASE_PREFIX##_DILATION_H},                                \
            .activation = {.min = CASE_PREFIX##_OUT_ACTIVATION_MIN, .max = CASE_PREFIX##_OUT_ACTIVATION_MAX},          \
            .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};                                                        \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .c = CASE_PREFIX##_IN_CH};                                                   \
        const cmsis_nn_dims filter_dims = {.n = CASE_PREFIX##_OUT_CH,                                                  \
                                           .w = CASE_PREFIX##_FILTER_W,                                                \
                                           .h = CASE_PREFIX##_FILTER_H,                                                \
                                           .c = CASE_PREFIX##_IN_CH};                                                  \
        const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CASE_PREFIX##_OUT_CH};                           \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                           \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
        const int32_t buf_size = arm_convolve_f32_get_buffer_size(                                                     \
            &conv_params, &input_dims, &filter_dims, &output_dims, CASE_PREFIX##_LAYOUT);                              \
        const int32_t wrapper_buf_size =                                                                               \
            arm_convolve_wrapper_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);           \
        TEST_ASSERT_EQUAL_INT32(buf_size, wrapper_buf_size);                                                           \
        TEST_ASSERT_TRUE(buf_size >= 0);                                                                               \
        if (buf_size > 0)                                                                                              \
        {                                                                                                              \
            ctx.buf = malloc((size_t)buf_size);                                                                        \
            ctx.size = buf_size;                                                                                       \
            TEST_ASSERT_NOT_NULL(ctx.buf);                                                                             \
        }                                                                                                              \
        TEST_ASSERT_CONV_SPECIALIZATION_BUFFER_SIZE(buf_size);                                                         \
                                                                                                                       \
        float32_t *packed_weights =                                                                                    \
            pack_rhs_nt_n_from_nt_t_f32(case_name##_weights_data,                                                      \
                                        CASE_PREFIX##_OUT_CH,                                                          \
                                        CASE_PREFIX##_FILTER_H * CASE_PREFIX##_FILTER_W * CASE_PREFIX##_IN_CH);        \
        TEST_ASSERT_NOT_NULL(packed_weights);                                                                          \
                                                                                                                       \
        const arm_cmsis_nn_status status = arm_convolve_f32(&ctx,                                                      \
                                                            &conv_params,                                              \
                                                            &input_dims,                                               \
                                                            case_name##_input_data,                                    \
                                                            &filter_dims,                                              \
                                                            packed_weights,                                            \
                                                            &bias_dims,                                                \
                                                            bias_data,                                                 \
                                                            &output_dims,                                              \
                                                            output,                                                    \
                                                            CASE_PREFIX##_LAYOUT);                                     \
                                                                                                                       \
        free(ctx.buf);                                                                                                 \
        free(packed_weights);                                                                                          \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS, status);                                                               \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), case_name##_output_ref_data[i], output[i]);                          \
        }                                                                                                              \
    }

RUN_CONV_F32_PACKED_CASE(CONV_K5_BATCH2_ROWS2_TAILS_NULL_BIAS_F32,
                         conv_k5_batch2_rows2_tails_null_bias_f32,
                         NULL,
                         5.0e-4f)
RUN_CONV_F32_PACKED_CASE(CONV_K7_OPT_F32, conv_k7_opt_f32, conv_k7_opt_f32_biases_data, 5.0e-4f)
RUN_CONV_F32_PACKED_CASE(CONV_K9_EXACT_SIX_F32, conv_k9_exact_six_f32, conv_k9_exact_six_f32_biases_data, 5.0e-4f)
RUN_CONV_F32_PACKED_CASE(CONV_K9_OPT_F32, conv_k9_opt_f32, conv_k9_opt_f32_biases_data, 5.0e-4f)
RUN_CONV_F32_PACKED_CASE(CONV_K2_PACKED_VALID_F32,
                         conv_k2_packed_valid_f32,
                         conv_k2_packed_valid_f32_biases_data,
                         5.0e-4f)
RUN_CONV_F32_PACKED_CASE(CONV_KERNEL_2X2_F32, conv_kernel_2x2_f32, conv_kernel_2x2_f32_biases_data, 5.0e-4f)
RUN_CONV_F32_PACKED_CASE(CONV_2X3_PACKED_VALID_F32,
                         conv_2x3_packed_valid_f32,
                         conv_2x3_packed_valid_f32_biases_data,
                         5.0e-4f)
RUN_CONV_F32_PACKED_CASE(CONV_2X5_PACKED_VALID_F32,
                         conv_2x5_packed_valid_f32,
                         conv_2x5_packed_valid_f32_biases_data,
                         5.0e-4f)

/* Binary fractions keep the reference exact in both native F16 and F32.
 * Distinct rows/channels exercise input reuse, packed offsets and every tile.
 */
static void check_packed_conv2d_tiles_f32(const int32_t kernel_w)
{
    enum
    {
        IN_CH = 3,
        OUT_H = 2,
        IN_H = OUT_H + 1,
        MAX_OUT_W = 17,
        MAX_IN_W = MAX_OUT_W + 4,
        MAX_OUT_CH = 5,
        MAX_PATCH_LEN = 2 * 5 * IN_CH
    };
    const int32_t patch_len = 2 * kernel_w * IN_CH;
    const int32_t widths[] = {1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 17};
    const int32_t channels[] = {1, 4, MAX_OUT_CH};
    static float32_t input[IN_H * MAX_IN_W * IN_CH];
    static float32_t weights[MAX_OUT_CH * MAX_PATCH_LEN];
    static float32_t bias[MAX_OUT_CH];
    static float32_t guarded_output[OUT_H * MAX_OUT_W * MAX_OUT_CH + 2];
    float32_t *output = guarded_output + 1;

    for (int32_t i = 0; i < IN_H * MAX_IN_W * IN_CH; ++i)
    {
        input[i] = (float32_t)((i % 17 - 8) * 0.0625f);
    }
    for (int32_t i = 0; i < MAX_OUT_CH * MAX_PATCH_LEN; ++i)
    {
        weights[i] = (float32_t)((i % 7 - 3) * 0.03125f);
    }
    for (int32_t i = 0; i < MAX_OUT_CH; ++i)
    {
        bias[i] = (float32_t)((i % 3 - 1) * 0.125f);
    }

    for (unsigned int c = 0; c < sizeof(channels) / sizeof(channels[0]); ++c)
    {
        const int32_t out_ch = channels[c];
        float32_t *packed = pack_rhs_nt_n_from_nt_t_f32(weights, out_ch, patch_len);
        TEST_ASSERT_NOT_NULL(packed);
        for (unsigned int w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w)
        {
            const int32_t out_w = widths[w];
            const int32_t in_w = out_w + kernel_w - 1;
            const int32_t output_size = OUT_H * out_w * out_ch;
            for (int32_t use_bias = 0; use_bias <= 1; ++use_bias)
            {
                for (int32_t i = 0; i < output_size + 2; ++i)
                {
                    guarded_output[i] = 17;
                }
                if (kernel_w == 2)
                {
                    arm_nn_conv2d_2x2_packed_f32(
                        input, IN_CH, IN_H, in_w, packed, use_bias ? bias : NULL, output, out_ch, OUT_H, out_w);
                }
                else if (kernel_w == 3)
                {
                    arm_nn_conv2d_2x3_packed_f32(
                        input, IN_CH, IN_H, in_w, packed, use_bias ? bias : NULL, output, out_ch, OUT_H, out_w);
                }
                else
                {
                    arm_nn_conv2d_2x5_packed_f32(
                        input, IN_CH, IN_H, in_w, packed, use_bias ? bias : NULL, output, out_ch, OUT_H, out_w);
                }
                for (int32_t oy = 0; oy < OUT_H; ++oy)
                {
                    for (int32_t ox = 0; ox < out_w; ++ox)
                    {
                        for (int32_t oc = 0; oc < out_ch; ++oc)
                        {
                            float32_t reference = use_bias ? (float32_t)bias[oc] : 0.0f;
                            for (int32_t ky = 0; ky < 2; ++ky)
                            {
                                for (int32_t kx = 0; kx < kernel_w; ++kx)
                                {
                                    for (int32_t ic = 0; ic < IN_CH; ++ic)
                                    {
                                        const int32_t input_index = ((oy + ky) * in_w + ox + kx) * IN_CH + ic;
                                        const int32_t weight_index = oc * patch_len + (ky * kernel_w + kx) * IN_CH + ic;
                                        reference += (float32_t)input[input_index] * (float32_t)weights[weight_index];
                                    }
                                }
                            }
                            TEST_ASSERT_EQUAL_FLOAT(reference, (float)output[(oy * out_w + ox) * out_ch + oc]);
                        }
                    }
                }
                TEST_ASSERT_EQUAL_FLOAT(17.0f, (float)guarded_output[0]);
                TEST_ASSERT_EQUAL_FLOAT(17.0f, (float)output[output_size]);
            }
        }
        free(packed);
    }
}

void conv_2x2_packed_tiles_f32(void) { check_packed_conv2d_tiles_f32(2); }
void conv_2x3_packed_tiles_f32(void) { check_packed_conv2d_tiles_f32(3); }
void conv_2x5_packed_tiles_f32(void) { check_packed_conv2d_tiles_f32(5); }

static void check_packed_conv_cancellation_f32(const int32_t kernel_w)
{
    enum
    {
        IN_CH = 2,
        MAX_OUT_CH = 5,
        MAX_KERNEL_W = 9,
        MAX_OUT_W = 7,
        MAX_IN_W = MAX_OUT_W + MAX_KERNEL_W - 1
    };
    static float32_t input[MAX_IN_W * IN_CH];
    static float32_t weights[MAX_OUT_CH * MAX_KERNEL_W * IN_CH];
    static float32_t output[MAX_OUT_W * MAX_OUT_CH];
    const int32_t channel_counts[] = {1, 4, 5};

    for (int32_t i = 0; i < MAX_IN_W * IN_CH; ++i)
    {
        input[i] = 1;
    }

    for (unsigned int channels = 0; channels < sizeof(channel_counts) / sizeof(channel_counts[0]); ++channels)
    {
        const int32_t out_ch = channel_counts[channels];
        const int32_t patch_len = kernel_w * IN_CH;
        for (int32_t i = 0; i < out_ch * patch_len; ++i)
        {
            weights[i] = 0;
        }
        for (int32_t oc = 0; oc < out_ch; ++oc)
        {
            /* Moderate binary fractions cancel exactly in either reduction order.
             * This checks all tiles without imposing bitwise agreement for large-value cancellation.
             */
            weights[oc * patch_len] = 1;
            weights[oc * patch_len + 1] = -1;
            weights[oc * patch_len + 2] = 0.125f;
        }
        float32_t *packed_weights = pack_rhs_nt_n_from_nt_t_f32(weights, out_ch, patch_len);

        /* Exercise every spatial tile and both full and partial channel vectors. */
        for (int32_t out_w = 1; out_w <= MAX_OUT_W; ++out_w)
        {
            /* Check the support kernel directly against the exact result, 1/8. */
            for (int32_t i = 0; i < out_w * out_ch; ++i)
            {
                output[i] = 17;
            }
            if (kernel_w == 7)
            {
                arm_nn_conv1d_k7_packed_f32(
                    input, IN_CH, out_w + kernel_w - 1, packed_weights, NULL, output, out_ch, out_w);
            }
            else
            {
                arm_nn_conv1d_k9_packed_f32(
                    input, IN_CH, out_w + kernel_w - 1, packed_weights, NULL, output, out_ch, out_w);
            }
            for (int32_t i = 0; i < out_w * out_ch; ++i)
            {
                TEST_ASSERT_EQUAL_FLOAT_MESSAGE(0.125f, (float)output[i], "packed kernel cancellation");
            }
        }
        free(packed_weights);
    }
}

void conv_k7_packed_cancellation_f32(void) { check_packed_conv_cancellation_f32(7); }
void conv_k9_packed_cancellation_f32(void) { check_packed_conv_cancellation_f32(9); }

/* Exercise every registration through the public API, including future entries. */
void conv_specialized_registry_f32(void)
{
#ifdef NN_DISABLE_SPECIALIZATION
    TEST_IGNORE_MESSAGE("Specialized kernels are disabled");
#else
    static const arm_conv_specialized_candidate cases[] = {
    #define ARM_CONV_1D_SPEC(W, PACKED, STANDARD, ...)                                                                 \
        {{.filter_h = 1, .filter_w = W, __VA_ARGS__}, (PACKED) != NULL, (STANDARD) != NULL},
    #define ARM_CONV_2D_SPEC(H, W, PACKED, ...) {{.filter_h = H, .filter_w = W, __VA_ARGS__}, (PACKED) != NULL, false},
    #include "Internal/arm_conv_specialized_registry_f32.h"
    #undef ARM_CONV_2D_SPEC
    #undef ARM_CONV_1D_SPEC
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        const int32_t in_c = 3;
        const int32_t out_c = 5;
        const arm_conv_specialized_requirements *spec = &cases[i].requirements;
        TEST_ASSERT_TRUE(spec->filter_h > 0 && spec->filter_w > 0);
        TEST_ASSERT_TRUE(spec->stride.h > 0 && spec->stride.w > 0);
        TEST_ASSERT_TRUE(spec->padding.h >= 0 && spec->padding.w >= 0);
        TEST_ASSERT_TRUE(spec->dilation.h > 0 && spec->dilation.w > 0);
        const cmsis_nn_dims input_dims = {
            .n = 2,
            .h = 2 * ((spec->filter_h - 1) * spec->dilation.h + 1) + 2 * spec->stride.h,
            .w = 2 * ((spec->filter_w - 1) * spec->dilation.w + 1) + 6 * spec->stride.w,
            .c = in_c,
        };
        const cmsis_nn_dims filter_dims = {.n = out_c, .h = spec->filter_h, .w = spec->filter_w, .c = in_c};
        const cmsis_nn_dims output_dims = {
            .n = input_dims.n,
            .h =
                (input_dims.h + 2 * spec->padding.h - (spec->filter_h - 1) * spec->dilation.h - 1) / spec->stride.h + 1,
            .w =
                (input_dims.w + 2 * spec->padding.w - (spec->filter_w - 1) * spec->dilation.w - 1) / spec->stride.w + 1,
            .c = out_c,
        };
        const int32_t patch_len = cases[i].requirements.filter_h * cases[i].requirements.filter_w * in_c;
        const int32_t input_count = input_dims.n * input_dims.h * input_dims.w * in_c;
        const int32_t output_count = output_dims.n * output_dims.h * output_dims.w * out_c;
        float32_t *input = malloc((size_t)input_count * sizeof(*input));
        float32_t *weights = malloc((size_t)patch_len * out_c * sizeof(*weights));
        float32_t *output = malloc((size_t)output_count * sizeof(*output));
        TEST_ASSERT_NOT_NULL(input);
        TEST_ASSERT_NOT_NULL(weights);
        TEST_ASSERT_NOT_NULL(output);
        for (int32_t j = 0; j < input_count; ++j)
        {
            input[j] = 1;
        }
        for (int32_t oc = 0; oc < out_c; ++oc)
        {
            for (int32_t k = 0; k < patch_len; ++k)
            {
                weights[oc * patch_len + k] = (float32_t)(oc + 1);
            }
        }
        float32_t *packed_weights = pack_rhs_nt_n_from_nt_t_f32(weights, out_c, patch_len);
        for (int32_t packed = 0; packed <= 1; ++packed)
        {
            char message[64];
            snprintf(message,
                     sizeof(message),
                     "%ldx%ld / %s",
                     (long)cases[i].requirements.filter_h,
                     (long)cases[i].requirements.filter_w,
                     packed ? "packed" : "standard");
            cmsis_nn_conv_params_f32 params = {.stride = spec->stride,
                                               .padding = spec->padding,
                                               .dilation = spec->dilation,
                                               .activation = {-10000, 10000},
                                               .weight_format = packed ? ARM_NN_WEIGHT_FORMAT_NT_N_PACKED
                                                                       : ARM_NN_WEIGHT_FORMAT_STANDARD};
            const bool available = packed ? cases[i].packed : cases[i].standard;
            arm_conv_selection_f32 selection;
            TEST_ASSERT_EQUAL_MESSAGE(
                available,
                arm_conv_select_specialized_f32(&params, &input_dims, &filter_dims, &output_dims, &selection),
                message);
            TEST_ASSERT_EQUAL_MESSAGE(
                available,
                arm_conv_select_specialized_f32(&params, &input_dims, &filter_dims, &output_dims, NULL),
                message);
            if (!available)
            {
                continue;
            }
            TEST_ASSERT_EQUAL_MESSAGE(
                0,
                arm_convolve_f32_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC),
                message);
            for (int32_t j = 0; j < output_count; ++j)
            {
                output[j] = -1;
            }
            TEST_ASSERT_EQUAL_MESSAGE(ARM_CMSIS_NN_SUCCESS,
                                      arm_convolve_f32(NULL,
                                                       &params,
                                                       &input_dims,
                                                       input,
                                                       &filter_dims,
                                                       packed ? packed_weights : weights,
                                                       NULL,
                                                       NULL,
                                                       &output_dims,
                                                       output,
                                                       ARM_NN_LAYOUT_NHWC),
                                      message);
            for (int32_t j = 0; j < output_count; ++j)
            {
                const int32_t y = (j / out_c / output_dims.w) % output_dims.h;
                const int32_t x = (j / out_c) % output_dims.w;
                int32_t valid_inputs = 0;
                for (int32_t ky = 0; ky < filter_dims.h; ++ky)
                {
                    const int32_t iy = y * params.stride.h - params.padding.h + ky * params.dilation.h;
                    for (int32_t kx = 0; kx < filter_dims.w; ++kx)
                    {
                        const int32_t ix = x * params.stride.w - params.padding.w + kx * params.dilation.w;
                        if (iy >= 0 && iy < input_dims.h && ix >= 0 && ix < input_dims.w)
                        {
                            valid_inputs += in_c;
                        }
                    }
                }
                TEST_ASSERT_EQUAL_FLOAT_MESSAGE((float)(valid_inputs * (j % out_c + 1)), (float)output[j], message);
            }

            /* Choose an undeclared value for each field, keeping output geometry valid.
             * Future registrations with this shape may legitimately support other values.
             */
            for (int32_t property = 0; property < 3; ++property)
            {
                for (int32_t axis = 0; axis < 2; ++axis)
                {
                    cmsis_nn_conv_params_f32 changed = params;
                    cmsis_nn_tile *tiles[] = {&changed.stride, &changed.padding, &changed.dilation};
                    int32_t unregistered = 0;
                    for (size_t k = 0; k < ARM_CONV_ARRAY_SIZE(cases); ++k)
                    {
                        const arm_conv_specialized_requirements *other = &cases[k].requirements;
                        const cmsis_nn_tile values[] = {other->stride, other->padding, other->dilation};
                        const int32_t value = axis == 0 ? values[property].h : values[property].w;
                        if (value >= unregistered)
                        {
                            unregistered = value + 1;
                        }
                    }
                    if (axis == 0)
                    {
                        tiles[property]->h = unregistered;
                    }
                    else
                    {
                        tiles[property]->w = unregistered;
                    }
                    cmsis_nn_dims changed_input = input_dims;
                    changed_input.h = 2 * ((filter_dims.h - 1) * changed.dilation.h + 1) + 2 * changed.stride.h;
                    changed_input.w = 2 * ((filter_dims.w - 1) * changed.dilation.w + 1) + 6 * changed.stride.w;
                    cmsis_nn_dims changed_output = output_dims;
                    changed_output.h =
                        (changed_input.h + 2 * changed.padding.h - (filter_dims.h - 1) * changed.dilation.h - 1) /
                            changed.stride.h +
                        1;
                    changed_output.w =
                        (changed_input.w + 2 * changed.padding.w - (filter_dims.w - 1) * changed.dilation.w - 1) /
                            changed.stride.w +
                        1;
                    TEST_ASSERT_TRUE(arm_conv_specialized_nhwc_geometry_is_valid(&changed.stride,
                                                                                 &changed.padding,
                                                                                 &changed.dilation,
                                                                                 &changed_input,
                                                                                 &filter_dims,
                                                                                 &changed_output));
                    TEST_ASSERT_FALSE(arm_conv_select_specialized_f32(
                        &changed, &changed_input, &filter_dims, &changed_output, &selection));
                    TEST_ASSERT_FALSE(
                        arm_conv_select_specialized_f32(&changed, &changed_input, &filter_dims, &changed_output, NULL));
                    TEST_ASSERT_EQUAL_UINT(0, selection.index);
                    TEST_ASSERT_FALSE(selection.is_1d);
                    TEST_ASSERT_FALSE(selection.packed);
                }
            }
            cmsis_nn_dims wrong_output = output_dims;
            ++wrong_output.w;
            TEST_ASSERT_FALSE(arm_conv_select_specialized_f32(&params, &input_dims, &filter_dims, &wrong_output, NULL));
            params.weight_format = (arm_nn_weight_format_flt)99;
            TEST_ASSERT_FALSE(arm_conv_select_specialized_f32(&params, &input_dims, &filter_dims, &output_dims, NULL));
        }
        free(packed_weights);
        free(output);
        free(weights);
        free(input);
    }

    /* Positive, valid geometry alone must not imply an available implementation. */
    int32_t unsupported_w = 1;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        if (cases[i].requirements.filter_w >= unsupported_w)
        {
            unsupported_w = cases[i].requirements.filter_w + 1;
        }
    }
    const cmsis_nn_conv_params_f32 params = {
        .stride = {1, 1}, .padding = {0, 0}, .dilation = {1, 1}, .weight_format = ARM_NN_WEIGHT_FORMAT_NT_N_PACKED};
    const cmsis_nn_dims input_dims = {.n = 1, .h = 3, .w = unsupported_w + 6, .c = 3};
    const cmsis_nn_dims filter_dims = {.n = 5, .h = 1, .w = unsupported_w, .c = 3};
    const cmsis_nn_dims output_dims = {.n = 1, .h = 3, .w = 7, .c = 5};
    TEST_ASSERT_FALSE(arm_conv_select_specialized_f32(&params, &input_dims, &filter_dims, &output_dims, NULL));
    TEST_ASSERT_TRUE(
        arm_convolve_f32_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims, ARM_NN_LAYOUT_NHWC) > 0);
#endif
}

void conv_specialized_geometry_f32(void)
{
    static const struct
    {
        int32_t input;
        int32_t filter;
        int32_t output;
        int32_t stride;
        int32_t padding;
        int32_t dilation;
        bool valid;
    } cases[] = {
        {9, 3, 4, 2, 0, 1, true},
        {9, 3, 5, 2, 1, 1, true},
        {9, 3, 3, 2, 0, 2, true},
        {9, 3, 4, 2, 0, 2, false}, /* Wrong output for the dilated filter. */
        {5, 3, 1, 4, 0, 1, true},  /* Division must round down. */
        {1, 1, 3, 1, 1, 1, true},  /* First and last windows contain only padding. */
        {1, 3, 1, 1, 0, 1, false}, /* Effective filter exceeds the padded input. */
        {1, 1, 1, 0, 0, 1, false},
        {1, 1, 1, -1, 0, 1, false},
        {1, 1, 1, 1, -1, 1, false},
        {1, 1, 1, 1, 0, 0, false},
        {1, 1, 1, 1, 0, -1, false},
        {0, 1, 1, 1, 0, 1, false},
        {1, 0, 1, 1, 0, 1, false},
        {1, 1, 0, 1, 0, 1, false},
        {INT32_MAX, 1, INT32_MAX, 1, 0, 1, true},
        {INT32_MAX, 1, 1, 1, 1, 1, false}, /* Padded extent exceeds int32_t. */
        {1, INT32_MAX, 1, 1, 0, INT32_MAX, false},
        {INT32_MAX, 2, 1, 1, 0, INT32_MAX - 1, true},
        {1, 1, 1, INT32_MAX, 0, INT32_MAX, true},
    };
    for (size_t i = 0; i < ARM_CONV_ARRAY_SIZE(cases); ++i)
    {
        for (int32_t axis = 0; axis < 2; ++axis)
        {
            cmsis_nn_dims input = {.n = 1, .h = 1, .w = 1, .c = 1};
            cmsis_nn_dims filter = input;
            cmsis_nn_dims output = input;
            cmsis_nn_tile stride = {.h = 1, .w = 1};
            cmsis_nn_tile padding = {.h = 0, .w = 0};
            cmsis_nn_tile dilation = {.h = 1, .w = 1};
            if (axis == 0)
            {
                input.h = cases[i].input;
                filter.h = cases[i].filter;
                output.h = cases[i].output;
                stride.h = cases[i].stride;
                padding.h = cases[i].padding;
                dilation.h = cases[i].dilation;
            }
            else
            {
                input.w = cases[i].input;
                filter.w = cases[i].filter;
                output.w = cases[i].output;
                stride.w = cases[i].stride;
                padding.w = cases[i].padding;
                dilation.w = cases[i].dilation;
            }
            char message[64];
            snprintf(message, sizeof(message), "geometry case %u, axis %ld", (unsigned)i, (long)axis);
            TEST_ASSERT_EQUAL_MESSAGE(
                cases[i].valid,
                arm_conv_specialized_nhwc_geometry_is_valid(&stride, &padding, &dilation, &input, &filter, &output),
                message);
        }
    }
}

#ifndef NN_DISABLE_SPECIALIZATION
/* Test-only fixed 1x3, horizontal-stride-2 kernel, using the existing row signature. */
static void conv_test_stride2_f32(const float32_t *input,
                                  int32_t input_c,
                                  int32_t input_w,
                                  const float32_t *weights,
                                  const float32_t *bias,
                                  float32_t *output,
                                  int32_t output_c,
                                  int32_t output_w)
{
    (void)input_w;
    for (int32_t x = 0; x < output_w; ++x)
    {
        for (int32_t oc = 0; oc < output_c; ++oc)
        {
            float32_t acc = bias ? (float32_t)bias[oc] : 0;
            for (int32_t k = 0; k < 3; ++k)
            {
                for (int32_t ic = 0; ic < input_c; ++ic)
                {
                    acc +=
                        (float32_t)input[(2 * x + k) * input_c + ic] * (float32_t)weights[(oc * 3 + k) * input_c + ic];
                }
            }
            output[x * output_c + oc] = (float32_t)acc;
        }
    }
}
#endif

void conv_specialized_stride2_adapter_f32(void)
{
#ifdef NN_DISABLE_SPECIALIZATION
    TEST_IGNORE_MESSAGE("Specialized kernels are disabled");
#else
    const arm_conv_specialized_requirements spec = {
        .filter_h = 1,
        .filter_w = 3,
        .stride = {.h = 2, .w = 2},
        .padding = {.h = 2, .w = 0},
        .dilation = {.h = 1, .w = 1},
    };
    const cmsis_nn_conv_params_f32 params = {
        .stride = spec.stride,
        .padding = spec.padding,
        .dilation = spec.dilation,
    };
    const cmsis_nn_dims input_dims = {.n = 1, .h = 3, .w = 7, .c = 1};
    const cmsis_nn_dims filter_dims = {.n = 1, .h = 1, .w = 3, .c = 1};
    const cmsis_nn_dims output_dims = {.n = 1, .h = 4, .w = 3, .c = 1};
    TEST_ASSERT_TRUE(arm_conv_specialized_nhwc_geometry_is_valid(
        &params.stride, &params.padding, &params.dilation, &input_dims, &filter_dims, &output_dims));
    TEST_ASSERT_TRUE(arm_conv_specialized_requirements_match(
        &spec, &filter_dims, &params.stride, &params.padding, &params.dilation));

    float32_t input[21];
    const float32_t weights[] = {1, 1, 1};
    const float32_t bias[] = {2};
    float32_t output[12];
    const float32_t expected[] = {0, 0, 0, 6, 12, 18, 48, 54, 60, 0, 0, 0};
    for (size_t i = 0; i < ARM_CONV_ARRAY_SIZE(input); ++i)
    {
        input[i] = (float32_t)(i + 1);
    }
    for (int32_t with_bias = 0; with_bias <= 1; ++with_bias)
    {
        for (size_t i = 0; i < ARM_CONV_ARRAY_SIZE(output); ++i)
        {
            output[i] = -1;
        }
        arm_conv_execute_1d_f32(
            conv_test_stride2_f32, &params, &input_dims, input, weights, with_bias ? bias : NULL, &output_dims, output);
        for (size_t i = 0; i < ARM_CONV_ARRAY_SIZE(output); ++i)
        {
            TEST_ASSERT_EQUAL_FLOAT(expected[i] + (with_bias ? 2 : 0), (float)output[i]);
        }
    }
#endif
}
