/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnfunctions.h>
#include "Internal/arm_depthwise_conv_specialized_f16.h"
#include <unity.h>

#include "../TestData/depthwise_2x3_opt_nhwc_f16/test_data.h"
#include "../TestData/depthwise_2x5_opt_batch2_f16/test_data.h"
#include "../TestData/depthwise_2x5_opt_nhwc_chmult16_f16/test_data.h"
#include "../TestData/depthwise_basic_f16/test_data.h"
#include "../TestData/depthwise_basic_smallc_nhwc_f16/test_data.h"
#include "../TestData/depthwise_ic1_to_conv_nhwc_f16/test_data.h"
#include "../TestData/depthwise_k2_1d_opt_nhwc_f16/test_data.h"
#include "../TestData/depthwise_k3_1d_opt_batch2_f16/test_data.h"
#include "../TestData/depthwise_k3_1d_opt_nhwc_f16/test_data.h"
#include "../TestData/depthwise_k5_1d_opt_nhwc_f16/test_data.h"
#include "../TestData/depthwise_k7_1d_opt_nhwc_f16/test_data.h"
#include "../TestData/depthwise_k9_1d_mult3_batch2_null_bias_f16/test_data.h"
#include "../TestData/depthwise_k9_1d_mult9_f16/test_data.h"
#include "../TestData/depthwise_k9_1d_opt_nhwc_f16/test_data.h"
#include "../TestData/depthwise_kernel_2x2_f16/test_data.h"
#include "../TestData/depthwise_kernel_3x3_f16/test_data.h"
#include "../TestData/depthwise_kernel_3x3_null_bias_f16/test_data.h"
#include "../TestData/depthwise_match_basic_f16/test_data.h"
#include "../TestData/depthwise_match_dilation_f16/test_data.h"
#include "../TestData/depthwise_match_out_activation_f16/test_data.h"
#include "../TestData/depthwise_match_stride2pad1_f16/test_data.h"
#include "../TestData/depthwise_match_sub_block_f16/test_data.h"

#define RUN_DEPTHWISE_F16_CASE(CASE_PREFIX, case_name, tolerance)                                                      \
    void case_name##_arm_depthwise_conv_f16(void)                                                                      \
    {                                                                                                                  \
        float16_t output[CASE_PREFIX##_DST_SIZE] = {0};                                                                \
        cmsis_nn_context ctx = {0};                                                                                    \
        const cmsis_nn_dw_conv_params_f16 dw_conv_params = {                                                           \
            .padding = {.w = CASE_PREFIX##_PADDING_W, .h = CASE_PREFIX##_PADDING_H},                                   \
            .stride = {.w = CASE_PREFIX##_STRIDE_W, .h = CASE_PREFIX##_STRIDE_H},                                      \
            .dilation = {.w = CASE_PREFIX##_DILATION_W, .h = CASE_PREFIX##_DILATION_H},                                \
            .ch_mult = CASE_PREFIX##_CH_MULT,                                                                          \
            .activation = {.min = CASE_PREFIX##_OUT_ACTIVATION_MIN, .max = CASE_PREFIX##_OUT_ACTIVATION_MAX}};         \
        const cmsis_nn_dims input_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                            \
                                          .w = CASE_PREFIX##_INPUT_W,                                                  \
                                          .h = CASE_PREFIX##_INPUT_H,                                                  \
                                          .c = CASE_PREFIX##_IN_CH};                                                   \
        const cmsis_nn_dims filter_dims = {.n = CASE_PREFIX##_IN_CH,                                                   \
                                           .w = CASE_PREFIX##_FILTER_W,                                                \
                                           .h = CASE_PREFIX##_FILTER_H,                                                \
                                           .c = CASE_PREFIX##_OUT_CH};                                                 \
        const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = CASE_PREFIX##_OUT_CH};                           \
        const cmsis_nn_dims output_dims = {.n = CASE_PREFIX##_INPUT_BATCHES,                                           \
                                           .w = CASE_PREFIX##_OUTPUT_W,                                                \
                                           .h = CASE_PREFIX##_OUTPUT_H,                                                \
                                           .c = CASE_PREFIX##_OUTPUT_C};                                               \
        const int32_t buf_size = arm_depthwise_conv_f16_get_buffer_size(                                               \
            &dw_conv_params, &input_dims, &filter_dims, &output_dims, CASE_PREFIX##_LAYOUT);                           \
                                                                                                                       \
        if (buf_size > 0)                                                                                              \
        {                                                                                                              \
            ctx.buf = malloc((size_t)buf_size);                                                                        \
            ctx.size = buf_size;                                                                                       \
            TEST_ASSERT_NOT_NULL(ctx.buf);                                                                             \
        }                                                                                                              \
                                                                                                                       \
        TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,                                                                        \
                          arm_depthwise_conv_f16(&ctx,                                                                 \
                                                 &dw_conv_params,                                                      \
                                                 &input_dims,                                                          \
                                                 case_name##_input_data,                                               \
                                                 &filter_dims,                                                         \
                                                 case_name##_weights_data,                                             \
                                                 &bias_dims,                                                           \
                                                 CASE_PREFIX##_USE_NULL_BIAS ? NULL : case_name##_biases_data,         \
                                                 &output_dims,                                                         \
                                                 output,                                                               \
                                                 CASE_PREFIX##_LAYOUT));                                               \
                                                                                                                       \
        for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                               \
        {                                                                                                              \
            TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref_data[i], (float)output[i]);            \
            output[i] = (float16_t)0.0f;                                                                               \
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
            const int32_t wrapper_buf_size = arm_depthwise_conv_wrapper_f16_get_buffer_size(                           \
                &dw_conv_params, &input_dims, &filter_dims, &output_dims);                                             \
            if (wrapper_buf_size > 0)                                                                                  \
            {                                                                                                          \
                ctx.buf = malloc((size_t)wrapper_buf_size);                                                            \
                ctx.size = wrapper_buf_size;                                                                           \
                TEST_ASSERT_NOT_NULL(ctx.buf);                                                                         \
            }                                                                                                          \
            TEST_ASSERT_EQUAL(                                                                                         \
                ARM_CMSIS_NN_SUCCESS,                                                                                  \
                arm_depthwise_conv_wrapper_f16(&ctx,                                                                   \
                                               &dw_conv_params,                                                        \
                                               &input_dims,                                                            \
                                               case_name##_input_data,                                                 \
                                               &filter_dims,                                                           \
                                               case_name##_weights_data,                                               \
                                               &bias_dims,                                                             \
                                               CASE_PREFIX##_USE_NULL_BIAS ? NULL : case_name##_biases_data,           \
                                               &output_dims,                                                           \
                                               output));                                                               \
            for (int i = 0; i < CASE_PREFIX##_DST_SIZE; ++i)                                                           \
            {                                                                                                          \
                TEST_ASSERT_FLOAT_WITHIN((tolerance), (float)case_name##_output_ref_data[i], (float)output[i]);        \
            }                                                                                                          \
            if (ctx.buf != NULL)                                                                                       \
            {                                                                                                          \
                memset(ctx.buf, 0, (size_t)wrapper_buf_size);                                                          \
                free(ctx.buf);                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
    }

RUN_DEPTHWISE_F16_CASE(DEPTHWISE_BASIC_F16, depthwise_basic_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_BASIC_SMALLC_NHWC_F16, depthwise_basic_smallc_nhwc_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_IC1_TO_CONV_NHWC_F16, depthwise_ic1_to_conv_nhwc_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_KERNEL_2X2_F16, depthwise_kernel_2x2_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_KERNEL_3X3_F16, depthwise_kernel_3x3_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_KERNEL_3X3_NULL_BIAS_F16, depthwise_kernel_3x3_null_bias_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_K3_1D_OPT_BATCH2_F16, depthwise_k3_1d_opt_batch2_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_K2_1D_OPT_NHWC_F16, depthwise_k2_1d_opt_nhwc_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_K3_1D_OPT_NHWC_F16, depthwise_k3_1d_opt_nhwc_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_K5_1D_OPT_NHWC_F16, depthwise_k5_1d_opt_nhwc_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_K7_1D_OPT_NHWC_F16, depthwise_k7_1d_opt_nhwc_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_K9_1D_OPT_NHWC_F16, depthwise_k9_1d_opt_nhwc_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_K9_1D_MULT3_BATCH2_NULL_BIAS_F16, depthwise_k9_1d_mult3_batch2_null_bias_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_K9_1D_MULT9_F16, depthwise_k9_1d_mult9_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_2X5_OPT_BATCH2_F16, depthwise_2x5_opt_batch2_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_2X3_OPT_NHWC_F16, depthwise_2x3_opt_nhwc_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_2X5_OPT_NHWC_CHMULT16_F16, depthwise_2x5_opt_nhwc_chmult16_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_MATCH_BASIC_F16, depthwise_match_basic_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_MATCH_SUB_BLOCK_F16, depthwise_match_sub_block_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_MATCH_DILATION_F16, depthwise_match_dilation_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_MATCH_OUT_ACTIVATION_F16, depthwise_match_out_activation_f16, 2.0e-2f)
RUN_DEPTHWISE_F16_CASE(DEPTHWISE_MATCH_STRIDE2PAD1_F16, depthwise_match_stride2pad1_f16, 2.0e-2f)

void depthwise_ic1_to_conv_nhwc_f16_arm_depthwise_conv_f16_no_ctx(void)
{
    float16_t output[DEPTHWISE_IC1_TO_CONV_NHWC_F16_DST_SIZE] = {0};
    const cmsis_nn_context ctx = {0};
    const cmsis_nn_dw_conv_params_f16 dw_conv_params = {
        .padding = {.w = DEPTHWISE_IC1_TO_CONV_NHWC_F16_PADDING_W, .h = DEPTHWISE_IC1_TO_CONV_NHWC_F16_PADDING_H},
        .stride = {.w = DEPTHWISE_IC1_TO_CONV_NHWC_F16_STRIDE_W, .h = DEPTHWISE_IC1_TO_CONV_NHWC_F16_STRIDE_H},
        .dilation = {.w = DEPTHWISE_IC1_TO_CONV_NHWC_F16_DILATION_W, .h = DEPTHWISE_IC1_TO_CONV_NHWC_F16_DILATION_H},
        .ch_mult = DEPTHWISE_IC1_TO_CONV_NHWC_F16_CH_MULT,
        .activation = {.min = DEPTHWISE_IC1_TO_CONV_NHWC_F16_OUT_ACTIVATION_MIN,
                       .max = DEPTHWISE_IC1_TO_CONV_NHWC_F16_OUT_ACTIVATION_MAX}};
    const cmsis_nn_dims input_dims = {.n = DEPTHWISE_IC1_TO_CONV_NHWC_F16_INPUT_BATCHES,
                                      .w = DEPTHWISE_IC1_TO_CONV_NHWC_F16_INPUT_W,
                                      .h = DEPTHWISE_IC1_TO_CONV_NHWC_F16_INPUT_H,
                                      .c = DEPTHWISE_IC1_TO_CONV_NHWC_F16_IN_CH};
    const cmsis_nn_dims filter_dims = {.n = DEPTHWISE_IC1_TO_CONV_NHWC_F16_IN_CH,
                                       .w = DEPTHWISE_IC1_TO_CONV_NHWC_F16_FILTER_W,
                                       .h = DEPTHWISE_IC1_TO_CONV_NHWC_F16_FILTER_H,
                                       .c = DEPTHWISE_IC1_TO_CONV_NHWC_F16_OUT_CH};
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = DEPTHWISE_IC1_TO_CONV_NHWC_F16_OUT_CH};
    const cmsis_nn_dims output_dims = {.n = DEPTHWISE_IC1_TO_CONV_NHWC_F16_INPUT_BATCHES,
                                       .w = DEPTHWISE_IC1_TO_CONV_NHWC_F16_OUTPUT_W,
                                       .h = DEPTHWISE_IC1_TO_CONV_NHWC_F16_OUTPUT_H,
                                       .c = DEPTHWISE_IC1_TO_CONV_NHWC_F16_OUTPUT_C};

    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_f16(&ctx,
                                             &dw_conv_params,
                                             &input_dims,
                                             depthwise_ic1_to_conv_nhwc_f16_input_data,
                                             &filter_dims,
                                             depthwise_ic1_to_conv_nhwc_f16_weights_data,
                                             &bias_dims,
                                             depthwise_ic1_to_conv_nhwc_f16_biases_data,
                                             &output_dims,
                                             output,
                                             DEPTHWISE_IC1_TO_CONV_NHWC_F16_LAYOUT));

    for (int i = 0; i < DEPTHWISE_IC1_TO_CONV_NHWC_F16_DST_SIZE; ++i)
    {
        TEST_ASSERT_FLOAT_WITHIN(2.0e-2f, (float)depthwise_ic1_to_conv_nhwc_f16_output_ref_data[i], (float)output[i]);
    }
}

/* All public entry points must reject bad geometry without touching output. */
void depthwise_invalid_geometry_f16(void)
{
    const float16_t input[1] = {2};
    const float16_t kernel[9] = {3};
    float16_t output[9];
    cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = 1, .c = 1};
    cmsis_nn_dims filter_dims = {.n = 1, .h = 3, .w = 3, .c = 1};
    cmsis_nn_dims output_dims = {.n = 1, .h = 3, .w = 3, .c = 1};
    cmsis_nn_dw_conv_params_f16 params = {
        .ch_mult = 1, .stride = {1, 1}, .padding = {1, 1}, .dilation = {1, 1}, .activation = {-100, 100}};
    const struct
    {
        int32_t *field;
        int32_t value;
    } invalid[] = {{&params.padding.h, 1073741824},
                   {&params.padding.w, 1073741824},
                   {&params.padding.h, -1},
                   {&params.padding.w, -1},
                   {&params.padding.h, INT32_MAX},
                   {&params.padding.w, INT32_MAX},
                   {&params.stride.h, 0},
                   {&params.stride.w, 0},
                   {&params.stride.h, -1},
                   {&params.stride.w, -1},
                   {&params.stride.h, 1073741824},
                   {&params.stride.w, 1073741824},
                   {&params.dilation.h, 0},
                   {&params.dilation.w, 0},
                   {&params.dilation.h, -1},
                   {&params.dilation.w, -1},
                   {&params.dilation.h, INT32_MAX},
                   {&params.dilation.w, INT32_MAX},
                   {&params.ch_mult, 0},
                   {&params.ch_mult, -1},
                   {&params.ch_mult, INT32_MAX},
                   {&input_dims.n, 0},
                   {&input_dims.h, 0},
                   {&input_dims.w, -1},
                   {&input_dims.c, 0},
                   {&input_dims.h, INT32_MAX},
                   {&filter_dims.h, 0},
                   {&filter_dims.w, INT32_MAX},
                   {&output_dims.n, 2},
                   {&output_dims.h, 0},
                   {&output_dims.w, INT32_MAX},
                   {&output_dims.c, 2}};

    for (unsigned int i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
    {
        const int32_t saved = *invalid[i].field;
        *invalid[i].field = invalid[i].value;
        for (int route = 0; route < 3; ++route)
        {
            for (int j = 0; j < 9; ++j)
            {
                output[j] = 17;
            }
            arm_cmsis_nn_status status;
            if (route == 0)
            {
                status = arm_depthwise_conv_f16(NULL,
                                                &params,
                                                &input_dims,
                                                input,
                                                &filter_dims,
                                                kernel,
                                                NULL,
                                                NULL,
                                                &output_dims,
                                                output,
                                                ARM_NN_LAYOUT_NHWC);
            }
            else if (route == 1)
            {
                status = arm_depthwise_nhwc_conv_f16(
                    NULL, &params, &input_dims, input, &filter_dims, kernel, NULL, NULL, &output_dims, output);
            }
            else
            {
                status = arm_depthwise_conv_wrapper_f16(
                    NULL, &params, &input_dims, input, &filter_dims, kernel, NULL, NULL, &output_dims, output);
            }
            TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR, status);
            for (int j = 0; j < 9; ++j)
            {
                TEST_ASSERT_EQUAL_FLOAT(17, (float)output[j]);
            }
        }
        *invalid[i].field = saved;
    }
}

void depthwise_geometry_limits_f16(void)
{
    const float16_t input[1] = {2};
    const float16_t kernel[9] = {3};
    float16_t output[4] = {17, 17, 17, 17};
    const cmsis_nn_dims input_dims = {.n = 1, .h = 1, .w = 1, .c = 1};
    cmsis_nn_dims filter_dims = {.n = 1, .h = 3, .w = 3, .c = 1};
    cmsis_nn_dims output_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
    cmsis_nn_dw_conv_params_f16 params = {.ch_mult = 1,
                                          .stride = {1073741824, 1073741824},
                                          .padding = {1073741824, 1073741824},
                                          .dilation = {1, 1},
                                          .activation = {-100, 100}};

    /* Original signed-overflow reproducer: reject before matching the 3x3 specialization. */
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_ARG_ERROR,
                      arm_depthwise_conv_f16(NULL,
                                             &params,
                                             &input_dims,
                                             input,
                                             &filter_dims,
                                             kernel,
                                             NULL,
                                             NULL,
                                             &output_dims,
                                             output,
                                             ARM_NN_LAYOUT_NHWC));
    for (int i = 0; i < 4; ++i)
    {
        TEST_ASSERT_EQUAL_FLOAT(17, (float)output[i]);
    }

    /* Immediately below the padding limit, the padded extent is exactly INT32_MAX. */
    params.padding.h = params.padding.w = 1073741823;
    params.stride.h = params.stride.w = 1073741823;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_f16(NULL,
                                             &params,
                                             &input_dims,
                                             input,
                                             &filter_dims,
                                             kernel,
                                             NULL,
                                             NULL,
                                             &output_dims,
                                             output,
                                             ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL_FLOAT(0, (float)output[0]);
    TEST_ASSERT_EQUAL_FLOAT(0, (float)output[1]);
    TEST_ASSERT_EQUAL_FLOAT(0, (float)output[2]);
    TEST_ASSERT_EQUAL_FLOAT(6, (float)output[3]);

    /* A large stride is safe when there is only one output position. */
    params.padding.h = params.padding.w = 0;
    params.stride.h = params.stride.w = INT32_MAX;
    filter_dims.h = filter_dims.w = 1;
    output_dims.h = output_dims.w = 1;
    TEST_ASSERT_EQUAL(ARM_CMSIS_NN_SUCCESS,
                      arm_depthwise_conv_f16(NULL,
                                             &params,
                                             &input_dims,
                                             input,
                                             &filter_dims,
                                             kernel,
                                             NULL,
                                             NULL,
                                             &output_dims,
                                             output,
                                             ARM_NN_LAYOUT_NHWC));
    TEST_ASSERT_EQUAL_FLOAT(6, (float)output[0]);
}

void depthwise_dilated_clipping_boundaries_f16(void)
{
    const float16_t input[1] = {2};
    const float16_t kernel[1] = {3};
    const cmsis_nn_dims dims = {.n = 1, .h = 1, .w = 1, .c = 1};
    static const struct
    {
        const char *name;
        int32_t padding;
        int32_t dilation;
        arm_cmsis_nn_status expected_status;
        float expected_output;
    } cases[] = {
        {"no padding, last valid", 0, INT32_MAX - 1, ARM_CMSIS_NN_SUCCESS, 6.0f},
        {"no padding, first invalid", 0, INT32_MAX, ARM_CMSIS_NN_ARG_ERROR, 17.0f},
        {"padding one, last valid", 1, INT32_MAX - 2, ARM_CMSIS_NN_SUCCESS, 0.0f},
        {"padding one, first invalid", 1, INT32_MAX - 1, ARM_CMSIS_NN_ARG_ERROR, 17.0f},
    };
    static const char *const route_names[] = {
        "arm_depthwise_conv", "arm_depthwise_nhwc_conv", "arm_depthwise_conv_wrapper"};

    /* A 1x1 kernel isolates clipping arithmetic from the effective-kernel-span check.
     * Valid cases produce 2 * 3 or sample padding; rejected cases preserve the sentinel 17.
     */
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        for (int32_t axis = 0; axis < 2; ++axis)
        {
            const char *axis_name = axis == 0 ? "height" : "width";
            cmsis_nn_dw_conv_params_f16 params = {.ch_mult = 1,
                                                  .stride = {1, 1},
                                                  .padding = {0, 0},
                                                  .dilation = {1, 1},
                                                  .activation = {-100, 100}};
            if (axis == 0)
            {
                params.padding.h = cases[i].padding;
                params.dilation.h = cases[i].dilation;
            }
            else
            {
                params.padding.w = cases[i].padding;
                params.dilation.w = cases[i].dilation;
            }

            for (size_t route = 0; route < sizeof(route_names) / sizeof(route_names[0]); ++route)
            {
                char message[128];
                snprintf(message, sizeof(message), "%s / %s / %s", cases[i].name, axis_name, route_names[route]);
                float16_t output[1] = {17};
                arm_cmsis_nn_status status;
                if (route == 0)
                {
                    status = arm_depthwise_conv_f16(NULL,
                                                    &params,
                                                    &dims,
                                                    input,
                                                    &dims,
                                                    kernel,
                                                    NULL,
                                                    NULL,
                                                    &dims,
                                                    output,
                                                    ARM_NN_LAYOUT_NHWC);
                }
                else if (route == 1)
                {
                    status = arm_depthwise_nhwc_conv_f16(
                        NULL, &params, &dims, input, &dims, kernel, NULL, NULL, &dims, output);
                }
                else
                {
                    status = arm_depthwise_conv_wrapper_f16(
                        NULL, &params, &dims, input, &dims, kernel, NULL, NULL, &dims, output);
                }

                TEST_ASSERT_EQUAL_MESSAGE(cases[i].expected_status, status, message);
                TEST_ASSERT_EQUAL_FLOAT_MESSAGE(cases[i].expected_output, (float)output[0], message);
            }
        }
    }
}

/* Check selection separately from output: a correct fallback can hide a wrong matcher. */
void depthwise_specialization_selection_f16(void)
{
    static const struct
    {
        const char *name;
        int32_t filter_h;
        int32_t filter_w;
        int32_t input_h;
        int32_t multiplier;
        cmsis_nn_tile stride;
        cmsis_nn_tile padding;
        cmsis_nn_tile dilation;
        bool specialized;
    } cases[] = {
        {"k3 rows and batches", 1, 3, 3, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, true},
        {"k3 multiplier fallback", 1, 3, 3, 3, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, false},
        {"k9 unit multiplier", 1, 9, 3, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, true},
        {"k9 multiple channels", 1, 9, 3, 3, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, true},
        {"2x5 unit multiplier", 2, 5, 2, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, true},
        {"2x5 multiple channels", 2, 5, 2, 3, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, true},
        {"2x5 extra row fallback", 2, 5, 3, 3, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, false},
        {"3x3 unit multiplier", 3, 3, 5, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, true},
        {"3x3 multiplier fallback", 3, 3, 5, 3, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, false},
        {"3x3 stride and padding", 3, 3, 5, 1, {.h = 2, .w = 2}, {.h = 1, .w = 1}, {.h = 1, .w = 1}, true},
        {"k9 stride fallback", 1, 9, 3, 3, {.h = 2, .w = 2}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, false},
        {"k9 padding fallback", 1, 9, 3, 3, {.h = 1, .w = 1}, {.h = 1, .w = 1}, {.h = 1, .w = 1}, false},
        {"k9 dilation fallback", 1, 9, 3, 3, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 2, .w = 2}, false},
        {"unregistered shape", 2, 2, 3, 3, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, false},
        {"3x3 stride without padding", 3, 3, 5, 1, {.h = 2, .w = 2}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, true},
        {"3x3 padding with unit stride", 3, 3, 5, 1, {.h = 1, .w = 1}, {.h = 1, .w = 1}, {.h = 1, .w = 1}, true},
        {"3x3 asymmetric stride", 3, 3, 5, 1, {.h = 2, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 1}, true},
        {"3x3 asymmetric padding", 3, 3, 5, 1, {.h = 1, .w = 1}, {.h = 1, .w = 0}, {.h = 1, .w = 1}, true},
        {"3x3 height dilation fallback", 3, 3, 5, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 2, .w = 1}, false},
        {"3x3 width dilation fallback", 3, 3, 5, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, {.h = 1, .w = 2}, false},
    };
    float16_t input[512];
    float16_t kernel[128];
    float16_t output[2048];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        cmsis_nn_dw_conv_params_f16 params = {
            .stride = cases[i].stride,
            .padding = cases[i].padding,
            .dilation = cases[i].dilation,
            .ch_mult = cases[i].multiplier,
            .activation = {.min = -1000, .max = 1000},
        };
        const cmsis_nn_dims in = {.n = 2, .h = cases[i].input_h, .w = 19, .c = 2};
        cmsis_nn_dims out = {
            .n = in.n,
            .h = (in.h + 2 * params.padding.h - (cases[i].filter_h - 1) * params.dilation.h - 1) / params.stride.h + 1,
            .w = (in.w + 2 * params.padding.w - (cases[i].filter_w - 1) * params.dilation.w - 1) / params.stride.w + 1,
            .c = in.c * params.ch_mult,
        };
        const cmsis_nn_dims filter = {.n = 1, .h = cases[i].filter_h, .w = cases[i].filter_w, .c = out.c};
#ifndef NN_DISABLE_SPECIALIZATION
        TEST_ASSERT_EQUAL_MESSAGE(cases[i].specialized,
                                  arm_dw_select_nhwc_f16(&params, &in, &filter, &out, ARM_NN_DW_KERNEL_KC) != NULL,
                                  cases[i].name);
        TEST_ASSERT_TRUE_MESSAGE(arm_dw_select_nhwc_f16(&params, &in, &filter, &out, ARM_NN_DW_KERNEL_CK) == NULL,
                                 cases[i].name);
        ++out.w;
        TEST_ASSERT_TRUE_MESSAGE(arm_dw_select_nhwc_f16(&params, &in, &filter, &out, ARM_NN_DW_KERNEL_KC) == NULL,
                                 cases[i].name);
        --out.w;
#endif
        for (int32_t j = 0; j < in.n * in.h * in.w * in.c; ++j)
        {
            input[j] = (float16_t)(j % 7 - 3);
        }
        for (int32_t j = 0; j < filter.h * filter.w * out.c; ++j)
        {
            kernel[j] = (float16_t)(j % 5 - 2);
        }
        for (int32_t j = 0; j < out.n * out.h * out.w * out.c; ++j)
        {
            output[j] = (float16_t)999;
        }
        TEST_ASSERT_EQUAL_MESSAGE(
            ARM_CMSIS_NN_SUCCESS,
            arm_depthwise_conv_f16(
                NULL, &params, &in, input, &filter, kernel, NULL, NULL, &out, output, ARM_NN_LAYOUT_NHWC),
            cases[i].name);
        /* Small integer products keep the reference exact in both precisions. */
        for (int32_t n = 0; n < out.n; ++n)
        {
            for (int32_t y = 0; y < out.h; ++y)
            {
                for (int32_t x = 0; x < out.w; ++x)
                {
                    for (int32_t oc = 0; oc < out.c; ++oc)
                    {
                        int32_t expected = 0;
                        for (int32_t ky = 0; ky < filter.h; ++ky)
                        {
                            const int32_t iy = y * params.stride.h - params.padding.h + ky * params.dilation.h;
                            for (int32_t kx = 0; kx < filter.w; ++kx)
                            {
                                const int32_t ix = x * params.stride.w - params.padding.w + kx * params.dilation.w;
                                if (iy >= 0 && iy < in.h && ix >= 0 && ix < in.w)
                                {
                                    const int32_t input_index =
                                        ((n * in.h + iy) * in.w + ix) * in.c + oc / params.ch_mult;
                                    const int32_t kernel_index = (ky * filter.w + kx) * out.c + oc;
                                    expected += (int32_t)input[input_index] * (int32_t)kernel[kernel_index];
                                }
                            }
                        }
                        const int32_t index = ((n * out.h + y) * out.w + x) * out.c + oc;
                        TEST_ASSERT_EQUAL_FLOAT_MESSAGE((float)expected, (float)output[index], cases[i].name);
                    }
                }
            }
        }
    }
}

/* Synthetic requirements exercise future combinations without adding production kernels. */
void depthwise_specialization_requirements_f16(void)
{
    static const struct
    {
        const char *name;
        arm_dw_stride_policy stride_policy;
        arm_dw_padding_policy padding_policy;
        cmsis_nn_tile stride;
        cmsis_nn_tile padding;
        cmsis_nn_tile dilation;
        int32_t input_h;
        bool require_input_height_equals_filter;
        bool expected_match;
    } cases[] = {
        {"exact requirements",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 2, .w = 2},
         {.h = 1, .w = 0},
         {.h = 2, .w = 1},
         8,
         false,
         true},
        {"exact height stride",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 1, .w = 2},
         {.h = 1, .w = 0},
         {.h = 2, .w = 1},
         8,
         false,
         false},
        {"exact width stride",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 2, .w = 1},
         {.h = 1, .w = 0},
         {.h = 2, .w = 1},
         8,
         false,
         false},
        {"exact height padding",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 2, .w = 2},
         {.h = 0, .w = 0},
         {.h = 2, .w = 1},
         8,
         false,
         false},
        {"zero width padding is exact",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 2, .w = 2},
         {.h = 1, .w = 1},
         {.h = 2, .w = 1},
         8,
         false,
         false},
        {"exact height dilation",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 2, .w = 2},
         {.h = 1, .w = 0},
         {.h = 1, .w = 1},
         8,
         false,
         false},
        {"exact width dilation",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 2, .w = 2},
         {.h = 1, .w = 0},
         {.h = 2, .w = 2},
         8,
         false,
         false},
        {"any stride, exact padding",
         ARM_DW_STRIDE_ANY,
         ARM_DW_PADDING_EXACT,
         {.h = 1, .w = 3},
         {.h = 1, .w = 0},
         {.h = 2, .w = 1},
         8,
         false,
         true},
        {"any stride still checks padding",
         ARM_DW_STRIDE_ANY,
         ARM_DW_PADDING_EXACT,
         {.h = 1, .w = 3},
         {.h = 0, .w = 0},
         {.h = 2, .w = 1},
         8,
         false,
         false},
        {"exact stride, any padding",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_ANY,
         {.h = 2, .w = 2},
         {.h = 0, .w = 2},
         {.h = 2, .w = 1},
         8,
         false,
         true},
        {"any padding still checks stride",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_ANY,
         {.h = 1, .w = 2},
         {.h = 0, .w = 2},
         {.h = 2, .w = 1},
         8,
         false,
         false},
        {"any stride and padding",
         ARM_DW_STRIDE_ANY,
         ARM_DW_PADDING_ANY,
         {.h = 1, .w = 3},
         {.h = 0, .w = 2},
         {.h = 2, .w = 1},
         8,
         false,
         true},
        {"height must equal filter",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 2, .w = 2},
         {.h = 1, .w = 0},
         {.h = 2, .w = 1},
         8,
         true,
         false},
        {"matching fixed height",
         ARM_DW_STRIDE_EXACT,
         ARM_DW_PADDING_EXACT,
         {.h = 2, .w = 2},
         {.h = 1, .w = 0},
         {.h = 2, .w = 1},
         2,
         true,
         true},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        const arm_dw_requirements spec = {
            .filter_h = 2,
            .filter_w = 3,
            .multiplier = ARM_DW_MULTIPLIER_ANY,
            .stride_policy = cases[i].stride_policy,
            .stride = {.h = 2, .w = 2},
            .padding_policy = cases[i].padding_policy,
            .padding = {.h = 1, .w = 0},
            .dilation = {.h = 2, .w = 1},
            .require_input_height_equals_filter = cases[i].require_input_height_equals_filter,
        };
        const cmsis_nn_dims input = {.n = 2, .h = cases[i].input_h, .w = 11, .c = 2};
        const cmsis_nn_dims filter = {.n = 1, .h = spec.filter_h, .w = spec.filter_w, .c = 6};
        cmsis_nn_dims output = {
            .n = input.n,
            .h = (input.h + 2 * cases[i].padding.h - (filter.h - 1) * cases[i].dilation.h - 1) / cases[i].stride.h + 1,
            .w = (input.w + 2 * cases[i].padding.w - (filter.w - 1) * cases[i].dilation.w - 1) / cases[i].stride.w + 1,
            .c = input.c * 3,
        };
        /* Matching follows public validation; rejected requests here still have valid indexing. */
        TEST_ASSERT_TRUE_MESSAGE(
            arm_depthwise_conv_axis_is_valid(
                input.h, filter.h, output.h, cases[i].stride.h, cases[i].padding.h, cases[i].dilation.h),
            cases[i].name);
        TEST_ASSERT_TRUE_MESSAGE(
            arm_depthwise_conv_axis_is_valid(
                input.w, filter.w, output.w, cases[i].stride.w, cases[i].padding.w, cases[i].dilation.w),
            cases[i].name);
        TEST_ASSERT_EQUAL_MESSAGE(
            cases[i].expected_match,
            arm_dw_spec_matches(
                &spec, &input, &filter, &output, cases[i].stride, cases[i].padding, cases[i].dilation, 3),
            cases[i].name);
        if (cases[i].expected_match)
        {
            ++output.h;
            TEST_ASSERT_FALSE_MESSAGE(
                arm_dw_spec_matches(
                    &spec, &input, &filter, &output, cases[i].stride, cases[i].padding, cases[i].dilation, 3),
                cases[i].name);
            --output.h;
            ++output.w;
            TEST_ASSERT_FALSE_MESSAGE(
                arm_dw_spec_matches(
                    &spec, &input, &filter, &output, cases[i].stride, cases[i].padding, cases[i].dilation, 3),
                cases[i].name);
        }
    }

    /* Width dilation also contributes to the effective filter; these extents are hand-calculated. */
    const arm_dw_requirements width_dilated = {
        .filter_h = 2,
        .filter_w = 3,
        .multiplier = ARM_DW_MULTIPLIER_ONE,
        .stride_policy = ARM_DW_STRIDE_EXACT,
        .stride = {.h = 2, .w = 2},
        .padding_policy = ARM_DW_PADDING_EXACT,
        .padding = {.h = 1, .w = 0},
        .dilation = {.h = 1, .w = 2},
        .require_input_height_equals_filter = false,
    };
    const cmsis_nn_dims input = {.n = 1, .h = 8, .w = 11, .c = 2};
    const cmsis_nn_dims filter = {.n = 1, .h = 2, .w = 3, .c = 2};
    const cmsis_nn_dims output = {.n = 1, .h = 5, .w = 4, .c = 2};
    TEST_ASSERT_TRUE(arm_dw_spec_matches(&width_dilated,
                                         &input,
                                         &filter,
                                         &output,
                                         width_dilated.stride,
                                         width_dilated.padding,
                                         width_dilated.dilation,
                                         1));
}

/* Cover vectors spanning input channels, output/channel tails, and the old multiplier path.
 * Binary-fraction data keeps the reference exactly representable in both precisions.
 */
void depthwise_small_multiplier_vectors_f16(void)
{
    static const int32_t channel_counts[] = {1, 2, 3, 5, 9};
    static const int32_t multipliers[] = {1, 2, 3, 4, 5, 8, 9};
    static const int32_t output_widths[] = {1, 3, 4, 5, 7};
    enum
    {
        BATCHES = 2,
        INPUT_H = 2,
        MAX_IN_W = 15,
        MAX_IN_C = 9,
        MAX_OUT_C = 81,
        MAX_OUT_W = 7
    };
    static float16_t input[BATCHES * INPUT_H * MAX_IN_W * MAX_IN_C];
    static float16_t kernel[10 * MAX_OUT_C];
    static float16_t bias[MAX_OUT_C];
    static float16_t output_storage[BATCHES * INPUT_H * MAX_OUT_W * MAX_OUT_C + 2];
    float16_t *output = output_storage + 1;

    for (int32_t shape = 0; shape < 2; ++shape)
    {
        const int32_t kernel_h = shape == 0 ? 1 : 2;
        const int32_t kernel_w = shape == 0 ? 9 : 5;
        for (size_t c = 0; c < sizeof(channel_counts) / sizeof(channel_counts[0]); ++c)
        {
            const int32_t in_c = channel_counts[c];
            for (size_t m = 0; m < sizeof(multipliers) / sizeof(multipliers[0]); ++m)
            {
                const int32_t ch_mult = multipliers[m];
                const int32_t out_c = in_c * ch_mult;
                for (size_t w = 0; w < sizeof(output_widths) / sizeof(output_widths[0]); ++w)
                {
                    const int32_t out_w = output_widths[w];
                    const cmsis_nn_dims input_dims = {.n = BATCHES, .h = INPUT_H, .w = out_w + kernel_w - 1, .c = in_c};
                    const cmsis_nn_dims filter_dims = {.n = 1, .h = kernel_h, .w = kernel_w, .c = out_c};
                    const cmsis_nn_dims output_dims = {
                        .n = BATCHES, .h = INPUT_H - kernel_h + 1, .w = out_w, .c = out_c};
                    const cmsis_nn_dw_conv_params_f16 params = {
                        .ch_mult = ch_mult,
                        .stride = {.h = 1, .w = 1},
                        .padding = {.h = 0, .w = 0},
                        .dilation = {.h = 1, .w = 1},
                        .activation = {.min = -0.25f, .max = 0.25f},
                    };
                    const int32_t input_count = BATCHES * INPUT_H * input_dims.w * in_c;
                    const int32_t output_count = BATCHES * output_dims.h * out_w * out_c;
                    for (int32_t i = 0; i < input_count; ++i)
                    {
                        input[i] = (float16_t)((i % 7 - 3) * 0.125f);
                    }
                    for (int32_t i = 0; i < kernel_h * kernel_w * out_c; ++i)
                    {
                        kernel[i] = (float16_t)((i % 5 - 2) * 0.0625f);
                    }
                    for (int32_t i = 0; i < out_c; ++i)
                    {
                        bias[i] = (float16_t)((i % 3 - 1) * 0.5f);
                    }
                    for (int32_t with_bias = 0; with_bias < 2; ++with_bias)
                    {
                        char message[128];
                        snprintf(message,
                                 sizeof(message),
                                 "%dx%d channels=%d multiplier=%d width=%d bias=%d",
                                 (int)kernel_h,
                                 (int)kernel_w,
                                 (int)in_c,
                                 (int)ch_mult,
                                 (int)out_w,
                                 (int)with_bias);
                        for (int32_t i = 0; i < output_count + 2; ++i)
                        {
                            output_storage[i] = (float16_t)99;
                        }
                        TEST_ASSERT_EQUAL_MESSAGE(ARM_CMSIS_NN_SUCCESS,
                                                  arm_depthwise_nhwc_conv_f16(NULL,
                                                                              &params,
                                                                              &input_dims,
                                                                              input,
                                                                              &filter_dims,
                                                                              kernel,
                                                                              NULL,
                                                                              with_bias ? bias : NULL,
                                                                              &output_dims,
                                                                              output),
                                                  message);
                        for (int32_t batch = 0; batch < BATCHES; ++batch)
                        {
                            for (int32_t oy = 0; oy < output_dims.h; ++oy)
                            {
                                for (int32_t ox = 0; ox < out_w; ++ox)
                                {
                                    for (int32_t oc = 0; oc < out_c; ++oc)
                                    {
                                        float expected = with_bias ? (float)bias[oc] : 0.0f;
                                        for (int32_t ky = 0; ky < kernel_h; ++ky)
                                        {
                                            for (int32_t kx = 0; kx < kernel_w; ++kx)
                                            {
                                                const int32_t index =
                                                    ((batch * INPUT_H + oy + ky) * input_dims.w + ox + kx) * in_c +
                                                    oc / ch_mult;
                                                expected += (float)input[index] *
                                                    (float)kernel[(ky * kernel_w + kx) * out_c + oc];
                                            }
                                        }
                                        expected = expected < -0.25f ? -0.25f : expected;
                                        expected = expected > 0.25f ? 0.25f : expected;
                                        const int32_t index = ((batch * output_dims.h + oy) * out_w + ox) * out_c + oc;
                                        TEST_ASSERT_EQUAL_FLOAT_MESSAGE(expected, (float)output[index], message);
                                    }
                                }
                            }
                        }
                        TEST_ASSERT_EQUAL_FLOAT_MESSAGE(99.0f, (float)output_storage[0], message);
                        TEST_ASSERT_EQUAL_FLOAT_MESSAGE(99.0f, (float)output_storage[output_count + 1], message);
                    }
                }
            }
        }
    }
}

/* Scratch sizing and execution must agree on every specialized family and nearby fallbacks. */
void depthwise_specialized_scratch_f16(void)
{
    static const struct
    {
        const char *name;
        int32_t filter_h;
        int32_t filter_w;
        cmsis_nn_dims input;
        int32_t multiplier;
        cmsis_nn_tile stride;
        cmsis_nn_tile padding;
        bool specialized;
    } cases[] = {
        {"k3 rows/batches", 1, 3, {.n = 2, .h = 3, .w = 11, .c = 5}, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, true},
        {"k9 64 channels", 1, 9, {.n = 1, .h = 1, .w = 9, .c = 64}, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, true},
        {"2x5", 2, 5, {.n = 2, .h = 2, .w = 11, .c = 5}, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, true},
        {"3x3 stride/padding", 3, 3, {.n = 2, .h = 7, .w = 11, .c = 5}, 1, {.h = 2, .w = 2}, {.h = 1, .w = 1}, true},
        {"k9 before conversion", 1, 9, {.n = 2, .h = 1, .w = 11, .c = 1}, 16, {.h = 1, .w = 1}, {.h = 0, .w = 0}, true},
        {"2x5 before conversion",
         2,
         5,
         {.n = 2, .h = 2, .w = 11, .c = 1},
         16,
         {.h = 1, .w = 1},
         {.h = 0, .w = 0},
         true},
        {"k9 stride fallback", 1, 9, {.n = 2, .h = 3, .w = 11, .c = 5}, 1, {.h = 2, .w = 2}, {.h = 0, .w = 0}, false},
        {"k9 padding fallback", 1, 9, {.n = 2, .h = 3, .w = 11, .c = 5}, 1, {.h = 1, .w = 1}, {.h = 1, .w = 1}, false},
        {"2x5 height fallback", 2, 5, {.n = 2, .h = 3, .w = 11, .c = 5}, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, false},
        {"unregistered 2x2", 2, 2, {.n = 2, .h = 3, .w = 11, .c = 5}, 1, {.h = 1, .w = 1}, {.h = 0, .w = 0}, false},
    };
    static float16_t input[1024];
    static float16_t kernel[1024];
    static float16_t output[2048];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        const cmsis_nn_dims *in = &cases[i].input;
        const cmsis_nn_dw_conv_params_f16 params = {
            .ch_mult = cases[i].multiplier,
            .stride = cases[i].stride,
            .padding = cases[i].padding,
            .dilation = {.h = 1, .w = 1},
            .activation = {.min = -100, .max = 100},
        };
        const cmsis_nn_dims filter = {
            .n = 1, .h = cases[i].filter_h, .w = cases[i].filter_w, .c = in->c * params.ch_mult};
        const cmsis_nn_dims out = {
            .n = in->n,
            .h = (in->h + 2 * params.padding.h - filter.h) / params.stride.h + 1,
            .w = (in->w + 2 * params.padding.w - filter.w) / params.stride.w + 1,
            .c = filter.c,
        };
        const int32_t size = arm_depthwise_conv_f16_get_buffer_size(&params, in, &filter, &out, ARM_NN_LAYOUT_NHWC);
        TEST_ASSERT_EQUAL_INT_MESSAGE(
            size, arm_depthwise_conv_wrapper_f16_get_buffer_size(&params, in, &filter, &out), cases[i].name);
        bool specialized = cases[i].specialized;
#ifdef NN_DISABLE_SPECIALIZATION
        specialized = false;
#endif
        if (specialized)
        {
            TEST_ASSERT_EQUAL_INT_MESSAGE(0, size, cases[i].name);
        }
        else if (params.ch_mult == 1)
        {
            /* Fallback NT-T uses four packed output rows; k9/64 requires 9216 F32 or 4608 F16 bytes. */
            TEST_ASSERT_EQUAL_INT_MESSAGE(
                4 * filter.h * filter.w * in->c * (int32_t)sizeof(float16_t), size, cases[i].name);
        }
        else
        {
#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
            /* With specialization disabled, the one-input-channel conversion needs workspace. */
            TEST_ASSERT_GREATER_THAN_INT_MESSAGE(0, size, cases[i].name);
#else
            TEST_ASSERT_EQUAL_INT_MESSAGE(0, size, cases[i].name);
#endif
        }
        const int32_t input_count = in->n * in->h * in->w * in->c;
        const int32_t output_count = out.n * out.h * out.w * out.c;
        TEST_ASSERT_LESS_OR_EQUAL_INT(1024, input_count);
        TEST_ASSERT_LESS_OR_EQUAL_INT(1024, filter.h * filter.w * out.c);
        TEST_ASSERT_LESS_OR_EQUAL_INT(2048, output_count);
        for (int32_t j = 0; j < input_count; ++j)
        {
            input[j] = 1;
        }
        for (int32_t j = 0; j < filter.h * filter.w * out.c; ++j)
        {
            kernel[j] = 1;
        }
        for (int32_t j = 0; j < output_count; ++j)
        {
            output[j] = -99;
        }
        TEST_ASSERT_EQUAL_MESSAGE(
            ARM_CMSIS_NN_SUCCESS,
            arm_depthwise_conv_wrapper_f16(NULL, &params, in, input, &filter, kernel, NULL, NULL, &out, output),
            cases[i].name);
        for (int32_t j = 0; j < output_count; ++j)
        {
            const int32_t ox = (j / out.c) % out.w;
            const int32_t oy = (j / (out.c * out.w)) % out.h;
            float expected = 0;
            for (int32_t ky = 0; ky < filter.h; ++ky)
            {
                const int32_t iy = oy * params.stride.h - params.padding.h + ky;
                for (int32_t kx = 0; kx < filter.w; ++kx)
                {
                    const int32_t ix = ox * params.stride.w - params.padding.w + kx;
                    if (iy >= 0 && iy < in->h && ix >= 0 && ix < in->w)
                    {
                        expected += 1;
                    }
                }
            }
            TEST_ASSERT_EQUAL_FLOAT_MESSAGE(expected, (float)output[j], cases[i].name);
        }
    }

    /* Malformed query geometry must not reach the specialization matcher's int32 arithmetic. */
    const cmsis_nn_dims in = {.n = 1, .h = 1, .w = 9, .c = 64};
    const cmsis_nn_dims filter = {.n = 1, .h = 1, .w = 9, .c = 64};
    cmsis_nn_dims out = {.n = 1, .h = 1, .w = 1, .c = 64};
    cmsis_nn_dw_conv_params_f16 params = {
        .ch_mult = 1, .stride = {.h = 1, .w = 1}, .padding = {.h = 0, .w = 0}, .dilation = {.h = 1, .w = 1}};
    params.padding.w = INT32_MAX;
    TEST_ASSERT_GREATER_THAN_INT(0, arm_depthwise_conv_wrapper_f16_get_buffer_size(&params, &in, &filter, &out));
    params.padding.w = 0;
    params.stride.w = 0;
    TEST_ASSERT_GREATER_THAN_INT(0, arm_depthwise_conv_wrapper_f16_get_buffer_size(&params, &in, &filter, &out));
    params.stride.w = 1;
    out.n = 2;
    TEST_ASSERT_GREATER_THAN_INT(0, arm_depthwise_conv_wrapper_f16_get_buffer_size(&params, &in, &filter, &out));
    out.n = 1;
    out.w = 2;
    TEST_ASSERT_GREATER_THAN_INT(0, arm_depthwise_conv_wrapper_f16_get_buffer_size(&params, &in, &filter, &out));
    TEST_ASSERT_EQUAL_INT(0, arm_depthwise_conv_wrapper_f16_get_buffer_size(NULL, &in, &filter, &out));
}
