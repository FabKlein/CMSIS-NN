/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* Picolibc streams for the standalone Clang FVP test build. */
#include <stdio.h>

#include "uart.h"

extern __attribute__((noreturn)) void UartEndSimulation(int code);

static int uart_stream_putc(char ch, FILE *stream)
{
    (void)stream;
    uart_putc((unsigned char)ch);
    return 0;
}

static int uart_stream_getc(FILE *stream)
{
    (void)stream;
    return uart_getc();
}

static FILE uart_stream = FDEV_SETUP_STREAM(uart_stream_putc, uart_stream_getc, NULL, _FDEV_SETUP_RW);
FILE *const stdin = &uart_stream;
FILE *const stdout = &uart_stream;
FILE *const stderr = &uart_stream;

__attribute__((noreturn)) void _exit(int code) { UartEndSimulation(code); }
