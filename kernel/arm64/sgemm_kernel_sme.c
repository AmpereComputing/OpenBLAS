/***************************************************************************
Copyright (c) 2024, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include <arm_sme.h>

#include "common.h"
#include "sme_abi.h"

// Outer product kernel.
// Computes a 2SVL x 2SVL block of C, utilizing all four FP32 tiles of ZA.
// This kernel is unpredicated, and assumes a full 2SVL x 2SVL block.
__attribute__((always_inline)) inline void
kernel_2x2(const float *A, const float *B, float *C, float alpha,
           size_t shared_dim, size_t a_step, size_t b_step, size_t c_step)
    __arm_out("za") __arm_streaming {
  const size_t svl = svcntw();

  // Predicate set-up
  svbool_t ptrue = svptrue_b32();

  // Load from C into ZA
  for (size_t i = 0; i < (svl >> 1); i++) {
    svld1_ver_za32(0, i, ptrue, &C[0 * svl + i * c_step]);
    svld1_ver_za32(1, i, ptrue, &C[1 * svl + i * c_step]);
    svld1_ver_za32(2, i, ptrue, &C[0 * svl + (i + svl) * c_step]);
    svld1_ver_za32(3, i, ptrue, &C[1 * svl + (i + svl) * c_step]);
  }

  svfloat32_t alpha_vec = svdup_f32(alpha);

  // Iterate through shared dimension (K)
  for (size_t k = 0; k < shared_dim; k++) {
    // Load column of A
    svfloat32x2_t cols_a = svld1_x2(svptrue_c32(), &A[k * a_step]);

    // Load row of B
    svfloat32x2_t rows_b = svld1_x2(svptrue_c32(), &B[k * b_step]);

    // Multiply B through by alpha
    svfloat32_t row_b_0 = svmul_x(ptrue, alpha_vec, svget2(rows_b, 0));
    svfloat32_t row_b_1 = svmul_x(ptrue, alpha_vec, svget2(rows_b, 1));

    // Perform outer products
    svmopa_za32_m(0, ptrue, ptrue, svget2(cols_a, 0), row_b_0);
    svmopa_za32_m(1, ptrue, ptrue, svget2(cols_a, 1), row_b_0);
    svmopa_za32_m(2, ptrue, ptrue, svget2(cols_a, 0), row_b_1);
    svmopa_za32_m(3, ptrue, ptrue, svget2(cols_a, 1), row_b_1);
  }

  // Store out to C from ZA
  for (size_t i = 0; i < (svl >> 1); i++) {
    // Store out one row of C per tile
    svst1_ver_za32(0, i, ptrue, &C[0 * svl + i * c_step]);
    svst1_ver_za32(1, i, ptrue, &C[1 * svl + i * c_step]);
    svst1_ver_za32(2, i, ptrue, &C[0 * svl + (i + svl) * c_step]);
    svst1_ver_za32(3, i, ptrue, &C[1 * svl + (i + svl) * c_step]);
  }
}

// Outer product kernel.
// Computes an SVL x SVL block of C, utilizing a single FP32 tile of ZA (ZA0).
// This kernel is predicated, and can handle under-filled blocks.
__attribute__((always_inline)) inline void
kernel_1x1(const float *A, const float *B, float *C, float alpha,
           size_t shared_dim, size_t a_len, size_t a_step, size_t b_len,
           size_t b_step, size_t c_step, size_t c_rows, size_t c_cols)
    __arm_out("za") __arm_streaming {

  // Predicate set-up
  svbool_t pg = svptrue_b32();
  svbool_t pg_a = svwhilelt_b32_u64(0, a_len);
  svbool_t pg_b = svwhilelt_b32_u64(0, b_len);
  svbool_t pg_c = svwhilelt_b32_u64(0, c_rows);

  // Load from C into ZA
  for (size_t i = 0; i < c_cols; i++) {
    svld1_ver_za32(0, i, pg_c, &C[i * c_step]);
  }

  svfloat32_t alpha_vec = svdup_f32_z(pg_b, alpha);

  // Iterate through shared dimension (K)
  for (size_t k = 0; k < shared_dim; k++) {
    // Load column of A
    svfloat32_t col_a = svld1(pg_a, &A[k * a_step]);
    // Load row of B
    svfloat32_t row_b = svld1(pg_b, &B[k * b_step]);
    // Multiply B through by alpha
    row_b = svmul_x(pg_b, alpha_vec, row_b);
    // Perform outer product
    svmopa_za32_m(0, pg, pg, col_a, row_b);
  }

  // Store out to C from ZA
  for (size_t i = 0; i < c_cols; i++) {
    svst1_ver_za32(0, i, pg_c, &C[i * c_step]);
  }
}

__arm_new("za") __arm_locally_streaming
    int CNAME(BLASLONG bm, BLASLONG bn, BLASLONG bk, FLOAT alpha0, FLOAT *ba,
              FLOAT *bb, FLOAT *C, BLASLONG ldc) {

  const BLASLONG num_rows = bm;
  const BLASLONG num_cols = bn;

  const FLOAT *a_ptr = ba;
  const FLOAT *b_ptr = bb;
  FLOAT *c_ptr = C;

  const BLASLONG svl = svcntw();

  const BLASLONG a_step = bm;
  const BLASLONG b_step = bn;
  const BLASLONG c_step = ldc;

  // Block over rows of C (panels of A)
  BLASLONG row_idx = 0;

  // 2x2 loop
  BLASLONG row_batch = 2 * svl;

  // Block over row dimension of C
  for (; row_idx + row_batch <= num_rows; row_idx += row_batch) {
    BLASLONG col_idx = 0;
    BLASLONG col_batch = 2 * svl;

    // Block over column dimension of C
    for (; col_idx + col_batch <= num_cols; col_idx += col_batch) {
      kernel_2x2(&a_ptr[row_idx], &b_ptr[col_idx],
                 &c_ptr[row_idx + col_idx * c_step], alpha0, bk, a_step, b_step,
                 c_step);
    }

    // Handle under-filled blocks w/ 2x(1x1) kernels
    col_batch = 1 * svl;
    for (; col_idx < num_cols; col_idx += col_batch) {
      col_batch = MIN(col_batch, num_cols - col_idx);

      kernel_1x1(&a_ptr[row_idx], &b_ptr[col_idx],
                 &c_ptr[row_idx + col_idx * c_step], alpha0, bk, svl, a_step,
                 col_batch, b_step, c_step, svl, col_batch);

      kernel_1x1(&a_ptr[row_idx + svl], &b_ptr[col_idx],
                 &c_ptr[(row_idx + svl) + col_idx * c_step], alpha0, bk, svl,
                 a_step, col_batch, b_step, c_step, svl, col_batch);
    }
  }

  // Handle under-filled blocks w/ 1x1 kernels
  row_batch = 1 * svl;
  for (; row_idx < num_rows; row_idx += row_batch) {
    row_batch = MIN(row_batch, num_rows - row_idx);
    // Block over column dimension of C
    BLASLONG col_batch = svl;
    for (BLASLONG col_idx = 0; col_idx < num_cols; col_idx += col_batch) {
      col_batch = MIN(col_batch, num_cols - col_idx);
      kernel_1x1(&a_ptr[row_idx], &b_ptr[col_idx],
                 &c_ptr[row_idx + col_idx * c_step], alpha0, bk, row_batch,
                 a_step, col_batch, b_step, c_step, row_batch, col_batch);
    }
  }
  return 0;
}
