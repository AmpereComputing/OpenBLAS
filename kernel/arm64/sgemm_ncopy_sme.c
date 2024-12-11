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

// Transpose 1SVL x N panel of A into B
__attribute__((always_inline)) inline static void
transpose_panel(const FLOAT *a, FLOAT *b, BLASLONG rows, BLASLONG cols,
                BLASLONG a_step, BLASLONG b_step)
    __arm_out("za") __arm_streaming {
  BLASLONG col_batch = svcntsw();
  const svbool_t pg_a = svwhilelt_b32_u64(0, rows);

  for (BLASLONG k = 0; k < cols; k += col_batch) {
    col_batch = MIN(col_batch, cols - k);
    for (BLASLONG col = 0; col < col_batch; col++) {
      svld1_ver_za32(0, col, pg_a, &a[(col + k) * a_step]);
    }

    const svbool_t pg_b = svwhilelt_b32_u64(k, cols);
    for (BLASLONG row = 0; row < rows; row++) {
      svst1_hor_za32(0, row, pg_b, &b[row * b_step + k]);
    }
  }
}

__arm_new("za") __arm_locally_streaming
    int CNAME(BLASLONG m, BLASLONG n, FLOAT *a, BLASLONG lda, FLOAT *b) {
  const BLASLONG num_rows = m;
  BLASLONG row_batch = svcntsw();
  for (BLASLONG row_idx = 0; row_idx < num_rows; row_idx += row_batch) {
    // Transpose 1xSVL panel
    row_batch = MIN(row_batch, num_rows - row_idx);
    transpose_panel(&a[row_idx], &b[row_idx * n], row_batch, n, lda, n);
  }
  return 0;
}
