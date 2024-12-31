#include "common.h"
#ifdef FUNCTION_PROFILE
#include "functable.h"
#endif

#if defined(RISCV_SIMD)
#if !defined(DOUBLE)
#define VSETVL(n)               __riscv_vsetvl_e32m8(n)
#define FLOAT_V_T               vfloat32m8_t
#define VLSEV_FLOAT             __riscv_vlse32_v_f32m8
#define VSSEV_FLOAT             __riscv_vsse32_v_f32m8
#define VFMACCVF_FLOAT          __riscv_vfmacc_vf_f32m8
#define VFMULVF_FLOAT           __riscv_vfmul_vf_f32m8
#define VFMSACVF_FLOAT          __riscv_vfmsac_vf_f32m8
#else
#define VSETVL(n)               __riscv_vsetvl_e64m8(n)
#define FLOAT_V_T               vfloat64m8_t
#define VLSEV_FLOAT             __riscv_vlse64_v_f64m8
#define VSSEV_FLOAT             __riscv_vsse64_v_f64m8
#define VFMACCVF_FLOAT          __riscv_vfmacc_vf_f64m8
#define VFMULVF_FLOAT           __riscv_vfmul_vf_f64m8
#define VFMSACVF_FLOAT          __riscv_vfmsac_vf_f64m8
#endif
#endif

#ifndef CBLAS

void NAME(blasint *N, FLOAT *dx, blasint *INCX, FLOAT *dy, blasint *INCY, FLOAT *dparam){

  blasint n = *N;
  blasint incx = *INCX;
  blasint incy = *INCY;

#else

void CNAME(blasint n, FLOAT *dx, blasint incx, FLOAT *dy, blasint incy, FLOAT *dparam){

#endif

  blasint i__1, i__2;

  blasint i__;
  FLOAT w, z__;
  blasint kx, ky;
  FLOAT dh11, dh12, dh22, dh21, dflag;
  blasint nsteps;

#if defined(RISCV_SIMD)
  FLOAT_V_T v_w, v_z__, v_dx, v_dy;
  blasint stride, stride_x, stride_y, offset;
#endif

#ifndef CBLAS
  PRINT_DEBUG_CNAME;
#else
  PRINT_DEBUG_CNAME;
#endif

  --dparam;
  --dy;
  --dx;

  dflag = dparam[1];
    if (n <= 0 || dflag == - 2.0) goto L140;

    if (! (incx == incy && incx > 0)) goto L70;

    nsteps = n * incx;
    if (dflag < 0.) {
	goto L50;
    } else if (dflag == 0) {
	goto L10;
    } else {
	goto L30;
    }
L10:
    dh12 = dparam[4];
    dh21 = dparam[3];
    i__1 = nsteps;
    i__2 = incx;
    #if !defined(RISCV_SIMD)
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	w = dx[i__];
	z__ = dy[i__];
	dx[i__] = w + z__ * dh12;
	dy[i__] = w * dh21 + z__;
/* L20: */
    }
    #else
    if(i__2 < 0){
        offset = i__1 - 2;
        dx += offset;
        dy += offset;
        i__1 = -i__1;
        i__2 = -i__2;
    }
    stride = i__2 * sizeof(FLOAT);
    n = i__1 / i__2;
    for (size_t vl; n > 0; n -= vl, dx += vl*i__2, dy += vl*i__2) {
        vl = VSETVL(n);

        v_w = VLSEV_FLOAT(&dx[1], stride, vl);
        v_z__ = VLSEV_FLOAT(&dy[1], stride, vl);

        v_dx = VFMACCVF_FLOAT(v_w, dh12, v_z__, vl);
        v_dy = VFMACCVF_FLOAT(v_z__, dh21, v_w, vl);

        VSSEV_FLOAT(&dx[1], stride, v_dx, vl);
        VSSEV_FLOAT(&dy[1], stride, v_dy, vl);
    }
    #endif
    goto L140;
L30:
    dh11 = dparam[2];
    dh22 = dparam[5];
    i__2 = nsteps;
    i__1 = incx;
    #if !defined(RISCV_SIMD)
    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
	w = dx[i__];
	z__ = dy[i__];
	dx[i__] = w * dh11 + z__;
	dy[i__] = -w + dh22 * z__;
/* L40: */
    }
    #else
    if(i__1 < 0){
        offset = i__2 - 2;
        dx += offset;
        dy += offset;
        i__1 = -i__1;
        i__2 = -i__2;
    }
    stride = i__1 * sizeof(FLOAT);
    n = i__2  / i__1;
    for (size_t vl; n > 0; n -= vl, dx += vl*i__1, dy += vl*i__1) {
        vl = VSETVL(n);

        v_w = VLSEV_FLOAT(&dx[1], stride, vl);
        v_z__ = VLSEV_FLOAT(&dy[1], stride, vl);

        v_dx = VFMACCVF_FLOAT(v_z__, dh11, v_w, vl);
        v_dy = VFMSACVF_FLOAT(v_w, dh22, v_z__, vl);

        VSSEV_FLOAT(&dx[1], stride, v_dx, vl);
        VSSEV_FLOAT(&dy[1], stride, v_dy, vl);
    }
    #endif
    goto L140;
L50:
    dh11 = dparam[2];
    dh12 = dparam[4];
    dh21 = dparam[3];
    dh22 = dparam[5];
    i__1 = nsteps;
    i__2 = incx;
    #if !defined(RISCV_SIMD)
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	w = dx[i__];
	z__ = dy[i__];
	dx[i__] = w * dh11 + z__ * dh12;
	dy[i__] = w * dh21 + z__ * dh22;
/* L60: */
    }
    #else
    if(i__2 < 0){
        offset = i__1 - 2;
        dx += offset;
        dy += offset;
        i__1 = -i__1;
        i__2 = -i__2;
    }
    stride = i__2 * sizeof(FLOAT);
    n = i__1 / i__2;
    for (size_t vl; n > 0; n -= vl, dx += vl*i__2, dy += vl*i__2) {
        vl = VSETVL(n);

        v_w = VLSEV_FLOAT(&dx[1], stride, vl);
        v_z__ = VLSEV_FLOAT(&dy[1], stride, vl);

        v_dx = VFMULVF_FLOAT(v_w, dh11, vl);
        v_dx = VFMACCVF_FLOAT(v_dx, dh12, v_z__, vl);
        VSSEV_FLOAT(&dx[1], stride, v_dx, vl);

        v_dy = VFMULVF_FLOAT(v_w, dh21, vl);
        v_dy = VFMACCVF_FLOAT(v_dy, dh22, v_z__, vl);
        VSSEV_FLOAT(&dy[1], stride, v_dy, vl);
    }
    #endif
    goto L140;
L70:
    kx = 1;
    ky = 1;
    if (incx < 0) {
	kx = (1 - n) * incx + 1;
    }
    if (incy < 0) {
	ky = (1 - n) * incy + 1;
    }

    if (dflag < 0.) {
	goto L120;
    } else if (dflag == 0) {
	goto L80;
    } else {
	goto L100;
    }
L80:
    dh12 = dparam[4];
    dh21 = dparam[3];
    i__2 = n;
    #if !defined(RISCV_SIMD)
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = dx[kx];
	z__ = dy[ky];
	dx[kx] = w + z__ * dh12;
	dy[ky] = w * dh21 + z__;
	kx += incx;
	ky += incy;
/* L90: */
    }
    #else
    if(incx < 0){
        incx = -incx;
        dx -= n*incx;
    }
    if(incy < 0){
        incy = -incy;
        dy -= n*incy;
    }
    stride_x = incx * sizeof(FLOAT);
    stride_y = incy * sizeof(FLOAT);
    for (size_t vl; n > 0; n -= vl, dx += vl*incx, dy += vl*incy) {
        vl = VSETVL(n);

        v_w = VLSEV_FLOAT(&dx[kx], stride_x, vl);
        v_z__ = VLSEV_FLOAT(&dy[ky], stride_y, vl);

        v_dx = VFMACCVF_FLOAT(v_w, dh12, v_z__, vl);
        v_dy = VFMACCVF_FLOAT(v_z__, dh21, v_w, vl);

        VSSEV_FLOAT(&dx[kx], stride_x, v_dx, vl);
        VSSEV_FLOAT(&dy[ky], stride_y, v_dy, vl);
    }
    #endif
    goto L140;
L100:
    dh11 = dparam[2];
    dh22 = dparam[5];
    i__2 = n;
    #if !defined(RISCV_SIMD)
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = dx[kx];
	z__ = dy[ky];
	dx[kx] = w * dh11 + z__;
	dy[ky] = -w + dh22 * z__;
	kx += incx;
	ky += incy;
/* L110: */
    }
    #else
    if(incx < 0){
        incx = -incx;
        dx -= n*incx;
    }
    if(incy < 0){
        incy = -incy;
        dy -= n*incy;
    }
    stride_x = incx * sizeof(FLOAT);
    stride_y = incy * sizeof(FLOAT);
    for (size_t vl; n > 0; n -= vl, dx += vl*incx, dy += vl*incy) {
        vl = VSETVL(n);

        v_w = VLSEV_FLOAT(&dx[kx], stride_x, vl);
        v_z__ = VLSEV_FLOAT(&dy[ky], stride_y, vl);

        v_dx = VFMACCVF_FLOAT(v_z__, dh11, v_w, vl);
        v_dy = VFMSACVF_FLOAT(v_w, dh22, v_z__, vl);

        VSSEV_FLOAT(&dx[kx], stride_x, v_dx, vl);
        VSSEV_FLOAT(&dy[ky], stride_y, v_dy, vl);
    }
    #endif
    goto L140;
L120:
    #if !defined(RISCV_SIMD)
    dh11 = dparam[2];
    dh12 = dparam[4];
    dh21 = dparam[3];
    dh22 = dparam[5];
    i__2 = n;
    for (i__ = 1; i__ <= i__2; ++i__) {
	w = dx[kx];
	z__ = dy[ky];
	dx[kx] = w * dh11 + z__ * dh12;
	dy[ky] = w * dh21 + z__ * dh22;
	kx += incx;
	ky += incy;
/* L130: */
    }
    #else
    if(incx < 0){
        incx = -incx;
        dx -= n*incx;
    }
    if(incy < 0){
        incy = -incy;
        dy -= n*incy;
    }
    stride_x = incx * sizeof(FLOAT);
    stride_y = incy * sizeof(FLOAT);
    for (size_t vl; n > 0; n -= vl, dx += vl*incx, dy += vl*incy) {
        vl = VSETVL(n);

        v_w = VLSEV_FLOAT(&dx[kx], stride_x, vl);
        v_z__ = VLSEV_FLOAT(&dy[ky], stride_y, vl);

        v_dx = VFMULVF_FLOAT(v_w, dh11, vl);
        v_dx = VFMACCVF_FLOAT(v_dx, dh12, v_z__, vl);
        VSSEV_FLOAT(&dx[kx], stride_x, v_dx, vl);

        v_dy = VFMULVF_FLOAT(v_w, dh21, vl);
        v_dy = VFMACCVF_FLOAT(v_dy, dh22, v_z__, vl);
        VSSEV_FLOAT(&dy[ky], stride_y, v_dy, vl);
    }
    #endif
L140:
    return;
}

