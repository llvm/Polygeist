//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin, zero.lin@amd.com
//    Yao Wang, bitwangyaoyao@gmail.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//

/* Scalar C specialization of OpenCV's public OpenCL morphology kernel.
 * Source: morph.cl MORPH_OP, PROCESS, VAL, and morph's output block;
 * morph.dispatch.cpp 874-880 constructs PROCESS_ELEMS in row-major order.
 * Specializations: one channel, all-ones 3x3 kernel, OP_DILATE, interior ROI.
 * GPU work-items become output-coordinate loops. Shared-memory cached pixel
 * loads become the same input coordinates. The dispatcher-generated row-major
 * list is represented by two fixed-bound loops; no arithmetic is added.
 * PVA is closed-source: this is a public equivalent reference, not PVA source.
 */
#include <stdint.h>
#include <limits.h>
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported reference"
#endif
#define max(a,b) ((a) > (b) ? (a) : (b))
#define MORPH_OP(A, B) max((A), (B))
void NETLIB_SPEC(int h, int w, const REFERENCE_TYPE input[][w],
                REFERENCE_TYPE output[][w]) {
  for (int y = 1; y < h - 1; ++y)
    for (int x = 1; x < w - 1; ++x) {
      REFERENCE_TYPE res = REFERENCE_MIN;
      for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx) {
          REFERENCE_TYPE temp = input[y + ky - 1][x + kx - 1];
          res = MORPH_OP(res, temp);
        }
      output[y][x] = res;
    }
}
