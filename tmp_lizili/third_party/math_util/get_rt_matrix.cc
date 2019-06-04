/*

 IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install,
 copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library

Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
Copyright (C) 2009, Willow Garage Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistribution's of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistribution's in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * The name of the copyright holders may not be used to endorse or promote products
    derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

*/

#include "third_party/math_util/get_rt_matrix.h"

namespace roadstar {
namespace third_party {

void icvGetRTMatrix(const cv::AutoBuffer<CvPoint2D32f> &a,
                    const cv::AutoBuffer<CvPoint2D32f> &b, 
                    const int count, CvMat* M, int full_affine) {
  if (full_affine) {
    double sa[36], sb[6];
    CvMat A = cvMat(6, 6, CV_64F, sa), B = cvMat(6, 1, CV_64F, sb);
    CvMat MM = cvMat(6, 1, CV_64F, M->data.db);

    int i;

    memset(sa, 0, sizeof(sa));
    memset(sb, 0, sizeof(sb));

    for (i = 0; i < count; i++) {
      sa[0] += a[i].x*a[i].x;
      sa[1] += a[i].y*a[i].x;
      sa[2] += a[i].x;

      sa[6] += a[i].x*a[i].y;
      sa[7] += a[i].y*a[i].y;
      sa[8] += a[i].y;

      sa[12] += a[i].x;
      sa[13] += a[i].y;
      sa[14] += 1;

      sb[0] += a[i].x*b[i].x;
      sb[1] += a[i].y*b[i].x;
      sb[2] += b[i].x;
      sb[3] += a[i].x*b[i].y;
      sb[4] += a[i].y*b[i].y;
      sb[5] += b[i].y;
    }

    sa[21] = sa[0];
    sa[22] = sa[1];
    sa[23] = sa[2];
    sa[27] = sa[6];
    sa[28] = sa[7];
    sa[29] = sa[8];
    sa[33] = sa[12];
    sa[34] = sa[13];
    sa[35] = sa[14];
    cvSolve(&A, &B, &MM, CV_SVD);
  } else {
    double sa[16], sb[4], m[4], *om = M->data.db;
    CvMat A = cvMat(4, 4, CV_64F, sa), B = cvMat(4, 1, CV_64F, sb);
    CvMat MM = cvMat(4, 1, CV_64F, m);

    int i;

    memset(sa, 0, sizeof(sa));
    memset(sb, 0, sizeof(sb));

    for (i = 0; i < count; i++) {
      sa[0] += a[i].x*a[i].x + a[i].y*a[i].y;
      sa[1] += 0;
      sa[2] += a[i].x;
      sa[3] += a[i].y;

      sa[4] += 0;
      sa[5] += a[i].x*a[i].x + a[i].y*a[i].y;
      sa[6] += -a[i].y;
      sa[7] += a[i].x;

      sa[8] += a[i].x;
      sa[9] += -a[i].y;
      sa[10] += 1;
      sa[11] += 0;

      sa[12] += a[i].y;
      sa[13] += a[i].x;
      sa[14] += 0;
      sa[15] += 1;

      sb[0] += a[i].x*b[i].x + a[i].y*b[i].y;
      sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
      sb[2] += b[i].x;
      sb[3] += b[i].y;
    }

    cvSolve(&A, &B, &MM, CV_SVD);

    om[0] = om[4] = m[0];
    om[1] = -m[1];
    om[3] = m[1];
    om[2] = m[2];
    om[5] = m[3];
  }
}

}
}