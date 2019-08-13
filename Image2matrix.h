#ifndef __IMAGE__COLM__H__
#define __IMAGE__COLM__H__

#include <stdio.h>

typedef int data_type;

extern void im2col_cpu(data_type* data_im, int channels, int height, int width,
		 int ksize,  int stride, int pad, data_type* data_col);

#endif
