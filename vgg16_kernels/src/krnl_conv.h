#ifndef __CONV_CORE_H__
#define __CONV_CORE_H__

#include <ap_int.h>
#include <iostream>

using namespace std;

#define K 2
#define Kx 3
#define Ky 3
extern "C"{
void Conv(
		ap_int<9> feature_in[],
		ap_int<8> W[],
		ap_int<9> feature_out[],
		unsigned int CHin,
		unsigned int Hin,
		unsigned int Win,
		unsigned int CHout,
		unsigned int Sx,
		unsigned int Sy,
		unsigned int mode,
		unsigned int relu_en,
		unsigned int pool_en,
		int scale,
		int offset,
		int layer,
		int invert_flag
	);//mode: 0:VALID, 1:SAME
}

#endif

