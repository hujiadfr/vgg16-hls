#include <ap_int.h>
#include <iostream>

#define K 2
#define Kx 3
#define Ky 3


/*max_ap:
 * Input: four elements
 * To do 2x2 max pool, choose the maximum of four elements
 * Output: the max value
 *
 * */
ap_int<25> max_ap(ap_int<25> a, ap_int<25> b, ap_int<25> c, ap_int<25> d){
	if(a>=b && a>=c && a>=d)
		return a;
	if(b>=a && b>=c && b>=d)
		return b;
	if(c>=a && c>=b && c>=d)
		return c;
	return d;
}

extern "C"{

/* Conv: Convolution kernel
 * Input: feature_in: the input of our image
 * 		  W: weight of our convolution kernel
 * 		  scale: scale factor to do quantization
 * 		  Chin: number of input channel
 * 		  Hin: Height of input image
 * 		  Win: Width of input image
 * 		  CHout: number of input channel
 * 		  Sx: Step in x axis
 * 		  Sy: Step in y axis
 * 		  mode: padding mode, valid or same
 * 		  relu_en: flag of enable relu activation function
 * 		  pool_en: flag of enable max pool
 * 		  layer: The index of our layer
 * 		  zero_point: zero point for quantization (not used now)
 * 		  invert_flag: flag to control whether invert feature_in and feature_out
 * Output: image after a layer
 *
 */

void Conv(
	ap_int<9> feature_in[],
	ap_int<8> W[],
	int scale[],
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
	int layer,
	int zero_point,
	int invert_flag
	)//mode: 0:VALID, 1:SAME
{
	#pragma HLS INTERFACE s_axilite port=return
	#pragma HLS INTERFACE s_axilite port=Sy
	#pragma HLS INTERFACE s_axilite port=Win
	#pragma HLS INTERFACE s_axilite port=Sx
	#pragma HLS INTERFACE s_axilite port=Hin
	#pragma HLS INTERFACE s_axilite port=relu_en
	#pragma HLS INTERFACE s_axilite port=CHin
	#pragma HLS INTERFACE s_axilite port=mode
	#pragma HLS INTERFACE s_axilite port=CHout
	#pragma HLS INTERFACE m_axi depth=524288 port=feature_in offset=slave bundle=bus1
	#pragma HLS INTERFACE m_axi depth=2359296 port=W offset=slave bundle=bus2
	#pragma HLS INTERFACE m_axi depth=6656 port=scale offset=slave bundle=bus3
	#pragma HLS INTERFACE m_axi depth=524288 port=feature_out offset=slave bundle=bus4


	unsigned int pad_x,pad_y;
	unsigned int out_truncate;

	ap_int<9> in_buf[512][32][32];
	ap_int<25> sum2[512][32][32];
	ap_int<8> W_buf[512][512][3][3]; // max W is 512 IN_CH 512 OUT_CH Kx, Ky
	ap_int<25> sum2_buf;

	#pragma HLS array_partition variable=W_buf block factor=128 dim=1

	ap_int<9> output_buf[512][32][32];

	#pragma HLS array_partition variable=sum2 block factor=128 dim=1
	#pragma HLS array_partition variable=output_buf block factor=64 dim=1

	// calculate padding for valid mode and zero mode
	if(mode==0)
	{
		pad_x=0;pad_y=0;
	}
	else
	{
		pad_x=(Kx-1)/2;pad_y=(Ky-1)/2;
	}
	unsigned int Hout,Wout;
	Wout=(Win+2*pad_x-Kx)/Sx+1;
	Hout=(Hin+2*pad_y-Ky)/Sy+1;

	/*prefetch data from ddr to buf*/
	for(unsigned int cin=0;cin<CHin;cin=cin+1){
		for(unsigned int i=0;i<Hin;i++)
		{
			for(unsigned int j=0; j<Win; j++)
				{
					if(!invert_flag)
						in_buf[cin][i][j]=feature_in[cin*Hin*Win+i*Win+j];
					else
						in_buf[cin][i][j]=feature_out[cin*Hin*Win+i*Win+j];
				}
		}
	}
	for (unsigned int cout = 0; cout < CHout; cout++)
	{
		for (unsigned int cin = 0; cin < CHin; cin++)
		{
			for (unsigned int i = 0; i < Ky; i++)
			{
				for (unsigned int j = 0; j < Kx; j++)

				{
					W_buf[cout][cin][i][j] = W[layer*512*Ky*512*Kx + cout*CHin*Kx*Ky + cin*Kx*Ky + i*Kx + j];
				}
			}
		}
	}

	//reset the sum2 buf for our convolution calculation
	for(unsigned int i=0; i<512; i++)
	{
		for(unsigned int j=0; j<32; j++)
			for(unsigned int m=0; m<32; m++)
				sum2[i][j][m] = 0;
	}


	//Convolution
	LOOP_ii:for(unsigned int ii=0;ii<Ky;ii++)
	{
		LOOP_jj:for(unsigned int jj=0;jj<Kx;jj++)
		{
			LOOP_i:for(unsigned int i=0;i<32;i++)
			{
				LOOP_j:for(unsigned int j=0;j<32;j++)
				{
					LOOP_cout:for(unsigned int cout=0;cout<512;cout=cout+1) // a chunk have channel 64
					{
						#pragma HLS unroll factor=128
						#pragma HLS LOOP_TRIPCOUNT min=5 max=5 avg=5
						sum2_buf = sum2[cout][i][j];
						LOOP_cin:for(unsigned int cin=0;cin<512;cin=cin+1)
						{
							if(cin >= CHin || cout >= CHout || j >= Wout || i >= Hout)
								break;
							#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
							int  h=i*Sy-pad_y+ii;
							int  w=j*Sx-pad_x+jj;

							ap_int<9>  dat;
							ap_int<8>  wt;
							if(h>=0 && w>=0 && h<Hin && w<Win)
							{
								dat = in_buf[cin][h][w];
								wt = W_buf[cout][cin][ii][jj];
							}
							else
							{
								dat=0;
								wt=0;
							}
							sum2_buf+=dat*wt;
						}
						sum2[cout][i][j] = sum2_buf;
					}
				}
			}
		}
	}
	//if pool enable, divide Wout and Hout by 2
	unsigned int Hout2,Wout2;
	if(pool_en){
		Wout2 = Wout/K;
		Hout2 = Hout/K;
	}
	else
	{
		Wout2 = Wout;
		Hout2 = Hout;
	}
	//  for max-pooling
	LOOP3_i:for(unsigned int i=0;i<32;i++)
	{
		LOOP3_j:for(unsigned int j=0;j<32; j++)
		{
			LOOP3_cout:for(unsigned int cout=0;cout<512;cout=cout+1)
			{
				#pragma HLS unroll factor=64
				if(cout>=CHout || j>=Wout2 || i>=Hout2)
					break;
				ap_int<25> temp;
				temp = 0;
				if(pool_en)
				{
					temp= max_ap(sum2[cout][2*i][2*j], sum2[cout][2*i+1][2*j], sum2[cout][2*i][2*j+1], sum2[cout][2*i+1][2*j+1]); // choose max value in a 2x2
					if(relu_en && temp<0) //relu activation
						temp=0;
					temp = temp/scale[layer*512+cout]; //do quantization
					if (temp>255)
						temp = 255;
					output_buf[cout][i][j] = (ap_int<9>)temp;
				}
				else{
					temp = sum2[cout][i][j];
					if(relu_en && temp<0) //relu activation
						temp = 0;
					temp = temp/scale[layer*512+cout]; //do quantization
					if (temp>255)
						temp = 255;
					output_buf[cout][i][j] = (ap_int<9>)temp;
				}
			}
		}
	}
	//load data from buf to our ddr
	LOOP_OUTPUT_CH: for(unsigned int cout=0; cout<CHout; cout=cout+1)
		LOOP_OUTPUT_i: for(unsigned int i=0; i<Hout2; ++i)
			LOOP_OUTPUT_j: for(unsigned int j=0; j<Wout2; ++j)
			{
				if(!invert_flag)
				{
					feature_out[cout*Wout2*Hout2+i*Wout2+j] = output_buf[cout][i][j];
				}
				else
				{
					feature_in[cout*Wout2*Hout2+i*Wout2+j] = output_buf[cout][i][j];
				}
			}
	}
}



