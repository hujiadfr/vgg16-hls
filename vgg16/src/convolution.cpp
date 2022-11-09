/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.cpp
Purpose: VITIS vector addition

*******************************************************************************
Copyright (C) 2019 XILINX, Inc.

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law:
(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX
HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising under
or in connection with these materials, including for any direct, or any indirect,
special, incidental, or consequential loss or damage (including loss of data,
profits, goodwill, or any type of loss or damage suffered as a result of any
action brought by a third party) even if such damage or loss was reasonably
foreseeable or Xilinx had been advised of the possibility of the same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES.

*******************************************************************************/
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "ap_int.h"
#include "vadd.h"
#include "event_timer.hpp"
#include <stdint.h>
#include <stdio.h>


#define K 8
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define X_STRIDE 1
#define Y_STRIDE 1
#define RELU_EN  1
#define MODE     1          //0:VALID, 1:SAME
#define X_PADDING (MODE?(KERNEL_WIDTH-1)/2:0)
#define Y_PADDING (MODE?(KERNEL_HEIGHT-1)/2:0)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

static const int DATA_SIZE = 4096;
static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";


int main(int argc, char* argv[]) {

	// Initialize an event timer we'll use for monitoring the application
	//EventTimer et;
	int IN_CH = 3;
	int IN_HEIGHT = 32;
	int IN_WIDTH = 32;
    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}

    char* xclbinFilename = argv[1];

    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device "
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE;
    }

    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);

    // This call will get the kernel object from program. A kernel is an
    // OpenCL function that is executed on the FPGA.
    cl::Kernel krnl_conv(program, "Conv");

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.

    unsigned int CHin = IN_CH;
	unsigned int Hin = IN_HEIGHT;
	unsigned int Win = IN_WIDTH;
	unsigned int CHout = 64;
	unsigned int Sx = X_STRIDE;
	unsigned int Sy = Y_STRIDE;
	unsigned int mode = MODE;
	unsigned int relu_en = RELU_EN;
	unsigned int pool_en = 0;
	int layer=0;
	int invert_flag = 0;
	int zero_point = 0;


	unsigned int OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
	unsigned int OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}


    size_t size_feature_in = sizeof(ap_int<9>)*512*32*32;
    size_t size_W = sizeof(ap_int<8>)*3*3*512*512*13;
    size_t size_feature_out = sizeof(ap_int<9>)*512*32*32;
    size_t size_scale = sizeof(int)*13*512;
    cl_mem_ext_ptr_t bank0_ext, bank1_ext, bank2_ext, bank3_ext;
	bank0_ext.flags = 0 | XCL_MEM_TOPOLOGY;
	bank0_ext.obj = NULL;
	bank0_ext.param = 0;
	bank1_ext.flags = 1 | XCL_MEM_TOPOLOGY;
	bank1_ext.obj = NULL;
	bank1_ext.param = 0;
	bank2_ext.flags = 2 | XCL_MEM_TOPOLOGY;
	bank2_ext.obj = NULL;
	bank2_ext.param = 0;
	bank3_ext.flags = 3 | XCL_MEM_TOPOLOGY;
	bank3_ext.obj = NULL;
	bank3_ext.param = 0;

    cl::Buffer buffer_feature_in(context, CL_MEM_READ_WRITE|CL_MEM_EXT_PTR_XILINX, size_feature_in, &bank0_ext);
    cl::Buffer buffer_W(context,CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, size_W, &bank1_ext);
    cl::Buffer buffer_scale(context, CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, size_scale, &bank2_ext);
    cl::Buffer buffer_feature_out(context, CL_MEM_READ_WRITE|CL_MEM_EXT_PTR_XILINX,size_feature_out, &bank3_ext);

    //We then need to map our OpenCL buffers to get the pointers
    ap_int<9> *feature_in = (ap_int<9>*)q.enqueueMapBuffer(buffer_feature_in, CL_TRUE , CL_MAP_WRITE , 0, size_feature_in);
    ap_int<8> *W = (ap_int<8>*)q.enqueueMapBuffer(buffer_W, CL_TRUE , CL_MAP_WRITE , 0, size_W);
    ap_int<9> *feature_out = (ap_int<9>*)q.enqueueMapBuffer(buffer_feature_out, CL_TRUE , CL_MAP_READ , 0, size_feature_out);
    int *scale_p = (int*)q.enqueueMapBuffer(buffer_scale, CL_TRUE , CL_MAP_WRITE , 0, size_scale);
    //setting input data

    FILE *f0, *f1, *f2, *f3, *f4, *f5, *f6, *f7, *f8, *f9, *f10, *f11, *f12;
    f0 = fopen("/home/jiaru/conv/0.txt", "r");
	f1 = fopen("/home/jiaru/conv/1.txt", "r");
	f2 = fopen("/home/jiaru/conv/2.txt", "r");
	f3 = fopen("/home/jiaru/conv/3.txt", "r");
	f4 = fopen("/home/jiaru/conv/4.txt", "r");
	f5 = fopen("/home/jiaru/conv/5.txt", "r");
	f6 = fopen("/home/jiaru/conv/6.txt", "r");
	f7 = fopen("/home/jiaru/conv/7.txt", "r");
	f8 = fopen("/home/jiaru/conv/8.txt", "r");
	f9 = fopen("/home/jiaru/conv/9.txt", "r");
	f10 = fopen("/home/jiaru/conv/10.txt", "r");
	f11 = fopen("/home/jiaru/conv/11.txt", "r");
	f12 = fopen("/home/jiaru/conv/12.txt", "r");

	FILE *f_image;
	f_image = fopen("/home/jiaru/conv/test.txt", "r");

    for(int ch=0; ch<3; ch++)
    	for(int i=0; i<32; i++)
    		for(int j=0; j<32; j++){
    			int temp;
    			fscanf(f_image, "%d", &temp);
    			//temp = temp - 128;
    			feature_in[ch*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j] = (ap_int<9>) temp;
    			//cout<<feature_in[ch*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<endl;

    		}



    /*read weight from file*/
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f0,"%d",&temp);
					W[0*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//1-2
	IN_CH=64;
	CHout=64;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f1,"%d",&temp);
					W[1*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//2-1
	IN_CH=64;
	CHout=128;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f2,"%d",&temp);
					W[2*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//2-2
	IN_CH=128;
	CHout=128;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f3,"%d",&temp);
					W[3*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//3-1
	IN_CH=128;
	CHout=256;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f4,"%d",&temp);
					W[4*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//3-2
	IN_CH=256;
	CHout=256;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f5,"%d",&temp);
					W[5*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//3-3
	IN_CH=256;
	CHout=256;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f6,"%d",&temp);
					W[6*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//4-1
	IN_CH=256;
	CHout=512;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f7,"%d",&temp);
					W[7*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//4-2
	IN_CH=512;
	CHout=512;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f8,"%d",&temp);
					W[8*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//4-3
	IN_CH=512;
	CHout=512;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f9,"%d",&temp);
					W[9*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//5-1
	IN_CH=512;
	CHout=512;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f10,"%d",&temp);
					W[10*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//5-2
	IN_CH=512;
	CHout=512;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f11,"%d",&temp);
					W[11*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}
	//5-3
	IN_CH=512;
	CHout=512;
	for (unsigned int i=0; i<CHout; i++){
		for (int j=0; j<IN_CH; j++){
			for (int k=0; k<KERNEL_HEIGHT; k++){
				for (int l=0; l<KERNEL_WIDTH;l++){
					int temp;
					fscanf(f12,"%d",&temp);
					W[12*512*KERNEL_WIDTH*512*KERNEL_HEIGHT+i* KERNEL_WIDTH*IN_CH*KERNEL_HEIGHT+j*KERNEL_HEIGHT*KERNEL_WIDTH+k*KERNEL_WIDTH+l] = temp;
				}
			}
		}
	}

	IN_CH=3;
	CHout=64;
    for(unsigned int cout=0;cout<CHout;cout++)
		for(unsigned int i=0;i<OUT_HEIGHT;i++)
			for(unsigned int j=0;j<OUT_WIDTH;j++)
				feature_out[cout*OUT_HEIGHT*OUT_WIDTH+i*OUT_WIDTH+j] = 0;


    /* read scale factor*/
    FILE *fscale;
	fscale = fopen("/home/jiaru/conv/scale_int.txt", "r");
	for(int layer=0; layer<13; layer++)
		for(int cout=0; cout<512; cout++){
			int temp;
			fscanf(fscale, "%d", &temp);
			scale_p[layer*512+cout] = temp;
		}
	fclose(fscale);
	ofstream outputfilei;
	outputfilei.open("/home/jiaru/input.txt");
    for(int cin=0;cin<IN_CH;cin++)
    		for(int i=0;i<IN_HEIGHT;i++)
    			for(int j=0;j<IN_WIDTH;j++)
    				outputfilei<<"OUT["<<cin<<"]["<<i<<"]["<<j<<"]="<<feature_in[cin*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
	outputfilei.close();


    std::cout<<"start kernel\n";
    //set the kernel Arguments
    int narg=0;
    krnl_conv.setArg(narg++, buffer_feature_in);
    krnl_conv.setArg(narg++, buffer_W);
    krnl_conv.setArg(narg++, buffer_scale);
    krnl_conv.setArg(narg++, buffer_feature_out);
    krnl_conv.setArg(narg++, CHin);
    krnl_conv.setArg(narg++, Hin);
    krnl_conv.setArg(narg++, Win);
    krnl_conv.setArg(narg++, CHout);
    krnl_conv.setArg(narg++, Sx);
    krnl_conv.setArg(narg++, Sy);
    krnl_conv.setArg(narg++, mode);
    krnl_conv.setArg(narg++, relu_en);
    krnl_conv.setArg(narg++, pool_en);
    krnl_conv.setArg(narg++, layer);
    krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);
    // Data will be migrated to kernel space
    //et.add("kernel run");
    q.enqueueMigrateMemObjects({buffer_feature_in,buffer_W,buffer_scale,buffer_feature_out},0);
    q.enqueueTask(krnl_conv);
    q.enqueueMigrateMemObjects({buffer_feature_out},CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    //1_2 convolution
    std::cout << "start 1-2 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 64;
	pool_en = 1;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	layer=1;
	invert_flag = 1;
	zero_point = 57;


	/*output of one layer*/
    ofstream outputfile1_1;
	outputfile1_1.open("/home/jiaru/output/output1-1.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile1_1<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_out[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile1_1.close();




	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);


	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_in},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();


    //2_1 convolution
	std::cout << "start 2-1 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 128;
	pool_en = 0;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 74;
	layer = 2;
	invert_flag = 0;

	ofstream outputfile1_2;
	outputfile1_2.open("/home/jiaru/output/output1-2.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile1_2<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_in[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile1_2.close();



	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);



	//q.enqueueMigrateMemObjects({buffer_feature_in,buffer_W},0);
	//q.enqueueMigrateMemObjects({buffer_W},0);
	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_out},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();



    //2-2 convolution
	std::cout << "start 2-2 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 128;
	pool_en = 1;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 78;
	layer = 3;
	invert_flag = 1;

	ofstream outputfile2_1;
	outputfile2_1.open("/home/jiaru/output/output2-1.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile2_1<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_out[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile2_1.close();


	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_in},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();

	//3-1 convolution
	std::cout << "start 3-1 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 256;
	pool_en = 0;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 72;
	layer = 4;
	invert_flag = 0;


	ofstream outputfile2_2;
	outputfile2_2.open("/home/jiaru/output/output2-2.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile2_2<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_in[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile2_2.close();


	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_out},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();

	//3-2 convolution
	std::cout << "start 3-2 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 256;
	pool_en = 0;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 66;
	layer = 5;
	invert_flag = 1;

	ofstream outputfile3_1;
	outputfile3_1.open("/home/jiaru/output/output3-1.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile3_1<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_out[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile3_1.close();



	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_in},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();

	//3-3 convolution
	std::cout << "start 3-3 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 256;
	pool_en = 1;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 69;
	layer = 6;
	invert_flag = 0;

	ofstream outputfile3_2;
	outputfile3_2.open("/home/jiaru/output/output3-2.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile3_2<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_in[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile3_2.close();



	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_out},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();


	//4-1 convolution
	std::cout << "start 4-1 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 512;
	pool_en = 0;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 70;
	layer = 7;
	invert_flag = 1;


	ofstream outputfile3_3;
	outputfile3_3.open("/home/jiaru/output/output3-3.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile3_3<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_out[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile3_3.close();

	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_in},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();

	//4-2 convolution
	std::cout << "start 4-2 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 512;
	pool_en = 0;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 74;
	layer = 8;
	invert_flag = 0;

	ofstream outputfile4_1;
	outputfile4_1.open("/home/jiaru/output/output4-1.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile4_1<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_in[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile4_1.close();

	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_out},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();

	//4-3 convolution
	std::cout << "start 4-3 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 512;
	pool_en = 1;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 72;
	layer = 9;
	invert_flag = 1;

	ofstream outputfile4_2;
	outputfile4_2.open("/home/jiaru/output/output4-2.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile4_2<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_out[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile4_2.close();

	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_in},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();

	//5-1 convolution
	std::cout << "start 5-1 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 512;
	pool_en = 0;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 72;
	layer = 10;
	invert_flag = 0;

	ofstream outputfile4_3;
	outputfile4_3.open("/home/jiaru/output/output4-3.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile4_3<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_in[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile4_3.close();

	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_out},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();

	//5-2 convolution
	std::cout << "start 5-2 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 512;
	pool_en = 0;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 67;
	layer = 11;
	invert_flag = 1;

	ofstream outputfile5_1;
	outputfile5_1.open("/home/jiaru/output/output5-1.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile5_1<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_out[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile5_1.close();

	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_in},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();


	//5-3 convolution
	std::cout << "start 5-3 convolution\n";
	IN_CH = CHout;
	CHin = IN_CH;
	CHout = 512;
	pool_en = 1;
	IN_WIDTH = OUT_WIDTH;
    IN_HEIGHT = OUT_HEIGHT;
    Hin = IN_HEIGHT;
    Win = IN_WIDTH;
    OUT_WIDTH=((IN_WIDTH+2*X_PADDING-KERNEL_WIDTH)/X_STRIDE+1);
    OUT_HEIGHT=((IN_HEIGHT+2*Y_PADDING-KERNEL_HEIGHT)/Y_STRIDE+1);
	if(pool_en){
		OUT_WIDTH = OUT_WIDTH/2;
		OUT_HEIGHT = OUT_HEIGHT/2;
	}
	zero_point = 69;
	layer = 12;
	invert_flag = 0;

	ofstream outputfile5_2;
	outputfile5_2.open("/home/jiaru/output/output5-2.txt");
	for(int cout=0;cout<IN_CH;cout++)
		for(int i=0;i<IN_HEIGHT;i++){
			for(int j=0;j<IN_WIDTH;j++){
				outputfile5_2<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_in[cout*IN_HEIGHT*IN_WIDTH+i*IN_WIDTH+j]<<std::endl;
			}
		}
	outputfile5_2.close();

	narg=0;
	krnl_conv.setArg(narg++, buffer_feature_in);
	krnl_conv.setArg(narg++, buffer_W);
	krnl_conv.setArg(narg++, buffer_scale);
	krnl_conv.setArg(narg++, buffer_feature_out);
	krnl_conv.setArg(narg++, CHin);
	krnl_conv.setArg(narg++, Hin);
	krnl_conv.setArg(narg++, Win);
	krnl_conv.setArg(narg++, CHout);
	krnl_conv.setArg(narg++, Sx);
	krnl_conv.setArg(narg++, Sy);
	krnl_conv.setArg(narg++, mode);
	krnl_conv.setArg(narg++, relu_en);
	krnl_conv.setArg(narg++, pool_en);
	krnl_conv.setArg(narg++, layer);
	krnl_conv.setArg(narg++, zero_point);
	krnl_conv.setArg(narg++, invert_flag);

	q.enqueueMigrateMemObjects({buffer_W},0);
	q.enqueueTask(krnl_conv);
	q.enqueueMigrateMemObjects({buffer_feature_in, buffer_feature_out},CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();

	ofstream outputfile;
	outputfile.open("/home/jiaru/output/output5-3.txt");
	for(unsigned int cout=0;cout<CHout;cout++)
		for(unsigned int i=0;i<OUT_HEIGHT;i++){
			for(unsigned int j=0;j<OUT_WIDTH;j++){
				outputfile<<"OUT["<<cout<<"]["<<i<<"]["<<j<<"]="<<feature_out[cout*OUT_HEIGHT*OUT_WIDTH+i*OUT_WIDTH+j]/512<<std::endl;
			}
		}


	//finish all layers
    q.enqueueUnmapMemObject(buffer_feature_in, feature_in);
	q.enqueueUnmapMemObject(buffer_W, W);
	q.enqueueUnmapMemObject(buffer_scale, scale_p);
	q.enqueueUnmapMemObject(buffer_feature_out, feature_out);
    std::cout<<"finish kernel\n";
    //et.finish();



	outputfile.close();
	std::cout<<"down\n";

	//et.print();
	return 0;
}

