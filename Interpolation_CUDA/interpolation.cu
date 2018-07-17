/*
#ifndef __CUDACC__  
	#define __NVCC__  
	#include "cuda_texture_types.h"  
#endif  
*/


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_math.h"

#include "interpolation.h"
#include<iostream>  
#include<malloc.h>  
#include<stdlib.h>  
#include<stdio.h>  
#include<string.h> 
#include<math.h>
#include<time.h>
#include<fstream>
#include <Windows.h>

#include "Pre_filter.cu"


//added by JR Young
texture<float, 2, cudaReadModeElementType> tex_coefficient;


template<class floatN>
void CubicBSplinePrefilter2D(floatN* image, uint pitch, uint width, uint height);


__device__ float interpolate_bspline(float x, float y) {
	
	//return tex2D(tex_coefficient, x + 0.5f, y + 0.5f);
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5] 
	const float2 coord_grid = make_float2(x , y );

	const float2 index = floorf(coord_grid);

	const float2 fraction = coord_grid - index;
	float2 one_frac = 1.0f - fraction;
	float2 one_frac2 = one_frac * one_frac;
	float2 fraction2 = fraction * fraction;

	float2 w0 = 1.0f / 6.0f * one_frac2 * one_frac;
	float2 w1 = 2.0f / 3.0f - 0.5f * fraction2 * (2.0f - fraction);
	float2 w2 = 2.0f / 3.0f - 0.5f * one_frac2 * (2.0f - one_frac);
	float2 w3 = 1.0f / 6.0f * fraction2 * fraction;

	const float2 g0 = w0 + w1;
	const float2 g1 = w2 + w3;

	// h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent] 
	const float2 h0 = (w1 / g0) - make_float2(0.5f) + index;
	const float2 h1 = (w3 / g1) + make_float2(1.5f) + index;
	// fetch the four linear interpolations 
	float tex00 = tex2D(tex_coefficient, h0.x, h0.y);
	float tex10 = tex2D(tex_coefficient, h1.x, h0.y);
	float tex01 = tex2D(tex_coefficient, h0.x, h1.y);
	float tex11 = tex2D(tex_coefficient, h1.x, h1.y);
	// weigh along the y-direction 
	
	//tex00 = g0.y * tex00 + g1.y * tex01;
	//tex10 = g0.y * tex10 + g1.y * tex11;
	//return (g0.x * tex00 + g1.x * tex10);

	tex00 = lerp(tex01, tex00, g0.y);
	tex10 = lerp(tex11, tex10, g0.y);
	// weigh along the x-direction 
	return lerp(tex10, tex00, g0.x);
}

__device__ float linear_(float x, float y) {

	float x_ = floor(x);
	float y_ = floor(y);

	float f1 = tex2D(tex_coefficient, x_ + 0.5f , y_ + 0.5f);
	float f2 = tex2D(tex_coefficient, x_ + 1.5f	, y_ + 0.5f);
	float f3 = tex2D(tex_coefficient, x_ + 0.5f , y_ + 1.5f);
	float f4 = tex2D(tex_coefficient, x_ + 1.5f	, y_ + 1.5f);

	float dx = x - x_;
	float dy = y - y_;

	float temp1 , temp2;
	temp1 = (1 - dx)*f1 + dx*f2;
	temp2 = (1 - dx)*f3 + dx*f4;

	return (1-dy)*temp1 + dy*temp2;

};

__device__ float interpolate_bspline_v2(float x, float y) {

	//return tex2D(tex_coefficient, x + 0.5f, y + 0.5f);
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5] 
	const float2 coord_grid = make_float2(x, y);

	const float2 index = floorf(coord_grid);

	const float2 fraction = coord_grid - index;
	float2 one_frac = 1.0f - fraction;
	float2 one_frac2 = one_frac * one_frac;
	float2 fraction2 = fraction * fraction;

	float2 w0 = 1.0f / 6.0f * one_frac2 * one_frac;
	float2 w1 = 2.0f / 3.0f - 0.5f * fraction2 * (2.0f - fraction);
	float2 w2 = 2.0f / 3.0f - 0.5f * one_frac2 * (2.0f - one_frac);
	float2 w3 = 1.0f / 6.0f * fraction2 * fraction;

	const float2 g0 = w0 + w1;
	const float2 g1 = w2 + w3;

	// h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent] 
	const float2 h0 = (w1 / g0) - make_float2(1.0f) + index;
	const float2 h1 = (w3 / g1) + make_float2(1.0f) + index;
	// fetch the four linear interpolations 
	float tex00 = linear_(h0.x, h0.y);
	float tex10 = linear_(h1.x, h0.y);
	float tex01 = linear_(h0.x, h1.y);
	float tex11 = linear_(h1.x, h1.y);
	// weigh along the y-direction 


	//tex00 = lerp(tex01, tex00, g0.y);
	//tex10 = lerp(tex11, tex10, g0.y);
	tex00 = tex01 + g0.y*(tex00 - tex01);
	tex10 = tex11 + g0.y*(tex10 - tex11);
	// weigh along the x-direction 
	return tex10 + g0.x*(tex00- tex10);
}



// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
	//    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
	return (1.0f / 6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
	//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
	return (1.0f / 6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
	//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
	return (1.0f / 6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
	return (1.0f / 6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
	return w0(a) + w1(a);
}

__device__ float g1(float a)
{
	return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
	// note +0.5 offset to compensate for CUDA linear filtering convention
	return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ float h1(float a)
{
	return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

__device__ float tex2DFastBicubic(float x, float y) {

	//x -= 0.5f;
	//y -= 0.5f;
	float px = floor(x);
	float py = floor(y);
	float fx = x - px;
	float fy = y - py;

	// note: we could store these functions in a lookup table texture, but maths is cheap
	float g0x = g0(fx);
	float g1x = g1(fx);
	float h0x = h0(fx);
	float h1x = h1(fx);
	float h0y = h0(fy);
	float h1y = h1(fy);

	float r = g0(fy) *	(g0x * tex2D(tex_coefficient, px + h0x, py + h0y) +
						g1x * tex2D(tex_coefficient, px + h1x, py + h0y))
							+
			g1(fy) *	(g0x * tex2D(tex_coefficient, px + h0x, py + h1y) +
						g1x * tex2D(tex_coefficient, px + h1x, py + h1y));
	return r;
};


//added by JR Young
__global__ void DIC_gBicubicBsplineTable_Kernel(uint8_t *d_output, DWORD d_width, DWORD d_height, DWORD s_width, DWORD s_height)
{
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (tid_x >= d_width)
		return;
	if (tid_y >= d_height)
		return;

	float s_x, s_y;

	s_x = (float)tid_x / (float)(d_width - 1) * (float)(s_width - 1);
	s_y = (float)tid_y / (float)(d_height - 1) * (float)(s_height - 1);

	float temp = interpolate_bspline( s_x  , s_y );//interpolate_bspline(s_x, s_y);
	//float temp = interpolate_bspline_v2(s_x, s_y);//interpolate_bspline(s_x, s_y);


	//float temp = tex2DFastBicubic(s_x, s_y);
	//float temp = tex2D(tex_coefficient, s_x+0.5, s_y + 0.5);
	temp += 0.5;
	if (temp >= 255)
		temp = 255.0;
	else if (temp <= 0)
		temp = 0;
	d_output[tid_y*d_width + tid_x] = (uint8_t)temp;


}

__global__ void DIC_gBicubicBsplineTable_Kernel_float(float *d_output, DWORD d_width, DWORD d_height, DWORD s_width, DWORD s_height)
{
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (tid_x >= d_width)
		return;
	if (tid_y >= d_height)
		return;

	float s_x, s_y;

	s_x = (float)tid_x / (float)(d_width - 1) * (float)(s_width - 1);
	s_y = (float)tid_y / (float)(d_height - 1) * (float)(s_height - 1);

	float temp = interpolate_bspline(s_x, s_y);//interpolate_bspline(s_x, s_y);
											   //float temp = tex2DFastBicubic(s_x, s_y);
											   //float temp = tex2D(tex_coefficient, s_x+0.5, s_y + 0.5);

	d_output[tid_y*d_width + tid_x] = temp;


}


void G_interpolation(const uint8_t *host_s_data, uint8_t *host_d_data, DWORD s_width, DWORD s_height, float width_scale, float height_scale) {

	if (width_scale <= 0 || height_scale <= 0)
		return;

	//prefix: d for destination  image; s for source image
	//
	clock_t start, end;
	double duration;
	start = clock();

	DWORD d_width = s_width * width_scale;
	DWORD d_height = s_height * height_scale;

	//added by JR Young
	float *m_g_d4pTBspline_Kernel_TMP;//存储Bspline_Kernel插值表的float4类型的一维矩阵
	cudaArray *m_g_d4pTBspline_Kernel;//存储Bspline_Kernel插值表的float4类型的一维矩阵
	uint8_t *d_output;
	
	//buffer with filling
	//float *f_data = nullptr;
	//f_data = (float *)malloc(sizeof(float)*(s_width + 4)*(s_height + 4));
	//fill_data(host_s_data, f_data, s_width, s_height);

	//allocation
	//added by JR Young
	cudaMalloc((void**)&m_g_d4pTBspline_Kernel_TMP, sizeof(float)*s_width*s_height);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&(m_g_d4pTBspline_Kernel), &channelDesc, s_width, s_height);
	cudaMalloc((void**)&d_output, sizeof(uint8_t)*d_width*d_height);
	//



	//coefficient
	float *temp_coeff; 
	temp_coeff = (float*)malloc(sizeof(float)*s_width*s_height);
	for (int i = 0; i < s_width*s_height; ++i)
		temp_coeff[i] = (float)host_s_data[i];

	cudaMemcpy(m_g_d4pTBspline_Kernel_TMP, temp_coeff, sizeof(float)*s_width*s_height, cudaMemcpyHostToDevice);



	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&t1);

	//coeff
	//IIR
	//CubicBSplinePrefilter2D<float>(m_g_d4pTBspline_Kernel_TMP, s_width * sizeof(float), s_width, s_height);
	//FIR
	CubicBSplinePrefilter2D_FIR<float>(m_g_d4pTBspline_Kernel_TMP, s_width * sizeof(float), s_width, s_height);

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&t2);
	std::cout << "Time Of Pre_Filtering：" << 1000*(t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << "ms" << std::endl;


	free(temp_coeff);
	//calculate coefficient
	cudaMemcpyToArray(m_g_d4pTBspline_Kernel, 0, 0, m_g_d4pTBspline_Kernel_TMP, sizeof(float)*s_width*s_height, cudaMemcpyDeviceToDevice);

	//
	//bind texture memory with m_g_d4pTBspline_Kernel
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaBindTextureToArray(&tex_coefficient,m_g_d4pTBspline_Kernel,&channelDesc);
	tex_coefficient.addressMode[0] = cudaAddressModeClamp;
	tex_coefficient.addressMode[1] = cudaAddressModeClamp;
	tex_coefficient.filterMode = cudaFilterModeLinear;


	//Timeing
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&t1);

	//interpolation
	DWORD Grid_Width = (d_width + 31) / 32;
	DWORD Grid_Height = (d_height + 31) / 32;

	dim3 DimBlock(32,32);
	dim3 DimGrid(Grid_Width, Grid_Height);

	DIC_gBicubicBsplineTable_Kernel<<<DimGrid, DimBlock >>>(d_output, d_width, d_height, s_width, s_height);
	//cudaMemcpyFromArray(temp_coeff, m_g_d4pTBspline_Kernel, 0, 0, sizeof(float)*s_width*s_height, cudaMemcpyDeviceToHost);

	//for (int i = 0; i < s_width*s_height; ++i) host_d_data[i] = (uint8_t)temp_coeff[i];
	//copy back
	cudaMemcpy(host_d_data, d_output, sizeof(uint8_t)*d_width*d_height, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&t2);
	std::cout << "Time Of Interpolating：" << 1000 * (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << "ms" << std::endl;

	float *d_temp_output, *h_temp_output;
	cudaMalloc((void**)&d_temp_output, sizeof(float)*d_width*d_height);
	h_temp_output = (float*)malloc(sizeof(float)*d_width*d_height);

	DIC_gBicubicBsplineTable_Kernel_float << <DimGrid, DimBlock >> >(d_temp_output , 
d_width, d_height, s_width, s_height);

	cudaMemcpy(h_temp_output, d_temp_output, sizeof(float)*d_width*d_height, cudaMemcpyDeviceToHost);

	//FILE *fs = fopen("CUDA_OUTPUT_LENA_tex_fir.bin", "wb");
	//fwrite(h_temp_output, sizeof(float), d_width*d_height, fs);
	//fclose(fs);

	cudaFree(d_temp_output);
	free(h_temp_output);

	//Free
	//added by JR Young
	cudaFree(m_g_d4pTBspline_Kernel_TMP);
	cudaFreeArray(m_g_d4pTBspline_Kernel);
	cudaFree(d_output);


	end = clock();
	duration = end - start;
	duration = duration / CLOCKS_PER_SEC;
	std::cout << "Time of scaling picture: "
		<< s_width << "x" << s_height
		<< " in rate:" << width_scale << "x" << height_scale
		<< " using method:" //<< ""
		<< "	finished in:" << duration << "s" << std::endl;


}


void CalBSplinePreFilter(float* h_image, int width, int height) {
	float* img;
	cudaMalloc((void**)&img, sizeof(float)*width*height);
	cudaMemcpy(img, h_image, sizeof(float)*width*height, cudaMemcpyHostToDevice);

	//IIR
	CubicBSplinePrefilter2D<float>(img, width * sizeof(float), width, height);
	//FIR
	//CubicBSplinePrefilter2D_FIR<float>(img, width * sizeof(float), width, height);

	cudaMemcpy(h_image, img, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
}
