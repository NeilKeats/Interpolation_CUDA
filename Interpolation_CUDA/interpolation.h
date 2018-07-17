#pragma once
#ifndef INTERPOLATION
#define INTERPOLATION

#include<Windows.h>   
#include<inttypes.h>
#include<string>

#define MODE_NEAREST_NEIGHBOUR 0x00
#define MODE_BILINEAR 0x01
#define MODE_BICUBIC 0x02
#define MODE_SPLINE 0x03
#define MODE_BC_KERNEL 0x04
#define MODE_BELL_KERNEL 0x05
#define MODE_HERMITE_KERNEL 0x06
#define MODE_MITCHELL_KERNEL 0x07
#define MODE_LANCZOS_KERNEL 0x08
#define MODE_WAD_BILINEAR 0X09
#define MODE_ADA_BILINEAR 0x0A
#define MODE_CFLS_BILINEAR 0x0B


static int A_MODE[] = { 
	MODE_NEAREST_NEIGHBOUR,
	MODE_BILINEAR,
	MODE_BICUBIC,
	MODE_SPLINE,
	MODE_BC_KERNEL,
	MODE_BELL_KERNEL,
	MODE_HERMITE_KERNEL,
	MODE_MITCHELL_KERNEL,
	MODE_LANCZOS_KERNEL,
	MODE_WAD_BILINEAR,
	MODE_ADA_BILINEAR,
	MODE_CFLS_BILINEAR };


static std::string MODE_NAME[] = { 
	"NN",
	"BILINEAR",
	"BICUBIC",
	"BSPLINE",
	"BC_KERNEL",
	"BELL_KERNEL",
	"HERMITE_KERNEL",
	"MITCHELL_KERNEL",
	"LANCZOS_KERNEL",
	"WAD_BILINEAR",
	"MODE_ADA_BILINEAR",
	"MODE_CFLS_BILINEAR"};

//order 3
float bubic_conv_kernel(const float *a, float x);

float lanczos_conv_kernel(const float *a, float x);

//order 2
float bell_conv_kernel(const float *a, float x);

//order 3
float hermite_conv_kernel(const float *a, float x);//a particular case of bicubic kernel. while a=0

//order 3
float mitchell_conv_kernel(const float *a, float x);

void gradientX(const float *Imdata, float * dx, float *dbuffer, DWORD width, DWORD height);

void gradientY(const float *Imdata, float * dx, float *dbuffer, DWORD width, DWORD height);

void bicubic_coeff(const float *Imdata,float * coeff, DWORD width, DWORD height);

void bicubic_spline_coeff(const float *f_data, float * coeff, DWORD s_width, DWORD s_height);

void kernel_coeff(const float *f_data, float * coeff, DWORD s_width, DWORD s_height);

float cal_bicubic(float *coeff, float s_x, float s_y, DWORD s_width, DWORD s_height, int MODE);

float cal_bicubic_kernel(const float *f_data, float s_x, float s_y, DWORD s_width, DWORD s_height, const float *a, int MODE);

float nearest_neighbour(const float *f_data, float s_x, float s_y, DWORD s_width, DWORD s_height);

float bilinear(const float* f_data, float s_x, float s_y, DWORD s_width, DWORD s_height);

/***********/
void wad_coeff(const float *Imdata, float * coeff, DWORD s_width, DWORD s_height);

float wad_bilinear(const float* f_data, const float* coeff, float s_x, float s_y, DWORD s_width, DWORD s_height);

void ada_coeff(const float *Imdata, float * coeff, DWORD s_width, DWORD s_height);

float ada_bilinear(const float* f_data, const float* coeff , float s_x, float s_y, DWORD s_width, DWORD s_height);

float cfls_bilinear(const float* f_data, float s_x, float s_y, DWORD s_width, DWORD s_height);

void fill_data(const uint8_t* s_data, float * f_data, DWORD s_width, DWORD s_height);

void fill_data(const float* s_data, float * f_data, DWORD s_width, DWORD s_height);


void interpolation(const uint8_t *s_data, uint8_t *d_data, DWORD s_width, DWORD s_height, float width_scale, float height_scale, int MODE);


//
void DIC_gBicubicBsplineTable_Kernel();

void G_interpolation(const uint8_t *host_s_data, uint8_t *host_d_data, DWORD s_width, DWORD s_height, float width_scale, float height_scale);

void CalBSplinePreFilter(float* h_image, int width, int height);

#endif