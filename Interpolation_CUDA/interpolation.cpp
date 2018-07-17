#include "interpolation.h"
#include<iostream>  
#include<malloc.h>  
#include<stdlib.h>  
#include<stdio.h>  
#include<string.h> 
#include<math.h>
#include<time.h>
#include<fstream>
//#include<cmath>

#define TIME_COUNT

const float M_PI=3.14159265358979f;

static float m_dBicubicMatrix[16][16] = { 
	{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0 },
	{ 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0 },
	{ -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0 },
	{ 9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1 },
	{ -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1 },
	{ 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 },
	{ 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0 },
	{ -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1 },
	{ 4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1 } };

static double m_dBSplineControlMatrix[4][4] = {
	{ 71 / 56.0, -19 / 56.0, 5 / 56.0, -1 / 56.0 },
	{ -19 / 56.0, 95 / 56.0, -25 / 56.0, 5 / 56.0 },
	{ 5 / 56.0, -25 / 56.0, 95 / 56.0, -19 / 56.0 },
	{ -1 / 56.0, 5 / 56.0, -19 / 56.0, 71 / 56.0 } };

static double m_dBSplineFunctionMatrix[4][4] = { 
	{ -1 / 6.0, 3 / 6.0, -3 / 6.0, 1 / 6.0 },
	{ 3 / 6.0, -6 / 6.0, 3 / 6.0, 0 },
	{ -3 / 6.0, 0, 3 / 6.0, 0 },
	{ 1 / 6.0, 4 / 6.0, 1 / 6.0, 0 } };

inline float bubic_conv_kernel(const float *a, float x) {
	// a = -0.75
	float tmp = x < 0 ? -x : x;
	if (tmp <= 1)
		//return ((*a + 2)*pow(tmp, 3) - (*a + 3)*pow(tmp, 2) + 1);
		return ((*a + 2)*tmp*tmp*tmp - (*a + 3)*tmp*tmp + 1);
	else if (1 < tmp < 2)
		//return ((*a)*pow(tmp, 3) - 5 * (*a)*pow(tmp, 2) + 8 * (*a)*tmp - 4 * (*a));
		return ((*a)*tmp*tmp*tmp - 5 * (*a)*tmp*tmp + 8 * (*a)*tmp - 4 * (*a));
	else
		return 0;
};

inline float lanczos_conv_kernel(const float *a, float x) {

	if (x == 0)
		return 1;
	else if (x<(*a) && x > -(*a)) {
		float result = ((*a)*sinf(M_PI*x)*sinf(M_PI*x/(*a)))/(M_PI*M_PI*x*x);
		return result;
	}
	else
		return 0;
}

inline float bell_conv_kernel(const float *a, float x) {
	float tmp = x < 0 ? -x : x;
	if (x < 0.5)
		return 0.75 - x*x;
	else if (x < 1.5)
		return 0.5*(1.5 - x)*(1.5 - x);
	else
		return 0;
}

inline float hermite_conv_kernel(const float *a, float x) {
	float tmp = 0;
	return bubic_conv_kernel( &tmp, x);
}

float mitchell_conv_kernel(const float *a, float x) {
	float B = a[0];
	float C = a[1];
	float x_ = x < 0 ? -x : x;
	
	float result = 0;
	if (x_ < 1)
		result =(
				x_*x_*x_*(12 - 9 * B - 6 * C)
				+ x_*x_*(-18 + 12 * B + 6 * C)
				+ (6 - 2 * B)
				) / 6;
	else if (x_ < 2)
		result = (
					x_*x_*x_*(-B - 6*C)
					+ x_*x_*(6*B + 30*C)
					+ x_*(-12*B -48*C)
					+ (8*B + 24*C)
					) / 6;
	return result;
}

void gradientX(const float *f_data, float * dx, float *dbuffer, DWORD width, DWORD height) {
	const DWORD local_width = width + 4;
	for(int i= 0; i<height+4;++i)
		for (int j = 2; j < width + 2; ++j) {
			dbuffer[local_width * i + j] =	 
				2.0 / 3 * f_data[local_width * i + j + 1] 
				- 1.0 / 12 * f_data[local_width * i + j + 2]
				- 2.0 / 3 * f_data[local_width * i + j -1]
				+ 1.0 / 12 * f_data[local_width * i + j -2 ];
		}	

	for(int i=0; i<height; ++i)
		for (int j = 0; j < width; ++j) {
			int row = i + 2;
			int col = j + 2;
			dx[i*width + j] = dbuffer[row*local_width+col];
		}
}

void gradientY(const float *f_data, float * dy, float *dbuffer, DWORD width, DWORD height) {
	const DWORD local_width = width + 4;
	for( int i =2 ; i< height+2 ; ++i)
		for (int j = 0; j < width + 4 ; ++j) {
			dbuffer[local_width * i + j] =   
				2.0 / 3 * f_data[local_width * (i - 1 ) + j ]
				- 1.0 / 12 * f_data[local_width * (i - 2 ) + j ]
				- 2.0 / 3 * f_data[local_width * (i + 1 ) + j ]
				+ 1.0 / 12 * f_data[local_width * (i + 2 ) + j ];
		}

	for (int i = 0; i<height; ++i)
		for (int j = 0; j < width; ++j) {
			int row = i + 2;
			int col = j + 2;
			dy[i*width + j] = dbuffer[row*local_width + col];
		}
}

void bicubic_coeff(const float *f_data, float * coeff, DWORD s_width, DWORD s_height){
	
	float *dx, *dy, *dxy,*d_buffx, *d_buffy;
	dx = (float*)malloc(sizeof(float)*s_width*s_height);
	dy = (float*)malloc(sizeof(float)*s_width*s_height);
	dxy = (float*)malloc(sizeof(float)*s_width*s_height);
	d_buffx = (float*)malloc(sizeof(float)*(s_width + 4)*(s_height + 4));
	d_buffy = (float*)malloc(sizeof(float)*(s_width + 4)*(s_height + 4));

	gradientY(f_data, dy, d_buffy, s_width, s_height);
	gradientX(f_data, dx, d_buffx, s_width, s_height);
	gradientY(d_buffx, dxy, d_buffy, s_width, s_height);

	float dTao[16], dAlpha[16];

	for(int i = 1 ; i<s_height ; ++i ){
		for (int j = 0; j < s_width-1 ; ++j) {
			dTao[0] = f_data[(i + 2)*(s_width + 4) + j + 2];
			dTao[1] = f_data[(i + 2)*(s_width + 4) + j + 3];
			dTao[2] = f_data[(i + 1)*(s_width + 4) + j + 2];
			dTao[3] = f_data[(i + 1)*(s_width + 4) + j + 3];

			
			dTao[4] = dx[(i)*s_width  + j ];
			dTao[5] = dx[(i)*s_width  + j + 1];
			dTao[6] = dx[(i - 1)*s_width  + j ];
			dTao[7] = dx[(i - 1)*s_width  + j + 1 ];

			dTao[8] = dy[(i)*s_width + j ];
			dTao[9] = dy[(i)*s_width + j + 1];
			dTao[10] = dy[(i - 1)*s_width + j];
			dTao[11] = dy[(i - 1)*s_width + j + 1];
			
			dTao[12] = dxy[(i)*s_width + j ];
			dTao[13] = dxy[(i)*s_width + j + 1];
			dTao[14] = dxy[(i - 1)*s_width + j];
			dTao[15] = dxy[(i - 1)*s_width + j + 1];
			
			/*
			dTao[4] = dTao[5] = dTao[6] = dTao[7] = 0;
			dTao[8] = dTao[9] = dTao[10] = dTao[11] = 0;
			dTao[12] = dTao[13] = dTao[14] = dTao[15] = 0;
			*/

			for (int k = 0; k < 16; k++)
			{
				dAlpha[k] = 0;
				for (int l = 0; l < 16; l++)
				{
					dAlpha[k] += (m_dBicubicMatrix[k][l] * dTao[l]);
				}
			}

			float *fBicubic = coeff + ((i*s_width) + j) * 16;
			for (int i = 0; i < 16; i++)
				fBicubic[i] = dAlpha[i];

		}
	}

	free(dx);
	free(dy);
	free(dxy);
	free(d_buffx);
	free(d_buffy);

}

void bicubic_spline_coeff(const float *f_data, float * coeff, DWORD s_width, DWORD s_height) {

	int i, j, k, l, m, n;
	float  Omiga[4][4], Beta[4][4];

	for (i = 0; i < s_height; ++i) {
		for (j = 0; j < s_width; ++j) {
			float *local_coeff = coeff + (i*s_width + j) * 16;

			//store neighbour value into Omeiga
			for (k = 0; k < 4; ++k)
				for (l = 0; l < 4; ++l)
					//reflect to original data (i + 1 - k, j - 1 + l)
					//reflect to filled data (i + 1 -k +2 , j - 1 +l +2)
					Omiga[k][l] = f_data[(i + 1 - k + 2)*(s_width + 4) + (j - 1 + l + 2)];

			//Beta
			for (k = 0; k < 4; k++)
			{
				for (l = 0; l < 4; l++)
				{
					Beta[k][l] = 0;
					for (m = 0; m < 4; m++)
					{
						for (n = 0; n < 4; n++)
						{
							Beta[k][l] +=
								m_dBSplineControlMatrix[k][m] *
								m_dBSplineControlMatrix[l][n] *
								Omiga[n][m];
						}
					}
				}
			}

			//calculate p array;
			for (k = 0; k < 4; k++)
			{
				for (l = 0; l < 4; l++)
				{
					//dTBspline[i][j][k][l] = 0;
					local_coeff[k * 4 + l] = 0;
					for (m = 0; m < 4; m++)
					{
						for (n = 0; n < 4; n++)
						{
							//dTBspline[i][j][k][l] += m_dBSplineFunctionMatrix[k][m] * m_dBSplineFunctionMatrix[l][n] * dBeta[n][m];
							local_coeff[k * 4 + l] += m_dBSplineFunctionMatrix[k][m] * m_dBSplineFunctionMatrix[l][n] * Beta[n][m];
						}
					}
				}
			}

			//trans p to a;
			for (k = 0; k < 2; k++)
			{
				for (l = 0; l < 4; l++)
				{
					float m_dTemp = local_coeff[k * 4 + l];
					local_coeff[k * 4 + l] = local_coeff[(3 - k) * 4 + (3 - l)];
					local_coeff[(3 - k) * 4 + (3 - l)] = m_dTemp;
				}
			}

			//for (int i = 4; i < 16; ++i)
			//	local_coeff[i] = 0;

		}

	}

}

void kernel_coeff(const float *f_data, float * coeff, DWORD s_width, DWORD s_height) {
	int i, j, index;
	for(i = 0; i<s_height; ++i)
		for (j = 0; j < s_width; ++j) {
			index = (i*s_width + j) * 16;

			coeff[index++] = f_data[(i + 1 + 2)*(s_width + 4) + j - 1 + 2];
			coeff[index++] = f_data[(i + 1 + 2)*(s_width + 4) + j + 2 ];
			coeff[index++] = f_data[(i + 1 + 2)*(s_width + 4) + j + 1 + 2 ];
			coeff[index++] = f_data[(i + 1 + 2)*(s_width + 4) + j + 2 + 2 ];

			coeff[index++] = f_data[(i + 2)*(s_width + 4) + j - 1 + 2 ];
			coeff[index++] = f_data[(i + 2)*(s_width + 4) + j + 2];
			coeff[index++] = f_data[(i + 2)*(s_width + 4) + j + 1 + 2];
			coeff[index++] = f_data[(i + 2)*(s_width + 4) + j + 2 + 2];

			coeff[index++] = f_data[(i - 1 + 2)*(s_width + 4) + j - 1 + 2];
			coeff[index++] = f_data[(i - 1 + 2)*(s_width + 4) + j + 2];
			coeff[index++] = f_data[(i - 1 + 2)*(s_width + 4) + j + 1 + 2];
			coeff[index++] = f_data[(i - 1 + 2)*(s_width + 4) + j + 2 + 2];

			coeff[index++] = f_data[(i - 2 + 2)*(s_width + 4) + j - 1 + 2];
			coeff[index++] = f_data[(i - 2 + 2)*(s_width + 4) + j + 2];
			coeff[index++] = f_data[(i - 2 + 2)*(s_width + 4) + j + 1 + 2];
			coeff[index++] = f_data[(i - 2 + 2)*(s_width + 4) + j + 2 + 2];

		}
}

float cal_bicubic(float *coeff, float s_x, float s_y, DWORD s_width, DWORD s_height, int MODE) {
	//coefficient array obtained from the left corner element.
	//for y, using ceil(),causes our image data start from left bottom
	//for x, using floor()

	//is_x  int s_x;  is_y  int s_y
	int is_y, is_x;
	float x, y;
	is_y = ceil(s_y);
	if (MODE == MODE_BICUBIC)
		is_y = is_y <= 0 ? 1 : is_y;
	else
		is_y = is_y <= 0 ? 0 : is_y;
	is_y = is_y >= s_height - 1 ? s_height - 1 : is_y;
	//is_y = is_y >= s_height ? : ;
	//is_y = is_y <= 0 ? 0 : is_y;
	y = (float)is_y - s_y;

	is_x = floor(s_x);
	if (s_x >= s_width -1 )
		is_x = is_x;
	is_x = is_x <= 0 ? 0 : is_x;	
	if (MODE == MODE_BICUBIC)
		is_x = is_x >= s_width - 1 ? s_width - 2 : is_x;
	else
		is_x = is_x >= s_width - 1 ? s_width - 1 : is_x;
	//is_x = is_x >= s_width - 1 ? s_width - 1 : is_x;
	x = s_x - (float)is_x;

	float *local_coeff = coeff + (is_y*s_width + is_x) * 16;
	
	float p_x[4],temp[4];
	
	p_x[0] = 1;
	p_x[1] = x;
	p_x[2] = x*x;
	p_x[3] = x*x*x;
	
	/*
	p_x[0] = pow(x, 0);
	p_x[1] = pow(x, 1);
	p_x[2] = pow(x, 2);
	p_x[3] = pow(x, 3);
	*/
	//
	float result = 0;

	temp[0] = 1;
	temp[1] = y;
	temp[2] = y*y;
	temp[3] = y*y*y;
	/*
	temp[0] = pow(y, 0);
	temp[1] = pow(y, 1);
	temp[2] = pow(y, 2);
	temp[3] = pow(y, 3);
	*/

	for(int k=0 ; k<4 ; ++k)
		for (int j = 0; j < 4; ++j) {
			result += local_coeff[k*4 + j] * temp[k] * p_x[j];
			//result += local_coeff[k * 4 + j] * temp[j] * p_x[k];
		}

	/*
	for(int i=0;i<4;++i)
		temp[i] = p_x[0] * local_coeff[0+i] 
				+ p_x[1] * local_coeff[4+i] 
				+ p_x[2] * local_coeff[8+i] 
				+ p_x[3] * local_coeff[12+i];
	*/

		/*
		temp[i] = p_x[0] * local_coeff[i * 4 + 1]
				+ p_x[1] * local_coeff[i * 4 + 2]
				+ p_x[2] * local_coeff[i * 4 + 3]
				+ p_x[3] * local_coeff[i * 4 + 4];
		*/
	//return (temp[0] + temp[1]*y + temp[2]*y*y + temp[3]*y*y*y);
	return result;
}

float cal_bicubic_kernel(const float *data, float s_x, float s_y, DWORD s_width, DWORD s_height, const float *a, int MODE) {
	int is_y, is_x;
	float x, y;
	is_y = ceil(s_y);
	//is_y = is_y <= 0 ? 1 : is_y;
	is_y = is_y <= 0 ? 0 : is_y;
	y = (float)is_y - s_y;

	is_x = floor(s_x);
	//is_x = is_x >= s_width - 1 ? s_width - 2 : is_x;
	is_x = is_x >= s_width - 1 ? s_width - 1 : is_x;
	x = s_x - (float)is_x;

	float c_j_[4], c_i_[4];
	float(*kernel_func)(const float*,float) = nullptr; 
	switch (MODE) {
	case MODE_BC_KERNEL: 
		kernel_func = &bubic_conv_kernel;
		break;
	case MODE_LANCZOS_KERNEL:
		kernel_func = &lanczos_conv_kernel;
		break;
	case MODE_BELL_KERNEL:
		kernel_func = &bell_conv_kernel;
		break;
	case MODE_HERMITE_KERNEL:
		kernel_func = &hermite_conv_kernel;
		break;
	case MODE_MITCHELL_KERNEL:
		kernel_func = &mitchell_conv_kernel;
		break;
	default:
		break;
	}

	c_j_[0] = kernel_func(a, (1 + x));
	c_j_[1] = kernel_func(a, (x));
	c_j_[2] = kernel_func(a, (1 - x));
	c_j_[3] = kernel_func(a, (2 - x));

	c_i_[0] = kernel_func(a, (1 + y));
	c_i_[1] = kernel_func(a, (y));
	c_i_[2] = kernel_func(a, (1 - y));
	c_i_[3] = kernel_func(a, (2 - y));
	/*
	if (MODE == MODE_BC_KERNEL) {
		c_j_[0] = bubic_conv_kernel(a, (1 + x));
		c_j_[1] = bubic_conv_kernel(a, (x));
		c_j_[2] = bubic_conv_kernel(a, (1 - x));
		c_j_[3] = bubic_conv_kernel(a, (2 - x));

		c_i_[0] = bubic_conv_kernel(a, (1 + y));
		c_i_[1] = bubic_conv_kernel(a, (y));
		c_i_[2] = bubic_conv_kernel(a, (1 - y));
		c_i_[3] = bubic_conv_kernel(a, (2 - y));
	}
	else if (MODE == MODE_LANCZOS_KERNEL) {
		c_j_[0] = lanczos_conv_kernel(a, (1 + x));
		c_j_[1] = lanczos_conv_kernel(a, (x));
		c_j_[2] = lanczos_conv_kernel(a, (1 - x));
		c_j_[3] = lanczos_conv_kernel(a, (2 - x));

		c_i_[0] = lanczos_conv_kernel(a, (1 + y));
		c_i_[1] = lanczos_conv_kernel(a, (y));
		c_i_[2] = lanczos_conv_kernel(a, (1 - y));
		c_i_[3] = lanczos_conv_kernel(a, (2 - y));
	}
	*/

	float result = 0;

	/*
	float sum = 0;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			sum += c_i_[i]*c_j_[j];
	*/

	//store neighbour value into Omeiga
	int index = (is_y*s_width + is_x) * 16;
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j){
			//reflect to original data (is_y + 1 - i, is_x - 1 +l)
			//reflect to filled data (is_y + 1 - i + 2 , is_x - 1 + l +2)
			//int yy = (is_y + 1 - i + 2);
			//int xx = (is_x - 1 + j + 2);
			//result += f_data[(is_y + 1 - i + 2)*(s_width + 4) + (is_x - 1 + j + 2)]*c_i_[i]*c_j_[j];
			
			float coeff = c_i_[i] * c_j_[j];
			//result += f_data[yy*(s_width + 4) + xx] * coeff;
			result += data[index++] * coeff;
		}
	return result;
}

float nearest_neighbour(const float *f_data, float s_x, float s_y, DWORD s_width, DWORD s_height) {
	int is_y, is_x;
	is_y = floor(s_y);
	is_y = s_y - (float)is_y >= 0.5 ? (is_y + 1) : is_y;
	//is_y = is_y <= 0 ? 1 : is_y;
	is_y = is_y <= 0 ? 0 : is_y;
	is_y = is_y >= s_height - 1 ?  s_height - 1 : is_y;

	is_x = floor(s_x);
	is_x = s_x - (float)is_x >= 0.5 ? (is_x + 1) : is_x;
	//is_x = is_x >= s_width - 1 ? s_width - 2 : is_x;
	//is_x = is_x >= s_width - 1 ? s_width - 1 : is_x;
	is_x = is_x <= 0 ? 0 : is_x;
	is_x = is_x >= s_width - 1 ? s_width - 1 : is_x;

	int location = (is_y+2)*(s_width+4) + is_x + 2;

	return f_data[location];
}

float bilinear(const float* f_data, float s_x, float s_y, DWORD s_width, DWORD s_height) {
	int y_0, x_0, y_1, x_1;
	float f_y0_x0, f_y0_x1, f_y1_x0, f_y1_x1;
	float f_y0, f_y1, f_inter;

	/*
	y_0 = floor(s_y);
	//y_0 = y_0 <= 0 ? 0 : y_0;
	//y_0 = y_0 >= s_height-1 ? s_height-1 : y_0;
	y_1 = y_0 + 1;


	x_0 = floor(s_x);
	//x_0 = x_0 <= 0 ? 0 : x_0;
	//x_0 = x_0 >= s_width - 1 ? s_width - 1 : x_0;
	x_1 = x_0 + 1;
	*/

	float dx, dy;

	//
	y_0 = floor(s_y);
	if (y_0 < 0) {
		y_0 = 0;
		dy = 0;
	}
	else if (y_0 > s_height - 1) {
		y_0 = s_height - 1;
		dy = 0;
	}
	else dy = s_y - float(y_0);

	y_1 = y_0 + 1;



	x_0 = floor(s_x);
	if (x_0 < 0) {
		x_0 = 0;
		dx = 0;
	}
	else if (x_0 > s_width - 1) {
		x_0 = s_width - 1;
		dx = 0;
	}
	else
		dx = s_x - float(x_0);

	x_1 = x_0 + 1;
	//

	f_y0_x0 = f_data[(y_0+2)*(s_width + 4) + (x_0 + 2)];
	f_y0_x1 = f_data[(y_0+2)*(s_width + 4) + (x_1 + 2)];
	f_y1_x0 = f_data[(y_1+2)*(s_width + 4) + (x_0 + 2)];
	f_y1_x1 = f_data[(y_1+2)*(s_width + 4) + (x_1 + 2)];

	f_y0 = f_y0_x0 + (dx)*(f_y0_x1 - f_y0_x0);
	f_y1 = f_y1_x0 + (dx)*(f_y1_x1 - f_y1_x0);
	f_inter = f_y0 + (dy)*(f_y1 - f_y0);

	return f_inter;
}

void wad_coeff(const float *Imdata, float * coeff, DWORD s_width, DWORD s_height) {
#define ABS(a) ((a)<0?-(a):(a))
	for (int i = 0; i < s_height; ++i) {
		for (int j = 0; j < s_width; ++j) {
			float *A = coeff + (i*s_width + j)*2 ;
			float f_y0_x1, f_y0_x_1, f_y0_x2, f_y0_x0;
			float f_y1_x0, f_y_1_x0, f_y2_x0;

			f_y0_x0 = Imdata[(i + 2)*(s_width + 4) + (j + 2)];
			f_y0_x1 = Imdata[(i + 2)*(s_width + 4) + (j + 3)];
			f_y1_x0 = Imdata[(i + 3)*(s_width + 4) + (j + 2)];

			f_y0_x2 = Imdata[(i + 2)*(s_width + 4) + (j + 4)];
			f_y0_x_1 = Imdata[(i + 2)*(s_width + 4) + (j + 1)];

			f_y2_x0 = Imdata[(i + 4)*(s_width + 4) + (j + 2)];
			f_y_1_x0 = Imdata[(i + 1)*(s_width + 4) + (j + 2)];

			A[0] = (ABS(f_y0_x1 - f_y0_x_1) - ABS(f_y0_x2 - f_y0_x0)) / 255;
			A[1] = (ABS(f_y1_x0 - f_y_1_x0) - ABS(f_y2_x0 - f_y0_x0)) / 255;
		}
	}
}

float wad_bilinear(const float* f_data, const float* coeff, float s_x, float s_y, DWORD s_width, DWORD s_height) {
	int y_0, x_0, y_1, x_1;
	float f_y0_x0, f_y0_x1, f_y1_x0, f_y1_x1;
	float f_y0, f_y1, f_inter;

	float dx, dy;
	
	y_0 = floor(s_y);
	if (y_0 < 0) {
		y_0 = 0;
		dy = 0;
	}
	else if (y_0 > s_height-1 ) {
		y_0 = s_height - 1;
		dy = 0;
	}
	else dy = s_y - float(y_0);
	y_1 = y_0 + 1;


	x_0 = floor(s_x);
	if (x_0 < 0) {
		x_0 = 0;
		dx = 0;
	}
	else if (x_0 > s_width - 1) {
		x_0 = s_width - 1;
		dx = 0;
	}
	else 
		dx = s_x - float(x_0);
	x_1 = x_0 + 1;
	

	/*
	y_0 = floor(s_y);
	y_0 = y_0 <= 0 ? 0 : y_0;
	y_0 = y_0 >= s_height - 1 ? s_height - 1 : y_0;
	y_1 = y_0 + 1;

	x_0 = floor(s_x);
	x_0 = x_0 <= 0 ? 0 : x_0;
	x_0 = x_0 >= s_width - 1 ? s_width - 1 : x_0;
	x_1 = x_0 + 1;
	*/

	f_y0_x0 = f_data[(y_0 + 2)*(s_width + 4) + (x_0 + 2)];
	f_y0_x1 = f_data[(y_0 + 2)*(s_width + 4) + (x_1 + 2)];
	f_y1_x0 = f_data[(y_1 + 2)*(s_width + 4) + (x_0 + 2)];
	f_y1_x1 = f_data[(y_1 + 2)*(s_width + 4) + (x_1 + 2)];

	/* cal warped distance */
	float x_, y_;
	float A_x, A_y;
	float k = 5.5;
/*
	float f_y0_x2, f_y0_x_1, f_y2_x0, f_y_1_x0;

	f_y0_x2 = f_data[(y_0 + 2)*(s_width + 4) + (x_1 + 3)];
	f_y0_x_1 = f_data[(y_0 + 2)*(s_width + 4) + (x_0 + 1)];

	f_y2_x0 = f_data[(y_1 + 3)*(s_width + 4) + (x_0 + 2)];
	f_y_1_x0 = f_data[(y_0 + 1)*(s_width + 4) + (x_0 + 2)];

#define ABS(a) ((a)<0?-(a):(a))

	A_x = (ABS(f_y0_x1 - f_y0_x_1) - ABS(f_y0_x2 - f_y0_x0))/255;
	A_y = (ABS(f_y1_x0 - f_y_1_x0) - ABS(f_y2_x0 - f_y0_x0))/255;
*/
	A_x = coeff[(y_0*(int)s_width+x_0)*2];
	A_y = coeff[(y_0*(int)s_width + x_0) * 2+1];
	//x_ = s_x - (float)x_0;
	x_ = dx;
	x_ = x_ - k*A_x*x_*(x_ - 1);
	//y_ = s_y - (float)y_0;
	y_ = dy;
	y_ = y_ - k*A_y*y_*(y_ - 1);

	x_ = x_ > 1 ? 1 : x_;
	x_ = x_ < 0 ? 0 : x_;

	y_ = y_ > 1 ? 1 : y_;
	y_ = y_ < 0 ? 0 : y_;
	/* until here */
	/*
	f_y0 = f_y0_x0 + (s_x - (float)x_0)*(f_y0_x1 - f_y0_x0);
	f_y1 = f_y1_x0 + (s_x - (float)x_0)*(f_y1_x1 - f_y1_x0);
	f_inter = f_y0 + (s_y - (float)y_0)*(f_y1 - f_y0);
	*/

	f_y0 = f_y0_x0 + (x_)*(f_y0_x1 - f_y0_x0);
	f_y1 = f_y1_x0 + (x_)*(f_y1_x1 - f_y1_x0);
	f_inter = f_y0 + (y_)*(f_y1 - f_y0);

	return f_inter;
}

void ada_coeff(const float *f_data, float * coeff, DWORD s_width, DWORD s_height) {

	float a = 0.05;
	float
		f_y_1_x0, f_y_1_x1,
		f_y0_x_1, f_y0_x0, f_y0_x1, f_y0_x2,
		f_y1_x_1, f_y1_x0, f_y1_x1, f_y1_x2,
		f_y2_x0, f_y2_x1;
	for (int i = 0; i < s_height; ++i) {
		for (int j = 0; j < s_width; ++j) {
			float *H = coeff + (i*s_width + j) * 4;
			/**/
			f_y_1_x0 = f_data[(i + 1)*(s_width + 4) + (j + 2)];
			f_y_1_x1 = f_data[(i + 1)*(s_width + 4) + (j + 3)];

			f_y0_x_1 = f_data[(i + 2)*(s_width + 4) + (j + 1)];
			f_y0_x0 = f_data[(i + 2)*(s_width + 4) + (j + 2)];
			f_y0_x1 = f_data[(i + 2)*(s_width + 4) + (j + 3)];
			f_y0_x2 = f_data[(i + 2)*(s_width + 4) + (j + 4)];

			f_y1_x_1 = f_data[(i + 3)*(s_width + 4) + (j + 1)];
			f_y1_x0 = f_data[(i + 3)*(s_width + 4) + (j + 2)];
			f_y1_x1 = f_data[(i + 3)*(s_width + 4) + (j + 3)];
			f_y1_x2 = f_data[(i + 3)*(s_width + 4) + (j + 4)];

			f_y2_x0 = f_data[(i + 4)*(s_width + 4) + (j + 2)];
			f_y2_x1 = f_data[(i + 4)*(s_width + 4) + (j + 3)];
			/**/

#define ABS(a) ((a)<0?-(a):(a))
			H[0] = 1.0 / sqrtf(1 + a*(ABS(f_y0_x0 - f_y0_x_1) + ABS(f_y1_x0 - f_y1_x_1)));
			H[1] = 1.0 / sqrtf(1 + a*(ABS(f_y0_x1 - f_y0_x2) + ABS(f_y1_x1 - f_y1_x2)));
			H[2] = 1.0 / sqrtf(1 + a*(ABS(f_y0_x0 - f_y_1_x0) + ABS(f_y0_x1 - f_y_1_x1)));
			H[3] = 1.0 / sqrtf(1 + a*(ABS(f_y1_x0 - f_y2_x0) + ABS(f_y1_x1 - f_y2_x1)));
		}
	}
}

float ada_bilinear(const float* f_data, const float* coeff , float s_x, float s_y, DWORD s_width, DWORD s_height) {
	int y_0, x_0, y_1, x_1;
	float 
		f_y_1_x0, f_y_1_x1,
		f_y0_x_1, f_y0_x0, f_y0_x1, f_y0_x2,
		f_y1_x_1, f_y1_x0, f_y1_x1, f_y1_x2,
		f_y2_x0, f_y2_x1;
	float Hl, Hr, Vu, Vl;
	float w0v, w0h, w1h, w1v;
	float Dh, Dv;
	float s, t;
	float a = 0.05;
	float  f_inter;

	/*
	y_0 = floor(s_y);
	y_1 = y_0 + 1;

	x_0 = floor(s_x);
	x_1 = x_0 + 1;
	*/

	//

	float dx, dy;

	y_0 = floor(s_y);
	if (y_0 < 0) {
		y_0 = 0;
		dy = 0;
	}
	else if (y_0 > s_height - 1) {
		y_0 = s_height - 1;
		dy = 0;
	}
	else dy = s_y - float(y_0);
	y_1 = y_0 + 1;


	x_0 = floor(s_x);
	if (x_0 < 0) {
		x_0 = 0;
		dx = 0;
	}
	else if (x_0 > s_width - 1) {
		x_0 = s_width - 1;
		dx = 0;
	}
	else
		dx = s_x - float(x_0);
	x_1 = x_0 + 1;
	//

	//t = s_y - (float)y_0;
	//s = s_x - float(x_0);
	t = dy;
	s = dx;

	/*
	f_y_1_x0 = f_data[(y_0 + 1)*(s_width + 4) + (x_0 + 2)];
	f_y_1_x1 = f_data[(y_0 + 1)*(s_width + 4) + (x_1 + 2)];

	f_y0_x_1 = f_data[(y_0 + 2)*(s_width + 4) + (x_0 + 1)];
	f_y0_x0 = f_data[(y_0 + 2)*(s_width + 4) + (x_0 + 2)];
	f_y0_x1 = f_data[(y_0 + 2)*(s_width + 4) + (x_1 + 2)];
	f_y0_x2 = f_data[(y_0 + 2)*(s_width + 4) + (x_1 + 3)];

	f_y1_x_1 = f_data[(y_1 + 2)*(s_width + 4) + (x_0 + 1)];
	f_y1_x0 = f_data[(y_1 + 2)*(s_width + 4) + (x_0 + 2)];
	f_y1_x1 = f_data[(y_1 + 2)*(s_width + 4) + (x_1 + 2)];
	f_y1_x2 = f_data[(y_1 + 2)*(s_width + 4) + (x_1 + 3)];

	f_y2_x0 = f_data[(y_1 + 3)*(s_width + 4) + (x_0 + 2)];
	f_y2_x1 = f_data[(y_1 + 3)*(s_width + 4) + (x_1 + 2)];


#define ABS(a) ((a)<0?-(a):(a))
	Hl = 1.0 / sqrtf(1 + a*(ABS(f_y0_x0- f_y0_x_1) + ABS(f_y1_x0- f_y1_x_1)));
	Hr = 1.0 / sqrtf(1 + a*(ABS(f_y0_x1- f_y0_x2) + ABS(f_y1_x1- f_y1_x2)));
	Vu = 1.0 / sqrtf(1 + a*(ABS(f_y0_x0- f_y_1_x0) + ABS(f_y0_x1- f_y_1_x1)));
	Vl = 1.0 / sqrtf(1 + a*(ABS(f_y1_x0- f_y2_x0) + ABS(f_y1_x1- f_y2_x1)));
	*/
	
	f_y0_x0 = f_data[(y_0 + 2)*(s_width + 4) + (x_0 + 2)];
	f_y0_x1 = f_data[(y_0 + 2)*(s_width + 4) + (x_1 + 2)];

	f_y1_x0 = f_data[(y_1 + 2)*(s_width + 4) + (x_0 + 2)];
	f_y1_x1 = f_data[(y_1 + 2)*(s_width + 4) + (x_1 + 2)];

	Hl = coeff[(y_0*(int)s_width + x_0) * 4];
	Hr = coeff[(y_0*(int)s_width + x_0) * 4+1];
	Vu = coeff[(y_0*(int)s_width + x_0) * 4+2];
	Vl = coeff[(y_0*(int)s_width + x_0) * 4+3];
	


	Dh = Hl*(1-s)+Hr*s;
	Dv = Vu*(1-t)+Vl*t;
	w0h = Hl*(1-s)/Dh;
	w1h = Hr*s/Dh;
	w0v = Vu*(1-t)/Dv;
	w1v = Vl*t/Dv;

	f_inter = w0v*(w0h*f_y0_x0 + w1h*f_y0_x1) + w1v*(w0h*f_y1_x0 + w1h*f_y1_x1);

	return f_inter;
}

float cfls_bilinear(const float* f_data, float s_x, float s_y, DWORD s_width, DWORD s_height) {
	int y_0, x_0, y_1, x_1;
	float f_y0_x0, f_y0_x1, f_y1_x0, f_y1_x1;
	float f_y0_x_1, f_y0_x2, f_y_1_x0, f_y2_x0;
	float f_y0, f_y1, f_inter;

	/*
	y_0 = floor(s_y);
	y_1 = y_0 + 1;

	x_0 = floor(s_x);
	x_1 = x_0 + 1;
	*/

	/*
	y_0 = floor(s_y);
	y_0 = y_0 <= 0 ? 0 : y_0;
	y_0 = y_0 >= s_height - 1 ? s_height - 1 : y_0;
	y_1 = y_0 + 1;

	x_0 = floor(s_x);
	x_0 = x_0 <= 0 ? 0 : x_0;
	x_0 = x_0 >= s_width - 1 ? s_width - 1 : x_0;
	x_1 = x_0 + 1;
	*/
	float dx, dy;

	y_0 = floor(s_y);
	if (y_0 < 0) {
		y_0 = 0;
		dy = 0;
	}
	else if (y_0 > s_height - 1) {
		y_0 = s_height - 1;
		dy = 0;
	}
	else dy = s_y - float(y_0);
	y_1 = y_0 + 1;


	x_0 = floor(s_x);
	if (x_0 < 0) {
		x_0 = 0;
		dx = 0;
	}
	else if (x_0 > s_width - 1) {
		x_0 = s_width - 1;
		dx = 0;
	}
	else
		dx = s_x - float(x_0);
	x_1 = x_0 + 1;

	f_y0_x_1= f_data[(y_0 + 2)*(s_width + 4) + (x_0 + 1)];
	f_y0_x0 = f_data[(y_0 + 2)*(s_width + 4) + (x_0 + 2)];
	f_y0_x1 = f_data[(y_0 + 2)*(s_width + 4) + (x_1 + 2)];
	f_y0_x2 = f_data[(y_0 + 2)*(s_width + 4) + (x_1 + 3)];

	f_y_1_x0 = f_data[(y_0 + 1)*(s_width + 4) + (x_0 + 2)];
	f_y1_x0 = f_data[(y_1 + 2)*(s_width + 4) + (x_0 + 2)];
	f_y2_x0 = f_data[(y_1 + 3)*(s_width + 4) + (x_0 + 2)];
	f_y1_x1 = f_data[(y_1 + 2)*(s_width + 4) + (x_1 + 2)];

	/**/
	float x_, y_, s;
	float c0, c1, c2, c3;

	//s = s_x - (float)x_0;
	s = dx;
	c0 = (1-s)*(f_y0_x1- f_y0_x0);
	c1 = s*(1-s)*(f_y0_x_1 -f_y0_x0);
	c2 = s*(f_y0_x1 - f_y0_x0);
	c3 = s*(f_y0_x0 - (2-s)*f_y0_x1 +(1-s)*f_y0_x2);
	if (c0 + c2 != 0) {
		x_ = -1.0*(c1 + c3) / (c2 + c0);
		x_ = x_ < 0 ? 0 : x_;
		x_ = x_ > 1 ? 1 : x_;
	}
	else
		x_ = s;

	//s = s_y - (float)y_0;
	s = dy;
	c0 = (1 - s)*(f_y1_x0 - f_y0_x0);
	c1 = s*(1 - s)*(f_y_1_x0 - f_y0_x0);
	c2 = s*(f_y1_x0 - f_y0_x0);
	c3 = s*(f_y0_x0 - (2 - s)*f_y1_x0 + (1 - s)*f_y2_x0);

	if (c0 + c2 != 0) {
		y_ = -1.0*(c1 + c3) / (c2 + c0);
		y_ = y_ < 0 ? 0 : y_;
		y_ = y_ > 1 ? 1 : y_;
	}
	else
		y_ = s;

	/**/

	f_y0 = f_y0_x0 + x_*(f_y0_x1 - f_y0_x0);
	f_y1 = f_y1_x0 + x_*(f_y1_x1 - f_y1_x0);
	f_inter = f_y0 + y_*(f_y1 - f_y0);

	return f_inter;
}

void fill_data(const uint8_t* s_data, float * f_data, DWORD s_width, DWORD s_height) {
//将原 w*h 大小的图像 填充到 (w+4) * (h+4)中，即上下左右增加了两行数据。
//原边界外的元素的值用近邻元素的像素值代替
	DWORD f_width = s_width + 4;
	DWORD f_height = s_height +4 ;

	int s_i, s_j;
	for(int i=0; i < f_height ;++i)
		for (int j = 0; j < f_width; ++j) {
			
			s_i = (i - 2) < 0 ? 0 : (i - 2);
			s_i = (i - 2) >= (int)s_height ? s_height-1 : s_i;

			s_j = (j - 2) < 0 ? 0 : (j - 2);
			s_j = (j - 2) >= (int)s_width ? s_width - 1 : s_j;
			f_data[i*f_width+j] = (float)s_data[s_i * s_width + s_j];
			
			/*
			if (i < 2 || (i + 2) >= s_height || j < 2 || j + 2 >= s_width)
				f_data[i*f_width + j] = 0;
			else
				f_data[i*f_width + j] = (float)s_data[(i-2) * s_width + (j-2)];
			*/
		}
}

//

void fill_data(const float* s_data, float * f_data, DWORD s_width, DWORD s_height) {
	//将原 w*h 大小的图像 填充到 (w+4) * (h+4)中，即上下左右增加了两行数据。
	//原边界外的元素的值用近邻元素的像素值代替
	DWORD f_width = s_width + 4;
	DWORD f_height = s_height + 4;

	int s_i, s_j;
	for (int i = 0; i < f_height; ++i)
		for (int j = 0; j < f_width; ++j) {

			s_i = (i - 2) < 0 ? 0 : (i - 2);
			s_i = (i - 2) >= (int)s_height ? s_height - 1 : s_i;

			s_j = (j - 2) < 0 ? 0 : (j - 2);
			s_j = (j - 2) >= (int)s_width ? s_width - 1 : s_j;
			f_data[i*f_width + j] = (float)s_data[s_i * s_width + s_j];

			/*
			if (i < 2 || (i + 2) >= s_height || j < 2 || j + 2 >= s_width)
			f_data[i*f_width + j] = 0;
			else
			f_data[i*f_width + j] = (float)s_data[(i-2) * s_width + (j-2)];
			*/
		}
}


// w0, w1, w2, and w3 are the four cubic B-spline basis functions
inline float w0_c(float a)
{
	//    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
	return (1.0f / 6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

inline float w1_c(float a)
{
	//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
	return (1.0f / 6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

inline float w2_c(float a)
{
	//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
	return (1.0f / 6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

inline float w3_c(float a)
{
	return (1.0f / 6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
inline float g0_c(float a)
{
	return w0_c(a) + w1_c(a);
}

inline float g1_c(float a)
{
	return w2_c(a) + w3_c(a);
}

// h0 and h1 are the two offset functions
inline float h0_c(float a)
{
	// note +0.5 offset to compensate for CUDA linear filtering convention
	return -1.0f + w1_c(a) / (w0_c(a) + w1_c(a));// +0.5f;
}

inline float h1_c(float a)
{
	return 1.0f + w3_c(a) / (w2_c(a) + w3_c(a));// +0.5f;
}



inline float w0_d(float a) {
	return -0.75 * a * (a * (a - 2.0f) + 1.0f);
}
inline float w1_d(float a) {
	return (a*a*(1.25f*a - 2.25f) + 1.0f);
}
inline float w2_d(float a) {
	a = 1.0 - a;
	return (a*a*(1.25f*a - 2.25f) + 1.0f);
}
inline float w3_d(float a) {
	return -0.75f * a * a * (1 - a);
}
// g0 and g1 are the two amplitude functions
inline float g0_d(float a)
{
	return w0_d(a) + w1_d(a);
}

inline float g1_d(float a)
{
	return w2_d(a) + w3_d(a);
}

// h0 and h1 are the two offset functions
inline float h0_d(float a)
{
	// note +0.5 offset to compensate for CUDA linear filtering convention
	if ((w0_d(a) + w1_d(a)) == 0.0f)
		return -1.0f ;// +0.5f;
	else
		return -1.0f + w1_d(a) / (w0_d(a) + w1_d(a));
}

inline float h1_d(float a)
{
	if ((w2_d(a) + w3_d(a)) == 0.0f)
		return 1.0f;// +0.5f;
	else
		return 1.0f + w3_d(a) / (w2_d(a) + w3_d(a));// +0.5f;
}


float fast_new_spline(const float* f_data, float s_x, float s_y, DWORD s_width, DWORD s_height) {
	float x = s_x ;
	float y = s_y ;
	float px = floor(x);
	float py = floor(y);
	float fx = x - px;
	float fy = y - py;

	// note: we could store these functions in a lookup table texture, but maths is cheap
	/*
	float g0x = g0_d(fx);
	float g1x = g1_d(fx);
	float h0x = h0_d(fx);
	float h1x = h1_d(fx);
	float h0y = h0_d(fy);
	float h1y = h1_d(fy);

	float temp_x0 = px + h0x;
	float temp_x1 = px + h1x;
	float temp_y0 = py + h0y;
	float temp_y1 = py + h1y;


	float r = g0_d(fy) * (	g0x * bilinear(f_data, temp_x0 , temp_y0, s_width,  s_height) +
							g1x * bilinear(f_data, temp_x1, temp_y0, s_width, s_height))
		+
			g1_d(fy) *	(	g0x * bilinear(f_data, temp_x0, temp_y1, s_width, s_height) +
							g1x * bilinear(f_data, temp_x1 , temp_y1, s_width, s_height));
	*/

	float g0x = g0_c(fx);
	float g1x = g1_c(fx);
	float h0x = h0_c(fx);
	float h1x = h1_c(fx);
	float h0y = h0_c(fy);
	float h1y = h1_c(fy);

	float temp_x0 = px + h0x;
	float temp_x1 = px + h1x;
	float temp_y0 = py + h0y;
	float temp_y1 = py + h1y;


	float r = g0_c(fy) * (g0x * bilinear(f_data, temp_x0, temp_y0, s_width, s_height) +
		g1x * bilinear(f_data, temp_x1, temp_y0, s_width, s_height))
		+
		g1_c(fy) *	(g0x * bilinear(f_data, temp_x0, temp_y1, s_width, s_height) +
			g1x * bilinear(f_data, temp_x1, temp_y1, s_width, s_height));
	
	return r;

};


float origin_spline(const float* f_data, float s_x, float s_y, DWORD s_width, DWORD s_height) {
	float x = s_x;
	float y = s_y;
	float px = floor(x);
	float py = floor(y);
	float fx = x - px;
	float fy = y - py;

	// note: we could store these functions in a lookup table texture, but maths is cheap
	float w_x[4];
	float w_y[4];
	float sum_x[4];
	float r;
	/*
	w_x[0] = w0_d(fx);
	w_x[1] = w1_d(fx);
	w_x[2] = w2_d(fx);
	w_x[3] = w3_d(fx);


	w_y[0] = w0_d(fy);
	w_y[1] = w1_d(fy);
	w_y[2] = w2_d(fy);
	w_y[3] = w3_d(fy);
	*/

	w_x[0] = w0_c(fx);
	w_x[1] = w1_c(fx);
	w_x[2] = w2_c(fx);
	w_x[3] = w3_c(fx);

	w_y[0] = w0_c(fy);
	w_y[1] = w1_c(fy);
	w_y[2] = w2_c(fy);
	w_y[3] = w3_c(fy);
	for (int i = 0; i < 4; ++i) {
		sum_x[i] = 
			w_x[0] * f_data[((int)py - 1 + i + 2)*(s_width + 4) + ((int)px - 1 + 2)]
			+ w_x[1] * f_data[((int)py - 1 + i + 2)*(s_width + 4) + ((int)px + 0 + 2)]
			+ w_x[2] * f_data[((int)py - 1 + i + 2)*(s_width + 4) + ((int)px + 1 + 2)]
			+ w_x[3] * f_data[((int)py - 1 + i + 2)*(s_width + 4) + ((int)px + 2 + 2)];
	}

	r = 
		w_y[0] * sum_x[0] 
		+ w_y[1] * sum_x[1]
		+ w_y[2] * sum_x[2]
		+ w_y[3] * sum_x[3];

	return r;
};



void interpolation(const uint8_t *s_data, uint8_t *d_data, DWORD s_width, DWORD s_height, float width_scale, float height_scale, int MODE) {
	//prefix: d for destination  image; s for source image
	//
	clock_t start, end;
	double duration;
	start = clock();

	if (width_scale <= 0 || height_scale <= 0)
		return;
	DWORD d_width = s_width * width_scale;
	DWORD d_height = s_height * height_scale;

	//buffer with filling
	float *f_data = nullptr;
	f_data = (float *)malloc(sizeof(float)*(s_width + 4)*(s_height + 4));
	fill_data(s_data, f_data, s_width, s_height);

	//coefficient
	float *coeff=nullptr;
	float *a=nullptr;

	//compute coeef
	if (MODE == MODE_BILINEAR) {
		float *s_tmp_data ;
		s_tmp_data = (float *)malloc(sizeof(float)*s_width*s_height);
		for (int i = 0; i < s_width*s_height; ++i)
			s_tmp_data[i] = (float)s_data[i];


		CalBSplinePreFilter(s_tmp_data, s_width, s_height);

		fill_data(s_tmp_data, f_data, s_width, s_height);
	}
	else if (MODE == MODE_BICUBIC) {
		clock_t gen_start = clock();
		coeff = (float *)malloc(sizeof(float)*s_width*s_height * 16);
		bicubic_coeff(f_data, coeff, s_width, s_height);

		clock_t gen_end = clock();
		double gen_duration = gen_end - gen_start;
		std::cout << "time of constructing coefficient: "
			<< gen_duration / CLOCKS_PER_SEC << std::endl;
	}
	else if (MODE == MODE_SPLINE) {
		clock_t gen_start = clock();
		coeff = (float *)malloc(sizeof(float)*s_width*s_height * 16);
		bicubic_spline_coeff(f_data, coeff, s_width, s_height);
		clock_t gen_end = clock();
		double gen_duration = gen_end - gen_start;
		std::cout <<"time of constructing coefficient: "
			<<gen_duration/CLOCKS_PER_SEC<<std::endl;
	}
	else if (MODE == MODE_BC_KERNEL) {
		a = new float(-0.75);
		coeff = (float*)malloc(sizeof(float)*s_width*s_height*16);
		kernel_coeff(f_data,coeff,s_width,s_height);
	}
	else if (MODE == MODE_LANCZOS_KERNEL){
		//using a = 3, casues some kind of stripes?. unknown
		a = new float(2.0);
		coeff = (float*)malloc(sizeof(float)*s_width*s_height * 16);
		kernel_coeff(f_data, coeff, s_width, s_height);
	}
	else if (MODE == MODE_MITCHELL_KERNEL){
		a = new float[2]{ 1.0f / 3.0f, 1.0f / 3.0f };
		coeff = (float*)malloc(sizeof(float)*s_width*s_height * 16);
		kernel_coeff(f_data, coeff, s_width, s_height);
	}
	else if (MODE == MODE_BELL_KERNEL || MODE == MODE_HERMITE_KERNEL){
		//useless but for indicating using kernel
		a = new float(0);
		coeff = (float*)malloc(sizeof(float)*s_width*s_height * 16);
		kernel_coeff(f_data, coeff, s_width, s_height);
	}
	//nearest neighbour\ bilinear
	else if (MODE == MODE_WAD_BILINEAR) {
		clock_t gen_start = clock();
		coeff = (float*)malloc(sizeof(float)*s_width*s_height * 2);
		wad_coeff(f_data, coeff, s_width, s_height);

		clock_t gen_end = clock();
		double gen_duration = gen_end - gen_start;
		std::cout << "time of constructing coefficient: "
			<< gen_duration / CLOCKS_PER_SEC << std::endl;
	}
	else if (MODE == MODE_ADA_BILINEAR) {
		clock_t gen_start = clock();
		coeff = (float*)malloc(sizeof(float)*s_width*s_height * 4);
		ada_coeff(f_data, coeff, s_width, s_height);

		clock_t gen_end = clock();
		double gen_duration = gen_end - gen_start;
		std::cout << "time of constructing coefficient: "
			<< gen_duration / CLOCKS_PER_SEC << std::endl;
	}
	else	;

	uint8_t *output = d_data;
	float s_x, s_y;
	float temp;

	for (int i = 0; i < d_height; ++i) {
		//destination y ->  source y
		s_y = (float)i / (float)(d_height - 1) * (float)(s_height - 1);
		//s_y = (float)(i + 1) / height_scale + 0.5*(1.0-1.0/height_scale) -1;
		//if (s_y > 255 || s_y <0)
		//	break;

		for (int j = 0; j < d_width; ++j) {

			//destination x -> source x
			s_x = (float)j / (float)(d_width - 1) * (float)(s_width - 1);
			//s_x = float(j + 1) / width_scale + 0.5*(1.0 - 1.0 / width_scale) -1 ;
			if (MODE == MODE_BICUBIC || MODE == MODE_SPLINE)
				temp = cal_bicubic(coeff, s_x, s_y, s_width, s_height, MODE);
			//else if (MODE == MODE_BC_KERNEL || MODE == MODE_LANCZOS_KERNEL)
			else if (a != nullptr)
				//temp = cal_bicubic_kernel(f_data, s_x, s_y, s_width, s_height, a, MODE);
				temp = cal_bicubic_kernel(coeff, s_x, s_y, s_width, s_height, a, MODE);
			else if (MODE == MODE_NEAREST_NEIGHBOUR)
				temp = nearest_neighbour(f_data, s_x, s_y, s_width, s_height);
				//temp = fast_new_spline(f_data, s_x, s_y, s_width, s_height);
			else if (MODE == MODE_BILINEAR)
				//temp = fast_new_spline(f_data, s_x, s_y, s_width, s_height);
			temp = origin_spline(f_data, s_x, s_y, s_width, s_height);
			else if (MODE == MODE_WAD_BILINEAR)
				temp = wad_bilinear(f_data, coeff, s_x, s_y, s_width, s_height);
			else if (MODE == MODE_ADA_BILINEAR)
				temp = ada_bilinear(f_data, coeff, s_x, s_y, s_width, s_height);
			else if (MODE == MODE_CFLS_BILINEAR)
				temp = cfls_bilinear(f_data, s_x, s_y, s_width, s_height);
			
			temp += 0.5;
			if (temp >= 255)
				temp = 255.0;
			else if (temp <= 0)
				temp = 0;
			output[i*d_width + j] = (uint8_t)temp;
		}
	}

	if (a != nullptr)
		free(a);
	free(f_data);
	if(coeff!=nullptr)
		free(coeff);
	end = clock();
	duration = end - start;
	duration = duration / CLOCKS_PER_SEC;
	std::cout	<< "Time of scaling picture: "
				<<s_width<<"x"<<s_height 
				<<" in rate:"<<width_scale<<"x"<<height_scale
				<<" using method:"<<MODE_NAME[MODE]
				<<"	finished in:"<<duration<<"s"<<std::endl;
	/* record time */

	/*
	std::ofstream fout;
	fout.open(("D:\\Codes\\VS\\CXX\\Interpolation\\Interpolation\\OUTPUT\\record.txt"), std::ios::app);
	if (fout.is_open()) {
		fout << duration<<" ";
		fout.close();
	}
	*/
}




