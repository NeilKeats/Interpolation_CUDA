#pragma once
#ifndef BMP_Image
#define BMP_Image

#include<iostream>  
#include<fstream>
#include<Windows.h>  
#include<malloc.h>  
#include<stdlib.h>  
#include<stdio.h>  
#include<string.h>  
#include<inttypes.h>
#include<string>


#define RED_WEIGHT 0.2989
#define GREEN_WEIGHT 0.5870
#define BLUE_WEIGHT 0.1440

class bmp_i {
public:
	uint8_t *buf;                                //定义文件读取缓冲区  
	uint8_t *table;
	//FILE *fp;                                 //定义文件指针  
	//FILE *fpw;                                //定义保存文件指针  
	DWORD w, h;                                //定义读取图像的长和宽  
	DWORD bitCorlorUsed;                      //定义  
	DWORD bitSize;                            //定义图像的大小  
	BITMAPFILEHEADER bf;                      //图像文件头  
	BITMAPINFOHEADER bi;                      //图像文件头信息  

	//build from another bmp_i object, allocate and copy memory;
	bmp_i(const bmp_i * obj);
	//build from file, allocate memory and read data
	bmp_i(FILE *fp);
	~bmp_i() {
		free(buf);
		free(table);
	}
	void write_image(FILE *fpw);
	void resize(float width_scale, float height_scale);
	void cutimage(DWORD n_width, DWORD n_height, DWORD w_offset, DWORD h_offset);
	void write_buffer(const char *filename);
	void converte_to_grey();

private:
	//forbid those usages
	bmp_i(const bmp_i&) {};
	void operator =(const bmp_i &) {};

};

bmp_i::bmp_i(const bmp_i * obj) {
	buf = nullptr;
	table = nullptr;

	w = obj->w;
	h = obj->h;
	bitCorlorUsed = obj->bitCorlorUsed;
	bitSize = obj->bitSize;
	bf = obj->bf;
	bi = obj->bi;

	int xxx = bf.bfOffBits - (sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER));
	buf = (uint8_t*)malloc(w*h);
	table = (uint8_t*)malloc(xxx);
	memcpy(buf,obj->buf,w*h);
	memcpy(table,obj->table,xxx);

}

bmp_i::bmp_i(FILE *fp) {

	fread(&bf, sizeof(BITMAPFILEHEADER), 1, fp);//读取BMP文件头文件  
	fread(&bi, sizeof(BITMAPINFOHEADER), 1, fp);//读取BMP文件头文件信息  
	w = bi.biWidth;                            //获取图像的宽  
	h = bi.biHeight;                           //获取图像的高  
	bitSize = bi.biSizeImage;                  //获取图像的size  
	//buf = (char*)malloc(w*h * 3);                //分配缓冲区大小  
	buf = (uint8_t*)malloc(w*h);                //分配缓冲区大小  
	int xxx = bf.bfOffBits - (sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER));
	int xxx_0 = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
	table = (uint8_t*)malloc(xxx);
	fseek(fp, xxx_0, 0);
	fread(table, xxx, 1, fp);

	fseek(fp, bf.bfOffBits, 0);//定位到像素起始位置  
	//fread(buf, 1, w*h * 3, fp);                   //开始读取数据  
	int aligned_width = (w+3)/4*4;
	for (int i = 0; i<h; ++i){
		fseek(fp, bf.bfOffBits+aligned_width*i, 0);
		fread(buf+w*i, 1, w, fp);                   //开始读取数据  
	}
}

void bmp_i::write_image(FILE *fpw) {

	if (buf == nullptr || fpw == nullptr)
		return;
	FILE *fp = fpw;
	fwrite(&(this->bf), sizeof(BITMAPFILEHEADER), 1, fp);  //写入文件头  
	fwrite(&(this->bi), sizeof(BITMAPINFOHEADER), 1, fp);  //写入文件头信息  
	int xxx = bf.bfOffBits - (sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER));

	/*
	RGBQUAD *board = new RGBQUAD[256];
	for (int i = 0; i < 256; ++i) {
		board[i].rgbBlue = i;
		board[i].rgbGreen = i;
		board[i].rgbRed = i;
		board[i].rgbReserved = 0;
	}
	*/

	//fwrite(board, sizeof(RGBQUAD), 256, fp);
	//delete board;
	fwrite(this->table, xxx, 1, fp);
	
	uint8_t *p = this->buf;
	int aligned_width = (w + 3) / 4 * 4;
	int i, j;
	uint8_t empty = 0;
	for ( i = 0; i<h; i++)
	{
		fseek(fp, bf.bfOffBits + aligned_width*i, 0);
		for ( j = 0; j<w; j++)
		{
			fwrite(p++, 1, 1, fp);
		}
		for (; j<aligned_width; j++)
			fwrite(&empty, 1, 1, fp);

	}
};

void bmp_i::resize(float width_scale, float height_scale) {

	if (width_scale <= 0 || height_scale<=0)
		return;

	this->bi.biWidth = this->bi.biWidth*width_scale;
	this->w = this->bi.biWidth;

	this->bi.biHeight = this->bi.biHeight*height_scale;
	this->h = this->bi.biHeight;

	//considering padding
	DWORD aligned_width = (this->w+3)/4*4;
	this->bi.biSizeImage = aligned_width*this->h;
	this->bitSize = this->bi.biSizeImage;

	this->bf.bfSize = 
		bf.bfOffBits + this->bitSize;

	if (buf != nullptr)
		free(buf);
	this->buf = (uint8_t*)malloc(w*h);
}

void bmp_i::write_buffer(const char *filename) {

	if (buf == nullptr)
		return;

	std::ofstream outfile;
	outfile.open(filename);

	uint8_t *p = this->buf;
	int i, j;
	std::string str_tmp;
	uint8_t empty = 0;
	for (i = 0; i<h; i++)
	{
		for (j = 0; j<w; j++)
		{
			str_tmp = std::to_string(*p++);
			outfile << str_tmp <<" ";
		}
		outfile << std::endl;
	}
	outfile.close();
}

void bmp_i::converte_to_grey() {

	uint8_t *p = buf;
	RGBQUAD *original_board = (RGBQUAD*)table;
	uint8_t temp;
	float color;

	for(int i=0 ; i<h; ++i)
		for (int j = 0; j < w; ++j) {
			temp = p[i*w+j];
			
			//if color_table[temp] is gray scale
			//then no color convert, in order to avoid bias due to conversion
			if (
				(original_board[temp].rgbRed == temp) &&
				(original_board[temp].rgbGreen == temp) &&
				(original_board[temp].rgbBlue == temp)
				)
				continue;

			
			//color conversion from RGB to gray scale
			color =
				  (float)original_board[temp].rgbRed*RED_WEIGHT
				+ (float)original_board[temp].rgbGreen*GREEN_WEIGHT
				+ (float)original_board[temp].rgbBlue*BLUE_WEIGHT;

			//range check
			color = color > 255 ? 255.0 : color;
			color = color < 0 ? 0 : color;
			p[i*w + j] = color;
		}

	//convert the color table
	for (int i = 0; i < 256; ++i) {
		original_board[i].rgbRed = i;
		original_board[i].rgbGreen = i;
		original_board[i].rgbBlue = i;
		original_board[i].rgbReserved = 0;
	}
}

void bmp_i::cutimage(DWORD n_width, DWORD n_height, DWORD w_offset, DWORD h_offset) {
	if (w_offset + n_width > w || h_offset + n_height > h || this->buf == nullptr)
		return;

	uint8_t *n_buffer = (uint8_t *)malloc(n_width*n_height);
	for (int i = 0; i < n_height; ++i)
		for (int j = 0; j < n_width; ++j) {
			n_buffer[i*n_width + j] =
				this->buf[(i + h_offset)*this->w + (w_offset + j)];
		}
	
	this->bi.biWidth = n_width;
	this->w = n_width;
	this->bi.biHeight = n_height;
	this->h = n_height;

	//considering padding
	DWORD aligned_width = (this->w + 3) / 4 * 4;
	this->bi.biSizeImage = aligned_width*this->h;
	this->bitSize = this->bi.biSizeImage;

	this->bf.bfSize =
		bf.bfOffBits + this->bitSize;


	free(this->buf);
	this->buf = n_buffer;
	return;

}

bmp_i* bmp_file_read(const char* fileName) {
	FILE *fp;
	if ((fp = fopen(fileName, "rb")) == NULL)
	{
		std::cerr << "文件未找到！" << std::endl;
		return nullptr;
	}
	bmp_i *temp = new bmp_i(fp);
	fclose(fp);
	return temp;
}

void bmp_file_write(bmp_i *bmp, const char* fileName) {
	if (bmp == nullptr || fileName == nullptr) {
		std::cerr << "Error: pointer is null！"<<std::endl;
		return;
	}

	FILE *fpw = nullptr;

	if ((fpw = fopen(fileName, "wb")) == NULL)
	{
		std::cerr << "文件未找到！"<<std::endl;
		return ;
	}
	bmp->write_image(fpw);
	fclose(fpw);
}

float bmp_compare(const bmp_i *tar, const bmp_i *ref, float *t_aver, float *r_aver) {
	if (tar->w != ref->w || tar->h != ref->h)
		return -1;

	unsigned long t_total, r_total , bias;
	t_total = r_total = bias = 0;
	int tar_value, ref_value , temp;
	for(int i=0; i< tar->h ; ++i)
		for (int j = 0; j < tar->w; ++j) {

			tar_value = tar->buf[i*tar->w + j];
			ref_value = ref->buf[i*tar->w + j];

			t_total += tar_value;
			r_total += ref_value;

			temp = tar_value - ref_value;
			temp = temp < 0 ? -temp : temp ;
			bias += temp*temp;
		}

	*t_aver = (double)t_total / (double)(tar->w*tar->h);
	*r_aver = (double)r_total / (double)(tar->w*tar->h);
	return (double)bias/(double)(tar->w*tar->h);;

}

#endif // !BMP_Image
