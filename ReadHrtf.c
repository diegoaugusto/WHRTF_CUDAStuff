/*
 *  ReadHrtf.c
 *  WHRTF
 *
 *  Created by Diego Gomes on 25/03/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "MsgUtil.h"
#include "MathUtil.h"

/* Inclusão do respectivo módulo de definição */
#define READHRTF_SER
#include "ReadHrtf.h"
#undef READHRTF_SER

// ########################################
// prototypes
float** readHrtf(int elev, int azim, char ear, int minCoef, int maxCoef);
char* hrtfpath(char* root, char dir_ch, char* subdir, char select, char* ext, int elev, int azim);
float* readraw(char* filePath, int* hrtfFileSize);

// ########################################
/**
 @elev is the elevation from -40 to 90 degrees
 @azim is azimuth from 0 to 360 derees
 @ear is:
 'L' use full data from left pinna
 'R' use full data from right pinna
 'H' use compact data 
 
 x - Returns stereo symmetrical hrtf in first two rows of x such that left is first row, right is second row.
 */
float** readHrtf(int elev, int azim, char ear, int minCoef, int maxCoef) {
	char* rootDir = "kemar";
	char dirChar = '/';
	char* ext = ".dat";
	int flipAzim = 0;
	char* pathname = NULL;
	
	//printf("elev = %d, azim = %d, ear = %c\n", elev, azim, ear);
	if ((elev < -40) || (elev > 90)) {
		char* msg = "Elevation must be between -40 and 90 degrees.\n";
		showError(msg);
		free(msg);
	}
	
	if (azim < 0) {
		azim = 360 + azim;
	}
	
	flipAzim = 360 - azim;
	if (flipAzim == 360) {
		flipAzim = 0;
	}
	
	float** x = (float**) calloc(2, sizeof(float*));
	int hrtfFileSize;
	
	if (ear == 'L') {
		pathname = hrtfpath(rootDir,dirChar,"full",ear,ext,elev,azim);
		//printf("PathName: %s\n\n", pathname);
		x[0] = readraw(pathname, &hrtfFileSize);
		pathname = hrtfpath(rootDir,dirChar,"full",ear,ext,elev,flipAzim);
		x[1] = readraw(pathname, &hrtfFileSize);
	} else if (ear == 'R') {
		pathname = hrtfpath(rootDir,dirChar,"full",ear,ext,elev,flipAzim);
		//printf("PathName: %s\n\n", pathname);
		x[0] = readraw(pathname, &hrtfFileSize);
		pathname = hrtfpath(rootDir,dirChar,"full",ear,ext,elev,azim);
		x[1] = readraw(pathname, &hrtfFileSize);
	} else if (ear == 'H') {
		pathname = hrtfpath(rootDir,dirChar,"compact",ear,ext,elev,azim);
		//tmp = readraw(pathname);
		//x(1,:) = tmp(1:2:length(tmp));
		//x(2,:) = tmp(2:2:length(tmp));
	} else {
		char* msg = (char*) malloc(40*sizeof(char));
		char temp[2];
		temp[0] = ear;
		temp[1] = '\0';
		strcpy(msg, temp);
		strcat(msg, " not a valid selection. Use L, R, or H.");
		showError(msg);
		free(msg);
	}
	free(pathname);
	
	//	NORMALIZATION OF HRTF (NORM OF DATA WAS > 1 AND WAS CAUSING DISTORTION IN WAVE FILES)
	char* AUXpath = hrtfpath(rootDir,dirChar,"full",'R',ext,40,45); 
	int auxFileSize;
	int size = maxCoef-minCoef;
	float* AUXx = readraw(AUXpath, &auxFileSize);
	float* aux = (float*) calloc(size, sizeof(float));
	
	// obtém coeficientes 24 a 173 (150 elementos) para normalização
	int j = 0;
	for (int i = minCoef; i < maxCoef; i++,j++) {
		aux[j] = AUXx[i];
	}
	
	float norma = norm(aux, size);
	free(aux);
	free(AUXpath);
	
	for (int i = 0; i < hrtfFileSize; i++) {
		x[0][i] = x[0][i]/norma;
		x[1][i] = x[1][i]/norma;
	}
	
	j = 0;
	float** y = (float**) calloc(2, sizeof(float*));
	y[0] = (float*) calloc(size, sizeof(float));
	y[1] = (float*) calloc(size, sizeof(float));
	
	for (int i = minCoef; i < maxCoef; i++, j++) {
		y[0][j] = x[0][i];
		y[1][j] = x[1][i];
	}

	free(x);
	
	return y;
}

/*
 Return pathanme for HRTF data file:
 @root is root directory.
 @dir_ch is directory character, '/' (unix) or ':' (mac).
 @subdir is 'compact', 'full', etc.
 @select is 'L', 'R' or 'H'.
 @ext is the filename extension '.dat', etc.
 @elev is elevation.
 @azim is azimuth.
 */
char* hrtfpath(char* root, char dir_ch, char* subdir, char ear, char* ext, int elev, int azim) {
	char* filePath = (char*) malloc(255*sizeof(char));
	char azimStr[4];
	
	// Convertendo caracteres e inteiros para string
	sprintf(azimStr,"%d",azim);
	
	if (azim < 10 || azim < 100) {
		char* zeros = (char*) malloc(3*sizeof(char));
		if (azim < 10) {
			strcpy(zeros, "00");
		} else if (azim < 100) {
			strcpy(zeros, "0");
		}
		sprintf(filePath, "%s%c%s%celev%d%c%c%de%s%da%s", root, dir_ch, subdir, dir_ch, elev, dir_ch, ear, elev, zeros, azim, ext);
		free(zeros);
	} else{
		sprintf(filePath, "%s%c%s%celev%d%c%c%de%da%s", root, dir_ch, subdir, dir_ch, elev, dir_ch, ear, elev, azim, ext);
	}
	
	return filePath;
}


float* readraw(char* filePath, int *hrtfFileSize) {
	FILE* pFile = NULL;
	long lSize;
	char* buffer = NULL;
	size_t result;
	
	// TODO remove POG
	char* basePath = "/Users/diego/Dev/Projetos/Mestrado/";
	char* endPath = (char*) calloc((strlen(basePath)+strlen(filePath)+1), sizeof(char));
	strcat(endPath, basePath);
	strcat(endPath, filePath);
	
	pFile = fopen ( endPath , "rb");
	if (pFile==NULL) {
		fputs ("File error",stderr); 
		exit(1);
	}
	
	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	lSize = ftell (pFile);
	rewind (pFile);
	
	// allocate memory to contain the whole file:
	buffer = (char*) calloc (lSize, sizeof(char));
	if (buffer == NULL) {
		fputs ("Memory error",stderr); 
		exit (2);
	}
	
	// copy the file into the buffer (byte by byte):
	result = fread (buffer,1,lSize,pFile);
	if (result != lSize) {
		fputs ("Reading error",stderr); 
		exit (3);
	}
	
	// create short value from two bytes
	float* rawValue = (float*) calloc((result/2), sizeof(float));
	for (int j = 0; j < result; j=j+2) {
		short val = (short)( ((buffer[j]&0xFF)<<8) | (buffer[j+1]&0xFF) );
		rawValue[j/2] = ((float)val)/32768.0f;
		//printf("j = %d, rawValue[%d] = %1.15f\n",j/2,j/2, rawValue[j/2]);
	}
	
	
	
	/* the whole file is now loaded in the memory buffer. */
	*hrtfFileSize = result/2;
	
	// terminate 
	fclose(pFile);
	free(buffer);
	free(endPath);
	
	return rawValue;
}