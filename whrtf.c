/*
 *  whrtf.c
 *  WHRTF
 *
 *  Created by Diego Gomes on 24/05/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include "whrtfConstants.h" 
#include "ReadHrtf.h"
#include "MathUtil.h"
#include "runtime.h"

/* Inclusão do respectivo módulo de definição */
#define WHRTF_SER
#include "whrtf.h"
#undef WHRTF_SER

// ########################################
// prototypes
float** getCoefSpars(int elev, int azim, char ear, int* G_size);
float* getRespImp(int numFiltros, float** G, int* G_size, int* resultLength);
float* whrtf (int elev, int azim, char ear, int* whrtfLength);


// Extern functions
extern void initCUDA(void);
extern short* findDelay(float** hrtf, int length);
extern float* shiftInParallel(float* vec, int vecLength, short delay, int maxLength);
extern void coef_spars(char* filtro[], int filtroLength, float* ho1d, int ho1dLength, float** G_aux, int* G_size);
extern float* resp_imp(char* filtros[], int numFiltros, float** G, int* G_size, int* resultLength);
extern void coef_spars2(char* filtro[], int numFiltros, float* ho1d, int ho1dLength, float** G_aux, int* G_size);


void printDelay(short* delay) {
	for (int i = 0; i < 2; i++) {
		printf("delay[%d] = %d\n", i, delay[i]);
	}
}

void printDelayedHrtf(float* h, int vecLength) {
	for (int i = 0; i < vecLength; i++) {
		printf("h[%d] = %1.15f\n", i, h[i]);
	}
}

/**
 *	Esta função retorna os coeficientes esparsos de uma hrtf com elevação e azimute conhecidos.
 */
float** getCoefSpars(int elev, int azim, char ear, int* G_size) {	
	float** hrtf = readHrtf(elev, azim, 'L', MIN_SAMPLE_NUMBER, MAX_SAMPLE_NUMBER);	
	short* delay = findDelay(hrtf, MAX_SAMPLE_NUMBER - MIN_SAMPLE_NUMBER);
	free(hrtf);
	
	//printDelay(delay);
	
	// hrtf sem os primeiros 24 coeficientes
	float** ho = readHrtf(elev, azim, 'R', MIN_SAMPLE_NUMBER, MAX_HRTF_SIZE);
	float* ho1d = NULL;
	
	int length = NUM_COEF_WITHOUT_BEGINNING - ((delay[0] > delay[1]) ? delay[0] : delay[1]);
	
	if (ear == 'L') {
		ho1d = shiftInParallel(ho[0], NUM_COEF_WITHOUT_BEGINNING, delay[0], length);
	} else if (ear == 'R') {
		ho1d = shiftInParallel(ho[1], NUM_COEF_WITHOUT_BEGINNING, delay[1], length);
	}
	
	free(ho);
	
	//printDelayedHrtf(ho1d, length);
	
	float** G_aux = NULL;
	G_aux = (float**) malloc((NUM_FILTROS+1) * sizeof(float*));
	
	//INIT_VARIABLES; INIT_RUNTIME;
	//coef_spars(WAVELET, NUM_FILTROS, ho1d, length, G_aux, G_size);
	//coef_spars2(WAVELET, NUM_FILTROS, ho1d, length, G_aux, G_size);
	coef_spars_host(WAVELET, NUM_FILTROS, ho1d, length, G_aux, G_size);
	//END_RUNTIME; printf("\n[coef_spars2]: "); PRINT_RUNTIME;
	
	// TODO: implementar atraso = calc_delta(Wavelet);
	int atraso[5] = {1, 1, 8, 22, 50};
	
	int novoG_size[NUM_FILTROS+1];
	for (int i = 0; i < NUM_FILTROS+1; i++) {
		novoG_size[i] = atraso[i] + G_size[i];
	}
	
	int maxValueOfGSize = max(novoG_size, NUM_FILTROS+1);
	float** G = (float**) malloc((NUM_FILTROS + 1) * sizeof(float*));
	
	for (int i = 0; i < (NUM_FILTROS + 1); i++) {
		G[i] = (float*) calloc(maxValueOfGSize, sizeof(float));
		
		int j = 0;
		for (j; j < atraso[i]; j++) {
			G[i][j] = 0.0;
		}
		for (j; j < (G_size[i]+atraso[i]); j++) {
			G[i][j] = G_aux[i][j-atraso[i]];
		}
		for (j; j < (maxValueOfGSize - (G_size[i] + atraso[i])); j++) {
			G[i][j] = 0.0;
		}
	}
	
	return G;
}

/**
 *	Esta função retorna a resposta impulsiva tendo como entrada os coeficientes esparsos
 */
float* getRespImp(int numFiltros, float** G, int* G_size, int* resultLength) {
	int auxWhrtfLength;
	float* respImp = resp_imp(WAVELET, NUM_FILTROS, G, G_size, &auxWhrtfLength);
	*resultLength = auxWhrtfLength;
	return respImp;
}

// Host code
float* whrtf (int elev, int azim, char ear, int* whrtfLength) {
	INIT_VARIABLES;
	
	int* G_size = (int*) calloc((NUM_FILTROS+1), sizeof(int));
	//INIT_RUNTIME;
	float** G = getCoefSpars(elev, azim, ear, G_size);
	//END_RUNTIME; printf("\n[getCoefSpars]: "); PRINT_RUNTIME;
	
	//INIT_RUNTIME;
	float* whrtf = getRespImp(NUM_FILTROS, G, G_size, &*whrtfLength);
	//END_RUNTIME; printf("\n[getRespImp]: "); PRINT_RUNTIME;
	return whrtf;
}