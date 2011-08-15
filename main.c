/*
*  main.cpp
*  WHRTF
*
*  Created by Diego Gomes on 24/05/11.
*  Copyright (c) 2011 __MyCompanyName__. All rights reserved.
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

#include "cuda.h"

#include "runtime.h"

/* Inclusão do respectivo módulo de definição */
#define WHRTF_SER
#include "whrtf.h"
#undef WHRTF_SER

#define SPARSECOEF_SER
#include "sparseCoefficients.h"
#undef SPARSECOEF_SER

extern float* convFFT(float* signal, int signalLength, float* filter, int filterLength);


#define NUM_FILTROS 4
#define NUM_THREADS	5

struct whrtf_parameters {
	int tid;
	int elev;
	int azim;
	char ear;
	int whrtfLength;
};

void *whrtfMain(void *arg) {
	struct whrtf_parameters *my_param;
	my_param = (struct whrtf_parameters *) arg;
	
	int tid = my_param->tid;
	int elev = my_param->elev;
	int azim = my_param->azim;
	char ear = my_param->ear;
	int whrtfLength = my_param->whrtfLength;
	
	cudaSetDevice(0);
	
	//CUdevice	hDevice;
	//CUcontext	hContext, currentContext;
	
	//cuDeviceGet(&hDevice, 0);
	//cuCtxCreate(&hContext, 0, hDevice);	
	
	float* whr = whrtf(elev, azim, ear, &whrtfLength);
	
	if (0) {
		for (int i = 0; i < 3; i=i+3) {
			//printf("\nColunas %d a %d \n", i, i+2);
			printf("[%d] - %1.15f, %1.15f, %1.15f\n", tid, whr[i], whr[i+1], whr[i+2]);
		}
	}
	
	//cuCtxDestroy(hContext);
	//cudaThreadExit();
}

int main0 (int argc, char * const argv[]) {
    int elev = 0;
	int azim = 90;
	char ear = 'L';
	int whrtfLength;
	
	initCUDA();
	
	/*
	INIT_VARIABLES; INIT_RUNTIME;
	float* whr = whrtf(elev, azim, ear, &whrtfLength);
	END_RUNTIME; printf("\n[whrtf]: "); PRINT_RUNTIME;
	 
	cudaDeviceReset();
	 
	if (0) {
		for (int i = 0; i < whrtfLength; i=i+3) {
			printf("\nColunas %d a %d \n", i, i+2);
			printf("%1.15f, %1.15f, %1.15f\n", whr[i], whr[i+1], whr[i+2]);
		}
	}*/
	 
	// OK - REVISADO ATÉ AQUI - 24/05/2011
	
	
	// Implementação com threads
	struct whrtf_parameters *param1, *param2;
	param1 = (struct whrtf_parameters *) malloc(sizeof(struct whrtf_parameters));
	param2 = (struct whrtf_parameters *) malloc(sizeof(struct whrtf_parameters));
	
	param1->tid = 0;
	param1->elev = 0;
	param1->azim = 90;
	param1->ear = 'L';
	
	param2->tid = 1;
	param2->elev = 0;
	param2->azim = 95;
	param2->ear = 'L';
 	
	int numThreads = 2;
	int rc[numThreads];
	pthread_t thread1, thread2, thread3, thread4;
	pthread_t threads[numThreads];
	
	INIT_VARIABLES; INIT_RUNTIME;
	for (int i = 0; i < numThreads; i++) {
		if (i % 2 == 0) {
			rc[i] = pthread_create(&threads[i], NULL, whrtfMain, (void *) param1);
		} else {
			rc[i] = pthread_create(&threads[i], NULL, whrtfMain, (void *) param2);
		}

	}
	
	for (int i = 0; i < numThreads; i++) {
		pthread_join( threads[i], NULL);
	}
	
	/*pthread_join( thread1, NULL);
	pthread_join( thread2, NULL);
	pthread_join( thread3, NULL);
	pthread_join( thread4, NULL);*/
	END_RUNTIME; printf("\n[%d threads]: ", numThreads); PRINT_RUNTIME;
		
	cudaDeviceReset();
	

	//printf("Thread 1 returns: %d\n", rc1);
	//printf("Thread 2 returns: %d\n", rc2);
	//printf("Thread 3 returns: %d\n", rc3);
	//printf("Thread 4 returns: %d\n", rc4);
	//exit(0);
	
	 
}


int main (int argc, char * const argv[]) {
	int elev = 0;
	int azim = 1;
	char ear = 'L';
	
	double** Gl = NULL;
	int* Gl_size = NULL;
	double** Gr = NULL;
	int* Gr_size = NULL;
	
	if (elev > 90) {
		elev = elev - (2 * (elev - 90));
	}
	
	initCUDA();
	
	INIT_VARIABLES;
	INIT_RUNTIME;
	
	//for (azim = 0; azim < 1; azim++) {
		int flipAzim = 360 - azim;
		if (flipAzim == 360) {
			flipAzim = 0;
		}		
		
		int whrtfLengthL, whrtfLengthR;
	
		Gl_size = (int*) calloc((NUM_FILTROS+1), sizeof(int));
		Gl = getSparseCoefficients(elev, azim, ear, Gl_size);
		float* whrtfL = getRespImp(NUM_FILTROS, Gl, Gl_size, &whrtfLengthL);

		Gr_size = (int*) calloc((NUM_FILTROS+1), sizeof(int));
		Gr = getSparseCoefficients(elev, flipAzim, ear, Gr_size);
		float* whrtfR = getRespImp(NUM_FILTROS, Gr, Gr_size, &whrtfLengthR);
	
	END_RUNTIME; printf("\n[2 whrtfs]: "); PRINT_RUNTIME;
	
	if (1) {
		for (int i = 0; i < 3; i=i+3) {
			//printf("\nColunas %d a %d \n", i, i+2);
			printf("%1.15f, %1.15f, %1.15f\n", whrtfL[i], whrtfL[i+1], whrtfL[i+2]);
		}
	}
	
	
		free(Gl);
		free(Gr);
		free(Gl_size);
		free(Gr_size);
		free(whrtfL);
		free(whrtfR);
		
		Gl = NULL;
		Gr = NULL;
		Gl_size = NULL;
		Gr_size = NULL;
		whrtfL = NULL;
		whrtfR = NULL;
 	//}
	
	
	
	// TODO implementar calc_delta
	int atrasos[5] = {1, 1, 8, 22, 50};

		
}