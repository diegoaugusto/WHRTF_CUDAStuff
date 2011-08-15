/*
 *  WHRTF.cu
 *  WHRTF
 *
 *  Created by Diego Gomes on 23/03/11.
 *  Copyright (c) 2011 __MyCompanyName__. All rights reserved.
 *
 */
 
 // Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>
#include "runtime.h"

// includes, project
#include <cufft.h>

#define DB8_CONST "db8"
#define NUM_STAGES 5;

// Global variables
int threadsPerBlock = 8;
int blocksPerGrid = 16;

// ################## Internal Functions

// Find delay
double sumSquaresOfArrayElements(float* hrtf, int vecLength);
double sumArrayElements(float* vec0, int vecLength);
float** getSquaresOfArrayElements(float* vec0, float* vec1, int vecLength);
float** sumColumns(float* vec0, float* vec1, int vecLength);
short* findIndexesGreaterThan(float* vec, int vecLength, double minValue);

// Sparse coefficients
float** leFiltros(char* filtro, int* filtroLength);
int max(int* vec, int vecLength);
float*** computeIpAux(float*** Hp, int hlength, float*** Fp, int flength);
float** calculateGp(float** Rp, int rpLength, float*** Fp, int fpLength, float*** Ip, int ipLength);
void dec_poly(double** h, int hlength, float* sist, int ho1dLength, float** G, int* G_size);
double* getPartialVector(double* vec, int numOfElements);

// convolution
float* convHost1(float* Gl_old, int filtroLength, float* sparsArray, int sparsLength);


// CUFFT
// Complex data type
typedef float2 Complex; 
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex*, const Complex*, int, float);

// Filtering functions
void Convolve(const Complex*, int, const Complex*, int, Complex*);

// Padding functions
int PadData(const Complex*, Complex**, int, const Complex*, Complex**, int);

// declaration, forward
//float* convFFT(float* signal, int signalLength, float* filter, int filterLength);
float* convSimple(float* signal, int signalLength, float* filter, int filterLength);


// Memory management
void cleanHostMemory(void* h);
void cleanDeviceMemory(void* d);

// CUDA Error
void checkCUDAError(const char *msg);


// Extern functions
extern "C" void initCUDA(void);
extern "C" short* findDelay(float** hrtf, int length);
extern "C" float* shiftInParallel(float* vec, int vecLength, short delay, int maxLength);
extern "C" void coef_spars(char* filtro[], int filtroLength, float* ho1d, int ho1dLength, float** G_aux, int* G_size);
extern "C" float* resp_imp(char* filtros[], int numFiltros, float** G, int* G_size, int* resultLength);
extern "C" float* convFFT(float* signal, int signalLength, float* filter, int filterLength);
extern "C" void coef_spars2(char* filtro[], int numFiltros, float* ho1d, int ho1dLength, float** G_aux, int* G_size);
extern "C" void coef_spars_host(char* filtro[], int numFiltros, float* ho1d, int ho1dLength, float** G_aux, int* G_size);


// ################## Device code

// Find delay
__global__ void VecMult(const float* vec0, const float* vec1, float* result0, float* result1, int vecLength) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < vecLength) {
        result0[i] = (vec0[i] * vec0[i]);
		result1[i] = (vec1[i] * vec1[i]);
	}
}

__global__ void sumColumnsInParallel(float* vec, float* result, int vecLength) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < vecLength) {
		float sum = 0.0;
		for (int j = 0; j < i; j++) {
			sum += vec[j];
		}		
		result[i] = (i == 0) ? vec[i] : sum;
	}
}


// Shift in parallel
__global__ void shiftArrayElementsInParallel(float* vec, int vecLength, float* result, short delay) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < vecLength-delay) {
        result[i] = vec[i+delay];
	}
}


// Sparse coefficients
__global__ void multiplyPolynomialCoefficientsKernel(const float* vecA, const int vecASize, const float* vecB, const int vecBSize, float* vecC) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < vecASize) {
        for (int j = 0; j < vecBSize; j++) {
			vecC[i+j] = vecC[i+j] + (vecA[i] * vecB[j]);
		}
	}
}

/**
 *	Kernel que inicia os valores de um array com 0.0.
 */
__global__ void initializeArray(float* array, int arrayLength) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < arrayLength) {
		array[tid] = 0.0;
	}
}

__global__ void sumColumns(float* dev_aux1, int aux1Length, float* dev_aux_sum, int resultSize) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < resultSize && i < aux1Length) {
		dev_aux_sum[i] += dev_aux1[i];
	}
}


__global__ void sumPolynomialCoefficientsKernel(float* dev_aux1, float* dev_aux2, float* dev_aux_sum, int resultSize) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < resultSize) {
		dev_aux_sum[i] = dev_aux1[i] + dev_aux2[i];
	}
}

/**
 *	Não é usada.
 */
__global__ void conv(float* u, int uLength, float* v, int vLength, float* w, int wLength, int maxLength, int minLength) {
	// seguindo a referência: http://www.mathworks.com/help/techdoc/ref/conv.html
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (tid < wLength) {
		for (int j = 0; j < maxLength; j++ ) {
			if ((tid - j) >= 0 && (tid - j) < minLength) {
				if (maxLength == uLength) {
					w[tid] += (u[j] * v[tid-j]);    // convolve: multiply and accumulate
				} else {
					w[tid] += (v[j] * u[tid-j]);    // convolve: multiply and accumulate
				}				
			}
		}
	}
}


void initCUDA(void) {
	char* d_char;	
	cudaMalloc((void**)&d_char, sizeof(char));
}

float* convSimple(float* signal, int signalLength, float* filter, int filterLength) {
    int new_size = signalLength + filterLength - 1;
	float *dev_result, *dev_signal, *dev_filter, *h_result;
	
	h_result = (float*) calloc(new_size, sizeof(float));

	cudaMalloc((void**)&dev_signal, signalLength * sizeof(float));
	cudaMalloc((void**)&dev_filter, filterLength * sizeof(float));
    cudaMalloc((void**)&dev_result, new_size * sizeof(float));
	
    // Copy host memory to device
    cudaMemcpy(dev_signal, signal, (signalLength * sizeof(float)), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_filter, filter, (filterLength * sizeof(float)), cudaMemcpyHostToDevice);
	
	initializeArray<<<32,32>>>(dev_result, new_size);

	int wLength = signalLength + filterLength - 1;	
	int maxLength = (signalLength >= filterLength ? signalLength : filterLength);
	int minLength = (signalLength <= filterLength ? signalLength : filterLength);

	conv<<<32, 32>>>(dev_signal, signalLength, dev_filter, filterLength, dev_result, wLength, maxLength, minLength);
    cudaMemcpy(h_result, dev_result, (new_size * sizeof(float)), cudaMemcpyDeviceToHost);

    // cleanup memory
	cudaFree(dev_result);
    cudaFree(dev_signal);
	cudaFree(dev_filter);
	
	return h_result;
}



/**
 *	CUFFT
 **/
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
float* convFFT(float* signal, int signalLength, float* filter, int filterLength) {
    // Allocate host memory for the signal
    Complex* h_signal = (Complex*) malloc(sizeof(Complex) * signalLength);
	
    // Initalize the memory for the signal
    for (unsigned int i = 0; i < signalLength; ++i) {
        h_signal[i].x = signal[i];
        h_signal[i].y = 0.0;
    }

    // Allocate host memory for the filter
    Complex* h_filter_kernel = (Complex*) malloc(sizeof(Complex) * filterLength);
	
    // Initalize the memory for the filter
    for (unsigned int i = 0; i < filterLength; ++i) {
        h_filter_kernel[i].x = filter[i];
        h_filter_kernel[i].y = 0.0;
    }

    // Pad signal and filter kernel
    Complex* h_padded_signal;
    Complex* h_padded_filter_kernel;
    int new_size = PadData(h_signal, &h_padded_signal, signalLength,
                           h_filter_kernel, &h_padded_filter_kernel, filterLength);
	
    int mem_size = sizeof(Complex) * new_size;

    // Allocate device memory for signal
    Complex* d_signal;
    cudaMalloc((void**)&d_signal, mem_size);
    // Copy host memory to device
    cudaMemcpy(d_signal, h_padded_signal, mem_size,
                              cudaMemcpyHostToDevice);

    // Allocate device memory for filter kernel
    Complex* d_filter_kernel;
    cudaMalloc((void**)&d_filter_kernel, mem_size);

    // Copy host memory to device
    cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
                              cudaMemcpyHostToDevice);

    // CUFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, new_size, CUFFT_C2C, 1);

    // Transform signal and kernel
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *)d_filter_kernel, (cufftComplex *)d_filter_kernel, CUFFT_FORWARD);

    // Multiply the coefficients together and normalize the result
    ComplexPointwiseMulAndScale<<<32, 256>>>(d_signal, d_filter_kernel, new_size, 1.0f / new_size);

    // Check if kernel execution generated and error
    //cutilCheckMsg("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

    // Transform signal back
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

    // Copy device memory to host
    Complex* h_convolved_signal = h_padded_signal;
    cudaMemcpy(h_convolved_signal, d_signal, mem_size,
                              cudaMemcpyDeviceToHost);

    // Allocate host memory for the convolution result
    Complex* h_convolved_signal_ref = (Complex*)malloc(sizeof(Complex) * signalLength);

    // Convolve on the host
	
	float* convolvedSignal = (float*) calloc(new_size, sizeof(float));
	for (int i = 0; i < new_size; i++) {
		convolvedSignal[i] = h_convolved_signal[i].x;
	}

    // check result


    //Destroy CUFFT context
    cufftDestroy(plan);

    // cleanup memory
    free(h_signal);
    free(h_filter_kernel);
    free(h_padded_signal);
    free(h_padded_filter_kernel);
    //free(h_convolved_signal_ref);
	//free(h_convolved_signal);
    cudaFree(d_signal);
    cudaFree(d_filter_kernel);
	
	return convolvedSignal;

}

// Pad data
int PadData(const Complex* signal, Complex** padded_signal, int signal_size,
            const Complex* filter_kernel, Complex** padded_filter_kernel, int filter_kernel_size) {
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;
    //int new_size = signal_size + maxRadius;
	int new_size = signal_size + filter_kernel_size - 1;
    
    // Pad signal
    Complex* new_data = (Complex*)malloc(sizeof(Complex) * new_size);
    memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
    memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
    *padded_signal = new_data;
    
    // Pad filter
    new_data = (Complex*)malloc(sizeof(Complex) * new_size);  
    /*memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
    memset(new_data + maxRadius, 0, (new_size - filter_kernel_size) * sizeof(Complex));   
    memcpy(new_data + new_size - minRadius, filter_kernel, minRadius * sizeof(Complex));*/
	
	memcpy(new_data + 0, filter_kernel, filter_kernel_size * sizeof(Complex));
    memset(new_data + filter_kernel_size, 0, (new_size - filter_kernel_size) * sizeof(Complex));
	
    *padded_filter_kernel = new_data;
    
    return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// Computes convolution on the host - NÃO É USADA
void Convolve(const Complex* signal, int signal_size,
              const Complex* filter_kernel, int filter_kernel_size,
              Complex* filtered_signal) {
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;
    // Loop over output element indices
    for (int i = 0; i < signal_size; ++i) {
        filtered_signal[i].x = filtered_signal[i].y = 0;
        // Loop over convolution indices
        for (int j = - maxRadius + 1; j <= minRadius; ++j) {
            int k = i + j;
            if (k >= 0 && k < signal_size) 
                filtered_signal[i] = ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex* a, const Complex* b, int size, float scale) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);     
} 



// ################## Host code

// Find Delay
/**
 Retorna o atraso (ITD) para cada ouvido usando algoritmo de busca diretamente 
 sobre a HRIR da direcao.
 D[0] = left;
 D[1] = right;
*/
short* findDelay(float** hrtf, int vecLength) {
	float** squaredElements = getSquaresOfArrayElements(hrtf[0], hrtf[1], vecLength);;

	double* et = (double*) malloc(2 * sizeof(double));
	et[0] = sumArrayElements(squaredElements[0], vecLength);
	et[1] = sumArrayElements(squaredElements[1], vecLength);
	
	float** ek = sumColumns(squaredElements[0], squaredElements[1], vecLength);;
	
	short* d = (short*) malloc(2*sizeof(short));
	for (int ear = 0; ear < 2; ear++) {	// ear
		for (int j = 0; j < vecLength; j++) {
			ek[ear][j] = ek[ear][j]/et[ear];
		}
		short* indexes = findIndexesGreaterThan(ek[ear], vecLength, 0.002f);
		d[ear] = indexes[0]-1;	
	}
	
	cleanHostMemory(ek);
	cleanHostMemory(et);
	cleanHostMemory(squaredElements);
	
	return d;
}

/**
 *	Soma paralelamente as colunas do vetor @vec de tamanho @vecLength.
 */
float** sumColumns(float* vec0, float* vec1, int vecLength) {
	// Variables
	float** h_result;

	int size = vecLength*sizeof(float);
	h_result = (float**)malloc(2 * sizeof(float*));
	h_result[0] = (float*)malloc(size);
	h_result[1] = (float*)malloc(size);
	
	for (int i = 0; i < vecLength; i++) {
		float sum0 = 0.0;
		float sum1 = 0.0;
		for (int j = 0; j < i; j++) {
			sum0 += vec0[j];
			sum1 += vec1[j];
		}		
		h_result[0][i] = (i == 0) ? vec0[i] : sum0;
		h_result[1][i] = (i == 0) ? vec1[i] : sum1;
	}
	

	/*
	float* d_vec;
	float* d_result;
	
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&d_vec, size);
    cudaMalloc((void**)&d_result, size);

	// Copy vector from host memory to device memory
    cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice);

	// Invoke kernel
    sumColumnsInParallel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_result, vecLength);

	// Copy result from device memory to host memory
    // h_result contains the result in host memory
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	printf("\n\n -- Elapsed time: %3.15f ms --\n\n", elapsedTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cleanDeviceMemory(d_vec);
	cleanDeviceMemory(d_result);*/
	
	return h_result;
}

/**
 *	Soma os quadrados de todos os elementos de um array.
 */
/*double sumSquaresOfArrayElements(float* vec, int vecLength) {
	float* h_result = getSquaresOfArrayElements(vec, vecLength);
	return sumArrayElements(h_result, vecLength);
}*/

/**
 *	Soma todos os elementos de um array.
 */
double sumArrayElements(float* vec, int vecLength) {
	// Sum result
	int i;
	double result = 0.0;
    for (i = 0; i < vecLength; ++i) {
		result += vec[i];
    }	
	
	return result;
}

/**
 *	Recupera os quadrados de cada elemento do array.
 */
float** getSquaresOfArrayElements(float* vec0, float* vec1, int vecLength) {
	// Variables
	float** h_result;
	int size = vecLength*sizeof(float);
	
	h_result = (float**)malloc(2 * sizeof(float*));
	h_result[0] = (float*) malloc(size);
	h_result[1] = (float*) malloc(size);
	
	for (int i = 0; i < vecLength; i++) {
		h_result[0][i] = (vec0[i] * vec0[i]);
		h_result[1][i] = (vec1[i] * vec1[i]);
	}
	

	/*
	float* d_vec0;
	float* d_vec1;
	float* d_result0;
	float* d_result1;
	
	cudaMalloc((void**)&d_vec0, size);
	cudaMalloc((void**)&d_vec1, size);
    cudaMalloc((void**)&d_result0, size);
	cudaMalloc((void**)&d_result1, size);

	// Copy vector from host memory to device memory
    cudaMemcpy(d_vec0, vec0, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec1, vec1, size, cudaMemcpyHostToDevice);

	// Invoke kernel
    VecMult<<<blocksPerGrid, threadsPerBlock>>>(d_vec0, d_vec1, d_result0, d_result1, vecLength);

	// Copy result from device memory to host memory
    // h_result contains the result in host memory
    cudaMemcpy(h_result[0], d_result0, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result[1], d_result1, size, cudaMemcpyDeviceToHost);
	
	cleanDeviceMemory(d_vec0);
	cleanDeviceMemory(d_vec1);
	cleanDeviceMemory(d_result0);
	cleanDeviceMemory(d_result1);*/
	
	return h_result;
}


/**
 *	Encontra os índices de um array em que os valores são maiores que @minValue.
 */
short* findIndexesGreaterThan(float* vec, int vecLength, double minValue) {
	short* indexes = (short*)malloc(vecLength*sizeof(short));
	int j = 0;
	for (short i = 0; i < vecLength; i++) {
		if (vec[i] >= minValue) {
			indexes[j++] = i;
		}
	}
	return indexes;
}


// Shift in parallel
/**
 *	Desloca os elementos de um array em paralelo.
 */
float* shiftInParallel(float* vec, int vecLength, short delay, int maxLength) {
	// Variables
	float* h_result = NULL;
	
	h_result = (float*) calloc(maxLength, sizeof(float));
	
	for (int i = 0; i < (vecLength-delay); i++) {
		h_result[i] = vec[i+delay];
	}
	
	
	/*
	float* d_vec = NULL;
	float* d_result = NULL;
	
	int vecSize = vecLength * sizeof(float);
	int size = maxLength * sizeof(float);
	
	cudaMalloc( (void**)&d_vec, vecSize );
    cudaMalloc((void**)&d_result, size);

	// Copy vector from host memory to device memory
    cudaMemcpy(d_vec, vec, vecLength*sizeof(float), cudaMemcpyHostToDevice);

	// Invoke kernel
    shiftArrayElementsInParallel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, vecLength, d_result, delay);

	// Copy result from device memory to host memory
    // h_result contains the result in host memory
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
	
	cleanDeviceMemory(d_vec);
	cleanDeviceMemory(d_result);*/
	
	return h_result;
}



// Sparse coefficients

/**
 *	Função que lê os coeficientes do filtro daubechies 8
 */
float** leFiltros(char* filtro, int* filtroLength) {
	float** h = NULL;
	
	if (strcmp(DB8_CONST, filtro) == 0) {
		float db8[2][8] = {
			{-0.0105, 0.0328, 0.0308, -0.1870, -0.0279, 0.6308, 0.7148, 0.2303},
			{-0.2303, 0.7148, -0.6308, -0.0279, 0.1870, 0.0308, -0.0328, -0.0105}};
	
		int channels = 2;	// número de canais nesse filtro
		int filterLength = 8;	// tamanho do maior filtro
		*filtroLength = filterLength;
		
		h = (float**) calloc(channels, sizeof(float*));
		h[0] = (float*) calloc(filterLength, sizeof(float));
		h[1] = (float*) calloc(filterLength, sizeof(float));
		
		for (int row = 0; row < channels; row++) {
			for (int col = 0; col < filterLength; col++) {
				h[row][col] = db8[row][col];
			}
		}
	}
	return h;
}

/**
 *	Função que retorna o maior elemento de um vetor
 */
int max(int* vec, int vecLength) {
	int max = 0;
	for (int k = 0; k < vecLength; k++) { 
		if (vec[k] > max) {
			max = vec[k];
		}
	}
	return max;
}

/**
 *	Função que retorna o vetor com elementos esparsos (separados por 0),
 *	de acordo com um coeficiente de esparsidade.
 */
float* spars(float* vec, int vecLength, int sparsity, int* resultLength) {
	*resultLength = (((vecLength - 1) * sparsity) + 1);
	float* y = (float*) calloc((((vecLength - 1) * sparsity) + 1) , sizeof(float));
	
	for (int i = 0; i < vecLength; i++) {
		y[(i * sparsity)] = vec[i];
	}
	
	return y;
}

/**
 *	Operação de cascateamento dos filtros utilizando a convolução
 *	simples implementada em CUDA.
 */
void cascataSimpleBKP(char* filtros[], int numFiltros, float** filterBank, int* filterBankLength) {
	float *convResult;

	int J = numFiltros;
	int filtroLength;
	float** h;
	int hLength;
	
	h = leFiltros(filtros[0], &hLength);
		
	float** filterBankAux = (float**) malloc((numFiltros + 1) * sizeof(float*));
	int* filterBankLengthAux = (int*) malloc((numFiltros + 1) * sizeof(int));
	filterBankAux[0] = (float*) malloc(hLength * sizeof(float));	// passa-altas
	filterBankLengthAux[0] = hLength;
	float* Gl = (float*) malloc(hLength * sizeof(float));	// passa-baixas
	
	for (int i = 0; i < hLength; i++) {
		filterBankAux[0][i] = h[1][i];
		Gl[i] = h[0][i];
	}

	float* Gl_old = NULL;
	filtroLength = hLength;
	
	for (int i = 1; i < numFiltros; i++) {
		Gl_old = (float*) calloc(filtroLength, sizeof(float));
		for (int j = 0; j < filtroLength; j++) {
			Gl_old[j] = Gl[j];
		}
		
		int sparsArray0Length, sparsArray1Length;
		float* sparsArray0 = spars(h[0], hLength, pow(2.0, i), &sparsArray0Length);
		float* sparsArray1 = spars(h[1], hLength, pow(2.0, i), &sparsArray1Length);
		
		int resultLength = (filtroLength + sparsArray0Length - 1);
		int resultSize = resultLength * sizeof(float);
	
		convResult = (float*) calloc(resultLength, sizeof(float));
		convResult = convSimple(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
		
		filterBankAux[i] = (float*) calloc(resultLength, sizeof(float));
		filterBankAux[i] = convSimple(Gl_old, filtroLength, sparsArray1, sparsArray1Length);
		
		filterBankLengthAux[i] = resultLength;

		if ((i+1) != numFiltros) {
			free(Gl_old);	// Gl_old é usado fora do for após a última iteração
			free(Gl);
			
			filtroLength = resultLength;
			Gl = (float*) malloc(resultSize);
			for (int i = 0; i < resultLength; i++) {
				Gl[i] = convResult[i];
			}
			
			free(convResult);
		}
	}
		
	int sparsArray0Length;
	float* sparsArray0 = spars(h[0], hLength, pow(2.0, J-1), &sparsArray0Length);
	
	int resultLength = (filtroLength + sparsArray0Length - 1);
	
	filterBankAux[J] = (float*) calloc(resultLength, sizeof(float));
	//filterBankAux[J] = convFFT(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
	filterBankAux[J] = convSimple(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
	
	filterBankLengthAux[J] = resultLength;

	free(Gl_old);
	free(Gl);
	
	int maxLength = max(filterBankLengthAux, (numFiltros + 1));
	
	for (int i = numFiltros; i >= 0; i--) {
		filterBank[i] = (float*) calloc(maxLength, sizeof(float));
		filterBankLength[i] = filterBankLengthAux[numFiltros-i];
		for (int j = 0; j < filterBankLengthAux[numFiltros - i]; j++) {
			filterBank[i][j] = filterBankAux[numFiltros - i][j];
		}
	}
}

/**
 *	Operação de cascateamento dos filtros utilizando a convolução
 *	simples implementada em CUDA.
 */
void cascataSimple(char* filtros[], int numFiltros, float** filterBank, int* filterBankLength) {
	float *convResult;

	int J = numFiltros;
	int filtroLength;
	float** h;
	int hLength;
	
	h = leFiltros(filtros[0], &hLength);
		
	float** filterBankAux = (float**) malloc((numFiltros + 1) * sizeof(float*));
	int* filterBankLengthAux = (int*) malloc((numFiltros + 1) * sizeof(int));
	filterBankAux[0] = (float*) malloc(hLength * sizeof(float));	// passa-altas
	filterBankLengthAux[0] = hLength;
	float* Gl = (float*) malloc(hLength * sizeof(float));	// passa-baixas
	
	for (int i = 0; i < hLength; i++) {
		filterBankAux[0][i] = h[1][i];
		Gl[i] = h[0][i];
	}

	float* Gl_old = NULL;
	filtroLength = hLength;
	
	for (int i = 1; i < numFiltros; i++) {
		Gl_old = (float*) calloc(filtroLength, sizeof(float));
		for (int j = 0; j < filtroLength; j++) {
			Gl_old[j] = Gl[j];
		}
		
		int sparsArray0Length, sparsArray1Length;
		float* sparsArray0 = spars(h[0], hLength, pow(2.0, i), &sparsArray0Length);
		float* sparsArray1 = spars(h[1], hLength, pow(2.0, i), &sparsArray1Length);
		
		int resultLength = (filtroLength + sparsArray0Length - 1);
		int resultSize = resultLength * sizeof(float);
	
		convResult = (float*) calloc(resultLength, sizeof(float));
		
		//convResult = convSimple(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
		convResult = convHost1(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
		
		filterBankAux[i] = (float*) calloc(resultLength, sizeof(float));
		//filterBankAux[i] = convSimple(Gl_old, filtroLength, sparsArray1, sparsArray1Length);
		filterBankAux[i] = convHost1(Gl_old, filtroLength, sparsArray1, sparsArray1Length);
		
		filterBankLengthAux[i] = resultLength;

		if ((i+1) != numFiltros) {
			free(Gl_old);	// Gl_old é usado fora do for após a última iteração
			free(Gl);
			
			filtroLength = resultLength;
			Gl = (float*) malloc(resultSize);
			for (int i = 0; i < resultLength; i++) {
				Gl[i] = convResult[i];
			}
			
			free(convResult);
		}
	}
		
	int sparsArray0Length;
	float* sparsArray0 = spars(h[0], hLength, pow(2.0, J-1), &sparsArray0Length);
	
	int resultLength = (filtroLength + sparsArray0Length - 1);
	
	filterBankAux[J] = (float*) calloc(resultLength, sizeof(float));
	//filterBankAux[J] = convFFT(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
	filterBankAux[J] = convSimple(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
	
	filterBankLengthAux[J] = resultLength;

	free(Gl_old);
	free(Gl);
	
	int maxLength = max(filterBankLengthAux, (numFiltros + 1));
	
	for (int i = numFiltros; i >= 0; i--) {
		filterBank[i] = (float*) calloc(maxLength, sizeof(float));
		filterBankLength[i] = filterBankLengthAux[numFiltros-i];
		for (int j = 0; j < filterBankLengthAux[numFiltros - i]; j++) {
			filterBank[i][j] = filterBankAux[numFiltros - i][j];
		}
	}
}


/**
 *	Operação de cascateamento dos filtros utilizando a convolução
 *	com FFT implementada em CUDA.
 */
void cascataFFT(char* filtros[], int numFiltros, float** filterBank, int* filterBankLength) {
	float *convResult;

	int J = numFiltros;
	int filtroLength;
	float** h;
	int hLength;
	
	h = leFiltros(filtros[0], &hLength);
		
	float** filterBankAux = (float**) malloc((numFiltros + 1) * sizeof(float*));
	int* filterBankLengthAux = (int*) malloc((numFiltros + 1) * sizeof(int));
	filterBankAux[0] = (float*) malloc(hLength * sizeof(float));	// passa-altas
	filterBankLengthAux[0] = hLength;
	float* Gl = (float*) malloc(hLength * sizeof(float));	// passa-baixas
	
	for (int i = 0; i < hLength; i++) {
		filterBankAux[0][i] = h[1][i];
		Gl[i] = h[0][i];
	}

	float* Gl_old = NULL;
	filtroLength = hLength;
	
	for (int i = 1; i < numFiltros; i++) {
		Gl_old = (float*) calloc(filtroLength, sizeof(float));
		for (int j = 0; j < filtroLength; j++) {
			Gl_old[j] = Gl[j];
		}
		
		int sparsArray0Length, sparsArray1Length;
		float* sparsArray0 = spars(h[0], hLength, pow(2.0, i), &sparsArray0Length);
		float* sparsArray1 = spars(h[1], hLength, pow(2.0, i), &sparsArray1Length);
		
		int resultLength = (filtroLength + sparsArray0Length - 1);
		int resultSize = resultLength * sizeof(float);
	
		convResult = (float*) calloc(resultLength, sizeof(float));
		convResult = convFFT(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
		
		filterBankAux[i] = (float*) calloc(resultLength, sizeof(float));
		filterBankAux[i] = convFFT(Gl_old, filtroLength, sparsArray1, sparsArray1Length);
		
		filterBankLengthAux[i] = resultLength;

		if ((i+1) != numFiltros) {
			free(Gl_old);	// Gl_old é usado fora do for após a última iteração
			free(Gl);
			
			filtroLength = resultLength;
			Gl = (float*) malloc(resultSize);
			for (int i = 0; i < resultLength; i++) {
				Gl[i] = convResult[i];
			}
			
			free(convResult);
		}
	}
		
	int sparsArray0Length;
	float* sparsArray0 = spars(h[0], hLength, pow(2.0, J-1), &sparsArray0Length);
	
	int resultLength = (filtroLength + sparsArray0Length - 1);
	
	filterBankAux[J] = (float*) calloc(resultLength, sizeof(float));
	//filterBankAux[J] = convFFT(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
	filterBankAux[J] = convSimple(Gl_old, filtroLength, sparsArray0, sparsArray0Length);
	
	filterBankLengthAux[J] = resultLength;

	free(Gl_old);
	free(Gl);
	
	int maxLength = max(filterBankLengthAux, (numFiltros + 1));
	
	for (int i = numFiltros; i >= 0; i--) {
		filterBank[i] = (float*) calloc(maxLength, sizeof(float));
		filterBankLength[i] = filterBankLengthAux[numFiltros-i];
		for (int j = 0; j < filterBankLengthAux[numFiltros - i]; j++) {
			filterBank[i][j] = filterBankAux[numFiltros - i][j];
		}
	}
}


float* multiplyPolynomialCoefficients(float* vecA, int vecASize, float* vecB, int vecBSize, int resultLength) {
	float* vecC = (float*) calloc(resultLength, sizeof(float));
	for (int i = 0; i < vecASize; i++) {
		for (int j = 0; j < vecBSize; j++) {
			vecC[i+j] = vecC[i+j] + (vecA[i] * vecB[j]);
		}
	}
	return vecC;
}


float* sumPolynomialCoefficients(float* dev_aux1, float* dev_aux2, int resultSize) {
	float* dev_aux_sum = (float*) calloc(resultSize, sizeof(float));
	for (int i = 0; i < resultSize; i++) {
		dev_aux_sum[i] = dev_aux1[i] + dev_aux2[i];
	}
	return dev_aux_sum;
}

float*** computeIpAuxHost(float*** Hp, int hlength, float*** Fp, int flength) {
	float *dev_aux1, *dev_aux2;
	int resultLength = (hlength + flength - 1);
	
	float*** Ip_aux = (float***) malloc(2 * sizeof(float**));
	Ip_aux[0] = (float**) malloc(2 * sizeof(float*));
	Ip_aux[1] = (float**) malloc(2 * sizeof(float*));
	
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			Ip_aux[i][j] = (float*) malloc(resultLength*sizeof(float*));
		}
	}
	
	/*
	 Ip_aux = Fp*Hp;
	 O resultado é uma matriz 2x2x7.
	 
	 aA+bC	aB+bD
	 cA+dC	cB+dD
	 */
	
	/* 00 */	
	dev_aux1 = multiplyPolynomialCoefficients(Fp[0][0], hlength, Hp[0][0], hlength, resultLength);
	dev_aux2 = multiplyPolynomialCoefficients(Fp[0][1], hlength, Hp[1][0], hlength, resultLength);
	Ip_aux[0][0] = sumPolynomialCoefficients(dev_aux1, dev_aux2, resultLength);
	
	
	/* 01 */
	// clean auxiliary variables
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	
	dev_aux1 = multiplyPolynomialCoefficients(Fp[0][0], hlength, Hp[0][1], hlength, resultLength);
	dev_aux2 = multiplyPolynomialCoefficients(Fp[0][1], hlength, Hp[1][1], hlength, resultLength);
	Ip_aux[0][1] = sumPolynomialCoefficients(dev_aux1, dev_aux2, resultLength);
	
	
	/* 10 */
	// clean auxiliary variables
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	
	dev_aux1 = multiplyPolynomialCoefficients(Fp[1][0], hlength, Hp[0][0], hlength, resultLength);
	dev_aux2 = multiplyPolynomialCoefficients(Fp[1][1], hlength, Hp[1][0], hlength, resultLength);
	Ip_aux[1][0] = sumPolynomialCoefficients(dev_aux1, dev_aux2, resultLength);
	
	
	/* 11 */
	// clean auxiliary variables
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	
	dev_aux1 = multiplyPolynomialCoefficients(Fp[1][0], hlength, Hp[0][1], hlength, resultLength);
	dev_aux2 = multiplyPolynomialCoefficients(Fp[1][1], hlength, Hp[1][1], hlength, resultLength);
	Ip_aux[1][1] = sumPolynomialCoefficients(dev_aux1, dev_aux2, resultLength);
	
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	
	return Ip_aux;
}

/**
 * Essa função computa a multiplicação das matrizes Fp e Hp. Ip_aux = Fp*Hp.
 * Resultado: [2][2][7]
 *
 *	Fp =	a	b
 *			c	d
 *
 *	Hp =	A	B
 *			C	D
 *
 *	Resultado mais rápido que método anterior com streams.
 */
float*** computeIpAux(float*** Hp, int hlength, float*** Fp, int flength) {
	float *dev_a0, *dev_b0, *dev_c0, *dev_d0, *dev_A0, *dev_B0, *dev_C0, *dev_D0; //GPU buffers for stream0 
	float *dev_aux1, *dev_aux2, *dev_aux_sum;
	
	cudaEvent_t	start, stop;
	float	elapsedTime;
	// start the timers
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop ); 
	cudaEventRecord( start, 0 );
	
	int size = hlength * sizeof(float);
	int resultLength = (hlength + flength - 1);
	int resultSize = resultLength * sizeof(float);
	
	float*** Ip_aux = (float***) malloc(2 * sizeof(float**));
	Ip_aux[0] = (float**) malloc(2 * sizeof(float*));
	Ip_aux[1] = (float**) malloc(2 * sizeof(float*));
	
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			Ip_aux[i][j] = (float*) malloc(resultLength*sizeof(float*));
		}
	}
	
	cudaMalloc( (void**)&dev_a0, size );
	cudaMalloc( (void**)&dev_b0, size );
	cudaMalloc( (void**)&dev_c0, size );
	cudaMalloc( (void**)&dev_d0, size );
	cudaMalloc( (void**)&dev_A0, size );
	cudaMalloc( (void**)&dev_B0, size );
	cudaMalloc( (void**)&dev_C0, size );
	cudaMalloc( (void**)&dev_D0, size );
	
	cudaMalloc( (void**)&dev_aux1, resultSize );
	cudaMalloc( (void**)&dev_aux2, resultSize );
	cudaMalloc( (void**)&dev_aux_sum, resultSize );
	
	/*
	Ip_aux = Fp*Hp;
	O resultado é uma matriz 2x2x7.
	
	aA+bC	aB+bD
	cA+dC	cB+dD
	*/
	
/* 00 */
	cudaMemcpy(dev_a0, Fp[0][0], size, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_b0, Fp[0][1], size, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_c0, Fp[1][0], size, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_d0, Fp[1][1], size, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_A0, Hp[0][0], size, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_B0, Hp[0][1], size, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_C0, Hp[1][0], size, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_D0, Hp[1][1], size, cudaMemcpyHostToDevice );
	
	// clean auxiliary variables
	initializeArray<<<4,4>>>(dev_aux1, resultLength);
	initializeArray<<<4,4>>>(dev_aux2, resultLength);
	
	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<4, 4>>>(dev_a0, hlength, dev_A0, hlength, dev_aux1);
    multiplyPolynomialCoefficientsKernel<<<4, 4>>>(dev_b0, hlength, dev_C0, hlength, dev_aux2);	
	
	sumPolynomialCoefficientsKernel<<<resultSize, 1>>>(dev_aux1, dev_aux2, dev_aux_sum, resultLength);
	
	// copy the data from device to locked memory
	cudaMemcpy(Ip_aux[0][0], dev_aux_sum, resultSize, cudaMemcpyDeviceToHost);	


/* 01 */
	// clean auxiliary variables
	initializeArray<<<4,4>>>(dev_aux1, resultLength);
	initializeArray<<<4,4>>>(dev_aux2, resultLength);

	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<4, 4>>>(dev_a0, hlength, dev_B0, hlength, dev_aux1);
    multiplyPolynomialCoefficientsKernel<<<4, 4>>>(dev_b0, hlength, dev_D0, hlength, dev_aux2);
	
	sumPolynomialCoefficientsKernel<<<resultSize, 1>>>(dev_aux1, dev_aux2, dev_aux_sum, resultLength);
	
	// copy the data from device to locked memory
	cudaMemcpy(Ip_aux[0][1], dev_aux_sum, resultSize, cudaMemcpyDeviceToHost);


/* 10 */
	// clean auxiliary variables
	initializeArray<<<4,4>>>(dev_aux1, resultLength);
	initializeArray<<<4,4>>>(dev_aux2, resultLength);
	
	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<4, 4>>>(dev_c0, hlength, dev_A0, hlength, dev_aux1);
    multiplyPolynomialCoefficientsKernel<<<4, 4>>>(dev_d0, hlength, dev_C0, hlength, dev_aux2);	
	
	sumPolynomialCoefficientsKernel<<<resultSize, 1>>>(dev_aux1, dev_aux2, dev_aux_sum, resultLength);
	
	// copy the data from device to locked memory
	cudaMemcpy(Ip_aux[1][0], dev_aux_sum, resultSize, cudaMemcpyDeviceToHost);	


/* 11 */
	// clean auxiliary variables
	initializeArray<<<4,4>>>(dev_aux1, resultLength);
	initializeArray<<<4,4>>>(dev_aux2, resultLength);

	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<4, 4>>>(dev_c0, hlength, dev_B0, hlength, dev_aux1);
    multiplyPolynomialCoefficientsKernel<<<4, 4>>>(dev_d0, hlength, dev_D0, hlength, dev_aux2);	
	
	sumPolynomialCoefficientsKernel<<<resultSize, 1>>>(dev_aux1, dev_aux2, dev_aux_sum, resultLength);
	
	// copy the data from device to locked memory
	cudaMemcpy(Ip_aux[1][1], dev_aux_sum, resultSize, cudaMemcpyDeviceToHost);	

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &elapsedTime, start, stop ); 
	//printf( "Time taken computeIpAux: %3.1f ms\n", elapsedTime );
	
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_d0);
	cudaFree(dev_A0);
	cudaFree(dev_B0);
	cudaFree(dev_C0);
	cudaFree(dev_D0);
	cudaFree(dev_aux1);
	cudaFree(dev_aux2);
	cudaFree(dev_aux_sum);	
	
	//printf("Error computeIpAux: %s\n\n", cudaGetErrorString(cudaGetLastError()));
	
	return Ip_aux;
}


/**
 * Essa função computa a multiplicação das matrizes Rp, Fp e Ip_aux. Ip_aux = Fp*Hp.
 * Resultado: [2][237]
 *
 *	Rp =	X
 *			Y
 *
 *	Fp =	a	b
 *			c	d
 *
 *	Ip =	A	B
 *			C	D
 *
 */
float** calculateGp(float** Rp, int rpLength, float*** Fp, int fpLength, float*** Ip, int ipLength) {
	// Rp
	float *dev_x0 = NULL;
	float *dev_y0 = NULL;

    // Fp
	float *dev_a0 = NULL;
	float *dev_b0 = NULL;
	float *dev_c0 = NULL;
	float *dev_d0 = NULL;
	
	// Ip
	float *dev_A0 = NULL;
	float *dev_B0 = NULL;
	float *dev_C0 = NULL;
	float *dev_D0 = NULL;
	
	float *dev_aux1, *dev_aux2, *dev_aux3, *dev_aux4, *dev_aux_sum1, *dev_aux_sum2 ;
	float **finalResult;
	
	cudaEvent_t	start, stop;
	float	elapsedTime;
	// start the timers
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop ); 
	cudaEventRecord( start, 0 );

	int lengthX = (rpLength % 2 == 0 ? rpLength/2 : (rpLength/2+1));
	int lengthY = rpLength/2;
	int sizeX = lengthX * sizeof(float);
	int sizeY = lengthY * sizeof(float);
	int sizeFpElement = fpLength * sizeof(float);
	int sizeIpElement = ipLength * sizeof(float);
	int partialResultLengthX = lengthX + fpLength - 1;
	int partialResultSizeX = partialResultLengthX * sizeof(float);
	int partialResultLengthY = lengthY + fpLength - 1;
	int partialResultSizeY = partialResultLengthY * sizeof(float);
	int finalResultLength = (partialResultLengthX + ipLength -1);
	int finalResultSize = finalResultLength * sizeof(float);
	
	cudaMalloc( (void**)&dev_x0, sizeX );
	cudaMalloc( (void**)&dev_y0, sizeY );
	cudaMalloc( (void**)&dev_a0, sizeFpElement );
	cudaMalloc( (void**)&dev_b0, sizeFpElement );
	cudaMalloc( (void**)&dev_c0, sizeFpElement );
	cudaMalloc( (void**)&dev_d0, sizeFpElement );
	cudaMalloc( (void**)&dev_A0, sizeIpElement );
	cudaMalloc( (void**)&dev_B0, sizeIpElement );
	cudaMalloc( (void**)&dev_C0, sizeIpElement );
	cudaMalloc( (void**)&dev_D0, sizeIpElement );
	
	cudaMalloc( (void**)&dev_aux1, partialResultSizeX );
	cudaMalloc( (void**)&dev_aux2, partialResultSizeY );
	cudaMalloc( (void**)&dev_aux3, partialResultSizeX );
	cudaMalloc( (void**)&dev_aux4, partialResultSizeY );
	
	cudaMalloc( (void**)&dev_aux_sum1, partialResultSizeX );
	cudaMalloc( (void**)&dev_aux_sum2, partialResultSizeX );
	
	// Inicialização de array no dispositivo para evitar erros em leituras
	initializeArray<<<16,16>>>(dev_aux1, partialResultLengthX);
	initializeArray<<<16,16>>>(dev_aux2, partialResultLengthX);
	initializeArray<<<16,16>>>(dev_aux3, partialResultLengthX);
	initializeArray<<<16,16>>>(dev_aux4, partialResultLengthX);
	initializeArray<<<16,16>>>(dev_aux_sum1, partialResultLengthX);
	initializeArray<<<16,16>>>(dev_aux_sum2, partialResultLengthX);
	
	/*
	Gp = Rp*Fp*Ip;
	O resultado é uma matriz 2x237.
	
	----- Parte 1
	  Rp	  Fp
	(x	y)	(a	c)
			(b	d)
	
	Rp*Fp
		xa+yb	xc+yd
	
	*/
	cudaMemcpy(dev_x0, Rp[0], sizeX, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_y0, Rp[1], sizeY, cudaMemcpyHostToDevice );
	
	cudaMemcpy(dev_a0, Fp[0][0], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_b0, Fp[0][1], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_c0, Fp[1][0], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_d0, Fp[1][1], sizeFpElement, cudaMemcpyHostToDevice );
	
	cudaMemcpy(dev_A0, Ip[0][0], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_B0, Ip[0][1], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_C0, Ip[1][0], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_D0, Ip[1][1], sizeIpElement, cudaMemcpyHostToDevice );
	
	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_x0, lengthX, dev_a0, fpLength, dev_aux1);	
    multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_y0, lengthY, dev_b0, fpLength, dev_aux2);
	multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_x0, lengthX, dev_c0, fpLength, dev_aux3);
    multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_y0, lengthY, dev_d0, fpLength, dev_aux4);
	
	sumPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux1, dev_aux2, dev_aux_sum1, partialResultLengthX);
	sumPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux3, dev_aux4, dev_aux_sum2, partialResultLengthX);
		
	// Re-Inicialização de array no dispositivo para evitar erros em leituras
	cudaFree(dev_aux1);
	cudaFree(dev_aux2);
	cudaFree(dev_aux3);
	cudaFree(dev_aux4);
	
	cudaMalloc( (void**)&dev_aux1, finalResultSize );
	cudaMalloc( (void**)&dev_aux2, finalResultSize );
	cudaMalloc( (void**)&dev_aux3, finalResultSize );
	cudaMalloc( (void**)&dev_aux4, finalResultSize );
	
	initializeArray<<<16,16>>>(dev_aux1, finalResultLength);
	initializeArray<<<16,16>>>(dev_aux2, finalResultLength);
	initializeArray<<<16,16>>>(dev_aux3, finalResultLength);
	initializeArray<<<16,16>>>(dev_aux4, finalResultLength);
	
	/*
	Gp = Rp*Fp*Ip;
	O resultado é uma matriz 2x237.
	
	----- Parte 1
	  Rp	  Fp
	(x	y)	(a	c)
			(b	d)
	
	Rp*Fp
		xa+yb	xc+yd
		  u		  v
	
	----- Parte 2	  	  
	Rp*Fp	  Hp
	(u	v)	(A	C)
			(B	D)
	
	Rp*Fp*Ip
		uA+vB	uC+vD
	
	*/
	multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux_sum1, partialResultLengthX, dev_A0, ipLength, dev_aux1);	
    multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux_sum2, partialResultLengthX, dev_B0, ipLength, dev_aux2);
	multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux_sum1, partialResultLengthX, dev_C0, ipLength, dev_aux3);
    multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux_sum2, partialResultLengthX, dev_D0, ipLength, dev_aux4);
	
	// Cleaning data of auxiliary vectors
	cudaFree(dev_aux_sum1);
	cudaFree(dev_aux_sum2);
	cudaMalloc( (void**)&dev_aux_sum1, finalResultSize );
	cudaMalloc( (void**)&dev_aux_sum2, finalResultSize );
	
	// Reinicializa arrays u e v (resultado de Rp*Fp)
	initializeArray<<<16,16>>>(dev_aux_sum1, finalResultLength);
	initializeArray<<<16,16>>>(dev_aux_sum2, finalResultLength);
	
	sumPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux1, dev_aux2, dev_aux_sum1, finalResultLength);
	sumPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux3, dev_aux4, dev_aux_sum2, finalResultLength);
	
	finalResult = (float**) malloc(2 * sizeof(float*));
	finalResult[0] = (float*) malloc(finalResultSize);
	finalResult[1] = (float*) malloc(finalResultSize);
	
	// copy the data from device to locked memory
	cudaMemcpy(finalResult[0], dev_aux_sum1, finalResultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(finalResult[1], dev_aux_sum2, finalResultSize, cudaMemcpyDeviceToHost);
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &elapsedTime, start, stop ); 
	//printf( "Time taken calculateGp: %3.1f ms\n", elapsedTime );
		
	cudaFree(dev_x0);
	cudaFree(dev_y0);
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_d0);
	cudaFree(dev_A0);
	cudaFree(dev_B0);
	cudaFree(dev_C0);
	cudaFree(dev_D0);
	
	cudaFree(dev_aux1);
	cudaFree(dev_aux2);
	cudaFree(dev_aux3);
	cudaFree(dev_aux4);
	cudaFree(dev_aux_sum1);
	cudaFree(dev_aux_sum2);

	//printf("Error calculateGp: %s\n\n", cudaGetErrorString(cudaGetLastError()));
	
	return finalResult;
}

/**
 *	Função que realiza a operação de decimação polinomial.
 *
 *  Retorna os coeficientes G (matriz 2xK) que correspondem ao 
 *	sistema R decomposto pelo banco de 2 canais H.
 */
void dec_poly(int hlength, float* sist, int sistLength, float** G, int* G_size, float*** Hp, float*** Fp) {	
	float** Rp = (float**) calloc(2, sizeof(float*));
	Rp[0] = (float*) calloc((sistLength % 2 == 0 ? (sistLength/2) : (sistLength/2 + 1)), sizeof(float));
	Rp[1] = (float*) calloc((sistLength/2), sizeof(float));

	for (int i = 0; i < sistLength; i++) {
		if (i % 2 == 0) {
			Rp[0][i/2] = sist[i];
		} else {
			Rp[1][i/2] = sist[i];
		}
	}
	
	/*
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% decomposicao polifasica do sistema R - matriz Rp
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	*/
	/*for (int i = 0; i < 10; i++) {
		printf("sist[%d] = %1.15f\n", i, sist[i]);
	}*/
	int m = 2;
	int n = hlength;

	/*
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% decomposicao polifasica dos bancos de sintese e analise
	% matrizes Fp e Hp
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	*/

	/*
	Ip_aux = Fp*Hp;
	O resultado é uma matriz 2x2x7.
	
	H[0][1]*F[0][0] + H[0][0]*F[1][0]	H[0][1]*F[0][1] + H[0][0]*F[1][1]
	H[1][1]*F[0][0] + H[1][0]*F[1][0]	H[1][1]*F[0][1] + H[1][0]*F[1][1]
	*/
	// Hp e Fp são matrizes (cubos) de dimensão: [2][2][7]	
	//float*** ip_aux_streamed = computeIpAuxStream(Hp, hlength/2, Fp, hlength/2);
	
	float*** ip_aux = computeIpAux(Hp, hlength/2, Fp, hlength/2);
	
	/*
	for (int dim1 = 0; dim1 < 2; dim1++) {
		for (int dim2 = 0; dim2 < 2; dim2++) {
			printf("\n\n[%d][%d]---------------------\n", dim1, dim2);
			for (int i = 0; i < 7; i++) {
				printf("ip_aux_streamed[%d] = %1.15f\t\tIp_aux[%d] = %1.15f\n", i, ip_aux_streamed[dim1][dim2][i], i, ip_aux[dim1][dim2][i]);
			}
		}
	}
	printf("\n\n");
	*/
	
	// Gp = Rp*Fp*Ip
	float** Gp = calculateGp(Rp, sistLength, Fp, (hlength/2), ip_aux, (hlength-1));
	
	int atraso = round((n-m)/2.0);
	double esparsidade = 2.0;
	
	*G_size = ceil( ((sistLength + atraso +1)/esparsidade) +1);
	
	G[0] = (float*) malloc(*G_size * sizeof(float));
	G[1] = (float*) malloc(*G_size * sizeof(float));
	
	for (int i = 0; i < *G_size; i++) {
		G[0][i] = Gp[0][i+atraso];
		G[1][i] = Gp[1][i+atraso];
	}
}


/**
 *	Obtém os coeficientes esparsos que equivalem o sistema ho1d.
 */
void coef_spars(char* filtro[], int numFiltros, float* ho1d, int ho1dLength, float** G_aux, int* G_size) {
	int mesmofiltro = 0;
	float* sist = ho1d;
	float** h;
	float** G;
	float** coefSpars;
	int filtroLength;
	int sistLength = ho1dLength;
	
	coefSpars = (float**) malloc((numFiltros+1) * sizeof(float*));
	
	G = (float**) malloc(2 * sizeof(float*));	// sempre tem tamanho 2
	
	if (!mesmofiltro) {
		h = leFiltros(filtro[0], &filtroLength);
		mesmofiltro = (0 < numFiltros-1 && strcmp(filtro[0], filtro[1]) == 0);
	}
	
	// Banco de análise
	double** H = (double**) malloc(2 * sizeof(double*));
	H[0] = (double*) malloc(filtroLength * sizeof(double));
	H[1] = (double*) malloc(filtroLength * sizeof(double));

	// Banco de síntese
	double** F = (double**) malloc(2 * sizeof(double*));
	F[0] = (double*) malloc(filtroLength * sizeof(double));
	F[1] = (double*) malloc(filtroLength * sizeof(double));
	
	/*
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Obtencao do banco de filtros
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	*/
	int power = 1;
	for (int i = filtroLength-1; i >= 0; i--) {
		int reverseIndex = filtroLength-i-1;
		double powerValue = pow((double)-1, (double)power++);
		if (h[0][reverseIndex] + h[1][i]*powerValue != 0) {
			break;	// Não é Daubechies
		} else {
			H[0][reverseIndex] = h[0][reverseIndex];
			H[1][reverseIndex] = h[0][i]*powerValue;
			F[0][reverseIndex] = h[1][reverseIndex]*powerValue;
			F[1][reverseIndex] = h[1][i];
		}
	}
	
	float*** Hp = (float***) malloc(2 * sizeof(float**));
	Hp[0] = (float**) malloc(2 * sizeof(float*));
	Hp[1] = (float**) malloc(2 * sizeof(float*));
	
	float*** Fp = (float***) malloc(2 * sizeof(float**));
	Fp[0] = (float**) malloc(2 * sizeof(float*));
	Fp[1] = (float**) malloc(2 * sizeof(float*));
	
	for (int i = 0; i < 2; i++) {
		Hp[0][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Hp[1][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Fp[0][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Fp[1][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		for (int j = 0; j < (filtroLength/2); j++) {
			Hp[0][i][j] = H[i][j*2];
			Hp[1][i][j] = H[i][j*2+1];
			
			if (i == 0) {
				Fp[0][i][j] = F[0][j*2+1];
				Fp[1][i][j] = F[1][j*2+1];
			} else {
				Fp[0][i][j] = (-1) * F[1][filtroLength*i - (j*2+1)];
				Fp[1][i][j] = F[0][filtroLength*i - (j*2+1)];
			}
		}
	}
	
	for (int j = 0; j < numFiltros; j++) {		
		dec_poly(filtroLength, sist, sistLength, G, &G_size[numFiltros-j], Hp, Fp);
		
		sistLength = G_size[numFiltros-j];
		coefSpars[numFiltros-j] = G[1];
		
		sist = NULL;
		sist = G[0];
		 
		// Sempre usando o mesmo filtro
		/*if (!mesmofiltro) {
			h = leFiltros(filtro[j], &filtroLength);
			mesmofiltro = (j < numFiltros-1 && strcmp(filtro[j], filtro[j+1]) == 0);
		}*/	
	}
	
	G_size[0] = G_size[1];
	coefSpars[0] = (float*) malloc(G_size[0] * sizeof(float));
	for (int k = 0; k < G_size[0]; k++) {
		coefSpars[0][k] = sist[k];
	}
	
	int maxG_size = max(G_size, (numFiltros+1));
	for (int k = 0; k < (numFiltros+1); k++) {
		G_aux[k] = (float*) malloc(maxG_size * sizeof(float));
		int cont = 0;
		while (cont < G_size[k]) {
			G_aux[k][cont] = coefSpars[k][cont];
			cont++;
		}
		while (cont < maxG_size) {
			G_aux[k][cont] = 0.0;
			cont++;
		}
	}	
}


/**
 * Essa função computa a multiplicação das matrizes Fp e Hp. Ip_aux = Fp*Hp.
 * Resultado: [2][2][7]
 *
 *	Fp =	a	b
 *			c	d
 *
 *	Hp =	A	B
 *			C	D
 *
 *	Resultado mais rápido que método anterior com streams.
 */
float*** computeIpAux_host(float*** Hp, int hlength, float*** Fp, int flength) {
	float *dev_aux1, *dev_aux2;
	int resultLength = (hlength + flength - 1);
	
	float*** Ip_aux = (float***) malloc(2 * sizeof(float**));
	Ip_aux[0] = (float**) malloc(2 * sizeof(float*));
	Ip_aux[1] = (float**) malloc(2 * sizeof(float*));
	
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			Ip_aux[i][j] = (float*) malloc(resultLength*sizeof(float*));
		}
	}
	
	/*
	 Ip_aux = Fp*Hp;
	 O resultado é uma matriz 2x2x7.
	 
	 aA+bC	aB+bD
	 cA+dC	cB+dD
	 */
	
	/* 00 */	
	dev_aux1 = multiplyPolynomialCoefficients(Fp[0][0], hlength, Hp[0][0], hlength, resultLength);
	dev_aux2 = multiplyPolynomialCoefficients(Fp[0][1], hlength, Hp[1][0], hlength, resultLength);
	Ip_aux[0][0] = sumPolynomialCoefficients(dev_aux1, dev_aux2, resultLength);
	
	
	/* 01 */
	// clean auxiliary variables
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	
	dev_aux1 = multiplyPolynomialCoefficients(Fp[0][0], hlength, Hp[0][1], hlength, resultLength);
	dev_aux2 = multiplyPolynomialCoefficients(Fp[0][1], hlength, Hp[1][1], hlength, resultLength);
	Ip_aux[0][1] = sumPolynomialCoefficients(dev_aux1, dev_aux2, resultLength);
	
	
	/* 10 */
	// clean auxiliary variables
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	
	dev_aux1 = multiplyPolynomialCoefficients(Fp[1][0], hlength, Hp[0][0], hlength, resultLength);
	dev_aux2 = multiplyPolynomialCoefficients(Fp[1][1], hlength, Hp[1][0], hlength, resultLength);
	Ip_aux[1][0] = sumPolynomialCoefficients(dev_aux1, dev_aux2, resultLength);
	
	
	/* 11 */
	// clean auxiliary variables
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	
	dev_aux1 = multiplyPolynomialCoefficients(Fp[1][0], hlength, Hp[0][1], hlength, resultLength);
	dev_aux2 = multiplyPolynomialCoefficients(Fp[1][1], hlength, Hp[1][1], hlength, resultLength);
	Ip_aux[1][1] = sumPolynomialCoefficients(dev_aux1, dev_aux2, resultLength);
	
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	
	return Ip_aux;
}

/**
 * Essa função computa a multiplicação das matrizes Rp, Fp e Ip_aux. Ip_aux = Fp*Hp.
 * Resultado: [2][237]
 *
 *	Rp =	X
 *			Y
 *
 *	Fp =	a	b
 *			c	d
 *
 *	Ip =	A	B
 *			C	D
 *
 */
float** calculateGp_host(float** Rp, int rpLength, float*** Fp, int fpLength, float*** Ip, int ipLength) {
	float *dev_aux1, *dev_aux2, *dev_aux3, *dev_aux4, *dev_aux_sum1, *dev_aux_sum2 ;
	float **finalResult;
	
	int lengthX = (rpLength % 2 == 0 ? rpLength/2 : (rpLength/2+1));
	int lengthY = rpLength/2;
	int partialResultLengthX = lengthX + fpLength - 1;
	int partialResultLengthY = lengthY + fpLength - 1;
	int finalResultLength = (partialResultLengthX + ipLength -1);
	int finalResultSize = finalResultLength * sizeof(float);
	
	/*
	 Gp = Rp*Fp*Ip;
	 O resultado é uma matriz 2x237.
	 
	 ----- Parte 1
	 Rp	  Fp
	 (x	y)	(a	c)
	 (b	d)
	 
	 Rp*Fp
	 xa+yb	xc+yd
	 
	 */
	// Invoke kernel
	dev_aux1 = multiplyPolynomialCoefficients(Rp[0], lengthX, Fp[0][0], fpLength, partialResultLengthX);
	dev_aux2 = multiplyPolynomialCoefficients(Rp[1], lengthY, Fp[0][1], fpLength, partialResultLengthX);
	dev_aux3 = multiplyPolynomialCoefficients(Rp[0], lengthX, Fp[1][0], fpLength, partialResultLengthX);
	dev_aux4 = multiplyPolynomialCoefficients(Rp[1], lengthY, Fp[1][1], fpLength, partialResultLengthX);
	
	dev_aux_sum1 = sumPolynomialCoefficients(dev_aux1, dev_aux2, partialResultLengthX);
	dev_aux_sum2 = sumPolynomialCoefficients(dev_aux3, dev_aux4, partialResultLengthX);
	
	// Re-Inicialização de array no dispositivo para evitar erros em leituras
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	free(dev_aux3); dev_aux3 = NULL;
	free(dev_aux4); dev_aux4 = NULL;
	
	/*
	 Gp = Rp*Fp*Ip;
	 O resultado é uma matriz 2x237.
	 
	 ----- Parte 1
	 Rp	  Fp
	 (x	y)	(a	c)
	 (b	d)
	 
	 Rp*Fp
	 xa+yb	xc+yd
	 u		  v
	 
	 ----- Parte 2	  	  
	 Rp*Fp	  Hp
	 (u	v)	(A	C)
	 (B	D)
	 
	 Rp*Fp*Ip
	 uA+vB	uC+vD
	 
	 */	
	finalResult = (float**) malloc(2 * sizeof(float*));
	finalResult[0] = (float*) malloc(finalResultSize);
	finalResult[1] = (float*) malloc(finalResultSize);
	
	dev_aux1 = multiplyPolynomialCoefficients(dev_aux_sum1, partialResultLengthX, Ip[0][0], ipLength, finalResultLength);
	dev_aux2 = multiplyPolynomialCoefficients(dev_aux_sum2, partialResultLengthX, Ip[0][1], ipLength, finalResultLength);
	dev_aux3 = multiplyPolynomialCoefficients(dev_aux_sum1, partialResultLengthX, Ip[1][0], ipLength, finalResultLength);
	dev_aux4 = multiplyPolynomialCoefficients(dev_aux_sum2, partialResultLengthX, Ip[1][1], ipLength, finalResultLength);
	
	finalResult[0] = sumPolynomialCoefficients(dev_aux1, dev_aux2, finalResultLength);
	finalResult[1] = sumPolynomialCoefficients(dev_aux3, dev_aux4, finalResultLength);
	
	// Cleaning data of auxiliary vectors
	free(dev_aux_sum1); dev_aux_sum1 = NULL;
	free(dev_aux_sum2); dev_aux_sum2 = NULL;
	free(dev_aux1); dev_aux1 = NULL;
	free(dev_aux2); dev_aux2 = NULL;
	free(dev_aux3); dev_aux3 = NULL;
	free(dev_aux4); dev_aux4 = NULL;
	
	return finalResult;
}

/**
 *	Função que realiza a operação de decimação polinomial.
 *
 *  Retorna os coeficientes G (matriz 2xK) que correspondem ao 
 *	sistema R decomposto pelo banco de 2 canais H.
 */
void dec_poly_host(int hlength, float* sist, int sistLength, float** G, int* G_size, float*** Hp, float*** Fp) {	
	float** Rp = (float**) calloc(2, sizeof(float*));
	Rp[0] = (float*) calloc((sistLength % 2 == 0 ? (sistLength/2) : (sistLength/2 + 1)), sizeof(float));
	Rp[1] = (float*) calloc((sistLength/2), sizeof(float));
	
	for (int i = 0; i < sistLength; i++) {
		if (i % 2 == 0) {
			Rp[0][i/2] = sist[i];
		} else {
			Rp[1][i/2] = sist[i];
		}
	}
	
	/*
	 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	 % decomposicao polifasica do sistema R - matriz Rp
	 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	 */
	int m = 2;
	int n = hlength;
	
	/*
	 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	 % decomposicao polifasica dos bancos de sintese e analise
	 % matrizes Fp e Hp
	 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	 */
	
	/*
	 Ip_aux = Fp*Hp;
	 O resultado é uma matriz 2x2x7.
	 
	 H[0][1]*F[0][0] + H[0][0]*F[1][0]	H[0][1]*F[0][1] + H[0][0]*F[1][1]
	 H[1][1]*F[0][0] + H[1][0]*F[1][0]	H[1][1]*F[0][1] + H[1][0]*F[1][1]
	 */
	// Hp e Fp são matrizes (cubos) de dimensão: [2][2][7]	
	float*** ip_aux = computeIpAux_host(Hp, hlength/2, Fp, hlength/2);
	
	// Gp = Rp*Fp*Ip
	float** Gp = calculateGp_host(Rp, sistLength, Fp, (hlength/2), ip_aux, (hlength-1));
	
	int atraso = round((n-m)/2.0);
	double esparsidade = 2.0;
	
	*G_size = ceil( ((sistLength + atraso +1)/esparsidade) +1);
	
	G[0] = (float*) malloc(*G_size * sizeof(float));
	G[1] = (float*) malloc(*G_size * sizeof(float));
	
	for (int i = 0; i < *G_size; i++) {
		G[0][i] = Gp[0][i+atraso];
		G[1][i] = Gp[1][i+atraso];
	}
}


/**
 *	Obtém os coeficientes esparsos que equivalem o sistema ho1d.
 */
void coef_spars_host(char* filtro[], int numFiltros, float* ho1d, int ho1dLength, float** G_aux, int* G_size) {
	int mesmofiltro = 0;
	float* sist = ho1d;
	float** h;
	float** G;
	float** coefSpars;
	int filtroLength;
	int sistLength = ho1dLength;
	
	coefSpars = (float**) malloc((numFiltros+1) * sizeof(float*));
	
	G = (float**) malloc(2 * sizeof(float*));	// sempre tem tamanho 2
	
	if (!mesmofiltro) {
		h = leFiltros(filtro[0], &filtroLength);
		mesmofiltro = (0 < numFiltros-1 && strcmp(filtro[0], filtro[1]) == 0);
	}
	
	// Banco de análise
	double** H = (double**) malloc(2 * sizeof(double*));
	H[0] = (double*) malloc(filtroLength * sizeof(double));
	H[1] = (double*) malloc(filtroLength * sizeof(double));
	
	// Banco de síntese
	double** F = (double**) malloc(2 * sizeof(double*));
	F[0] = (double*) malloc(filtroLength * sizeof(double));
	F[1] = (double*) malloc(filtroLength * sizeof(double));
	
	/*
	 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	 % Obtencao do banco de filtros
	 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	 */
	int power = 1;
	for (int i = filtroLength-1; i >= 0; i--) {
		int reverseIndex = filtroLength-i-1;
		double powerValue = pow((double)-1, (double)power++);
		if (h[0][reverseIndex] + h[1][i]*powerValue != 0) {
			break;	// Não é Daubechies
		} else {
			H[0][reverseIndex] = h[0][reverseIndex];
			H[1][reverseIndex] = h[0][i]*powerValue;
			F[0][reverseIndex] = h[1][reverseIndex]*powerValue;
			F[1][reverseIndex] = h[1][i];
		}
	}
	
	float*** Hp = (float***) malloc(2 * sizeof(float**));
	Hp[0] = (float**) malloc(2 * sizeof(float*));
	Hp[1] = (float**) malloc(2 * sizeof(float*));
	
	float*** Fp = (float***) malloc(2 * sizeof(float**));
	Fp[0] = (float**) malloc(2 * sizeof(float*));
	Fp[1] = (float**) malloc(2 * sizeof(float*));
	
	for (int i = 0; i < 2; i++) {
		Hp[0][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Hp[1][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Fp[0][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Fp[1][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		for (int j = 0; j < (filtroLength/2); j++) {
			Hp[0][i][j] = H[i][j*2];
			Hp[1][i][j] = H[i][j*2+1];
			
			if (i == 0) {
				Fp[0][i][j] = F[0][j*2+1];
				Fp[1][i][j] = F[1][j*2+1];
			} else {
				Fp[0][i][j] = (-1) * F[1][filtroLength*i - (j*2+1)];
				Fp[1][i][j] = F[0][filtroLength*i - (j*2+1)];
			}
		}
	}
	
	for (int j = 0; j < numFiltros; j++) {		
		dec_poly_host(filtroLength, sist, sistLength, G, &G_size[numFiltros-j], Hp, Fp);
		
		sistLength = G_size[numFiltros-j];
		coefSpars[numFiltros-j] = G[1];
		
		sist = NULL;
		sist = G[0];
	}
	
	G_size[0] = G_size[1];
	coefSpars[0] = (float*) malloc(G_size[0] * sizeof(float));
	for (int k = 0; k < G_size[0]; k++) {
		coefSpars[0][k] = sist[k];
	}
	
	int maxG_size = max(G_size, (numFiltros+1));
	for (int k = 0; k < (numFiltros+1); k++) {
		G_aux[k] = (float*) malloc(maxG_size * sizeof(float));
		int cont = 0;
		while (cont < G_size[k]) {
			G_aux[k][cont] = coefSpars[k][cont];
			cont++;
		}
		while (cont < maxG_size) {
			G_aux[k][cont] = 0.0;
			cont++;
		}
	}	
}



/**
 *	2
 */
void dec_poly2BKP(float** coefSpars, int numFiltros, int hlength, float* sist, int sistLength, int* G_size, float*** Hp, float*** Fp) {	
	// Rp
	float *dev_x0 = NULL;
	float *dev_y0 = NULL;
	
	// Fp
	float *dev_a0 = NULL;
	float *dev_b0 = NULL;
	float *dev_c0 = NULL;
	float *dev_d0 = NULL;
	
	// Ip
	float *dev_A0 = NULL;
	float *dev_B0 = NULL;
	float *dev_C0 = NULL;
	float *dev_D0 = NULL;
	
	float *dev_aux1, *dev_aux2, *dev_aux3, *dev_aux4, *dev_aux_sum1, *dev_aux_sum2 ;
	float **Gp;

	float** G = (float**) malloc(2 * sizeof(float*));	// sempre tem tamanho 2
	float** Rp = (float**) calloc(2, sizeof(float*));
	int m = 2;
	int n = hlength;
	int atraso = round((n-m)/2.0);
	double esparsidade = 2.0;
	
	/*
		Decomposicao polifasica dos bancos de sintese e analise
		matrizes Fp e Hp
	
	Ip_aux = Fp*Hp;
	O resultado é uma matriz 2x2x7.
	
	H[0][1]*F[0][0] + H[0][0]*F[1][0]	H[0][1]*F[0][1] + H[0][0]*F[1][1]
	H[1][1]*F[0][0] + H[1][0]*F[1][0]	H[1][1]*F[0][1] + H[1][0]*F[1][1]
	*/
	// Hp e Fp são matrizes (cubos) de dimensão: [2][2][7]
	
	//float*** ip_aux = computeIpAux(Hp, hlength/2, Fp, hlength/2);
	float*** ip_aux = computeIpAuxHost(Hp, hlength/2, Fp, hlength/2);
	
	int fpLength = (hlength/2);
	int ipLength = hlength - 1;
	int sizeFpElement = fpLength * sizeof(float);
	int sizeIpElement = ipLength * sizeof(float);
	
	cudaMalloc( (void**)&dev_a0, sizeFpElement );
	cudaMalloc( (void**)&dev_b0, sizeFpElement );
	cudaMalloc( (void**)&dev_c0, sizeFpElement );
	cudaMalloc( (void**)&dev_d0, sizeFpElement );
	cudaMalloc( (void**)&dev_A0, sizeIpElement );
	cudaMalloc( (void**)&dev_B0, sizeIpElement );
	cudaMalloc( (void**)&dev_C0, sizeIpElement );
	cudaMalloc( (void**)&dev_D0, sizeIpElement );
	
	cudaMemcpy(dev_a0, Fp[0][0], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_b0, Fp[0][1], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_c0, Fp[1][0], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_d0, Fp[1][1], sizeFpElement, cudaMemcpyHostToDevice );
	
	cudaMemcpy(dev_A0, ip_aux[0][0], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_B0, ip_aux[0][1], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_C0, ip_aux[1][0], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_D0, ip_aux[1][1], sizeIpElement, cudaMemcpyHostToDevice );	
	
	for (int j = 0; j < numFiltros; j++) {
		Rp[0] = (float*) calloc((sistLength % 2 == 0 ? (sistLength/2) : (sistLength/2 + 1)), sizeof(float));
		Rp[1] = (float*) calloc((sistLength/2), sizeof(float));

		for (int i = 0; i < sistLength; i++) {
			if (i % 2 == 0) {
				Rp[0][i/2] = sist[i];
			} else {
				Rp[1][i/2] = sist[i];
			}
		}
		
		int rpLength = sistLength;
		int lengthX = (rpLength % 2 == 0 ? rpLength/2 : (rpLength/2+1));
		int lengthY = rpLength/2;
		int sizeX = lengthX * sizeof(float);
		int sizeY = lengthY * sizeof(float);
		int partialResultLengthX = lengthX + fpLength - 1;
		int partialResultSizeX = partialResultLengthX * sizeof(float);
		int partialResultLengthY = lengthY + fpLength - 1;
		int partialResultSizeY = partialResultLengthY * sizeof(float);
		int finalResultLength = (partialResultLengthX + ipLength -1);
		int finalResultSize = finalResultLength * sizeof(float);
		
		cudaMalloc( (void**)&dev_x0, sizeX );
		cudaMalloc( (void**)&dev_y0, sizeY );
		
		cudaMalloc( (void**)&dev_aux1, partialResultSizeX );
		cudaMalloc( (void**)&dev_aux2, partialResultSizeY );
		cudaMalloc( (void**)&dev_aux3, partialResultSizeX );
		cudaMalloc( (void**)&dev_aux4, partialResultSizeY );
	
		cudaMalloc( (void**)&dev_aux_sum1, partialResultSizeX );
		cudaMalloc( (void**)&dev_aux_sum2, partialResultSizeX );
	
		// Inicialização de array no dispositivo para evitar erros em leituras´
//		cudaMemset(dev_aux1, 0, partialResultLengthX * sizeof(float));
//		cudaMemset(dev_aux2, 0, partialResultLengthX * sizeof(float));
//		cudaMemset(dev_aux3, 0, partialResultLengthX * sizeof(float));
//		cudaMemset(dev_aux4, 0, partialResultLengthX * sizeof(float));
//		cudaMemset(dev_aux_sum1, 0, partialResultLengthX * sizeof(float));
//		cudaMemset(dev_aux_sum2, 0, partialResultLengthX * sizeof(float));
		
		
		initializeArray<<<16,16>>>(dev_aux1, partialResultLengthX);
		initializeArray<<<16,16>>>(dev_aux2, partialResultLengthX);
		initializeArray<<<16,16>>>(dev_aux3, partialResultLengthX);
		initializeArray<<<16,16>>>(dev_aux4, partialResultLengthX);
		initializeArray<<<16,16>>>(dev_aux_sum1, partialResultLengthX);
		initializeArray<<<16,16>>>(dev_aux_sum2, partialResultLengthX);
		
		cudaMemcpy(dev_x0, Rp[0], sizeX, cudaMemcpyHostToDevice );
		cudaMemcpy(dev_y0, Rp[1], sizeY, cudaMemcpyHostToDevice );
		
		/*
		Gp = Rp*Fp*Ip;
		O resultado é uma matriz 2x237.
	
		----- Parte 1
		Rp		Fp
		(x	y)	(a	c)
				(b	d)
	
		Rp*Fp
			xa+yb	xc+yd
	
		*/
		// Invoke kernel
		multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_x0, lengthX, dev_a0, fpLength, dev_aux1);	
		multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_y0, lengthY, dev_b0, fpLength, dev_aux2);
		multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_x0, lengthX, dev_c0, fpLength, dev_aux3);
		multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_y0, lengthY, dev_d0, fpLength, dev_aux4);
	
		sumPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux1, dev_aux2, dev_aux_sum1, partialResultLengthX);
		sumPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux3, dev_aux4, dev_aux_sum2, partialResultLengthX);
		
		// Re-Inicialização de array no dispositivo para evitar erros em leituras
		cudaFree(dev_aux1);
		cudaFree(dev_aux2);
		cudaFree(dev_aux3);
		cudaFree(dev_aux4);
	
		cudaMalloc( (void**)&dev_aux1, finalResultSize );
		cudaMalloc( (void**)&dev_aux2, finalResultSize );
		cudaMalloc( (void**)&dev_aux3, finalResultSize );
		cudaMalloc( (void**)&dev_aux4, finalResultSize );
	
		initializeArray<<<16,16>>>(dev_aux1, finalResultLength);
		initializeArray<<<16,16>>>(dev_aux2, finalResultLength);
		initializeArray<<<16,16>>>(dev_aux3, finalResultLength);
		initializeArray<<<16,16>>>(dev_aux4, finalResultLength);
		
		
		/*
		Gp = Rp*Fp*Ip;
		O resultado é uma matriz 2x237.
	
		----- Parte 1
		Rp		Fp
		(x	y)	(a	c)
				(b	d)
	
		Rp*Fp
			xa+yb	xc+yd
			  u		  v
	
		----- Parte 2	  
		Rp*Fp	  Hp
		(u	v)	(A	C)
				(B	D)
	
		Rp*Fp*Ip
			uA+vB	uC+vD
	
		*/
		multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux_sum1, partialResultLengthX, dev_A0, ipLength, dev_aux1);	
		multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux_sum2, partialResultLengthX, dev_B0, ipLength, dev_aux2);
		multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux_sum1, partialResultLengthX, dev_C0, ipLength, dev_aux3);
		multiplyPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux_sum2, partialResultLengthX, dev_D0, ipLength, dev_aux4);
	
		// Cleaning data of auxiliary vectors
		cudaFree(dev_aux_sum1);
		cudaFree(dev_aux_sum2);
		cudaMalloc( (void**)&dev_aux_sum1, finalResultSize );
		cudaMalloc( (void**)&dev_aux_sum2, finalResultSize );
	
		// Reinicializa arrays u e v (resultado de Rp*Fp)
		initializeArray<<<16,16>>>(dev_aux_sum1, finalResultLength);
		initializeArray<<<16,16>>>(dev_aux_sum2, finalResultLength);
	
		sumPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux1, dev_aux2, dev_aux_sum1, finalResultLength);
		sumPolynomialCoefficientsKernel<<<16, 16>>>(dev_aux3, dev_aux4, dev_aux_sum2, finalResultLength);
		
		Gp = (float**) malloc(2 * sizeof(float*));
		Gp[0] = (float*) malloc(finalResultSize);
		Gp[1] = (float*) malloc(finalResultSize);
	
		// copy the data from device to locked memory
		cudaMemcpy(Gp[0], dev_aux_sum1, finalResultSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(Gp[1], dev_aux_sum2, finalResultSize, cudaMemcpyDeviceToHost);
		
		G_size[numFiltros-j] = ceil( ((sistLength + atraso +1)/esparsidade) +1);
	
		G[0] = (float*) malloc(G_size[numFiltros-j] * sizeof(float));
		G[1] = (float*) malloc(G_size[numFiltros-j] * sizeof(float));
	
		for (int i = 0; i < G_size[numFiltros-j]; i++) {
			G[0][i] = Gp[0][i+atraso];
			G[1][i] = Gp[1][i+atraso];
		}
		
		sistLength = G_size[numFiltros-j];
		coefSpars[numFiltros-j] = G[1];
		
		sist = NULL;
		sist = G[0];
		
		cudaFree(dev_x0);
		cudaFree(dev_y0);
		cudaFree(dev_aux1);
		cudaFree(dev_aux2);
		cudaFree(dev_aux3);
		cudaFree(dev_aux4);
		cudaFree(dev_aux_sum1);
		cudaFree(dev_aux_sum2);
	}
	
	G_size[0] = G_size[1];
	coefSpars[0] = (float*) malloc(G_size[0] * sizeof(float));
	for (int k = 0; k < G_size[0]; k++) {
		coefSpars[0][k] = sist[k];
	}
	
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_d0);
	cudaFree(dev_A0);
	cudaFree(dev_B0);
	cudaFree(dev_C0);
	cudaFree(dev_D0);

	//printf("Error dec_poly2: %s\n\n", cudaGetErrorString(cudaGetLastError()));
	
}

__device__ float dev_aux1[256];
__device__ float dev_aux2[256];
__device__ float dev_aux3[256];
__device__ float dev_aux4[256];

__global__ void MultiplyRpAndFp(float* dev_x0, int lengthX, float* dev_y0, int lengthY, float* dev_a0, float* dev_b0, float* dev_c0, float* dev_d0, int fpLength, 
		float* dev_aux_sum1, float* dev_aux_sum2, int partialResultLengthX) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < partialResultLengthX) {
		dev_aux1[i] = 0.0;
		dev_aux2[i] = 0.0;
		dev_aux3[i] = 0.0;
		dev_aux4[i] = 0.0;
		dev_aux_sum1[i] = 0.0;
		dev_aux_sum2[i] = 0.0;
	}
	
    if (i < lengthX) {
        for (int j = 0; j < fpLength; j++) {
			dev_aux1[i+j] = dev_aux1[i+j] + (dev_x0[i] * dev_a0[j]);
			dev_aux3[i+j] = dev_aux3[i+j] + (dev_x0[i] * dev_c0[j]);
		}
	}

	if (i < lengthY) {
        for (int j = 0; j < fpLength; j++) {
			dev_aux2[i+j] = dev_aux2[i+j] + (dev_y0[i] * dev_b0[j]);
			dev_aux4[i+j] = dev_aux4[i+j] + (dev_y0[i] * dev_d0[j]);
		}
	}

    if (i < partialResultLengthX) {
		dev_aux_sum1[i] = dev_aux1[i] + dev_aux2[i];
		dev_aux_sum2[i] = dev_aux3[i] + dev_aux4[i];
	}
}

__global__ void MultiplyPart2(float* dev_aux_sum1, float* dev_aux_sum2, int partialResultLengthX, 
		float* dev_A0, float* dev_B0, float* dev_C0, float* dev_D0, int ipLength, 
		float* dev_aux_sum3, float* dev_aux_sum4, int finalResultLength) {
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < finalResultLength) {
		dev_aux1[i] = 0.0;
		dev_aux2[i] = 0.0;
		dev_aux3[i] = 0.0;
		dev_aux4[i] = 0.0;
		dev_aux_sum3[i] = 0.0;
		dev_aux_sum4[i] = 0.0;
	}
	
	if (i < partialResultLengthX) {
        for (int j = 0; j < ipLength; j++) {
			dev_aux1[i+j] = dev_aux1[i+j] + (dev_aux_sum1[i] * dev_A0[j]);
			dev_aux2[i+j] = dev_aux2[i+j] + (dev_aux_sum2[i] * dev_B0[j]);
			dev_aux3[i+j] = dev_aux3[i+j] + (dev_aux_sum1[i] * dev_C0[j]);
			dev_aux4[i+j] = dev_aux4[i+j] + (dev_aux_sum2[i] * dev_D0[j]);
		}
	}
	
	if (i < finalResultLength) {
		dev_aux_sum3[i] = dev_aux1[i] + dev_aux2[i];
		dev_aux_sum4[i] = dev_aux3[i] + dev_aux4[i];
	}
	
}

	// Fp
	__constant__ float *dev_a0 = NULL;
	__constant__ float *dev_b0 = NULL;
	__constant__ float *dev_c0 = NULL;
	__constant__ float *dev_d0 = NULL;
	
	// Ip
	__constant__ float *dev_A0 = NULL;
	__constant__ float *dev_B0 = NULL;
	__constant__ float *dev_C0 = NULL;
	__constant__ float *dev_D0 = NULL;

void dec_poly2(float** coefSpars, int numFiltros, int hlength, float* sist, int sistLength, int* G_size, float*** Hp, float*** Fp) {	
	// Rp
	float *dev_x0 = NULL;
	float *dev_y0 = NULL;
	
	float *dev_aux_sum1, *dev_aux_sum2, *dev_aux_sum3, *dev_aux_sum4;
	float **Gp;

	float** G = (float**) malloc(2 * sizeof(float*));	// sempre tem tamanho 2
	float** Rp = (float**) calloc(2, sizeof(float*));
	int m = 2;
	int n = hlength;
	int atraso = round((n-m)/2.0);
	double esparsidade = 2.0;

	INIT_VARIABLES; 
	
	/*
		Decomposicao polifasica dos bancos de sintese e analise
		matrizes Fp e Hp
	
	Ip_aux = Fp*Hp;
	O resultado é uma matriz 2x2x7.
	
	H[0][1]*F[0][0] + H[0][0]*F[1][0]	H[0][1]*F[0][1] + H[0][0]*F[1][1]
	H[1][1]*F[0][0] + H[1][0]*F[1][0]	H[1][1]*F[0][1] + H[1][0]*F[1][1]
	*/
	// Hp e Fp são matrizes (cubos) de dimensão: [2][2][7]
	
	//float*** ip_aux = computeIpAux(Hp, hlength/2, Fp, hlength/2);	
	float*** ip_aux = computeIpAuxHost(Hp, hlength/2, Fp, hlength/2);
	
	int fpLength = (hlength/2);
	int ipLength = hlength - 1;
	int sizeFpElement = fpLength * sizeof(float);
	int sizeIpElement = ipLength * sizeof(float);	
	
	INIT_RUNTIME;
	cudaMalloc( (void**)&dev_a0, sizeFpElement );
	cudaMalloc( (void**)&dev_b0, sizeFpElement );
	cudaMalloc( (void**)&dev_c0, sizeFpElement );
	cudaMalloc( (void**)&dev_d0, sizeFpElement );
	cudaMalloc( (void**)&dev_A0, sizeIpElement );
	cudaMalloc( (void**)&dev_B0, sizeIpElement );
	cudaMalloc( (void**)&dev_C0, sizeIpElement );
	cudaMalloc( (void**)&dev_D0, sizeIpElement );
	
	cudaMemcpy(dev_a0, Fp[0][0], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_b0, Fp[0][1], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_c0, Fp[1][0], sizeFpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_d0, Fp[1][1], sizeFpElement, cudaMemcpyHostToDevice );
	
	cudaMemcpy(dev_A0, ip_aux[0][0], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_B0, ip_aux[0][1], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_C0, ip_aux[1][0], sizeIpElement, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_D0, ip_aux[1][1], sizeIpElement, cudaMemcpyHostToDevice );
	END_RUNTIME; printf("\n[cudaMalloc Memcpy 1]: "); PRINT_RUNTIME;
	
	//INIT_RUNTIME;
	for (int j = 0; j < numFiltros; j++) {
		Rp[0] = (float*) calloc((sistLength % 2 == 0 ? (sistLength/2) : (sistLength/2 + 1)), sizeof(float));
		Rp[1] = (float*) calloc((sistLength/2), sizeof(float));
		
		for (int i = 0; i < sistLength; i++) {
			if (i % 2 == 0) {
				Rp[0][i/2] = sist[i];
			} else {
				Rp[1][i/2] = sist[i];
			}
		}
		
		int rpLength = sistLength;
		int lengthX = (rpLength % 2 == 0 ? rpLength/2 : (rpLength/2+1));
		int lengthY = rpLength/2;
		int sizeX = lengthX * sizeof(float);
		int sizeY = lengthY * sizeof(float);
		int partialResultLengthX = lengthX + fpLength - 1;
		int partialResultSizeX = partialResultLengthX * sizeof(float);
		int partialResultLengthY = lengthY + fpLength - 1;
		int partialResultSizeY = partialResultLengthY * sizeof(float);
		int finalResultLength = (partialResultLengthX + ipLength -1);
		int finalResultSize = finalResultLength * sizeof(float);
		
		
		cudaMalloc( (void**)&dev_x0, sizeX );
		cudaMalloc( (void**)&dev_y0, sizeY );
		cudaMalloc( (void**)&dev_aux_sum1, partialResultSizeX );
		cudaMalloc( (void**)&dev_aux_sum2, partialResultSizeX );
		
		cudaMemcpy(dev_x0, Rp[0], sizeX, cudaMemcpyHostToDevice );
		cudaMemcpy(dev_y0, Rp[1], sizeY, cudaMemcpyHostToDevice );
		
		/*
		Gp = Rp*Fp*Ip;
		O resultado é uma matriz 2x237.
	
		----- Parte 1
		Rp		Fp
		(x	y)	(a	c)
				(b	d)
	
		Rp*Fp
			xa+yb	xc+yd
	
		*/
		// Invoke kernel
		MultiplyRpAndFp<<<16,16>>>(dev_x0, lengthX, dev_y0, lengthY, dev_a0, dev_b0, dev_c0, dev_d0, fpLength, dev_aux_sum1, dev_aux_sum2, partialResultLengthX/*, dev_aux1, dev_aux2, dev_aux3, dev_aux4*/);
		
		cudaMalloc( (void**)&dev_aux_sum3, finalResultSize );
		cudaMalloc( (void**)&dev_aux_sum4, finalResultSize );		
		
		/*
		Gp = Rp*Fp*Ip;
		O resultado é uma matriz 2x237.
	
		----- Parte 1
		Rp		Fp
		(x	y)	(a	c)
				(b	d)
	
		Rp*Fp
			xa+yb	xc+yd
			  u		  v
	
		----- Parte 2	  
		Rp*Fp	  Hp
		(u	v)	(A	C)
				(B	D)
	
		Rp*Fp*Ip
			uA+vB	uC+vD
	
		*/
		MultiplyPart2<<<16,16>>>(dev_aux_sum1, dev_aux_sum2, partialResultLengthX, dev_A0, dev_B0, dev_C0, dev_D0, ipLength, /*dev_aux1, dev_aux2, dev_aux3, dev_aux4,*/ dev_aux_sum3, dev_aux_sum4, finalResultLength);
		
		Gp = (float**) malloc(2 * sizeof(float*));
		Gp[0] = (float*) malloc(finalResultSize);
		Gp[1] = (float*) malloc(finalResultSize);
	
		// copy the data from device to locked memory
		cudaMemcpy(Gp[0], dev_aux_sum3, finalResultSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(Gp[1], dev_aux_sum4, finalResultSize, cudaMemcpyDeviceToHost);
		
		G_size[numFiltros-j] = ceil( ((sistLength + atraso +1)/esparsidade) +1);
	
		G[0] = (float*) malloc(G_size[numFiltros-j] * sizeof(float));
		G[1] = (float*) malloc(G_size[numFiltros-j] * sizeof(float));
	
		for (int i = 0; i < G_size[numFiltros-j]; i++) {
			G[0][i] = Gp[0][i+atraso];
			G[1][i] = Gp[1][i+atraso];
		}
		
		sistLength = G_size[numFiltros-j];
		coefSpars[numFiltros-j] = G[1];
		
		sist = NULL;
		sist = G[0];
		
		dev_x0 = NULL;
		dev_y0 = NULL;
		dev_aux_sum1 = NULL;
		dev_aux_sum2 = NULL;
		dev_aux_sum3 = NULL;
		dev_aux_sum4 = NULL;
	}
	//END_RUNTIME; printf("\n[for]: "); PRINT_RUNTIME;
	
	G_size[0] = G_size[1];
	coefSpars[0] = (float*) malloc(G_size[0] * sizeof(float));
	for (int k = 0; k < G_size[0]; k++) {
		coefSpars[0][k] = sist[k];
	}
	
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_d0);
	cudaFree(dev_A0);
	cudaFree(dev_B0);
	cudaFree(dev_C0);
	cudaFree(dev_D0);

	//printf("Error dec_poly2: %s\n\n", cudaGetErrorString(cudaGetLastError()));
	
}


/**
 *	Obtém os coeficientes esparsos que equivalem o sistema ho1d.
 */
void coef_spars2(char* filtro[], int numFiltros, float* ho1d, int ho1dLength, float** G_aux, int* G_size) {
	int mesmofiltro = 0;
	float* sist = ho1d;
	float** h;
	float** coefSpars;
	int filtroLength;
	int sistLength = ho1dLength;

	coefSpars = (float**) malloc((numFiltros+1) * sizeof(float*));
	
	if (!mesmofiltro) {
		h = leFiltros(filtro[0], &filtroLength);
		mesmofiltro = (0 < numFiltros-1 && strcmp(filtro[0], filtro[1]) == 0);
	}
	
	// Banco de análise
	double** H = (double**) malloc(2 * sizeof(double*));
	H[0] = (double*) malloc(filtroLength * sizeof(double));
	H[1] = (double*) malloc(filtroLength * sizeof(double));

	// Banco de síntese
	double** F = (double**) malloc(2 * sizeof(double*));
	F[0] = (double*) malloc(filtroLength * sizeof(double));
	F[1] = (double*) malloc(filtroLength * sizeof(double));
	
	/*
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Obtencao do banco de filtros
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	*/
	int power = 1;
	for (int i = filtroLength-1; i >= 0; i--) {
		int reverseIndex = filtroLength-i-1;
		double powerValue = pow((double)-1, (double)power++);
		if (h[0][reverseIndex] + h[1][i]*powerValue != 0) {
			break;	// Não é Daubechies
		} else {
			H[0][reverseIndex] = h[0][reverseIndex];
			H[1][reverseIndex] = h[0][i]*powerValue;
			F[0][reverseIndex] = h[1][reverseIndex]*powerValue;
			F[1][reverseIndex] = h[1][i];
		}
	}
	
	float*** Hp = (float***) malloc(2 * sizeof(float**));
	Hp[0] = (float**) malloc(2 * sizeof(float*));
	Hp[1] = (float**) malloc(2 * sizeof(float*));
	
	float*** Fp = (float***) malloc(2 * sizeof(float**));
	Fp[0] = (float**) malloc(2 * sizeof(float*));
	Fp[1] = (float**) malloc(2 * sizeof(float*));
	
	for (int i = 0; i < 2; i++) {
		Hp[0][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Hp[1][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Fp[0][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		Fp[1][i] = (float*) malloc((filtroLength/2) * sizeof(float));
		for (int j = 0; j < (filtroLength/2); j++) {
			Hp[0][i][j] = H[i][j*2];
			Hp[1][i][j] = H[i][j*2+1];
			
			if (i == 0) {
				Fp[0][i][j] = F[0][j*2+1];
				Fp[1][i][j] = F[1][j*2+1];
			} else {
				Fp[0][i][j] = (-1) * F[1][filtroLength*i - (j*2+1)];
				Fp[1][i][j] = F[0][filtroLength*i - (j*2+1)];
			}
		}
	}
	
	//INIT_VARIABLES; INIT_RUNTIME;
	dec_poly2(coefSpars, numFiltros, filtroLength, sist, sistLength, G_size, Hp, Fp);
	//END_RUNTIME; printf("\n[dec_poly2]: "); PRINT_RUNTIME;
	
	int maxG_size = max(G_size, (numFiltros+1));
	for (int k = 0; k < (numFiltros+1); k++) {
		G_aux[k] = (float*) malloc(maxG_size * sizeof(float));
		int cont = 0;
		while (cont < G_size[k]) {
			G_aux[k][cont] = coefSpars[k][cont];
			cont++;
		}
		while (cont < maxG_size) {
			G_aux[k][cont] = 0.0;
			cont++;
		}
	}
}



/**
 *	Função que retorna os primeiros X elementos de um vetor.
 */
float* getPartialVector(float* vec, int numOfElements) {
    float* partialVec = (float*) malloc(numOfElements * sizeof(float));
	for (int i = 0; i < numOfElements; i++) {
		partialVec[i] = vec[i];
	}
	return partialVec;
}

__global__ void getL(int width, int* row) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < width) {
		if (i == 0) {
			row[0] = powf(2.0, width-1);
		} else {
			for (int c = 1; c < width; ++c) {
				row[c] = powf(2.0, width-c);
			}
		}
	}
}


__shared__ int L;		// 16 16 8 4 2 - coeficientes de esparsidade
__shared__ float gaux[608]; // 512+32+32+32
__shared__ float fbaux[256];
__shared__ int partialVecLength;
__shared__ int resultLength;
__shared__ int convLength;
__shared__ int maxLength;
__shared__ int minLength;
__device__ float convolutionResult[1024 * 5];

__device__ float calcConv(int index, float* dev_filterBank, int filterBankLength, float* gaux, int gauxLength) {
	float r;
	
	int maxLength = (filterBankLength >= resultLength ? filterBankLength : resultLength);
	int minLength = (filterBankLength <= resultLength ? filterBankLength : resultLength);
	
	for (int j = 0; j < maxLength; j++ ) {
		if ((index - j) >= 0 && (index - j) < minLength) {
			if (maxLength == filterBankLength) {
				r += (dev_filterBank[j] * gaux[index - j]);    // convolve: multiply and accumulate
			} else {
				r += (gaux[j] * dev_filterBank[index - j]);    // convolve: multiply and accumulate
			}				
		}
	}
	
	return r;
}

__device__ float convCalc(int index, float* u, int uLength, float* v, int vLength, int maxLength, int minLength) {
	float w = 0.0;
	for (int j = 0; j < maxLength; j++ ) {
		if ((index - j) >= 0 && (index - j) < minLength) {
			if (maxLength == uLength) {
				w += (u[j] * v[index-j]);    // convolve: multiply and accumulate
			} else {
				w += (v[j] * u[index-j]);    // convolve: multiply and accumulate
			}				
		}
	}
	return w;
}


__global__ void getR(int height, size_t pitchG, int* G_size, float* dev_G, int widthG, int* filterBankLength, float* dev_filterBank, int* r_size, float* r, int widthR, float* test) {
	int tidX = blockDim.x * blockIdx.x + threadIdx.x;
	int atrasos[5] = {1, 1, 8, 22, 50};

	//L[blockIdx.x] = (int) (blockIdx.x == 0) ? powf(2.0, height-1) : powf(2.0, height-blockIdx.x);
	partialVecLength = (G_size[blockIdx.x] + atrasos[blockIdx.x]);
	//resultLength = (((partialVecLength - 1) * L[blockIdx.x]) + 1);
	convLength = (filterBankLength[blockIdx.x] + resultLength - 1);
	r_size[blockIdx.x] = convLength;
		
/*	while (i < widthR) {
		if (i < r_size[blockIdx.x]) {
			r[i] = i;
//			for (int j = 0; j < maxLength; j++ ) {
//				if ((tidX - j) >= 0 && (tidX - j) < minLength) {
//					if (maxLength == filterBankLength[blockIdx.x]) {
//						r[tidX] += (dev_filterBank[j] * gaux[tidX - j]);    // convolve: multiply and accumulate
//					} else {
//						r[tidX] += (gaux[j] * dev_filterBank[tidX - j]);    // convolve: multiply and accumulate
//					}				
//				}
//			}
		} else {
			r[i] = 0.0;
		}

		i += blockDim.x;
	}
*/

}

__global__ void test(int height, int widthG, float* dev_G, int* G_size, float* filterBank, int* filterBankLength, int* r_size, float* resultado) {	
	int tidX = threadIdx.x + blockIdx.x * blockDim.x;
	int atrasos[5] = {1, 1, 8, 22, 50};

	L = (int) (blockIdx.x == 0) ? powf(2.0, height-1) : powf(2.0, height-blockIdx.x);
	partialVecLength = (G_size[blockIdx.x] + atrasos[blockIdx.x]);
	resultLength = (((partialVecLength - 1) * L) + 1);
	convLength = (filterBankLength[blockIdx.x] + resultLength - 1);
	r_size[blockIdx.x] = convLength;
	
	// Esparsando os coeficientes de G	
	if (threadIdx.x % L == 0) {
		gaux[threadIdx.x] = dev_G[(threadIdx.x / L) + (blockIdx.x * blockDim.x)];
	} else {
		gaux[threadIdx.x] = 0.0;
	}
	if ((threadIdx.x + blockDim.x) < resultLength) {
		int index = threadIdx.x + blockDim.x;
		
		if (index % L == 0) {
			gaux[index] = dev_G[(index / L) + (blockIdx.x * blockDim.x)];
		} else {
			gaux[index] = 0.0;
		}			
	}

	if (threadIdx.x < filterBankLength[blockIdx.x]) {
		fbaux[threadIdx.x] = filterBank[tidX];
	}

	if (threadIdx.x == 0) {	// This is executed just once in a block
		maxLength = (filterBankLength[blockIdx.x] >= resultLength ? filterBankLength[blockIdx.x] : resultLength);
		minLength = (filterBankLength[blockIdx.x] <= resultLength ? filterBankLength[blockIdx.x] : resultLength);
	}
	
	__syncthreads();
	
	int index = 0;
	if (threadIdx.x < convLength) {
		index = threadIdx.x;
		
		float convResult = convCalc(index, fbaux, filterBankLength[blockIdx.x], gaux, resultLength, maxLength, minLength);
		convolutionResult[index + blockIdx.x * blockDim.x * 2] = convResult;
		
		if (index + blockDim.x < convLength) {
			index = index + blockDim.x;
			convResult = convCalc(index, fbaux, filterBankLength[blockIdx.x], gaux, resultLength, maxLength, minLength);
			convolutionResult[index + blockIdx.x * blockDim.x * 2] = convResult;
		}

	}
	
	index = threadIdx.x + blockDim.x;
	
	resultado[threadIdx.x] = convolutionResult[threadIdx.x + (0 * blockDim.x * 2)] + 
							convolutionResult[threadIdx.x + (1 * blockDim.x * 2)] +
							convolutionResult[threadIdx.x + (2 * blockDim.x * 2)] +
							convolutionResult[threadIdx.x + (3 * blockDim.x * 2)] +
							convolutionResult[threadIdx.x + (4 * blockDim.x * 2)];
	
	resultado[index] = convolutionResult[index+ (0 * blockDim.x * 2)] + 
						convolutionResult[index + (1 * blockDim.x * 2)] +
						convolutionResult[index + (2 * blockDim.x * 2)] +
						convolutionResult[index + (3 * blockDim.x * 2)] +
						convolutionResult[index + (4 * blockDim.x * 2)];
}

const int MAX_NUM_OF_THREADS = 512;
/**
 *	Função que retorna a resposta impulsiva a partir dos coeficientes
 *	esparsos.
 */
float* resp_imp(char* filtros[], int numFiltros, float** G, int* G_size, int* resultLength) {
	float** filterBank = (float**) calloc((numFiltros + 1), sizeof(float));
	int* filterBankLength = (int*) calloc((numFiltros + 1), sizeof(int));
	
	//cascataFFT(filtros, numFiltros, filterBank, filterBankLength);
	cascataSimple(filtros, numFiltros, filterBank, filterBankLength);
	
	//INIT_VARIABLES; INIT_RUNTIME;
	// TODO implementar calc_delta
	int atrasos[5] = {1, 1, 8, 22, 50};
	int widthG = max(G_size, numFiltros+1) + max(atrasos, numFiltros+1);
	int height = numFiltros + 1;
	
	float *d_resultado, *h_resultado;
	float *d_G, *d_filterBank, *h_G, *h_filterBank;
	int *d_GSize, *d_filterBankLength, *h_rSize, *d_rSize;
	
	h_G = (float*) malloc(MAX_NUM_OF_THREADS * height * sizeof(float));
	h_filterBank = (float*) malloc(MAX_NUM_OF_THREADS * height * sizeof(float));
	h_rSize = (int*) calloc(height, sizeof(int));
	h_resultado = (float*) calloc(2 * MAX_NUM_OF_THREADS, sizeof(float));
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < MAX_NUM_OF_THREADS; j++) {
			if (j < widthG) {
				h_G[j + i*MAX_NUM_OF_THREADS] = G[i][j];
			} else {
				h_G[j + i*MAX_NUM_OF_THREADS] = 0.0;
			}
			
			if (j < filterBankLength[i]) {
				h_filterBank[j + i*MAX_NUM_OF_THREADS] = filterBank[i][j];
			} else {
				h_filterBank[j + i*MAX_NUM_OF_THREADS] = 0.0;
			}
		}
	}
	
	//INIT_RUNTIME;
	cudaMalloc((void**) &d_GSize, height * sizeof(int));
	cudaMalloc((void**) &d_filterBankLength, height * sizeof(int));
	cudaMalloc((void**) &d_rSize, height * sizeof(int));
	cudaMalloc((void**) &d_resultado, 2 * MAX_NUM_OF_THREADS * sizeof(float));
	
	cudaMalloc((void**) &d_G, MAX_NUM_OF_THREADS * height * sizeof(float));
	cudaMalloc((void**) &d_filterBank, MAX_NUM_OF_THREADS * height * sizeof(float));
	
	cudaMemcpy(d_G, h_G, MAX_NUM_OF_THREADS * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_GSize, G_size, height * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_filterBank, h_filterBank, MAX_NUM_OF_THREADS * height * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_filterBankLength, filterBankLength, height * sizeof(int), cudaMemcpyHostToDevice);
	//END_RUNTIME; printf("\n[Malloc e memcpy]: "); PRINT_RUNTIME;
	
	//INIT_RUNTIME;
	test<<<height, MAX_NUM_OF_THREADS>>>(height, widthG, d_G, d_GSize, d_filterBank, d_filterBankLength, d_rSize, d_resultado);
	//END_RUNTIME; printf("\n[kernel]: "); PRINT_RUNTIME;

	//INIT_RUNTIME;
	cudaMemcpy(h_resultado, d_resultado, 2 * MAX_NUM_OF_THREADS * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rSize, d_rSize, height * sizeof(int), cudaMemcpyDeviceToHost);
	//END_RUNTIME; printf("\n[memcpy result]: "); PRINT_RUNTIME;
	
	
	int maxRSize = max(h_rSize, height);
	int maxFilterBankLength = max(filterBankLength, height);
	*resultLength = (maxRSize - maxFilterBankLength);
	float* respostaImpulsiva = (float*) calloc(*resultLength, sizeof(float));
	int j = 0;
	for (int i = maxFilterBankLength; i < maxRSize; i++) {
		respostaImpulsiva[j++] = h_resultado[i];
	}
	
	//INIT_RUNTIME;
	cudaFree(d_resultado);
	cudaFree(d_G);
	cudaFree(d_filterBank);
	cudaFree(d_GSize);
	cudaFree(d_filterBankLength);
	cudaFree(d_rSize);
	
	free(h_resultado);
	free(filterBank);
	free(filterBankLength);
	free(h_G);
	free(h_filterBank);
	free(h_rSize);
	//END_RUNTIME; printf("\n[free]: "); PRINT_RUNTIME;
	//END_RUNTIME; printf("\n[1]: "); PRINT_RUNTIME;
	
	return respostaImpulsiva;
}


/**
 *	Função que retorna a resposta impulsiva a partir dos coeficientes
 *	esparsos.
 */
float* resp_imp2(char* filtros[], int numFiltros, float** G, int* G_size, int* resultLength) {
	float** filterBank = (float**) calloc((numFiltros + 1), sizeof(float));
	int* filterBankLength = (int*) calloc((numFiltros + 1), sizeof(int));
	
	cascataFFT(filtros, numFiltros, filterBank, filterBankLength);
	
	// TODO implementar calc_delta
	int atrasos[5] = {1, 1, 8, 22, 50};
	
	
	// Start changes ##########################################################
/*	INIT_RUNTIME;
	int* dev_L, *h_L;
	int widthL = (numFiltros + 1);
	h_L = (int*) malloc(widthL * sizeof(int));
	cudaMalloc( (void**)&dev_L, widthL * sizeof(int) );
	getL<<<32, 1>>>(widthL, dev_L);
	//cudaMemcpy(h_L, dev_L, widthL * sizeof(int), cudaMemcpyDeviceToHost);
	END_RUNTIME; printf("\n[L1]: "); PRINT_RUNTIME;
	
	INIT_RUNTIME;
	int* L = (int*) calloc(numFiltros + 1, sizeof(int));
	L[0] = (int) pow(2.0, numFiltros);
	for (int i = 1; i < (numFiltros + 1); i++) {
		L[i] = (int) pow(2.0, (numFiltros+1)-i);
		//printf("h_L[%d] = %d, L[%d] = %d\n", i, h_L[i], i, L[i]);
	}
	END_RUNTIME; printf("\n[L2]: "); PRINT_RUNTIME;
*/	
	int maxValueOfGSize = max(G_size, numFiltros+1) + max(atrasos, numFiltros+1);
	int maxValueOfFilterBankLength = max(filterBankLength, numFiltros+1);
	int widthG = maxValueOfGSize;
	int widthFilterBank = maxValueOfFilterBankLength;
	int widthR = 1024;
	int height = (numFiltros + 1);
	
	int *dev_GSize, *dev_filterBankLength, *dev_RSize;
	float* dev_G, *dev_filterBank, *dev_R;
	size_t pitchG, pitchFilterBank, pitchR;
	
	// copying G data
	cudaMalloc( (void**)&dev_GSize, height * sizeof(int) );
	cudaMemcpy( dev_GSize, G_size, height * sizeof(int), cudaMemcpyHostToDevice);	
	//cudaMallocPitch(&dev_G, &pitchG, widthG * sizeof(float), height);
	//cudaMemcpy2D(dev_G, widthG * sizeof(float), G, pitchG, widthG * sizeof(float), height, cudaMemcpyHostToDevice);	
	
	float* Gaux = (float*) calloc(widthG * height, sizeof(float));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < 2; j++) {
			Gaux[j + i*2] = G[i][j];
		}
	}
	cudaMalloc( (void**)&dev_G, 2 * height * sizeof(float) );
	cudaMemcpy( dev_G, Gaux, 2 * height * sizeof(float), cudaMemcpyHostToDevice );
	
	float *a, *d_test1;
	cudaMalloc( (void**)&d_test1, 2 * height * sizeof(float));
	printf("height = %d, widthG = %d\n", height, widthG);
	//test<<<height, 2>>>(height, 2, d_test1, dev_G);
	//printf("Error test: %s\n\n", cudaGetErrorString(cudaGetLastError()));

	a = (float*) calloc(2*height, sizeof(float));
	
	//cudaMemcpy2D(a, widthG * sizeof(float), d_test1, pitchTest, widthG * sizeof(float), height, cudaMemcpyDeviceToHost);
	cudaMemcpy(a, d_test1, 2 * height * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Error cudaMemcpy : %s\n\n", cudaGetErrorString(cudaGetLastError()));
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < 2; j++) {
			printf("G[%d][%d] = %1.15f, a[%d][%d] = %1.15f\n", i, j, G[i][j], i, j, a[i*height + j]);
		}
	}
	exit(0);
	
	// copying filter bank data
	cudaMalloc( (void**)&dev_filterBankLength, height * sizeof(int) );
	cudaMemcpy( dev_filterBankLength, filterBankLength, height * sizeof(int), cudaMemcpyHostToDevice);
	cudaMallocPitch(&dev_filterBank, &pitchFilterBank, widthFilterBank * sizeof(float), height);
	cudaMemcpy2D(dev_filterBank, pitchFilterBank, filterBank, widthFilterBank * sizeof(float), widthFilterBank * sizeof(float), height, cudaMemcpyHostToDevice);
	
	// creating R
	cudaMalloc( (void**)&dev_RSize, height * sizeof(int) );
	cudaMalloc( (void**)&dev_R, widthR * height * sizeof(float) );
	//cudaMallocPitch(&dev_R, &pitchR, widthR * sizeof(float), height);
	
	float* d_test, *h_test;
	cudaMalloc((void**) &d_test, 5*sizeof(float));
	
	getR<<<height,512>>>(height, pitchG, dev_GSize, dev_G, widthG, dev_filterBankLength, dev_filterBank, dev_RSize, dev_R, widthR, d_test);
	printf("Error kernel: %s\n\n", cudaGetErrorString(cudaGetLastError()));
	
	h_test = (float*) malloc(5*sizeof(float));
	cudaMemcpy(h_test, d_test, 5*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 5; i++) {
		printf("h_test[%d] = %1.15f\n", i, h_test[i]);
	}
	
	int* r_sizes = (int*) calloc(height, sizeof(int));
	cudaMemcpy(r_sizes, dev_RSize, height * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Error cudaMemcpy r_sizes: %s\n\n", cudaGetErrorString(cudaGetLastError()));
	
	for (int i = 0; i < height; i++) {
		printf("r_sizes[%d] = %d\n", i, r_sizes[i]);
	}
	
	//cudaMemcpy2D(r, widthR * sizeof(float), dev_R, pitchR, widthR * sizeof(float), height, cudaMemcpyDeviceToHost);
	//printf("Error cudaMemcpy2D: %s\n\n", cudaGetErrorString(cudaGetLastError()));

	//cudaMemcpy2D(r, widthR * sizeof(float), dev_R, pitchR, widthR * sizeof(float), height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(r, dev_R, widthR * height * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy2D(test, widthR * sizeof(float), dev_test, pitchTest, widthR * sizeof(float), height, cudaMemcpyDeviceToHost);
	
	
	float *r;
	//cudaMalloc((void**) &d_test, widthR * height * sizeof(float));
	//test<<<height, 512>>>(height, widthR, d_test);
	//cudaMemcpy(t, d_test, widthR * height * sizeof(float), cudaMemcpyDeviceToHost);
	
	r = (float*) calloc(widthR * height, sizeof(float));
	cudaMemcpy(r, dev_R, widthR * height * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < widthR; j++) {
			printf("r[%d][%d] = %1.15f\n", i, j, r[j + widthR*i]);
			//printf("t[%d][%d] = %1.15f\n", i, j, t[i][j]);
		}
	}
	
	cudaFree(dev_RSize);
	cudaFree(dev_R);
	cudaFree(dev_filterBank);
	cudaFree(dev_filterBankLength);
	cudaFree(dev_GSize);
	cudaFree(dev_G);
	
	exit(0);
	
//	TESTE
//	float** G2 = (float**) malloc(height * sizeof(float*));
//	for (int i = 0; i < height; i++) {
//		G2[i] = (float*) malloc(widthG * sizeof(float));
//	}
//	cudaMemcpy2D(G2, widthG * height * sizeof(float), dev_G, pitchG, widthG, height, cudaMemcpyDeviceToHost);
//	for (int i = 0; i < 5; i++) {
//		for (int j = 0; j < widthG; j++) {
//			if (G[i][j] == G2[i][j]) {
//				printf("G[%d][%d] = %1.15f, G2[%d][%d] = %1.15f\n", i, j, G[i][j], i, j, G2[i][j]);
//			}
//		}
//	}	
//	printf("Error cudaMemcpy2D: %s\n\n", cudaGetErrorString(cudaGetLastError()));
//	exit(0);
	

/*	
	float** gaux = (float**) calloc((numFiltros+1), sizeof(float*));
	
	// TODO copiar G, G_size, filterBank, filterBankLength, atrasos para device
	// TODO calcular gaux, L, r, r_sizes no device
	
	for (int i = 0; i < numFiltros+1; i++) {
		int partialVecLength = (G_size[i]+atrasos[i]);
		int resultLength;
		gaux[i] = spars(getPartialVector(G[i], partialVecLength ), partialVecLength, L[i], &resultLength);
		
		// conv(filterBank[i], gaux[i])	
		int convLength = (filterBankLength[i] + resultLength - 1);
		printf("filterBankLength[%d] = %d, resultLength = %d, convLength = %d\n", i, filterBankLength[i], resultLength, convLength);
	
		//r[i] = convFFT(filterBank[i], filterBankLength[i], gaux[i], resultLength);
		r[i] = convSimple(filterBank[i], filterBankLength[i], gaux[i], resultLength);
		
		r_sizes[i] = convLength;	
	}
*/


/*
	int maxR = max(r_sizes, numFiltros+1);
	float* res = (float*) calloc(maxR, sizeof(float));
	float** raux = (float**) malloc((numFiltros + 1) * sizeof(float*));
	
	for (int i = 0; i < (numFiltros + 1); i++) {
		raux[i] = (float*) malloc(maxR * sizeof(float));
		for (int j = 0; j < maxR; j++) {
			printf("r[%d][%d] = %1.15f\n", i, j, r[i][j]);
			if (j < r_sizes[i]) {
				raux[i][j] = r[i][j];
			} else {
				raux[i][j] = 0.0;
			}
		}
	}
	
	float* dev_r;
	float* dev_aux_sum;
	
	INIT_RUNTIME;
	cudaMalloc( (void**)&dev_aux_sum, maxR * sizeof(float) );
	initializeArray<<<32,ceil(maxR/32.0)>>>(dev_aux_sum, maxR);
	
	for (int i = 0; i < (numFiltros + 1); i++) {
		cudaMalloc( (void**)&dev_r, r_sizes[i] * sizeof(float) );
		initializeArray<<<32,ceil(maxR/32.0)>>>(dev_r, r_sizes[i]);
		cudaMemcpy(dev_r, r[i], r_sizes[i] * sizeof(float), cudaMemcpyHostToDevice );
		sumColumns<<<32,ceil(maxR/32.0)>>>(dev_r, r_sizes[i], dev_aux_sum, maxR);
		cudaFree(dev_r);
	}
		
	cudaMemcpy(res, dev_aux_sum, maxR * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_aux_sum);	
	cudaFree(dev_r);
	END_RUNTIME; printf("\n[loop2]: "); PRINT_RUNTIME;
	
	int maxHlength = max(filterBankLength, numFiltros+1);
	float* resFinal = (float*) calloc((maxR - maxHlength), sizeof(float));
	
	for (int i = 0; i < (maxR-maxHlength); i++) {
		resFinal[i] = res[maxHlength + i];
	}
	
	*resultLength = (maxR-maxHlength);
*/
	float* resFinal;
	return resFinal;
}




// Memory management
/**
 *	Libera memória do Host.
 */
void cleanHostMemory(void* h) {
    // Free host memory
    if (h) {
        free(h);
	}
}

/**
 *	Libera memória do Device.
 */
void cleanDeviceMemory(void* d) {
    // Free device memory
    if (d) {
        cudaFree(d);
	}
	
    cudaThreadExit();	
}


// CUDA Error
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)  {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}










/*
 *	NÃO ESTÁ SENDO USADO
 *
 */
float* multiplyPolynomialCoefficients_OLD(float* vecA, const int vecASize, float* vecB, const int vecBSize) {
	int vecCSize = vecASize + vecBSize - 1;
	int size = vecCSize * sizeof(float);
	
	float* vecC = (float*) malloc(size);
	
	for (int i = 0; i < vecASize; i++) {
		for (int j = 0; j < vecBSize; j++) {
			vecC[i+j] = vecC[i+j] + (vecA[i] * vecB[j]);
		}
	}

	return vecC;
}

/**
 *	Operação de convolução executada no HOST.
 */
float* convHost(float* Gl_old, int filtroLength, float* sparsArray1, int resultLength) {
	float* convResult = (float*) calloc(resultLength, sizeof(float));
	for (int k = 0; k < resultLength; k++ ) {
		convResult[k] = 0;                 // set to zero before sum
		for (int j = 0; j < filtroLength; j++ ) {
			if ((k - j) >= 0) {
				convResult[k] += (Gl_old[j] * sparsArray1[k-j]);    // convolve: multiply and accumulate
			}
		}
	}
	return convResult;
}

float* convHost1(float* Gl_old, int filtroLength, float* sparsArray, int sparsLength) {	
	int resultLength = filtroLength + sparsLength - 1;	
	int maxLength = (filtroLength >= sparsLength ? filtroLength : sparsLength);
	int minLength = (filtroLength <= sparsLength ? filtroLength : sparsLength);
	
	float* convResult = (float*) calloc(resultLength, sizeof(float));
	for (int k = 0; k < resultLength; k++ ) {
		for (int j = 0; j < maxLength; j++ ) {
			if ((k - j) >= 0 && (k-j) < minLength) {
				if (maxLength == filtroLength) {
					convResult[k] += (Gl_old[j] * sparsArray[k-j]);    // convolve: multiply and accumulate
				} else {
					//printf("[%d] sparsArray1[%d] = %1.15f, Gl_old[%d] = %1.15f\n", k, j, sparsArray1[j], (k-j), Gl_old[k-j]);
					convResult[k] += (sparsArray[j] * Gl_old[k-j]);    // convolve: multiply and accumulate
				}				
			}
		}
	}
	return convResult;
}


/**
 *	Operação de cascateamento dos filtros usando CUDA e a operação de
 *	convolução em paralelo:
 *  __global__ void conv(float* u, int uLength, float* v, int vLength, float* w, int wLength);
 */
/*void cascata(char* filtros[], int numFiltros, float** filterBank, int* filterBankLength) {
	float *dev_filtro, *dev_spars_array0, *dev_spars_array1, *dev_result0, *dev_result1;
	float *convResult;

	int J = numFiltros;
	int filtroLength;
	double** h;
	int hLength;
	
	h = leFiltros(filtros[0], &hLength);
		
	float** filterBankAux = (float**) malloc((numFiltros + 1) * sizeof(float*));
	int* filterBankLengthAux = (int*) malloc((numFiltros + 1) * sizeof(int));
	filterBankAux[0] = (float*) malloc(hLength * sizeof(float));	// passa-altas
	filterBankLengthAux[0] = hLength;
	float* Gl = (float*) malloc(hLength * sizeof(float));	// passa-baixas
	
	for (int i = 0; i < hLength; i++) {
		filterBankAux[0][i] = h[1][i];
		Gl[i] = h[0][i];
	}

	float* Gl_old = NULL;
	filtroLength = hLength;
	
	cudaEvent_t	start, stop;
	float	elapsedTime;
	// start the timers
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop ); 
	cudaEventRecord( start, 0 );
	
	for (int i = 1; i < numFiltros; i++) {
		Gl_old = (float*) calloc(filtroLength, sizeof(float));
		for (int j = 0; j < filtroLength; j++) {
			Gl_old[j] = Gl[j];
		}
		
		int sparsArray0Length, sparsArray1Length;
		float* sparsArray0 = spars(h[0], hLength, pow(2.0, i), &sparsArray0Length);
		float* sparsArray1 = spars(h[1], hLength, pow(2.0, i), &sparsArray1Length);
		
		int filtroSize = filtroLength * sizeof(float);
		int sparsArray0Size = sparsArray0Length * sizeof(float);
		int sparsArray1Size = sparsArray1Length * sizeof(float);
		int resultLength = (filtroLength + sparsArray0Length - 1);
		int resultSize = resultLength * sizeof(float);
	
		cudaMalloc( (void**)&dev_filtro, filtroSize );
		cudaMalloc( (void**)&dev_spars_array0, sparsArray0Size );
		cudaMalloc( (void**)&dev_spars_array1, sparsArray1Size );
		cudaMalloc( (void**)&dev_result0, resultSize );
		cudaMalloc( (void**)&dev_result1, resultSize );
	
		cudaMemcpy(dev_filtro, Gl_old, filtroSize, cudaMemcpyHostToDevice );
		cudaMemcpy(dev_spars_array0, sparsArray0, sparsArray0Size, cudaMemcpyHostToDevice );
		cudaMemcpy(dev_spars_array1, sparsArray1, sparsArray1Size, cudaMemcpyHostToDevice );
		
		initializeArray<<<16,16>>>(dev_result0, resultLength);
		initializeArray<<<16,16>>>(dev_result1, resultLength);
		
		conv<<<16,16>>>(dev_filtro, filtroLength, dev_spars_array0, sparsArray0Length, dev_result0, resultLength);
		conv<<<16,16>>>(dev_filtro, filtroLength, dev_spars_array1, sparsArray1Length, dev_result1, resultLength);
		
		convResult = (float*) calloc(resultLength, sizeof(float));
		filterBankAux[i] = (float*) calloc(resultLength, sizeof(float));
		cudaMemcpy(convResult, dev_result0, resultSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(filterBankAux[i], dev_result1, resultSize, cudaMemcpyDeviceToHost);
		filterBankLengthAux[i] = resultLength;

		if ((i+1) != numFiltros) {
			free(Gl_old);	// Gl_old é usado fora do for após a última iteração
			free(Gl);
			
			filtroLength = resultLength;
			Gl = (float*) malloc(resultSize);
			for (int i = 0; i < resultLength; i++) {
				Gl[i] = convResult[i];
			}
			
			free(convResult);
		}
		
		cudaFree(dev_filtro);
		cudaFree(dev_spars_array0);
		cudaFree(dev_spars_array1);
		cudaFree(dev_result0);
		cudaFree(dev_result1);
	}
		
	int sparsArray0Length;
	float* sparsArray0 = spars(h[0], hLength, pow(2.0, J-1), &sparsArray0Length);
		
	int filtroSize = filtroLength * sizeof(float);
	int sparsArray0Size = sparsArray0Length * sizeof(float);
	int resultLength = (filtroLength + sparsArray0Length - 1);
	int resultSize = resultLength * sizeof(float);
	
	cudaMalloc( (void**)&dev_filtro, filtroSize );
	cudaMalloc( (void**)&dev_spars_array0, sparsArray0Size );
	cudaMalloc( (void**)&dev_result0, resultSize );
	
	cudaMemcpy(dev_filtro, Gl_old, filtroSize, cudaMemcpyHostToDevice );
	cudaMemcpy(dev_spars_array0, sparsArray0, sparsArray0Size, cudaMemcpyHostToDevice );
		
	initializeArray<<<16,16>>>(dev_result0, resultLength);
		
	conv<<<16,16>>>(dev_filtro, filtroLength, dev_spars_array0, sparsArray0Length, dev_result0, resultLength);
	
	filterBankAux[J] = (float*) calloc(resultLength, sizeof(float));
	cudaMemcpy(filterBankAux[J], dev_result0, resultSize, cudaMemcpyDeviceToHost);
	filterBankLengthAux[J] = resultLength;

	free(Gl_old);
	free(Gl);
		
	cudaFree(dev_filtro);
	cudaFree(dev_spars_array0);
	cudaFree(dev_result0);
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &elapsedTime, start, stop ); 
	//printf( "Time taken cascata: %3.1f ms\n", elapsedTime );
	
	int maxLength = max(filterBankLengthAux, (numFiltros + 1));
	
	for (int i = numFiltros; i >= 0; i--) {
		filterBank[i] = (float*) calloc(maxLength, sizeof(float));
		filterBankLength[i] = filterBankLengthAux[numFiltros-i];
		for (int j = 0; j < filterBankLengthAux[numFiltros - i]; j++) {
			filterBank[i][j] = filterBankAux[numFiltros - i][j];
		}
	}
}/*

/**
 * Essa função computa a multiplicação das matrizes Fp e Hp. Ip_aux = Fp*Hp.
 * Resultado: [2][2][7]
 *
 *	Fp =	a	b
 *			c	d
 *
 *	Hp =	A	B
 *			C	D
 *
 *
 *	NÃO ESTÁ SENDO USADO
 */
float*** computeIpAuxStream(float*** Hp, int hlength, float*** Fp, int flength) {
	float *host_A = NULL;
	float *host_B = NULL;
	float *host_C = NULL;
	float *host_D = NULL;
	float *host_a = NULL;
	float *host_b = NULL;
	float *host_c = NULL;
	float *host_d = NULL;
	float *dev_a0, *dev_b0, *dev_c0, *dev_d0, *dev_A0, *dev_B0, *dev_C0, *dev_D0; //GPU buffers for stream0 
	float *dev_aux1, *dev_aux2, *dev_aux_sum1, *dev_aux_sum2, *dev_aux_sum3, *dev_aux_sum4;
	
	cudaDeviceProp prop; 
	int whichDevice; 
	cudaGetDevice( &whichDevice ); 
	cudaGetDeviceProperties( &prop, whichDevice );
	
	if (!prop.deviceOverlap) { 
		//printf( "Device will not handle overlaps, so no speed up from streams\n" );
		return NULL;
	}
	
	cudaEvent_t	start, stop;
	float	elapsedTime;
	// start the timers
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop ); 
	cudaEventRecord( start, 0 );
	
	// initialize the streams
	cudaStream_t stream0, stream1; 
	cudaStreamCreate( &stream0 );
	cudaStreamCreate( &stream1 );
	
	int size = hlength * sizeof(float);
	int resultLength = (hlength + flength - 1);
	int resultSize = resultLength * sizeof(float);
	
	float*** Ip_aux = (float***) malloc(2 * sizeof(float**));
	Ip_aux[0] = (float**) malloc(2 * sizeof(float*));
	Ip_aux[1] = (float**) malloc(2 * sizeof(float*));
	
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			Ip_aux[i][j] = (float*) malloc(resultLength*sizeof(float*));
		}
	}
	
	cudaMalloc( (void**)&dev_a0, size );
	cudaMalloc( (void**)&dev_b0, size );
	cudaMalloc( (void**)&dev_c0, size );
	cudaMalloc( (void**)&dev_d0, size );
	cudaMalloc( (void**)&dev_A0, size );
	cudaMalloc( (void**)&dev_B0, size );
	cudaMalloc( (void**)&dev_C0, size );
	cudaMalloc( (void**)&dev_D0, size );
	
	cudaMalloc( (void**)&dev_aux1, resultSize );
	cudaMalloc( (void**)&dev_aux2, resultSize );
	cudaMalloc( (void**)&dev_aux_sum1, resultSize );
	cudaMalloc( (void**)&dev_aux_sum2, resultSize );
	cudaMalloc( (void**)&dev_aux_sum3, resultSize );
	cudaMalloc( (void**)&dev_aux_sum4, resultSize );
	
	/*
	Ip_aux = Fp*Hp;
	O resultado é uma matriz 2x2x7.
	
	aA+bC	aB+bD
	cA+dC	cB+dD
	
	*/
	
	// allocate page-locked memory, used to stream
	cudaHostAlloc( (void**)&host_A, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&host_B, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&host_C, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&host_D, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&host_a, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&host_b, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&host_c, size, cudaHostAllocDefault );
	cudaHostAlloc( (void**)&host_d, size, cudaHostAllocDefault );
		
	//host_aux_sum = (float*) malloc(resultSize);
		
	for (int i = 0; i < hlength; i++) {
		host_a[i] = Fp[0][0][i];
		host_b[i] = Fp[0][1][i];
		host_c[i] = Fp[1][0][i];
		host_d[i] = Fp[1][1][i];
		host_A[i] = Hp[0][0][i];
		host_B[i] = Hp[0][1][i];
		host_C[i] = Hp[1][0][i];
		host_D[i] = Hp[1][1][i];
	}
	
/* 00 */
	cudaMemcpyAsync(dev_a0, host_a, size, cudaMemcpyHostToDevice, stream0 );
	cudaMemcpyAsync(dev_b0, host_b, size, cudaMemcpyHostToDevice, stream1 );
	cudaMemcpyAsync(dev_A0, host_A, size, cudaMemcpyHostToDevice, stream0 );
	cudaMemcpyAsync(dev_C0, host_C, size, cudaMemcpyHostToDevice, stream1 );
	
	// clean auxiliary variables
	initializeArray<<<4,4,0,stream0>>>(dev_aux1, resultLength);
	initializeArray<<<4,4,0,stream1>>>(dev_aux2, resultLength);

	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<hlength, 1, 0, stream0>>>(dev_a0, hlength, dev_A0, hlength, dev_aux1);
    multiplyPolynomialCoefficientsKernel<<<hlength, 1, 0, stream1>>>(dev_b0, hlength, dev_C0, hlength, dev_aux2);
				
	sumPolynomialCoefficientsKernel<<<resultSize, 1, 0, stream0>>>(dev_aux1, dev_aux2, dev_aux_sum1, resultLength);
	
	
/* 01 */
	// clean auxiliary variables
	initializeArray<<<4,4,0,stream1>>>(dev_aux2, resultLength);
	initializeArray<<<4,4,0,stream0>>>(dev_aux1, resultLength);

	cudaMemcpyAsync(dev_B0, host_B, size, cudaMemcpyHostToDevice, stream0 );
	cudaMemcpyAsync(dev_D0, host_D, size, cudaMemcpyHostToDevice, stream1 );
	
	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<hlength, 1, 0, stream1>>>(dev_b0, hlength, dev_D0, hlength, dev_aux2);
    multiplyPolynomialCoefficientsKernel<<<hlength, 1, 0, stream0>>>(dev_a0, hlength, dev_B0, hlength, dev_aux1);
			
	sumPolynomialCoefficientsKernel<<<resultSize, 1, 0, stream1>>>(dev_aux1, dev_aux2, dev_aux_sum2, resultLength);
	

/* 10 */
	// clean auxiliary variables
	initializeArray<<<4,4,0,stream0>>>(dev_aux1, resultLength);
	initializeArray<<<4,4,0,stream1>>>(dev_aux2, resultLength);
	
	cudaMemcpyAsync(dev_c0, host_c, size, cudaMemcpyHostToDevice, stream0 );
	cudaMemcpyAsync(dev_d0, host_d, size, cudaMemcpyHostToDevice, stream1 );
	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<hlength, 1, 0, stream0>>>(dev_c0, hlength, dev_A0, hlength, dev_aux1);
    multiplyPolynomialCoefficientsKernel<<<hlength, 1, 0, stream1>>>(dev_d0, hlength, dev_C0, hlength, dev_aux2);
				
	sumPolynomialCoefficientsKernel<<<resultSize, 1, 0, stream0>>>(dev_aux1, dev_aux2, dev_aux_sum3, resultLength);
	
/* 11 */
	// clean auxiliary variables
	initializeArray<<<4,4,0,stream1>>>(dev_aux2, resultLength);
	initializeArray<<<4,4,0,stream0>>>(dev_aux1, resultLength);

	// Invoke kernel
    multiplyPolynomialCoefficientsKernel<<<hlength, 1, 0, stream1>>>(dev_d0, hlength, dev_D0, hlength, dev_aux2);
    multiplyPolynomialCoefficientsKernel<<<hlength, 1, 0, stream0>>>(dev_c0, hlength, dev_B0, hlength, dev_aux1);
			
	sumPolynomialCoefficientsKernel<<<resultSize, 1, 0, stream1>>>(dev_aux1, dev_aux2, dev_aux_sum4, resultLength);
	
	cudaStreamSynchronize( stream0 );
	cudaStreamSynchronize( stream1 );
	
	cudaMemcpy(Ip_aux[0][0], dev_aux_sum1, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(Ip_aux[0][1], dev_aux_sum2, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(Ip_aux[1][0], dev_aux_sum3, resultSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(Ip_aux[1][1], dev_aux_sum4, resultSize, cudaMemcpyDeviceToHost);
	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop ); 
	cudaEventElapsedTime( &elapsedTime, start, stop ); 
	//printf( "Time taken getIpAux: %3.1f ms\n", elapsedTime );
	
	cudaFreeHost(host_A);
	cudaFreeHost(host_B);
	cudaFreeHost(host_C);
	cudaFreeHost(host_D);
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFreeHost(host_d);
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c0);
	cudaFree(dev_d0);
	cudaFree(dev_A0);
	cudaFree(dev_B0);
	cudaFree(dev_C0);
	cudaFree(dev_D0);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	
	return Ip_aux;
}