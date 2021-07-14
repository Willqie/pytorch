#include <vector>
#include <torch/extension.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"


torch::Tensor BMMExt_forward_biased(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor sizemap,
    torch::Tensor result,
    torch::Tensor bias,
    torch::Tensor bias_multiplier,
    int op_batch_num, int op_base_size) {

    static float** weight_arr = nullptr;
    static float** result_arr = nullptr;
    static float** input_arr = nullptr;
    static float** bias_arr = nullptr;
    static float** bias_multiplier_arr = nullptr;
    static float** weight_arr_cpu = nullptr;
    static float** result_arr_cpu = nullptr;
    static float** input_arr_cpu = nullptr;
    static float** bias_arr_cpu = nullptr;
    static float** bias_multiplier_cpu = nullptr;
    static int cur_batch_num = -1;
    static cublasStatus_t stat;
    static cublasHandle_t handle;

    if (cur_batch_num == -1) {
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "Cannot init cublas handle\n");
            exit(1);
        }
    }
    
    if (weight_arr == nullptr || cur_batch_num != op_batch_num) {
        if (cudaMalloc(&weight_arr, sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        if (cudaMalloc(&result_arr, sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        if (cudaMalloc(&input_arr, sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        if (cudaMalloc(&bias_arr, sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        if (cudaMalloc(&bias_multiplier_arr, sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        weight_arr_cpu = (float**)malloc(sizeof(float*)*op_batch_num);
        result_arr_cpu = (float**)malloc(sizeof(float*)*op_batch_num);
        input_arr_cpu = (float**)malloc(sizeof(float*)*op_batch_num);
        bias_arr_cpu = (float**)malloc(sizeof(float*)*op_batch_num);
        bias_multiplier_cpu = (float**)malloc(sizeof(float*)*op_batch_num);
        cur_batch_num = op_batch_num;
    }
    
    float* weight_ptr = (float*)weights.data_ptr();
    float* result_ptr = (float*)result.data_ptr();
    float* input_ptr = (float*)input.data_ptr();
    float* bias_ptr = (float*)bias.data_ptr();
    float* bias_multiplier_ptr = (float*)bias_multiplier.data_ptr();
    auto weight_shape = weights.sizes();
    int num_features = weight_shape[2];
    int num_in = weight_shape[1];
    float* sizemap_ptr = (float*)sizemap.data_ptr();
    int pos = 0;
    for (int i = 0; i < (int)sizemap.sizes()[0]; i++) {
        for (int j = 0; j < ((int)(sizemap_ptr[i])) / op_base_size; j++) {
            input_arr_cpu[pos] = input_ptr + pos * op_base_size * num_in;
            weight_arr_cpu[pos] = weight_ptr + i * num_in * num_features;
            bias_arr_cpu[pos] = bias_ptr + i * num_features;
            result_arr_cpu[pos] = result_ptr + pos * op_base_size * num_features;
            bias_multiplier_cpu[pos] = bias_multiplier_ptr;
            pos++;
        }
    }
    if (cudaMemcpy(weight_arr, weight_arr_cpu, sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    if (cudaMemcpy(result_arr, result_arr_cpu, sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    if (cudaMemcpy(input_arr, input_arr_cpu, sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    if (cudaMemcpy(bias_arr, bias_arr_cpu, sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    if (cudaMemcpy(bias_multiplier_arr, bias_multiplier_cpu, sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    float alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    stat = cublasSgemmBatched(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    num_features, op_base_size, num_in,
    &alpha,
    weight_arr, num_features,
    input_arr, num_in,
    &beta, result_arr, num_features, op_batch_num);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Cannot perform compute \n");
        exit(1);
    }

    beta = 1.0;
    stat = cublasSgemmBatched(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    num_features, op_base_size, 1,
    &alpha,
    bias_arr, num_features,
    bias_multiplier_arr, 1,
    &beta, result_arr, num_features, op_batch_num);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Cannot perform compute \n");
        exit(1);
    }


    return result;
}

torch::Tensor BMMExt_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor sizemap,
    torch::Tensor result,int op_batch_num, int op_base_size) {

    static float** weight_arr = nullptr;
    static float** result_arr = nullptr;
    static float** input_arr = nullptr;
    static float** weight_arr_cpu = nullptr;
    static float** result_arr_cpu = nullptr;
    static float** input_arr_cpu = nullptr;
    static int cur_batch_num = -1;
    static cublasStatus_t stat;
    static cublasHandle_t handle;

    if (cur_batch_num == -1) {
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "Cannot init cublas handle\n");
            exit(1);
        }
    }
    
    if (weight_arr == nullptr || cur_batch_num != op_batch_num) {
        if (cudaMalloc(&weight_arr, sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        if (cudaMalloc(&result_arr, sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        if (cudaMalloc(&input_arr, sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        weight_arr_cpu = (float**)malloc(sizeof(float*)*op_batch_num);
        result_arr_cpu = (float**)malloc(sizeof(float*)*op_batch_num);
        input_arr_cpu = (float**)malloc(sizeof(float*)*op_batch_num);
        cur_batch_num = op_batch_num;
    }
    
    float* weight_ptr = (float*)weights.data_ptr();
    float* result_ptr = (float*)result.data_ptr();
    float* input_ptr = (float*)input.data_ptr();
    auto weight_shape = weights.sizes();
    int num_features = weight_shape[2];
    int num_in = weight_shape[1];
    float* sizemap_ptr = (float*)sizemap.data_ptr();
    int pos = 0;
    for (int i = 0; i < (int)sizemap.sizes()[0]; i++) {
        for (int j = 0; j < ((int)(sizemap_ptr[i])) / op_base_size; j++) {
            input_arr_cpu[pos] = input_ptr + pos * op_base_size * num_in;
            weight_arr_cpu[pos] = weight_ptr + i * num_in * num_features;
            result_arr_cpu[pos] = result_ptr + pos * op_base_size * num_features;
            pos++;
        }
    }
    if (cudaMemcpy(weight_arr, weight_arr_cpu, sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    if (cudaMemcpy(result_arr, result_arr_cpu, sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    if (cudaMemcpy(input_arr, input_arr_cpu, sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    float alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    stat = cublasSgemmBatched(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    num_features, op_base_size, num_in,
    &alpha,
    weight_arr, num_features,
    input_arr, num_in,
    &beta, result_arr, num_features, op_batch_num);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Cannot perform compute \n");
        exit(1);
    }
    return result;
}

torch::Tensor BMMExt_forward_optimized(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor sizemap,
    torch::Tensor result,int op_batch_num, int op_base_size) {

    static float** weight_arr = nullptr;
    static float** result_arr = nullptr;
    static float** input_arr = nullptr;
    static float** weight_arr_cpu = nullptr;
    static float** result_arr_cpu = nullptr;
    static float** input_arr_cpu = nullptr;
    static int cur_batch_num = -1;
    static cublasStatus_t stat;
    static cublasHandle_t handle;

    if (cur_batch_num == -1) {
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "Cannot init cublas handle\n");
            exit(1);
        }
    }
    
    if (weight_arr == nullptr || cur_batch_num != op_batch_num) {
        float** tmp_ptr = nullptr;
        if (cudaMalloc(&tmp_ptr, 3*sizeof(float*)*op_batch_num) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc fail\n");
            exit(1);
        }
        weight_arr = tmp_ptr + 0 * op_batch_num;
        result_arr = tmp_ptr + 1 * op_batch_num;
        input_arr  = tmp_ptr + 2 * op_batch_num;
        weight_arr_cpu = (float**)malloc(3*sizeof(float*)*op_batch_num);
        result_arr_cpu = weight_arr_cpu + 1 * op_batch_num;
        input_arr_cpu = weight_arr_cpu + 2 * op_batch_num;
        cur_batch_num = op_batch_num;
    }
    
    float* weight_ptr = (float*)weights.data_ptr();
    float* result_ptr = (float*)result.data_ptr();
    float* input_ptr = (float*)input.data_ptr();
    auto weight_shape = weights.sizes();
    int num_features = weight_shape[2];
    int num_in = weight_shape[1];
    float* sizemap_ptr = (float*)sizemap.data_ptr();
    int pos = 0;
    for (int i = 0; i < (int)sizemap.sizes()[0]; i++) {
        for (int j = 0; j < ((int)(sizemap_ptr[i])) / op_base_size; j++) {
            input_arr_cpu[pos] = input_ptr + pos * op_base_size * num_in;
            weight_arr_cpu[pos] = weight_ptr + i * num_in * num_features;
            result_arr_cpu[pos] = result_ptr + pos * op_base_size * num_features;
            pos++;
        }
    }
    if (cudaMemcpy(weight_arr, weight_arr_cpu, 3*sizeof(float*)*op_batch_num, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Copy failed\n");
        exit(1);
    }
    float alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    stat = cublasSgemmBatched(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    num_features, op_base_size, num_in,
    &alpha,
    weight_arr, num_features,
    input_arr, num_in,
    &beta, result_arr, num_features, op_batch_num);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Cannot perform compute \n");
        exit(1);
    }
    return result;
}

PYBIND11_MODULE(BMMExt, m) {
  m.def("op", &BMMExt_forward, "BMMExt forward");
  m.def("op_opt", &BMMExt_forward_optimized, "BMMExt forward optimized");
  m.def("op_biased", &BMMExt_forward_biased, "BMMExt forward biased");
}