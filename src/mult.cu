/* -*- mode: c -*- */
#include <stdio.h>

/* GPU で(並列に)実行される関数 */
__global__ void mult(float* out, float* a, float* b, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

int main(int argc, char* argv[])
{
    int n = 1000000;

    /* データを用意する。 */
    float* a = (float*)malloc(sizeof(float) * n);
    float* b = (float*)malloc(sizeof(float) * n);
    float* out = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        a[i] = b[i] = i;
    }

    /* GPU上にデータ領域を割り当てる。 */
    float* c_a;
    float* c_b;
    float* c_out;
    cudaMalloc(&c_a, sizeof(float) * n);
    cudaMalloc(&c_b, sizeof(float) * n);
    cudaMalloc(&c_out, sizeof(float) * n);

    /* CPU→GPU にデータを転送する。 */
    cudaMemcpy(c_a, a, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(c_b, b, sizeof(float) * n, cudaMemcpyHostToDevice);

    /* GPU 上で関数を実行する。 */
    mult<<<1, 1>>>(c_out, c_a, c_b, n);

    /* 計算結果を GPU→CPU に転送する。 */
    cudaMemcpy(out, c_out, sizeof(float) * n, cudaMemcpyDeviceToHost);

    /* 計算結果を表示する。 */
    printf("out[0]=%f\n", out[0]);
    printf("out[n-1]=%f\n", out[n-1]);

    /* 領域を開放。 */
    cudaFree(c_a);
    cudaFree(c_b);
    cudaFree(c_out);
    free(a);
    free(b);
    free(out);
    return 0;
}
