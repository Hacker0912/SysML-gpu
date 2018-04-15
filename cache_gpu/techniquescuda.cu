//
//  techniques.cpp
//  Coordinate_descent
//
//  Created by Huawei on 04/07/18.
//  Copyright © 2015 Zhiwei Fan. All rights reserved.
//

#include <cuda.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <stdlib.h>
#include <string.h>
#include "techniques.h"
#include "DataManagement.h"
#include "gradientkl.cu"
#include "linear_models.h"

techniques::techniques(){};

/**
 Coordinate Descent/Block Coordinate Descent:
 (CD/BCD)
 Materialize, Stream, Factorize
 
 Stochastic Gradient Descent/Batch Gradient Descent:
 (SGD/BGD)
 Materialize only
*/

#pragma mark - Stochastic Coordiante Descent
/**
 Stochastic Coordinate Descent - Materialize
 
 @param table_T The output table results from the join of the "entity" table S and the "attribute" table R
 @param _setting
 @param model
 @param avail_mem The available memory measured by "sizeof(double)"
 */

//  Actually the choice of computation on CPU or GPU should take into memory space on GPU into consideration
//  Currently just followed the calculation on CPU, if not avaiable on CPU memory, definitly can't continue to compute on GPU
void techniques::materialize(string table_T, setting _setting, double *&model, double avail_mem, const char *lm)
{
    // Object for reading data into memory
    DataManagement DM;
    DM.message("Start materialize");
    
    // Get the table information and column names
    vector<long> tableInfo(3);
    vector<string> fields = DM.getFieldNames(table_T, tableInfo);
    int feature_num = (int)tableInfo[1];
    long row_num = (long)tableInfo[2];
    
    // For cache, avail_mem_total in Bytes, avail_mem in GB
    double avail_mem_total = 1024*1024*1024*avail_mem;
    int avail_col = 0;
    int avail_cache = 0;
    double *cache;

    // Primary main memory space: three columns
    // Whenever time, must have space for label Y, 1 feature column X, residual H
    // Label array
    double *Y;
    // Residual vector
    double *H;
    // Buffer for 1 column reading
    double *X;
    
    // Setting
    double step_size = _setting.step_size;

    // Calculate the available memory measured by size of a single column, total available column
    avail_col = avail_mem_total/(sizeof(double)*row_num);
    
    // Calculate the available remaining space measured by size of a single column for cache
    avail_cache = avail_col - 3;
    if(avail_cache < 0)
    {
        DM.errorMessage("Insufficient memory space for training");
        exit(1);
    }
    else if (avail_cache == 0)
    {
        DM.message("No space for caching");
    }
    else
    {
        // can cache all feature column
        if( avail_cache >= feature_num - 1 )
        {
            // cache = new double*[feature_num];
            // for(int i = 0; i < feature_num; i ++)
            // {
            //     cache[i] = new double[row_num];
            // }
            cache = (double*)malloc(sizeof(double)*feature_num*row_num);
            // No need to reserve the X buffer to read single column, all in cache
            avail_cache = feature_num;
        }
        else
        {
            cache = (double*)malloc(sizeof(double)*avail_cache*row_num);            
        }
    }
    
    // Dynamic memory allocation
    if(avail_cache < feature_num)
    {
        // Allocate the memory to X
        X = new double[row_num];
    }
    Y = new double[row_num];
    H = new double[row_num];
    model = new double[feature_num];

    // Dynamic allocation for device variables
    // H, Y definitly need to be allocated & cached
    double* dH;
    double* dY;
    double* dX;
    double* cuda_cache;
    double* dmul;
    double* mul;
    mul = new double[row_num];
    size_t pitch;
    
    if(avail_cache < feature_num) {
        if(cudaSuccess != cudaMalloc((void**)&dX, row_num*sizeof(double))) {
            DM.message("No space on device for single column feature");
            exit(1);
        }
    }
    if(cudaSuccess != cudaMalloc((void**)&dY, row_num * sizeof(double))) {
        DM.message("No space on device for class labels");
        exit(1);
    }
    if(cudaSuccess != cudaMalloc((void**)&dH, row_num * sizeof(double))) {
        DM.message("No space on device for remain variables");
        exit(1);
    }
    if(cudaSuccess != cudaMalloc((void**)&dmul, row_num * sizeof(double))) {
        DM.message("No space on device for Intermediate variables");
        exit(1);
    }
	// Actually need to consider the extra variable malloc for reducing
    // May need further calculation for cache in the future
    printf("avaiable cache number is %d\n", avail_cache);    
    if(avail_cache > 0) {
        if(cudaSuccess != cudaMallocPitch((void**)&cuda_cache, &pitch, row_num * sizeof(double), avail_cache)) {
            DM.message("No space on device for variable cache");
        } else {
            DM.message("Allocate GPU-Cache successfully on GPU");
            printf("Malloc width after padding is %zu bytes\n", pitch);
        }
    }

    // Initialization of variables for loss and gradient
    // Transfer data between GPU & CPU, dY label & dH always remains in GPU memory
    double F = 0.00;
    double F_partial = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int iters = 0;
    memset(model, 0.00, sizeof(double)*feature_num);
    
    DM.fetchColumn(fields[1], row_num, Y);
    cudaMemcpy(dY, Y, row_num*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dH, 0, row_num*sizeof(double));

    // Kernal call parameters
    const int threadsPerBlock = 512;
    const int blocksPerGrid = row_num/threadsPerBlock + 1;

    // Shuffling process
    vector<int> original_index_set;
    vector<int> shuffling_index;
    for(long i = 0; i < feature_num; i ++)
    {
        original_index_set.push_back(i);
    }
    shuffling_index = shuffle(original_index_set, (unsigned)time(NULL));
	shuffling_index = original_index_set;    

    // Caching & copy data for training
    printf("\n");
    printf("Avail_col: %d\n", avail_cache);
    for(int i = 0; i < avail_cache; i ++)
    {
        DM.fetchColumn(fields[i+2], row_num, cache + i*row_num);
    }
    // printf("Test cache data is %lf\n", *(cache + (avail_cache - 1)*row_num));
    if(cudaSuccess != cudaMemcpy2D(cuda_cache, pitch, cache, row_num*sizeof(double), row_num*sizeof(double), avail_cache, cudaMemcpyHostToDevice)) {
        DM.message("No enough space on GPU memory for caching feature");
    } else {
        DM.message("GPU can cache all data into main memory");
    }
    // Training process, maybe too much judge logic to improve the performance
    do {
        // Update one coordinate each time
        for(int j = 0; j < feature_num; j ++)
        {
            int cur_index = shuffling_index.at(j);
            F_partial = 0.00;
            
            // If the column corresponding to the current updating coordinate is in the cache, no extra I/O is needed
            if(cur_index < avail_cache)
            {
                G_lrcache<<<blocksPerGrid, threadsPerBlock>>>(dY, dH, cuda_cache, dmul, cur_index, row_num, pitch);
                cudaDeviceSynchronize();
                cudaMemcpy(mul, dmul, sizeof(double)*row_num, cudaMemcpyDeviceToHost);
                for(long k = 0; k < row_num; k++)
                    F_partial += mul[k];
                // printf("F_partial value is %lf\n", F_partial);
            }
            else
            {
                // Fetch the column and store the current column into X, not in cache
                DM.fetchColumn(fields[cur_index+2], row_num, X);
                cudaMemcpy(dX, X, sizeof(double) * row_num, cudaMemcpyHostToDevice);
                // Compute the partial gradient
                G_lrkl<<<blocksPerGrid, threadsPerBlock>>>(dY, dH, dX, dmul, row_num);
                cudaDeviceSynchronize();
                cudaMemcpy(mul, dmul, sizeof(double)*row_num, cudaMemcpyDeviceToHost);
                for(long k = 0; k < row_num; k++)
                    F_partial += mul[k];
            }
            // Store the old W(j)
            double W_j = model[cur_index];
            
            // Update the current coordinate
            model[cur_index] = model[cur_index] - step_size * F_partial;
            std::cout << "model[" << cur_index << "]: " << model[cur_index] << std::endl;
            double diff = model[cur_index] - W_j;
    
            // Update the intermediate variable
            // H = H + (Wj - old_Wj)* X(,j)
            if( cur_index < avail_cache)
            {
                H_cache<<<blocksPerGrid, threadsPerBlock>>>(dH, cuda_cache, diff, cur_index, pitch, row_num);
                cudaDeviceSynchronize();                
            }
            else
            {
                Hkl<<<blocksPerGrid, threadsPerBlock>>>(dH, dX, diff, row_num);
                cudaDeviceSynchronize();                
            }
        }
        cudaMemcpy(H, dH, sizeof(double)*row_num, cudaMemcpyDeviceToHost);
        r_prev = F;
        // Caculate F
        F = 0.00;
        G_lrloss<<<blocksPerGrid, threadsPerBlock>>>(dY, dH, dmul, row_num);
        cudaDeviceSynchronize();                
        cudaMemcpy(mul, dmul, sizeof(double)*row_num, cudaMemcpyDeviceToHost);
        for(long k = 0; k < row_num; k++)
        {
            F += mul[k];
        }
        // for(long i = 0; i < row_num ; i ++)
        // {
        //     double tmp = lossCompute(Y[i],H[i],lm);
        //     F += tmp;
        // }
        cout<<"loss: " <<F<<endl;
		cout<< F << endl;
        r_curr = F;
        iters ++;
    }
    while(!stop(iters , r_prev, r_curr, _setting));
    
    delete [] Y;
    delete [] H;
    cudaFree(dY);
    cudaFree(dH);
    cudaFree(dmul);
    if( avail_cache < feature_num ){
        delete [] X;
        cudaFree(dX);
    }
    
    // Clear the cache
    if( avail_cache > 0) {
        // for(int i = 0; i < avail_cache; i ++)
        // {
        //     delete [] cache[i];
        // }
        delete [] cache;
        cudaFree(cuda_cache);
    } 
    printf("\n");
    outputResults(r_curr, feature_num, iters, model); 
    DM.message("Finish materialize");

}


/* oid-oid mapping is Not stored in memory */
void techniques::stream(string table_S, string table_R, setting _setting, double *&model, double avail_mem, const char *lm)
{
    DataManagement DM;
    DM.message("Start stream");
    
    // Set Timer
    clock_t c_start;
    clock_t c_end;
    
    c_start = clock();
    // Get the table information and column names
    vector<long> tableInfo_S(3);
    vector<long> tableInfo_R(3);
    vector<string> fields_S = DM.getFieldNames(table_S, tableInfo_S);
    vector<string> fields_R = DM.getFieldNames(table_R, tableInfo_R);
    int feature_num_S = (int)tableInfo_S[1];
    int feature_num_R = (int)tableInfo_R[1];
    int feature_num = feature_num_S + feature_num_R;
    long row_num_S = tableInfo_S[2];
    long row_num_R = tableInfo_R[2];
    
    // For Cache
    long avail_mem_total = 1024*1024*1024*avail_mem;
    long avail_cache;
    int avail_col_S = 0;
    int avail_col_R = 0;
    double **cache_R;
    double **cache_S;
    
    // Label array
    double *Y;
    // Residual vector
    double *H;
    // Buffer for column reading in S
    double *X_S;
    // Buffer for column reading in R
    double *X_R;
    // OID-OID Mapping (Key Foreign-Key Mapping Reference)
    double *KKMR;
    
    // Setting
    double step_size = _setting.step_size;

    // Calculate the available memory measured by size of each column in R and S
    avail_cache = avail_mem_total - sizeof(double)*(4*row_num_S + row_num_R);

    if(avail_cache < 0)
    {
    	DM.errorMessage("Insufficient memory space");
	    exit(1);
    }
    else if(avail_cache == 0)
    {
    	DM.message("No space for caching");
    }
    else
    {
    	// First consider caching columns in S
        avail_col_S = avail_cache/(sizeof(double)*row_num_S);
        if(avail_col_S == 0)
        {
            DM.message("No space for caching S");
            // Then consider caching columns in R
            avail_col_R = avail_cache/(sizeof(double)*row_num_R);
            if(avail_col_R == 0)
            {
                DM.message("No space for caching R");
            }
            else
            {
                if(avail_col_R >= feature_num_R - 1)
                {
                    cache_R = new double*[feature_num_R];
                    for(int i = 0; i < feature_num_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                    // No need to reserve the X_R buffer to read a single column in R
                    avail_col_R = feature_num_R;
                }
                else
                {
                    cache_R = new double*[avail_col_R];
                    for(int i = 0; i < avail_col_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                }
            }
        }
        else
        {
            if(avail_col_S >= feature_num_S)
            {
                cache_S = new double*[feature_num_S];
                for(int i = 0; i < feature_num_S; i ++)
                {
                    cache_S[i] = new double[row_num_S];
                }
                // X_S is still needed to "reconstruct" the complete column from single column fetched from R
                avail_col_S = feature_num_S;
            }
            else
            {
                cache_S = new double*[avail_col_S];
                for(int i = 0; i < avail_col_S; i ++)
                {
                    cache_S[i] = new double[row_num_S];
                }
            }
		
            // Then consider the caching for R using the remaining caching space
            avail_cache = avail_cache - avail_col_S*sizeof(double)*row_num_S;
            avail_col_R = avail_cache/(sizeof(double)*row_num_R);
            if(avail_col_R == 0)
            {
                DM.message("No space for caching R");
            }
            else
            {
                if(avail_col_R >= feature_num_R - 1)
                {
                    cache_R = new double*[feature_num_R];
                    for(int i = 0; i < feature_num_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                    // No need to reserve the X_R buffer to read a single column in R
                    avail_col_R = feature_num_R;
                }
                else
                {
                    cache_R = new double*[avail_col_R];
                    for(int i = 0; i < avail_col_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                }
            }
        }
        
    }
    
    // Dynamic memory allocation
    if(avail_col_R < feature_num_R)
    {
         X_R = new double[row_num_R];
    }
    X_S = new double[row_num_S];
    Y = new double[row_num_S];
    H = new double[row_num_S];
    KKMR = new double[row_num_S];
    model = new double[feature_num];
   
    // Initialization of variables for loss and gradient
    double F = 0.00;
    double F_partial = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int iters = 0;
    
    // Initialization
    memset(model, 0.00, sizeof(float)*feature_num);
    memset(H, 0.00, sizeof(float)*row_num_S);
    DM.fetchColumn(fields_S[1], row_num_S, Y);
    
    // Shuffling process
    vector<int> original_index_set;
    vector<int> shuffling_index;
    // Initialize the original_index_set
    for(int i = 0; i < feature_num; i ++)
    {
        original_index_set.push_back(i);
    }
    shuffling_index = shuffle(original_index_set, (unsigned)time(NULL));
    
    // Caching S
    printf("\n");
    printf("Avail_col_S: %d\n", avail_col_S);
    for(int i = 0; i < avail_col_S; i ++)
    {
        //printf("Cache %d th column in S\n", i);
        DM.fetchColumn(fields_S[3+i], row_num_S, cache_S[i]);
    }
   
    // Caching R
    printf("\n");
    printf("Avail_col_R: %d\n", avail_col_R);
    for(int k = 0; k < avail_col_R; k ++)
    {
        //printf("Cache %d th column in R\n", k);
        DM.fetchColumn(fields_R[1+k],row_num_R, cache_R[k]);
    }
    
    c_end = clock();
    cout<<"Caching:"<<1000*(c_end-c_start)/CLOCKS_PER_SEC<<"ms\n";

    do
    {
        c_start = clock();
        printf("\n");
        DM.message("Start fetching KKMR reference");
        // Read the fk column(referred rid in R) in table S, rid column in R
        ifstream fk;
        // Load the fk to KKMR
        fk.open(fields_S[2], ios::in | ios::binary);
        // rid.open(table2_fields[0], ios::in | ios::binary);
        if(!fk.is_open())
        {
            DM.errorMessage("Error Message: Cannot load the fk column.");
            exit(1);
        }
        fk.read((char *)KKMR, row_num_S*(sizeof(double)));
        fk.close();
        DM.message("Finish fetchig KKMR reference");
       
        // Update one coordinate each time
        for(int j = 0; j < feature_num; j ++)
        {
            int cur_index = shuffling_index.at(j);
            //printf("Current feature index: %d\n", cur_index);
            
            F_partial = 0.00;
            
            if(cur_index < feature_num_S)
            {
                // Check cache for S
                if(cur_index < avail_col_S)
                {
                    // Compute the partial gradient
                    for(long i = 0; i < row_num_S; i ++)
                    {
                        F_partial += gradientCompute(Y[i],H[i],lm)*cache_S[cur_index][i];
                    }
                }
                else
                {
                    // Fetch the corresponding column in S and store in X_S
                    DM.fetchColumn(fields_S[3+cur_index], row_num_S, X_S);
                    // Compute the partial gradient
                    for(long i = 0; i < row_num_S; i ++)
                    {
                        F_partial += gradientCompute(Y[i],H[i], lm)*X_S[i];
                    }
                }
            }
            else
            {
                // Check cache for R
                int col_index_R = cur_index - feature_num_S;
                //printf("col_index_R: %d\n", col_index_R);
                if(col_index_R < avail_col_R)
                {
                    for(long m = 0; m < row_num_S; m ++)
                    {
                        long fk = KKMR[m];
                        X_S[m]= cache_R[col_index_R][fk-1];
                    }
                }
                else
                {
                    DM.fetchColumn(fields_R[1+col_index_R], row_num_R, X_R);
                    for(long m = 0; m < row_num_S; m ++)
                    {
                        long fk = KKMR[m];
                        X_S[m]= X_R[fk-1];
                    }
                }
                
                // Compute the partial gradient
                for(long i = 0; i < row_num_S; i ++)
                {
                    F_partial += gradientCompute(Y[i],H[i],lm)*X_S[i];
                }
            }
            // Store the old W(j)
            double W_j = model[cur_index];
            
            // Update the current coordinate
            model[cur_index] = model[cur_index] - step_size * F_partial;
            
            double diff = model[cur_index] - W_j;

            //Update the intermediate variable
            //H = H + (Wj - old_Wj)* X(,j)
            if( cur_index < avail_col_S)
            {
                for(long m = 0; m < row_num_S; m ++ )
                {
                    H[m] = H[m] + diff*cache_S[cur_index][m];
                }
            }
            else
            {
                for(long m = 0; m < row_num_S; m ++ )
                {
                    H[m] = H[m] + diff*X_S[m];
                }
            }
        }
        
        r_prev = F;
        // Caculate F
        F = 0.00;
        for(long i = 0; i < row_num_S; i ++)
        {
            double tmp = lossCompute(Y[i],H[i],lm);
            F += tmp;
        }
        
        r_curr = F;
        iters ++;
        c_end = clock();
        cout<<"Iteration:"<<1000*(c_end-c_start)/CLOCKS_PER_SEC<<"ms\n";
    }
    while(!stop(iters , r_prev, r_curr, _setting));
    
    delete [] Y;
    delete [] H;
    delete [] X_S;
    delete [] KKMR;
    
    if(avail_col_R < feature_num_R)
    {
        delete [] X_R;
    }
    
    // Clear the cache
    if(avail_col_S > 0)
    {
        for(int i  = 0; i < avail_col_S; i ++)
        {
            delete [] cache_S[i];
        }
        delete [] cache_S;
    }
    if(avail_col_R > 0)
    {
        for(int i  = 0; i < avail_col_R; i ++)
        {
            delete [] cache_R[i];
        }
        delete [] cache_R;
    }
    
    printf("\n");
    outputResults(r_curr, feature_num, iters, model);
    
    DM.message("Finish stream");
}


void techniques::factorize(string table_S, string table_R, setting _setting, double *&model, double avail_mem, const char *lm)
{
    DataManagement DM;
    DM.message("Start factorize");
    
    // Set Timer
    clock_t c_start;
    clock_t c_end;
    
    c_start = clock();
    // Get the table information and column names
    vector<long> tableInfo_S(3);
    vector<long> tableInfo_R(3);
    vector<string> fields_S = DM.getFieldNames(table_S, tableInfo_S);
    vector<string> fields_R = DM.getFieldNames(table_R, tableInfo_R);
    int feature_num_S = (int)tableInfo_S[1];
    int feature_num_R = (int)tableInfo_R[1];
    int feature_num = feature_num_S + feature_num_R;
    long row_num_S = tableInfo_S[2];
    long row_num_R = tableInfo_R[2];
    
    // For Cache
    long avail_mem_total = 1024*1024*1024*avail_mem;
    long avail_cache;
    int avail_col_S = 0;
    int avail_col_R = 0;
    double **cache_R;
    double **cache_S;
    
    // Label array
    double *Y;
    // Residual vector
    double *H;
    // Buffer for column reading in S
    double *X_S;
    // Buffer for column reading in R
    double *X_R;
    // Buffer to store factorized factor when considering column R
    double *X_R_f;
    // OID-OID Mapping (Key Foreign-Key Mapping Reference, to be kept in memory)
    double *KKMR;
    
    // Setting
    double step_size = _setting.step_size;
    
    // Calculate the available memory measured by size of each column in R and S
    avail_cache = avail_mem_total - sizeof(double)*(4*row_num_S + 2*row_num_R);
    
    if(avail_cache < 0)
    {
        DM.errorMessage("Insufficient memory space");
        exit(1);
    }
    else if(avail_cache == 0)
    {
        DM.message("No space for caching");
    }
    else
    {
        // First consider caching columns in S
        avail_col_S = avail_cache/(sizeof(double)*row_num_S);
        if(avail_col_S == 0)
        {
            DM.message("No space for caching S");
            X_S = new double[row_num_S];
            // Then consider caching columns in R
            avail_col_R = avail_cache/(sizeof(double)*row_num_R);
            if(avail_col_R == 0)
            {
                DM.message("No space for caching R");
            }
            else
            {
                if(avail_col_R >= feature_num_R - 1)
                {
                    cache_R = new double*[feature_num_R];
                    for(int i = 0; i < feature_num_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                    // No need to reserve the X_R buffer to read a single column in R
                    avail_col_R = feature_num_R;
                }
                else
                {
                    cache_R = new double*[avail_col_R];
                    for(int i = 0; i < avail_col_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                }
            }
        }
        else
        {
            if(avail_col_S >= (feature_num_S - 1))
            {
                cache_S = new double*[feature_num_S];
                for(int i = 0; i < feature_num_S; i ++)
                {
                    cache_S[i] = new double[row_num_S];
                }
                //No need to reserve X_S for single column reading
                avail_col_S = feature_num_S;
            }
            else
            {
                cache_S = new double*[avail_col_S];
                for(int i = 0; i < avail_col_S; i ++)
                {
                    cache_S[i] = new double[row_num_S];
                }
            }
            
            //Then consider the caching for R using the remaining caching space
            if(avail_col_S == feature_num_S)
            {
                avail_cache = avail_cache - (avail_col_S - 1)*sizeof(double)*row_num_S;
            }
            else
            {
                avail_cache = avail_cache - avail_col_S*sizeof(double)*row_num_S;
            }
            avail_col_R = avail_cache/(sizeof(double)*row_num_R);
            if(avail_col_R == 0)
            {
                DM.message("No space for caching R");
            }
            else
            {
                if(avail_col_R >= feature_num_R - 1)
                {
                    cache_R = new double*[feature_num_R];
                    for(int i = 0; i < feature_num_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                    // No need to reserve the X_R buffer to read a single column in R
                    avail_col_R = feature_num_R;
                }
                else
                {
                    cache_R = new double*[avail_col_R];
                    for(int i = 0; i < avail_col_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                }
            }
        }
        
    }
    
    
    //Dynamic memory alloaction
    if(avail_col_S < feature_num_S)
    {
        X_S = new double[row_num_S];
    }
    if(avail_col_R < feature_num_R)
    {
        X_R = new double[row_num_R];
    }
    model = new double[feature_num];
    Y = new double[row_num_S];
    H = new double[row_num_S];
    X_R_f = new double[row_num_R];
    KKMR = new double[row_num_S];
    
    // Initialization of variables for loss and gradient
    double F = 0.00;
    double F_partial = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int iters = 0;
    
    // Initialization
    memset(model, 0.00, sizeof(float)*feature_num);
    memset(H, 0.00, sizeof(float)*row_num_S);
    
    DM.fetchColumn(fields_S[1], row_num_S, Y);
    printf("\n");
    DM.message("Start fetching KKMR reference");
    // Read the fk column(referred rid in R) in table S, rid column in R
    ifstream fk;
    // Load the fk to KKMR
    fk.open(fields_S[2], ios::in | ios::binary);
    // rid.open(table2_fields[0], ios::in | ios::binary);
    if(!fk.is_open())
    {
        DM.errorMessage("Error Message: Cannot load the fk column.");
        exit(1);
    }
    fk.read((char *)KKMR, row_num_S*(sizeof(double)));
    fk.close();
    DM.message("Finished fetching KKMR reference");
    
    vector<int> original_index_set;
    vector<int> shuffling_index;
    // Initialize the original_index_set
    for(int i = 0; i < feature_num; i ++)
    {
        original_index_set.push_back(i);
    }
    // Shuffling
    shuffling_index = shuffle(original_index_set, (unsigned)time(NULL));
    
    // Caching S
    printf("\n");
    printf("Avail_col_S: %d\n", avail_col_S);
    for(int i = 0; i < avail_col_S; i ++)
    {
       // printf("Cache %d th column in S\n", i);
        DM.fetchColumn(fields_S[3+i], row_num_S, cache_S[i]);
    }
    
    // Caching R
    printf("\n");
    printf("Avail_col_R: %d\n", avail_col_R);
    for(int k = 0; k < avail_col_R; k ++)
    {
        //printf("Cache %d th column in R\n", k);
        DM.fetchColumn(fields_R[1+k],row_num_R, cache_R[k]);
    }
    
    c_end = clock();
    cout<<"Caching:"<<1000*(c_end-c_start)/CLOCKS_PER_SEC<<"ms\n";

    do
    {
        c_start = clock();
        // Update one coordinate each time
        for(int j = 0; j < feature_num; j ++)
        {
            int cur_index = shuffling_index.at(j);
            //printf("Current feature index: %d\n", cur_index);
            
            F_partial = 0.00;
            
            if(cur_index < feature_num_S)
            {
                // Check cache for S
                if(cur_index < avail_col_S)
                {
                    // Compute the partial gradient
                    for(long i = 0; i < row_num_S; i ++)
                    {
                        F_partial += gradientCompute(Y[i],H[i],lm)*cache_S[cur_index][i];
                    }
                }
                else
                {
                    // Fetch the corresponding column in S and store in X_S
                    DM.fetchColumn(fields_S[3+cur_index], row_num_S, X_S);
                    // Compute the partial gradient
                    for(int i = 0; i < row_num_S; i ++)
                    {
                        F_partial += gradientCompute(Y[i],H[i],lm)*X_S[i];
                    }
                }
                
                // Store the old W(j)
                double W_j = model[cur_index];
                
                // Update the current coordinate
                model[cur_index] = model[cur_index] - step_size * F_partial;
                
                double diff = model[cur_index] - W_j;
                
                // Update the intermediate variable
                // H = H + (Wj - old_Wj)* X(,j)
                if(cur_index < avail_col_S)
                {
                    for(long m = 0; m < row_num_S; m ++ )
                    {
                        H[m] = H[m] + diff*cache_S[cur_index][m];
                    }
                }
                else{
                    for(long m = 0; m < row_num_S; m ++ )
                    {
                        H[m] = H[m] + diff*X_S[m];
                    }
                }
            }
            else
            {
                memset(X_R_f, 0.00, sizeof(float)*row_num_R);
                
                // Check cache for R
                int col_index_R = cur_index - feature_num_S;
                
                // Compute the factorized factor
                for(long m = 0; m < row_num_S; m ++)
                {
                    long fk = KKMR[m];
                    X_R_f[fk-1] += gradientCompute(Y[m],H[m],lm);
                }
                
                if(col_index_R < avail_col_R)
                {
                    // Compute the partial gradient
                    for(long k = 0; k < row_num_R; k ++)
                    {
                        F_partial += cache_R[col_index_R][k]*X_R_f[k];
                    }
                }
                else
                {
                    DM.fetchColumn(fields_R[1+col_index_R], row_num_R, X_R);
                    for(long k = 0; k < row_num_R; k ++)
                    {
                        F_partial += X_R[k]*X_R_f[k];
                    }
                }
                
                // Store the old W(j)
                double W_j = model[cur_index];
                
                // Update the current coordinate
                model[cur_index] = model[cur_index] - step_size * F_partial;
                
                double diff = model[cur_index] - W_j;
                
                // Factorized computation
                if(col_index_R < avail_col_R)
                {
                    for(long k = 0; k < row_num_R; k ++)
                    {
                        X_R_f[k] = diff*cache_R[col_index_R][k];
                    }
                }
                else
                {
                    for(long k = 0; k < row_num_R; k ++)
                    {
                        X_R_f[k] = diff*X_R[k];
                    }
                }
                
                // Update the intermediate variable
                // H = H + (Wj - old_Wj)* X(,j)
                for(long m = 0; m < row_num_S; m ++ )
                {
                    long fk = KKMR[m];
                    H[m] = H[m] + X_R_f[fk-1];
                }
            }
        }
        
        r_prev = F;
        // Caculate F
        F = 0.00;
        for(int i = 0; i < row_num_S; i ++)
        {
            double tmp = lossCompute(Y[i],H[i],lm);
            F += tmp;
        }
        
        r_curr = F;
        iters ++;
        c_end = clock();
        cout<<"Iteration:"<<1000*(c_end-c_start)/CLOCKS_PER_SEC<<" ms\n";
    }
    while(!stop(iters, r_prev, r_curr, _setting));
    
    delete [] Y;
    delete [] H;
    delete [] X_R_f;
    delete [] KKMR;
    
    if(avail_col_S < feature_num_S)
    {
        delete [] X_S;
    }
    if(avail_col_R < feature_num_R)
    {
        delete [] X_R;
    }
    
    // Clear the cache
    if(avail_col_S > 0)
    {
        for(int i  = 0; i < avail_col_S; i ++)
        {
            delete [] cache_S[i];
        }
        delete [] cache_S;
    }
    if(avail_col_R > 0)
    {
        for(int i  = 0; i < avail_col_R; i ++)
        {
            delete [] cache_R[i];
        }
        delete [] cache_R;
    }
    
    printf("\n");
    outputResults(r_curr, feature_num, iters, model);
    
    DM.message("Finish factorize");
}

#pragma mark - Block Coordinate Descent
void techniques::materializeBCD(string table_T, setting _setting, double *&model, int block_size, double avail_mem, const char *lm)
{
    DataManagement DM;
    DM.message("Start materializeBCD");
    
    // Get the table information and column names
    vector<long> tableInfo(3);
    vector<string> fields = DM.getFieldNames(table_T, tableInfo);
    int feature_num = (int)tableInfo[1];
    long row_num = tableInfo[2];
    
    // Block Info
    int block_num = feature_num/block_size;
    int block_residual = feature_num%block_size;
    block_num = block_residual > 0 ? (block_num + 1) : block_num;
    
    // For cache
    long avail_mem_total = 1024*1024*1024*avail_mem;
    int avail_col = 0;
    int avail_cache = 0;
    double **cache;
    
    // Label Array
    double *Y;
    // Residual Vector
    double *H;
    // Buffer for column reading
    double *X;
    // Additional columns space reserved for gradient computation
    double *G;
    double *difference;
    
    // Setting
    double step_size = _setting.step_size;
    
    // Calculate the available memory measured by size of each column
    avail_col = avail_mem_total/(sizeof(double)*row_num);
    
    // Calculate the available remaining space for cache
    avail_cache = avail_col - 5;
    
    if(avail_cache < 0)
    {
        DM.errorMessage("Insufficient memory space");
        exit(1);
    }
    else if (avail_cache == 0)
    {
        DM.message("No space for caching");
    }
    else
    {
        if( avail_cache >= feature_num - 1 )
        {
            cache = new double*[feature_num];
            for(int i = 0; i < feature_num; i ++)
            {
                cache[i] = new double[row_num];
            }
            //No need to reserve the X buffer to read single column
            avail_cache = feature_num;
        }
        else
        {
            cache = new double*[avail_cache];
            for(int i = 0; i < avail_cache; i ++)
            {
                cache[i] = new double[row_num];
            }
        }
    }
    
    // Dynamic memory allocation
    if(avail_cache < feature_num)
    {
        // Allocate the memory to X
        X = new double[row_num];
    }
    
    Y = new double[row_num];
    H = new double[row_num];
    G = new double[row_num];
    difference = new double[row_num];
    model = new double[feature_num];
   
    // Initialization of variables for loss and gradient
    double F = 0.00;
    double F_partial[block_size];
    // Initialize the partial graident for every block
    for(int i = 0; i < block_size; i ++)
    {
        F_partial[i] = 0.00;
    }
    
    double r_curr = 0.00;
    double r_prev = 0.00;
    int iters = 0;
    
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
    }
    
    for(long i = 0; i < row_num; i ++)
    {
        H[i] = 0.00;
        G[i] = 0.00;
        difference[i] = 0.00;
    }
    
    DM.fetchColumn(fields[1], row_num, Y);
    
    // Two level shuffling: first shuffling all columns, then all blocks
    vector<int> original_index;
    vector<int> shuffling_index;
    vector<int> original_block_index;
    vector<int> shuffling_block_index;
    
    // Initialize the original_index_set
    for(int i = 0; i < feature_num; i ++)
    {
        original_index.push_back(i);
    }
    
    for(int i = 0; i < block_num; i ++)
    {
        original_block_index.push_back(i);
    }
    
    // Shuffling
    shuffling_index = shuffle(original_index, (unsigned)time(NULL));
    shuffling_block_index = shuffle(original_block_index, (unsigned)time(NULL));
    
    
    // Print the shuffling_index and shuffling_block_index
    /**
    printf("After shuffling, the feature indexes:\n");
    for(int i = 0; i < feature_num; i ++)
    {
        printf("[%d]\n",shuffling_index.at(i));
    }
    
    printf("After shuffling, the block indexes:\n");
    for(int i = 0; i < block_num; i ++)
    {
        printf("[%d]\n",shuffling_block_index.at(i));
    }
    **/
    
    // Caching
    printf("\n");
    printf("Avail_col: %d\n", avail_cache);
    for(int i = 0; i < avail_cache; i ++)
    {
        printf("Cache %d th column\n", i);
        DM.fetchColumn(fields[i+2],row_num, cache[i]);
    }
    
    do
    {
    
        // Update one "block" each time
        // "Cumulative" difference in H caused by block
        for(int j = 0; j < block_num; j ++)
        {
            int cur_block_index = shuffling_block_index.at(j);
            //printf("Current_block_index: %d\n",cur_block_index);
            
            int cur_block_size = 0;
            
            //Check whether the current block is the "residual"
            if( (cur_block_index == block_num - 1) && block_residual > 0 )
            {
                cur_block_size = block_residual;
            }
            else
            {
                cur_block_size = block_size;
            }
            
            for(long d = 0; d < row_num; d ++)
            {
                difference[d] = 0.00;
            }
            
            // Start with "first level" block index
            int block_start_index= 0;
            
            // Double indexing: here, the index is the "index" of the "real index"
            // Update each 'block' by starting with getting the block index
            block_start_index = cur_block_index*block_size;
            
            //printf("Block_start_index: %d\n",shuffling_index.at(block_start_index));
            
            // First calculate the statistics used for gradient
            for(long g = 0; g < row_num; g ++)
            {
                G[g] = gradientCompute(Y[g],H[g],lm);
            }
            
            for(int b = 0; b < cur_block_size; b ++)
            {
                int cur_index = shuffling_index.at(block_start_index+b);
                //printf("Current feature index: %d\n", cur_index);
                
                F_partial[b] = 0.00;
                
                // Check for Cache
                if(cur_index < avail_cache)
                {
                    // Compute the partial gradient from cache
                    for(long i = 0; i < row_num ; i ++)
                    {
                        F_partial[b] += G[i]*cache[cur_index][i];
                    }
                }
                else
                {
                    // Fetch the column and store the current column into X
                    DM.fetchColumn(fields[cur_index+2], row_num, X);
                
                    // Compute the partial gradient
                    for(long i = 0; i < row_num ; i ++)
                    {
                        F_partial[b] += G[i]*X[i];
                    }
                }
              
                // Store the old W(j)
                int cur_model_index = cur_index;
                
                double diff = model[cur_model_index];
                
                // Update the current coordinate
                model[cur_model_index] = model[cur_model_index] - step_size * F_partial[b];
                
                // Compute the difference on current coordinate
                diff = model[cur_model_index] - diff;
                
                // Update the cumulative difference
                if(cur_index < avail_cache)
                {
                    for(long m = 0; m < row_num; m ++)
                    {
                        difference[m] += diff*cache[cur_index][m];
                    }

                }
                else
                {
                    for(long m = 0; m < row_num; m ++)
                    {
                        difference[m] += diff*X[m];
                    }

                }
               
            }
            
            for(long m = 0; m < row_num; m ++ )
            {
                H[m] = H[m] + difference[m];
            }
            
        }
        
        
        r_prev = F;
        // Caculate F
        F = 0.00;
        for(long i = 0; i < row_num ; i ++)
        {
            double tmp = lossCompute(Y[i],H[i], lm);
            F += tmp;
        }
        
        r_curr = F;
        iters ++;
    }
    while(!stop(iters, r_prev, r_curr, _setting));
    
    delete [] Y;
    delete [] H;
    delete [] G;
    delete [] difference;
    
    if(avail_cache < feature_num)
    {
        delete [] X;
    }
    
    // Clear the cache
    if(avail_cache > 0)
    {
        for(int i = 0; i < avail_cache; i ++)
        {
            delete [] cache[i];
        }
        
        delete [] cache;
    }
    
    printf("\n");
    outputResults(r_curr, feature_num, iters, model);
   
    DM.message("Finish materializeBCD");
    
}

void techniques::factorizeBCD(string table_S, string table_R, setting _setting, double *&model, int block_size, double avail_mem, const char *lm)
{
    DataManagement DM;
    DM.message("Start factorizeBCD");
    
    // Get the table information and column names
    vector<long> tableInfo_S(3);
    vector<long> tableInfo_R(3);
    vector<string> fields_S = DM.getFieldNames(table_S, tableInfo_S);
    vector<string> fields_R = DM.getFieldNames(table_R, tableInfo_R);
    int feature_num_S = (int)tableInfo_S[1];
    int feature_num_R = (int)tableInfo_R[1];
    int feature_num = feature_num_S + feature_num_R;
    int row_num_S = tableInfo_S[2];
    int row_num_R = tableInfo_R[2];
    
    // Block Info
    int block_num = feature_num/block_size;
    int block_residual = feature_num%block_size;
    block_num = block_residual > 0 ? (block_num + 1) : block_num;
    
    // For Cache
    long avail_mem_total = 1024*1024*1024*avail_mem;;
    long avail_cache = 0;
    int avail_col_S = 0;
    int avail_col_R = 0;
    double **cache_R;
    double **cache_S;
    
    // Label array
    double *Y;
    // Residual vector
    double *H;
    // Buffer for column reading in S
    double *X_S;
    // Buffer for column reading in R
    double *X_R;
    // Buffer to store factorized factor when considering column R
    double *X_R_f;
    // OID-OID Mapping (Key Foreign-Key Mapping Reference, to be kept in memory)
    double *KKMR;
    // Additional column space reserved for gradient computation
    double *G;
    double *difference;
    
    // Setting
    double step_size = _setting.step_size;
    
    // Calculate the available memory measured by size of each column in R and S
    avail_cache = avail_mem_total - sizeof(double)*(6*row_num_S + 2*row_num_R);
    
    if(avail_cache < 0)
    {
        DM.errorMessage("Insufficient memory space");
        exit(1);
    }
    else if(avail_cache == 0)
    {
        DM.message("No space for caching");
    }
    else
    {
        // First consider caching columns in S
        avail_col_S = avail_cache/(sizeof(double)*row_num_S);
        if(avail_col_S == 0)
        {
            DM.message("No space for caching S");
            // Then consider caching columns in R
            avail_col_R = avail_cache/(sizeof(double)*row_num_R);
            if(avail_col_R == 0)
            {
                DM.message("No space for caching R");
            }
            else
            {
                if(avail_col_R >= feature_num_R - 1)
                {
                    cache_R = new double*[feature_num_R];
                    for(int i = 0; i < feature_num_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                    // No need to reserve the X_R buffer to read a single column in R
                    avail_col_R = feature_num_R;
                }
                else
                {
                    cache_R = new double*[avail_col_R];
                    for(int i = 0; i < avail_col_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                }
            }
        }
        else
        {
            if(avail_col_S >= feature_num_S)
            {
                cache_S = new double*[feature_num_S];
                for(int i = 0; i < feature_num_S; i ++)
                {
                    cache_S[i] = new double[row_num_S];
                }
                // No need to reserve X_S for single column reading
                avail_col_S = feature_num_S;
                
            }
            else
            {
                X_S = new double[row_num_S];
                cache_S = new double*[avail_col_S];
                for(int i = 0; i < avail_col_S; i ++)
                {
                    cache_S[i] = new double[row_num_S];
                }
            }
            
            // Then consider the caching for R using the remaining caching space
            if(avail_col_S == feature_num_S)
            {
                avail_cache = avail_cache - (avail_col_S - 1)*sizeof(double)*row_num_S;
            }
            else
            {
                avail_cache = avail_cache - avail_col_S*sizeof(double)*row_num_S;
            }
            avail_col_R = avail_cache/(sizeof(double)*row_num_R);
            if(avail_col_R == 0)
            {
                DM.message("No space for caching R");
            }
            else
            {
                if(avail_col_R >= feature_num_R - 1)
                {
                    cache_R = new double*[feature_num_R];
                    for(int i = 0; i < feature_num_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                    //No need to reserve the X_R buffer to read a single column in R
                    avail_col_R = feature_num_R;
                }
                else
                {
                    cache_R = new double*[avail_col_R];
                    for(int i = 0; i < avail_col_R; i ++)
                    {
                        cache_R[i] = new double[row_num_R];
                    }
                }
            }
        }
        
    }
    
    // Dynamic memory allocation
    if(avail_col_S < feature_num_S)
    {
        X_S = new double[row_num_S];
    }
    if(avail_col_R < feature_num_R)
    {
        X_R = new double[row_num_R];
    }
    Y = new double[row_num_S];
    H = new double[row_num_S];
    X_R_f = new double[row_num_R];
    G = new double[row_num_S];
    difference = new double[row_num_S];
    KKMR = new double[row_num_S];
    model = new double[feature_num];
    
    // Initialization of variables for loss and gradient
    double F = 0.00;
    double F_partial[block_size];
    double r_curr = 0.00;
    double r_prev = 0.00;
    int iters = 0;
    // Initialize the partial graident for every block
    for(int i = 0; i < block_size; i ++)
    {
        F_partial[i] = 0.00;
    }
    
    // Initialization
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
    }
    for(long i = 0; i < row_num_S; i ++)
    {
        H[i] = 0.00;
        G[i] = 0.00;
        difference[i] = 0.00;
    }
    for(long i = 0; i < row_num_R; i ++)
    {
        X_R_f[i] = 0.00;
    }
    DM.fetchColumn(fields_S[1], row_num_S, Y);
    printf("\n");
    DM.message("Start fetching KKMR reference");
    // Read the fk column(referred rid in R) in table S, rid column in R
    ifstream fk;
    // Load the fk to KKMR
    fk.open(fields_S[2], ios::in | ios::binary);
    // rid.open(table2_fields[0], ios::in | ios::binary);
    if(!fk.is_open())
    {
        DM.errorMessage("Error Message: Cannot load the fk column.");
        exit(1);
    }
    fk.read((char *)KKMR, row_num_S*(sizeof(double)));
    fk.close();
    DM.message("Finished fetching KKMR reference");
    
    //Two level shuffling: first shuffling all columns, then all blocks
    vector<int> original_index;
    vector<int> shuffling_index;
    vector<int> original_block_index;
    vector<int> shuffling_block_index;
    
    // Initialize the original_index_set
    for(int i = 0; i < feature_num; i ++)
    {
        original_index.push_back(i);
    }
    
    for(int i = 0; i < block_num; i ++)
    {
        original_block_index.push_back(i);
    }
    
    // Shuffling
    shuffling_index = shuffle(original_index, (unsigned)time(NULL));
    shuffling_block_index = shuffle(original_block_index, (unsigned)time(NULL));
    
    // Print the shuffling_index and shuffling_block_index
    /**
    printf("After shuffling, the feature indexes:\n");
    for(int i = 0; i < feature_num; i ++)
    {
        printf("[%d]\n",shuffling_index.at(i));
    }
    
    //printf("After shuffling, the block indexes:\n");
    for(int i = 0; i < block_num; i ++)
    {
        printf("[%d]\n",shuffling_block_index.at(i));
    }
    **/
    
    // Caching S
    printf("\n");
    printf("Avail_col_S: %d\n", avail_col_S);
    for(int i = 0; i < avail_col_S; i ++)
    {
        printf("Cache %d th column in S\n", i);
        DM.fetchColumn(fields_S[3+i], row_num_S, cache_S[i]);
    }
    
    // Caching R
    printf("\n");
    printf("Avail_col_R: %d\n", avail_col_R);
    for(int k = 0; k < avail_col_R; k ++)
    {
        printf("Cache %d th column in R\n", k);
        DM.fetchColumn(fields_R[1+k],row_num_R, cache_R[k]);
    }

    do
    {
        // Update one "block" each time
        // "Cumulative" difference in H caused by block
        for(int j = 0; j < block_num; j ++)
        {
            int cur_block_index = shuffling_block_index.at(j);
            //printf("Current_block_index: %d\n",cur_block_index);
            
            int cur_block_size = 0;
            
            //Check whether the current block is the "residual"
            if( (cur_block_index == block_num - 1) && block_residual > 0 )
            {
                cur_block_size = block_residual;
            }
            else
            {
                cur_block_size = block_size;
            }
            
            for(long d = 0; d < row_num_S; d ++)
            {
                difference[d] = 0.00;
            }
            
            // Start with "first level" block index
            int block_start_index= 0;
            
            // Double indexing: here, the index is the "index" of the "real index"
            // Update each 'block' by starting with getting the block index
            block_start_index = cur_block_index*block_size;
            
            //printf("Block_start_index: %d\n", shuffling_index.at(block_start_index));
            
            // First calculate the statistics used for gradient
            for(long g = 0; g < row_num_S; g ++)
            {
                G[g] = gradientCompute(Y[g],H[g],lm);
            }
            
            for(int b = 0; b < cur_block_size; b ++)
            {
                int cur_index = shuffling_index.at(block_start_index + b);
                //printf("Current feature index: %d\n", cur_index);;
                F_partial[b] = 0.00;
                
                // Check whether the column is in table R. If it is, applied factorized learning
                if(cur_index < feature_num_S)
                {
                    // Check cache for S
                    if(cur_index < avail_col_S)
                    {
                        // Compute the partial gradient
                        for(long i = 0; i < row_num_S; i ++)
                        {
                            F_partial[b] += G[i]*cache_S[cur_index][i];
                        }
                    }
                    else
                    {
                        // Fetch each column and store the column into X
                        DM.fetchColumn(fields_S[cur_index+3], row_num_S, X_S);
                        // Compute the partial gradient
                        for(long i = 0; i < row_num_S; i ++)
                        {
                            F_partial[b] += G[i]*X_S[i];
                        }
                    }
                    
                    // Store the old Wj
                    int cur_model_index = cur_index;
                    double W_j = model[cur_model_index];
                    
                    // Update the current coordinate
                    model[cur_model_index] = model[cur_model_index] - step_size * F_partial[b];
                    
                    // Compute the difference
                    double diff = model[cur_model_index] - W_j;
             
                    // Update the cumulative difference
                    if(cur_index < avail_col_S)
                    {
                        for(long m = 0; m < row_num_S; m ++)
                        {
                            difference[m] += diff*cache_S[cur_index][m];
                        }
                    }
                    else
                    {
                        for(long m = 0; m < row_num_S; m ++)
                        {
                            difference[m] += diff*X_S[m];
                        }
                        
                    }
                }
                else
                {
                    for(long i = 0; i < row_num_R; i ++)
                    {
                        X_R_f[i] = 0.00;
                    }
                    
                    // Check cache for R
                    int col_index_R = cur_index - feature_num_S;
                    //printf("col_index_R: %d\n",col_index_R);

                    // Apply factorized learning to gradient computation
                    for(long m = 0; m < row_num_S; m ++)
                    {
                        long fk = KKMR[m];
                        X_R_f[fk-1] += G[m];
                    }
                    
                    if(col_index_R < avail_col_R)
                    {
                        for(long j = 0; j < row_num_R; j ++)
                        {
                            F_partial[b] += cache_R[col_index_R][j]*X_R_f[j];
                        }
                    }
                    else
                    {
                        // Fetch the corresponding column in R
                        DM.fetchColumn(fields_R[1+col_index_R],row_num_R, X_R);
                        for(long j = 0; j < row_num_R; j ++)
                        {
                            F_partial[b] += X_R[j]*X_R_f[j];
                        }
                    }
                    
                    int cur_model_index = cur_index;
                    double W_j = model[cur_model_index];
                   
                    model[cur_model_index] = model[cur_model_index] - step_size * F_partial[b];
                   
                    double diff = model[cur_model_index] - W_j;
                    
                    // Apply factorized learning to difference (of model/coordinate) computation
                    if(col_index_R < avail_col_R)
                    {
                        for(int i = 0; i < row_num_R; i ++ )
                        {
                            X_R_f[i] = diff*cache_R[col_index_R][i];
                        }
                    }
                    else
                    {
                        for(int i = 0; i < row_num_R; i ++ )
                        {
                            X_R_f[i] = diff*X_R[i];
                        }
                    }
                    for(long m = 0; m < row_num_S; m ++)
                    {
                        long fk = KKMR[m];
                        difference[m] += X_R_f[fk-1];
                    }
                }
                
            }
            
            for(long m = 0; m < row_num_S; m ++)
            {
                H[m] = H[m] + difference[m];
            }
        }
        
        r_prev = F;
        // Caculate F
        F = 0.00;
        for(long i = 0; i < row_num_S; i ++)
        {
            double tmp = lossCompute(Y[i],H[i],lm);
            F += tmp;
        }
        
        r_curr = F;
        iters ++;
    }
    while(!stop(iters, r_prev, r_curr, _setting));
    
    delete [] Y;
    delete [] H;
    delete [] X_R_f;
    delete [] KKMR;
    delete [] G;
    delete [] difference;
    
    if(avail_col_S < feature_num_S)
    {
        delete [] X_S;
    }
    if(avail_col_R < feature_num_R)
    {
        delete [] X_R;
    }
    
    // Clear Cache
    if(avail_col_R > 0)
    {
        for(int i = 0; i < avail_col_R; i ++)
        {
            delete [] cache_R[i];
        }
        delete [] cache_R;
    }
    if(avail_col_S > 0)
    {
        for(int i = 0; i < avail_col_S; i ++)
        {
            delete [] cache_S[i];
        }
        delete [] cache_S;
    }
    
    printf("\n");
    outputResults(r_curr, feature_num, iters, model);
    
    DM.message("Finish factorizeBCD");
}


#pragma mark - Gradient descent
/*
 Read a single file the columns of which are in format like: id, label, feature
 The offset entry for W0 is not considered for now
 Logistic Regression for now
 */

// Specific techniques selection: flag (for generalization purpose)
// Stochastic Gradient Descent
void techniques::SGD(vector< vector<double> > data, setting _setting, double *&model, int feature_num)
{
    DataManagement::message("Start SGD");
    long data_size = data.size();
    vector<long> original_index_set;
    vector<long> shuffling_index;
    //Initialize the original_index_set
    std::cout << "Start building the index set" << std::endl;
	for(long i = 0; i < data_size; i ++)
    {
        original_index_set.push_back(i);
    }
    
    // Shuffling 
    shuffling_index = shuffle(original_index_set, (unsigned)time(NULL));
    
    // Setting
    double step_size = _setting.step_size;
    
    // Allocate the memory to model
    model = new double[feature_num];
    
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
    }
    
    // Loss Function
    double F = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int iters = 0;
   
	std::cout << "Start training" << std::endl;
    do
    {
        r_prev = F;
        F = 0.00;
        vector<double> gradient(feature_num,0.00);
        
        for(long j = 0; j < data_size; j ++)
        {
            long cur_index = shuffling_index[j];
            
            // Update the model
            double output = 0.00;
            for(int k = 0; k < feature_num; k ++)
            {
                output += model[k]*data[cur_index][k+2];
            }
            
            for(int k = 0; k < feature_num; k ++)
            {
                gradient[k] = gradientCompute(data[cur_index][1],output, "lr")*data[cur_index][2+k];
                model[k] = model[k]-step_size*gradient[k];
            }
        }
        
        // Calculate F
        for(long j = 0; j < data_size; j ++)
        {
            double output = 0.00;
            for(int k = 0; k < feature_num; k ++)
            {
                output += model[k]*data[j][k+2];
            }
            double tmp = lossCompute(data[j][1], output, "lr");
            F += tmp;
        }
        
        r_curr = F;
		std::cout << "Loss: " << F << std::endl;
        iters ++;
    }
    while(!stop(iters ,r_prev,r_curr,_setting));
    
    printf("\n");
    outputResults(r_curr, feature_num, iters, model);
    
    DataManagement::message("Finish SGD");

}

#pragma mark - Batch Gradient Descent
// Batch Gradient Descent
void techniques::BGD(vector< vector<double> > data, setting _setting, double *&model, int feature_num)
{
    DataManagement::message("Start BGD");
    long data_size = data.size();
    
    // Setting
    double step_size = _setting.step_size;
    
    // Allocate the memory to the model
    model = new double[feature_num];
    
    for(int i = 0; i < feature_num; i ++)
    {
        model[i] = 0.00;
    }
    
    // Loss Function
    double F = 0.00;
    double r_curr = 0.00;
    double r_prev = 0.00;
    int iters = 0;
    
    do
    {
        r_prev = F;
        F = 0.00;
        vector<double> gradient(feature_num,0.00);
        
        for(long j = 0; j < data_size; j ++)
        {
            
            // Update the model
            double output = 0.00;
            for(int k = 0; k < feature_num; k ++)
            {
                output += model[k]*data[j][2+k];
            }
            
            for(int k = 0; k < feature_num; k ++)
            {
                gradient[k] += gradientCompute(data[j][1],output, "lm")*data[j][2+k];
            }
        }
        
        
        for(int k = 0; k < feature_num; k ++)
        {
            model[k] = model[k]-step_size*gradient[k];
        }
        
        for(long j = 0; j < data_size; j ++)
        {
            double output = 0.00;
            for(int k = 0; k < feature_num; k ++)
            {
                output += model[k]*data[j][2+k];
            }
            double tmp = lossCompute(data[j][1], output, "lm");
            printf("tmp loss: %f\n", tmp);
            F += tmp;
        }
        
        
        r_curr = F;
        printf("The loss: %lf\n",F);
        iters ++;
    }
    while(!stop(iters ,r_prev,r_curr,_setting));
    
    printf("\n");
    outputResults(r_curr, feature_num, iters, model);
    
    DataManagement::message("Finish BGD");

}

void techniques::classify(vector< vector<double> > data, vector<double> model)
{
    // Count the number of correct classifcation
    long count = 0;
    long data_size =  data.size();
    if(data.at(0).size() != model.size()+2)
    {
        DataManagement::errorMessage("Inconsistent file provided");
    }
    
    int featureNum = (int)model.size();
    for(long i = 0; i < data_size; i ++)
    {
        double actual_label = data[i][1];
        double predicted_label = 0.00;
        double confidence = 0.00;
        double output = 0.00;
        for(int j = 0; j < featureNum; j ++)
        {
            output += model[j]*data[i][2+j];
        }
        printf("W^TX: %f\n", output);
        confidence = C_lr(output);
        if(confidence > 0.5)
        {
            predicted_label = 1.00;
        }
        else
        {
            predicted_label = -1.00;
        }
        if(actual_label == predicted_label)
        {
            printf("Prediction Correct\n");
            count++;
        }
        else
        {
            printf("Prediction Wrong\n");
        }
        printf("Confidence: %f\n", confidence);
        printf("Actual Label: %f , Predicted Label: %f\n", actual_label, predicted_label);
    }
    printf("Correcteness: %f \n", (double)count/(double)data_size);
    
}

#pragma mark - shuffling
vector<int> techniques::shuffle(vector<int> &index_set, unsigned seed)
{
    vector<int> original_set = index_set;
    int size = (int)index_set.size();
    vector<int> new_index_set;
    srand (seed);
    for(int i = 0; i < size; i ++)
    {
        int cur_size = (int)original_set.size();
        int rand_index = random()%cur_size;
        new_index_set.push_back(original_set.at(rand_index));
        original_set.erase(original_set.begin()+rand_index);
    }
    
    return new_index_set;
}

vector<long> techniques::shuffle(vector<long> &index_set, unsigned seed)
{
    vector<long> original_set = index_set;
    long size = (long)index_set.size();
    vector<long> new_index_set;
    srand(seed);
    for(long i = 0; i < size; i ++)
    {
        long cur_size = original_set.size();
        long rand_index = random()%cur_size;
        new_index_set.push_back(original_set.at(rand_index));
        original_set.erase(original_set.begin()+rand_index);
    }
    
    return new_index_set;
}

#pragma mark - stop creteria
bool techniques::stop(int k, double r_prev, double r_curr, setting &setting)
{
    double iter_num = k;
    double difference = abs(r_prev - r_curr);
    
    if( iter_num == setting.iter_num || difference <= setting.error)
    {
        return true;
    }
    else
    {
        return false;
    }
}

#pragma mark - print the final result
void techniques::outputResults(double r_curr, int feature_num, int k, double *&model)
{
    printf("The final loss: %lf\n", r_curr);
    printf("Number of iteration: %d\n", k);
    printf("Model: ");
    for(int i = 0; i < feature_num; i ++)
    {
        if(i == feature_num - 1)
        {
            printf("%.20f\n",model[i]);
        }
        else
        {
            printf("%.20f, ",model[i]);
        }
    }
}
