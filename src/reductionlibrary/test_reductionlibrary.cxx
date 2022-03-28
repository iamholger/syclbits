#include <gtest/gtest.h>
#include <stdexcept>

#include "reductionlibrary.h"

#include <CL/sycl.hpp>
using namespace sycl;


static queue mQ(default_selector{});


TEST(ReductionLibraryTest, BasicAssertions)
{

    const size_t N(100);
    
    auto input = malloc_shared<int>(N, mQ);
    for (int i=0;i<N;i++) input[i] = i+1;

    // initial value
    int sum(0);
    int max(-1);
    //reduce<double>(input, output, N);
    reduce_to_scalar<int>(mQ, input, sum, N, plus<>());
    reduce_to_scalar<int>(mQ, input, max, N, maximum<>());

    EXPECT_EQ(sum, 5050);
    EXPECT_EQ(max, 100);

    free(input, mQ);
}



TEST(ReductionLibraryTest, CheckRecursive3D)
{
    constexpr size_t n(40), N(2000), nitems(3*n*n*n*N);
    auto input   = malloc_shared<double>(nitems,mQ);
    for (int i =0;i<nitems;i++) input[i] = i;


    auto output  = malloc_shared<double>(N,mQ);
    for (int i =0;i<N;i++) output[i] = -1;
    reduce_to_array<double>(mQ, N, n, input, output, 4, 3, maximum<>());
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], *std::max_element(input + 3*n*n*n*i, input + 3*n*n*n*(i+1)));
    }
    
    
    auto output2  = malloc_shared<double>(N,mQ);
    for (int i =0;i<N;i++) output2[i] = -1;
    reduce_to_array2<double>(mQ, N, n, input, output2, maximum<>());
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], output2[i]);
    }
   
    auto output3  = malloc_shared<double>(N,mQ);
    for (int i =0;i<N;i++) output3[i] = -1;
    auto redbuf = malloc_device<double>(N*3*n*n*n, mQ); 
    reduce_to_array<double>(mQ, N, 3*n*n*n, n, input, output3, redbuf, maximum<>());
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], output3[i]);
    }

    std::vector<size_t> wgsizes = {1,2,3,4,5,6,8,10,12,16,20};

    for (auto ws:wgsizes)
    {
      for (int i=0;i<N;i++) output[i] = -1;
      reduce_to_array<double>(mQ, N, 3*n*n*n, ws, input, output, redbuf, maximum<>());
      for (int i=0;i<N;i++)
      {
        EXPECT_EQ(output[i], output3[i]);
      }
    }
    free(redbuf,  mQ);
    free(input,   mQ);
    free(output,  mQ);
    free(output2, mQ);
    free(output3, mQ);
}

TEST(ReductionLibraryTest, CheckRecursive2D)
{
    constexpr size_t n(100), N(2000), nitems(2*n*n*N);

    auto input   = malloc_shared<double>(nitems,mQ);
    for (int i=0;i<nitems;i++) input[i] = i;

    auto output   = malloc_shared<double>(N,mQ);
    for (int i=0;i<N;i++) output[i] = -1;
    reduce_to_array<double>(mQ, N, nitems, input, output, maximum<>());

    for (int i=0;i<nitems;i++) input[i] = i;
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], *std::max_element(input + 2*n*n*i, input + 2*n*n*(i+1)));
    }
    

    auto output2  = malloc_shared<double>(N,mQ);
    for (int i=0;i<N;i++) output2[i] = -1;
    auto redbuf = malloc_device<double>(N*2*n*n, mQ); 
    reduce_to_array<double>(mQ, N, 2*n*n, n, input, output2, redbuf, maximum<>());
    
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], output2[i]);
    }
    free(input,   mQ);
    free(redbuf,  mQ);
    free(output,  mQ);
    free(output2, mQ);
}

TEST(ReductionLibraryTest, CheckPadded)
{
    const size_t gathersize(20), Ntotal(123456);
    // Padding 
    size_t Npadded(0);
    while (Npadded < Ntotal) Npadded +=gathersize;

    // Allocate memory and generate data on device --- note this would be the job of the eigenvalue function
    auto input = malloc_device<double>(Npadded, mQ);
    mQ.parallel_for(Ntotal,           [=](id<1> idx) {input[  idx] = idx   ;});
    // Set all padding data to sth really small (Alternatively, use myQueue.memset(input, -1e100, (Npadded-N)*sizeof(double))
    if ( (Npadded-Ntotal) > 0)
    mQ.parallel_for(Npadded - Ntotal, [=](id<1> idx) {input[Ntotal+idx] = -1e100 ;});
      
    auto output = malloc_host<double>(1, mQ);
    output[0] = -1;

    padded_reduction_to_scalar_with_fixed_gather_size<double>(mQ, Ntotal, Npadded, gathersize, input, output, maximum<>()); 

    
    EXPECT_EQ(output[0], 123455);
    free(input,   mQ);
    free(output,  mQ);
}
