#include <gtest/gtest.h>
#include <stdexcept>
#include <numeric>
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
TEST(ReductionLibraryTest, Check3Dpure)
{
  std::vector<size_t> nvalues = {2,3,4,5,8,9,12,30,31};
  const size_t N(201);

  for (auto n : nvalues)
  {
    size_t nitems(3*n*n*n*N);
    std::vector<double> test(nitems);
    std::iota(test.begin(), test.end(), 0);
    
    auto input = malloc_shared<double>(nitems,mQ);
    mQ.memcpy(input, test.data(), sizeof(double)*nitems).wait();

     //NOTE: this is observed to not be correct on CPU, possible race condition somewhere
    reduce_to_array<double>(mQ, N, nitems, input, maximum<>());
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(input[i], *std::max_element(test.begin() + 3*n*n*n*i, test.begin() + 3*n*n*n*(i+1)));
    }
    
    free(input,   mQ);
  }
}


TEST(ReductionLibraryTest, Check3DwithoutBuf)
{
  std::vector<size_t> nvalues = {2,3,4,5,8,9,12,30,31};
  const size_t N(201);

  for (auto n : nvalues)
  {
    size_t nitems(3*n*n*n*N);
    std::vector<double> test(nitems);
    std::iota(test.begin(), test.end(), 0);
    
    auto input = malloc_shared<double>(nitems,mQ);
    mQ.memcpy(input, test.data(), sizeof(double)*nitems).wait();

    auto output = malloc_shared<double>(N,mQ);
    for (int i =0;i<N;i++) output[i] = -1;

     //NOTE: this is observed to not be correct on CPU, possible race condition somewhere
    reduce_to_array<double>(mQ, N, nitems, input, output, maximum<>());
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], *std::max_element(test.begin() + 3*n*n*n*i, test.begin() + 3*n*n*n*(i+1)));
    }
    
    free(input,   mQ);
    free(output,  mQ);
  }
}

TEST(ReductionLibraryTest, Check3DwithBuf)
{
  std::vector<size_t> nvalues = {2,3,4,5,8,9,12,30,31};
  const size_t N(201);

  for (auto n : nvalues)
  {
    size_t nitems(3*n*n*n*N);
    std::vector<double> test(nitems);
    std::iota(test.begin(), test.end(), 0);
    
    auto input = malloc_device<double>(nitems,mQ);
    mQ.memcpy(input, test.data(), sizeof(double)*nitems).wait();

    auto output = malloc_shared<double>(N,mQ);
    for (int i =0;i<N;i++) output[i] = -1;

    auto redbuf = malloc_device<double>(nitems, mQ); 
   
    reduce_to_array<double>(mQ, N, nitems, input, output, redbuf, maximum<>());
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], *std::max_element(test.begin() + 3*n*n*n*i, test.begin() + 3*n*n*n*(i+1)));
    }
    
    free(redbuf,  mQ);
    free(input,   mQ);
    free(output,  mQ);
  }
}

TEST(ReductionLibraryTest, Check2D)
{
  std::vector<size_t> nvalues = {2,3,4,5,8,9,12,30,31,40,45,100};
  const size_t N(201);

  for (auto n : nvalues)
  {
    size_t nitems(2*n*n*N);
    auto input = malloc_shared<double>(nitems,mQ);
    for (int i =0;i<nitems;i++) input[i] = i;
    
    std::vector<double> test(nitems);
    std::iota(test.begin(), test.end(), 0);

    auto output = malloc_shared<double>(N,mQ);
    for (int i =0;i<N;i++) output[i] = -1;

    auto redbuf = malloc_device<double>(nitems, mQ); 
    
    reduce_to_array<double>(mQ, N, nitems, input, output, redbuf, maximum<>());
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], *std::max_element(test.begin() + 2*n*n*i, test.begin() + 2*n*n*(i+1)));
    }

    reduce_to_array<double>(mQ, N, nitems, input, output, maximum<>());
    for (int i=0;i<N;i++)
    {
      EXPECT_EQ(output[i], *std::max_element(test.begin() + 2*n*n*i, test.begin() + 2*n*n*(i+1)));
    }
    
    free(redbuf,  mQ);
    free(input,   mQ);
    free(output,  mQ);
  }
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
