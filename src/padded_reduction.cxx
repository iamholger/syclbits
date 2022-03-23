// padded_reduction.cxx
//
//
// Padded tree reduction to find the maximum of an
// array of lenth N with n reduction partners.

#include <numeric>
#include <random>
#include <CL/sycl.hpp>
using namespace sycl;

#include <chrono>

static queue myQueue(default_selector{});

// Explicitly copy Npadded by value as we manipulate it
void padded_reduction(const size_t N, size_t Npadded, const size_t n, double * input, double * output)
{
    // The number of reduction rounds --- N = n^rounds
    size_t rounds = ceil(log(N)/log(n));

#ifdef DEBUG
    std::cout << "Will do " << rounds << " rounds of reduction\n";
    std::cout << "Npadded: " << Npadded << "\n";
    std::cout << "N: " << N << "\n";
#endif
   
    for (int round=0;round<rounds;round++)
    {
        
#ifdef DEBUG
        std::cout << "Round " << round+1 << " reduces " << Npadded << " items\n";
        std::cout << "The reduction size is " << n << "\n";
        std::cout << "The reduced size is " << Npadded/n << "\n";
#endif

        auto myrange = nd_range<1>{{Npadded}, {n}};
        myQueue.submit([&](handler & cgh)
        {

#ifdef DEBUG
          std::cout << "This range has a global range size of " << myrange.get_global_range().size() << std::endl;
          std::cout << "This range has a group  range size of " << myrange.get_group_range().size()  << std::endl;
          std::cout << "This range has a local  range size of " << myrange.get_local_range().size()  << std::endl;
          
          sycl::stream out(4096, 256, cgh);
#endif
          cgh.parallel_for(myrange, [=](nd_item<1> it) 
          {
            auto wg =  it.get_group();
            input[it.get_group_linear_id()] = reduce_over_group(wg, input[it.get_global_linear_id()], maximum<>());
          });
        }).wait();

        Npadded /= n;

        if (Npadded%n != 0) Npadded = (Npadded/n + 1) * n;
    }

    // Copy data to host --- maybe do this with memcopy instead
    //myQueue.submit([&](handler & cgh)
    //{
    //cgh.single_task(
        //[=](){output[0] = input[0];});
    //}).wait();
    myQueue.memcpy(output, input, sizeof(double));
    myQueue.wait();
}


int main(int argc, char** argv)
{
    const size_t N = std::atoi(argv[1]); // number of data elements to reduce
    const size_t n = std::atoi(argv[2]); // Reduction partners

    // Padding 
    size_t Npadded(0);
    while (Npadded < N) Npadded +=n;

    // Allocate memory and generate data on device --- note this would be the job of the eigenvalue function
    auto input = malloc_device<double>(Npadded, myQueue);
    myQueue.parallel_for(N,           [=](id<1> idx) {input[  idx] = idx   ;});
    // Set all padding data to sth really small (Alternatively, use myQueue.memset(input, -1e100, (Npadded-N)*sizeof(double))
    if ( (Npadded-N) > 0)
    myQueue.parallel_for(Npadded - N, [=](id<1> idx) {input[N+idx] = -1e100 ;});
      
    // This is the output data, alternativel, memcopy should suffice
    auto output = malloc_host<double>(1, myQueue);
    output[0] = -1; 


    padded_reduction(N,Npadded,n, input, output);
#ifndef MEASURE
    std::cout << "Result: " << output[0] << "\n";
#endif

#ifdef MEASURE
    auto start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[3]);i++) 
      padded_reduction(N,Npadded,n, input, output);
    auto end = std::chrono::steady_clock::now();
    std::cout << N << "," << n << "," << argv[3] << "," << "padded_reduction" << "," <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "," << myQueue.get_device().get_info<info::device::name>() << "\n";
    //std::cout <<    << "," << "," << argv[3] << std::endl;
#endif
   
   return 0; 
}
