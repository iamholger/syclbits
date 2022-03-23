#include <numeric>
#include <random>
#include <CL/sycl.hpp>
using namespace sycl;

#include <chrono>

static queue myQueue(default_selector{});


void array_reduction(const size_t NPT, const size_t numVPAIP, double * input, double * output, const size_t alpha, const size_t beta)
{

    const auto   WGMAX =  myQueue.get_device().get_info<info::device::max_work_group_size>();
    const auto   mysize = numVPAIP*numVPAIP*numVPAIP*3;
    const size_t nitems = NPT*mysize;
    size_t gathersize(alpha*beta), reduced_size(nitems/gathersize);
    auto redbuf = malloc_device<double>(reduced_size, myQueue);
   
    // First reduction 
    myQueue.submit([&](handler & cgh) {
          cgh.parallel_for(nd_range<3>{{NPT, numVPAIP*numVPAIP, numVPAIP*3}, {1,alpha,beta}},
            [=](nd_item<3> it) 
          {
              auto wg =  it.get_group();
              const auto grlid = it.get_group_linear_id();
              auto ps =  reduce_over_group(wg, input[it.get_global_linear_id()], maximum<>());
              redbuf[grlid] = ps;
          });
    }).wait();

    // Further reductions
    while ( (reduced_size/NPT) > WGMAX )
    {
#ifdef DEBUG
        std::cout << "The next gather size  " << reduced_size/NPT << " is >WGMAX -> need to iterate\n";
#endif
        size_t start = NPT;
        bool done(false);
        while (not done)
        {
            gathersize = reduced_size/start;
            start += NPT;
            if ( (gathersize < WGMAX) and (gathersize >= NPT) and (reduced_size%gathersize ==0) ) done=true;
        }
#ifdef DEBUG
        std::cout << "reduce " << reduced_size << " " << gathersize << "\n";
#endif
        
        myQueue.submit([&](handler & cgh)
        {
          cgh.parallel_for(nd_range<1>{{reduced_size}, {gathersize}}, [=](nd_item<1> it) 
          {
            auto wg =  it.get_group();
            redbuf[it.get_group_linear_id()] = reduce_over_group(wg, redbuf[it.get_global_linear_id()], maximum<>());
          });
        }).wait();
        reduced_size /= gathersize;
    }
#ifdef DEBUG
    std::cout << "The next gather size " << reduced_size/NPT << " is <= WGMAX (" << WGMAX << ") so final reduction\n";
    std::cout << "reduce " << reduced_size << " " << gathersize << "\n";
#endif
    gathersize = reduced_size/NPT;

    // Note how in the final reduction we can reduce directly into the output buffer
    myQueue.submit([&](handler & cgh)
    {
      cgh.parallel_for(nd_range<1>{{reduced_size}, {gathersize}}, [=](nd_item<1> it) 
      {
        auto wg =  it.get_group();
        output[it.get_group_linear_id()] = reduce_over_group(wg, redbuf[it.get_global_linear_id()], maximum<>());
      });
    }).wait();

}



int main(int argc, char** argv)
{
    const auto   WGMAX =  myQueue.get_device().get_info<info::device::max_work_group_size>();
    const size_t NPT      = std::atoi(argv[1]);
    const size_t numVPAIP = std::atoi(argv[2]);

    const size_t alpha      = std::atoi(argv[3]);
    const size_t beta       = std::atoi(argv[4]);

    if ( (numVPAIP*numVPAIP)%alpha != 0 ) std::cout << "alpha bad\n";
    if ( (numVPAIP*3)%beta != 0 )         std::cout << "beta bad\n";
    if ( (alpha*beta > WGMAX) )           std::cout << "alpha*beta bad\n";


    //std::uniform_real_distribution<double> unif(-100, 100);
    //std::default_random_engine re(time(NULL));
    
    const auto   mysize = numVPAIP*numVPAIP*numVPAIP*3;
    const size_t nitems = NPT*mysize;
#ifdef DEBUG
    std::cout <<"Items per patch: " << nitems << " Total: " << mysize << "\n";
#endif
    
    auto input   = malloc_shared<double>(nitems,myQueue);
    for (int i =0;i<nitems;i++) input[i] = i;//unif(re);

    auto output  = malloc_shared<double>(NPT,myQueue);
    for (int i =0;i<NPT;i++) output[i] = -1;


    array_reduction(NPT, numVPAIP, input, output, alpha, beta);

#ifdef MEASURE
    auto start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[5]);i++) 
    array_reduction(NPT, numVPAIP, input, output, alpha, beta);
    auto end = std::chrono::steady_clock::now();
    std::cout << N << "," << n << "," << argv[5] << "," << "array_reduction" << "," <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "," << myQueue.get_device().get_info<info::device::name>() << "\n";
#endif

//TODO measure all possible combinations of alpha, beta

#ifndef MEASURE
    for (int i=0;i<NPT;i++) std::cout << " " << output[i] << "\n";
#endif

}
