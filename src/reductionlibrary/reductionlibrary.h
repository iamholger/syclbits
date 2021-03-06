#include <CL/sycl.hpp>
using namespace sycl;

// Input array and output reference of the same type
// Need total number of elements N
// operation is any of these supported std::functional operations
// std::plus<>(), std::multiplies<>(), std::bit_and, bit_or, bit_xor, logical_and, logical_or
// sycl further defines maximum<>() and minimum<>()
template <class T, class F>
void reduce_to_scalar(sycl::queue Q, T * input, T & output, const size_t N, F operation)
{
    T temp(output);
    buffer<T> BUF (&temp, 1);

    Q.submit([&](handler& cgh)
    {
      auto RED = reduction(BUF, cgh, operation);

      cgh.parallel_for(range<1>(N), RED, [=](id<1> i, auto & red)
      {
          red.combine(input[i]);
      });
    }).wait();
    output = BUF.get_host_access()[0];
}

template <class T, class F>
void reduce_to_array(sycl::queue Q, const size_t Nwant, size_t Nhave, T * input, F operation)
{
    static const auto WGMAX = Q.get_device().get_info<info::device::max_work_group_size>();
    while ( (Nhave/Nwant) > WGMAX ) // Keep reducing until the gathersize is <= WGMAX (device specific)
    {
#ifdef DEBUG
        std::cout << Nhave << " The next gather size  " << Nhave/Nwant << " is > WGMAX ---> need to iterate\n";
#endif
        size_t gathersize(WGMAX);
        while ( not ( (Nhave/gathersize)%Nwant == 0 and (Nhave%gathersize==0) ) )  gathersize--;
#ifdef DEBUG
        std::cout << "reduce " << Nhave << " " << gathersize << " " << Nhave%gathersize << "\n";
#endif
        Q.submit([&](handler & cgh)
        {
          cgh.parallel_for(nd_range<1>{{Nhave}, {gathersize}}, [=](nd_item<1> it)
          {
            input[it.get_group(0)] = reduce_over_group(it.get_group(), input[it.get_global_id(0)], operation);
          });
        }).wait();
        Nhave /= gathersize;
    }

    const size_t gathersize = Nhave/Nwant;
#ifdef DEBUG
    std::cout << "The final gather size " << gathersize << " is <= WGMAX (" << WGMAX << ") for the remaining " << Nhave << " items\n";
#endif

    const size_t Nhavefinal(Nhave);
    Q.submit([&](handler & cgh)
    {
      cgh.parallel_for(nd_range<1>{{Nhavefinal}, {gathersize}}, [=](nd_item<1> it)
      {
        input[it.get_group_linear_id()] = reduce_over_group(it.get_group(), input[it.get_global_id(0)], operation);
      });
    }).wait();
}



// Main reducer --- keep in mind that this one changes input, also Nhave is modified!
template <class T, class F>
void reduce_to_array(sycl::queue Q, const size_t Nwant, size_t Nhave, T * input, T * output, F operation)
{
    static const auto WGMAX =  Q.get_device().get_info<info::device::max_work_group_size>();
    while ( (Nhave/Nwant) > WGMAX ) // Keep reducing until the gathersize is <= WGMAX (device specific)
    {
#ifdef DEBUG
        std::cout << Nhave << " The next gather size  " << Nhave/Nwant << " is > WGMAX ---> need to iterate\n";
#endif
        size_t gathersize(WGMAX);
        while ( not ( (Nhave/gathersize)%Nwant == 0 and (Nhave%gathersize==0) ) )  gathersize--;
#ifdef DEBUG
        std::cout << "reduce " << Nhave << " " << gathersize << " " << Nhave%gathersize << "\n";
#endif
        Q.submit([&](handler & cgh)
        {
          cgh.parallel_for(nd_range<1>{{Nhave}, {gathersize}}, [=](nd_item<1> it) 
          {
            //input[it.get_group_linear_id()] = reduce_over_group(it.get_group(), input[it.get_global_linear_id()], operation);
            auto temp = reduce_over_group(it.get_group(), input[it.get_global_linear_id()], operation);
            input[it.get_group_linear_id()] = temp;
          });
        }).wait();
        Nhave /= gathersize;
    }
#ifdef DEBUG
    std::cout << "The next gather size " << Nhave/Nwant << " is <= WGMAX (" << WGMAX << ") so final reduction\n";
#endif

    std::cout << "oha: " << input[0] << " and " << output[0] << " now: " << Nhave << "\n";

    size_t gathersize = Nhave/Nwant;
    // Note how in the final reduction we can reduce directly into the output buffer
    Q.submit([&](handler & cgh)
    {
      cgh.parallel_for(nd_range<1>{{Nhave}, {gathersize}}, [=](nd_item<1> it) 
      {
        auto wg =  it.get_group();
        //output[it.get_group_linear_id()] = reduce_over_group(wg, input[it.get_global_linear_id()], operation);
        auto temp = reduce_over_group(wg, input[it.get_global_linear_id()], operation);
        output[it.get_group_linear_id()] = temp;
      });
    }).wait();
}

// Main reducer --- uses explicit reduction buffer i.e. input is unchanged, also Nhave is modified!
template <class T, class F>
void reduce_to_array(sycl::queue Q, const size_t Nwant, size_t Nhave, T * input, T * output, T * redbuf, F operation)
{
    static const auto WGMAX =  Q.get_device().get_info<info::device::max_work_group_size>();
    int round(0);
    while ( (Nhave/Nwant) > WGMAX ) // Keep reducing until the gathersize is <= WGMAX (device specific)
    {
#ifdef DEBUG
        std::cout << "The next gather size  " << Nhave/Nwant << " is > WGMAX ---> need to iterate\n";
#endif
        size_t gathersize(WGMAX);
        while ( not ( (Nhave/gathersize)%Nwant == 0 and (Nhave%gathersize==0)) )  gathersize--;
#ifdef DEBUG
        std::cout << "reduce " << Nhave << " " << gathersize << "\n";
#endif
        if (round==0)
        {
          Q.submit([&](handler & cgh)
          {
            cgh.parallel_for(nd_range<1>{{Nhave}, {gathersize}}, [=](nd_item<1> it) 
            {
              redbuf[it.get_group_linear_id()] = reduce_over_group(it.get_group(), input[it.get_global_linear_id()], operation);
            });
          }).wait();
        }
        else {
          Q.submit([&](handler & cgh)
          {
            cgh.parallel_for(nd_range<1>{{Nhave}, {gathersize}}, [=](nd_item<1> it) 
            {
              redbuf[it.get_group_linear_id()] = reduce_over_group(it.get_group(), redbuf[it.get_global_linear_id()], operation);
            });
          }).wait();
        }
        Nhave /= gathersize;
        round++;
    }
#ifdef DEBUG
    std::cout << "The next gather size " << Nhave/Nwant << " is <= WGMAX (" << WGMAX << ") so final reduction\n";
#endif

    size_t gathersize = Nhave/Nwant;
    // Note how in the final reduction we can reduce directly into the output buffer
    Q.submit([&](handler & cgh)
    {
      cgh.parallel_for(nd_range<1>{{Nhave}, {gathersize}}, [=](nd_item<1> it) 
      {
        auto wg =  it.get_group();
        if (round==0) output[it.get_group_linear_id()] = reduce_over_group(wg, input[it.get_global_linear_id()], operation);
        else          output[it.get_group_linear_id()] = reduce_over_group(wg, redbuf[it.get_global_linear_id()], operation);
      });
    }).wait();
}

// Deprecated
//
// The most flexible one using nd_range<2>
// Initially, we have an array of length N*n
// We reduce to an array of length N where the reduction happens over the n sub elements
// We need to specify an initial gathersize (workgroup size to be precise)
// The only requirements are that n%wgsize==0 and wgsize < device's max work group size
template <class T, class F>
void reduce_to_array(sycl::queue Q, const size_t N, const size_t n, const size_t wgsize, T * input, T * output, F operation)
{

    const size_t nitems = N*n;
    size_t reduced_size(nitems/wgsize);
   
    std::cout<< reduced_size << " red_size\n";
    // Reducing nitems -> nitems/(wgsize) 
    Q.submit([&](handler & cgh) {
          cgh.parallel_for(nd_range<2>{{N, n}, {1, wgsize}},
            [=](nd_item<2> it) 
          {
              input[it.get_group_linear_id()] = reduce_over_group(it.get_group(), input[it.get_global_linear_id()], operation);
          });
    }).wait();

    for (int i=0;i<reduced_size;i++) std::cout << "See " << input[i] << "\n";

    reduce_to_array<T>(Q, N, reduced_size, input, output, operation);
}

// Deprecated
// Uses a reduction buffer, i.e. won't change input
template <class T, class F>
void reduce_to_array(sycl::queue Q, const size_t N, const size_t n, const size_t wgsize, T * input, T * output, T* redbuf, F operation)
{

    const size_t nitems = N*n;
    // Device allocated reduction buffer
    size_t reduced_size(nitems/wgsize);
   
    // Reducing nitems -> nitems/(wgsize) 
    Q.submit([&](handler & cgh) {
          cgh.parallel_for(nd_range<2>{{N, n}, {1, wgsize}},
            [=](nd_item<2> it) 
          {
              redbuf[it.get_group_linear_id()] = reduce_over_group(it.get_group(), input[it.get_global_linear_id()], operation);
          });
    }).wait();

    reduce_to_array<double>(Q, N, reduced_size, redbuf, output, operation);
}



// Deprecated
//
// Reduce data that is layed out flat in N pieces of size 3*n^3 into an array of size N
template <class T, class F>
void reduce_to_array(sycl::queue Q, const size_t N, const size_t n, T * input, T * output, const size_t alpha, const size_t beta, F operation)
{

    const size_t nitems = 3*n*n*n*N;
    size_t gathersize(alpha*beta), reduced_size(nitems/gathersize);
    auto redbuf = malloc_device<T>(reduced_size, Q);
   
    // First reduction --- reducing nitems -> nitems/(alpha*beta) 
    Q.submit([&](handler & cgh) {
          cgh.parallel_for(nd_range<3>{{N, n*n, 3*n}, {1,alpha,beta}},
            [=](nd_item<3> it) 
          {
              redbuf[it.get_group_linear_id()] = reduce_over_group(it.get_group(), input[it.get_global_linear_id()], operation);
          });
    }).wait();

    reduce_to_array<double>(Q, N, reduced_size, redbuf, output, operation);
}

// Deprecated
//
template <class T, class F>
void reduce_to_array2(sycl::queue Q, const size_t N, const size_t n, T * input, T * output, F operation)
{

    const auto   WGMAX =  Q.get_device().get_info<info::device::max_work_group_size>();
    const size_t nitems = 3*n*n*n*N;
    size_t reduced_size(nitems/n);
    auto redbuf = malloc_device<T>(reduced_size, Q);
   
    // First reduction --- reducing nitems -> nitems/(alpha*beta) 
    Q.submit([&](handler & cgh) {
          cgh.parallel_for(nd_range<2>{{N, 3*n*n*n}, {1,n}},
            [=](nd_item<2> it) 
          {
              redbuf[it.get_group_linear_id()] = reduce_over_group(it.get_group(), input[it.get_global_linear_id()], operation);
          });
    }).wait();

    reduce_to_array<double>(Q, N, reduced_size, redbuf, output, operation);
}




// Expects and input array of N meaningful data elements and a total size of Npadded
// We require the gathersize to be a factor of Npadded
//
// Explicitly copy Npadded by value as we manipulate it
template<class T, class F>
void padded_reduction_to_scalar_with_fixed_gather_size(sycl::queue Q, const size_t N, size_t Npadded, const size_t gathersize, T * input, T * output, F operation)
{
    // The number of reduction rounds --- N = n^rounds
    const size_t rounds = ceil(log(N)/log(gathersize));

#ifdef DEBUG
    std::cout << "Will do " << rounds << " rounds of reduction\n";
    std::cout << "Npadded: " << Npadded << "\n";
    std::cout << "N: " << N << "\n";
#endif
   
    for (int round=0;round<rounds;round++)
    {
        
#ifdef DEBUG
        std::cout << "Round " << round+1 << " reduces " << Npadded << " items\n";
        std::cout << "The reduction size is " << gathersize << "\n";
        std::cout << "The reduced size is " << Npadded/gathersize << "\n";
#endif

        auto myrange = nd_range<1>{{Npadded}, {gathersize}};
        Q.submit([&](handler & cgh)
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
            input[it.get_group_linear_id()] = reduce_over_group(wg, input[it.get_global_linear_id()], operation);
          });
        }).wait();

        Npadded /= gathersize;

        if (Npadded%gathersize != 0) Npadded = (Npadded/gathersize + 1) * gathersize;
    }

    Q.memcpy(output, input, sizeof(T));
    Q.wait();
}

