#define cimg_display  0 // This gets around need to link libX11
#define cimg_use_png 1  // required for PNG images, needs linking to libpng
#include "CImg.h"
using namespace cimg_library;

#include <CL/sycl.hpp>
using namespace sycl;
static queue mQ(default_selector{});

template <class T>
T* read_png(const char* fname, int & width, int & height)
{
  // Read an image with 3 channels (3 values per pixel)
  CImg<unsigned char> img = CImg<unsigned char>(fname);
  width  = img.width();
  height = img.height();

  // linear array with enough space for all rgb values
  auto buffer = malloc_shared<T>(width*height*3,mQ);
  
  int pos(0);
  for (int r = 0; r < height; r++)
  for (int c = 0; c < width; c++)
  for (int x = 0; x < 3; x++) // innermost loop over RGB
  {
    buffer[pos] = (int)img(c,r,0,x); // copy image data into array
    pos++;
  }
  return buffer;
} 

template <class T>
void gray_array(sycl::queue Q, T * input, T* output, const size_t width, const size_t height)
{
  Q.submit([&](handler & cgh) // Kernel submission
  {
    cgh.parallel_for(range<2>{height, width}, [=](item<2> it) // 2D iteration space
    {
      int r = it.get_id(0); // that item's row (global)
      int c = it.get_id(1); // that item's column (global)
      // The RGB values are next to each other in the input array
      // Per pixel, write 0.3*R + 0.59*G + 0.11*B
      output[r*width + c] = 0.3*input[3*(r*width + c)] + 0.59*input[3*(r*width + c) + 1] + 0.11*input[3*(r*width + c) + 2];
    });
  }).wait(); // Synchronisation
} 

template <class T>
void write_grayscale_png(const char* fname, T * buffer, int & width, int & height)
{
  CImg<unsigned char> img(width, height, 1, 1, 0); // output image, all black
  for (int r = 0; r < height; r++)
  for (int c = 0; c < width;  c++) img(c,r) = buffer[r*width + c]; // set pixel (c,r)
  img.save(fname); // Write to file
} 

int main(int argc, char * argv[])
{
  assert(strcmp(argv[1], argv[2]) !=0); // Make sure we don't overwrite the input file

  int width(0), height(0);
  auto input = read_png<int>(argv[1], width, height); // Read in data
  auto output = malloc_shared<int>(width*height,mQ);  // USM array for grayscale image 
  
  gray_array<int>(mQ, input, output, width, height);  // The actual kernel
  write_grayscale_png<int>(argv[2], output, width, height); // Write output
  
#ifdef MEASURE
    auto start = std::chrono::steady_clock::now();   
    for (int i=0;i<std::atoi(argv[3]);i++) 
    gray_array<int>(mQ, input, output, width, height);
    auto end = std::chrono::steady_clock::now();
    std::cout << argv[3] << " iterations took " <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms on device" << mQ.get_device().get_info<info::device::name>() << "\n";
#endif

  free(input, mQ);
  free(output, mQ);
  return 0;
}

