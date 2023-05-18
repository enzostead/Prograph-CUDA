#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

/**
 * Kernel pour transformer l'image RGB en niveaux de gris.
 */
__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }
}

/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris.
 */
__global__ void sobel( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows )
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if( i > 1 && i < cols && j > 1 && j < rows )
  {
    auto h =     g[ (j-1)*cols + i - 1 ] -     g[ (j-1)*cols + i + 1 ]
           + 2 * g[ (j  )*cols + i - 1 ] - 2 * g[ (j  )*cols + i + 1 ]
           +     g[ (j+1)*cols + i - 1 ] -     g[ (j+1)*cols + i + 1 ];

    auto v =     g[ (j-1)*cols + i - 1 ] -     g[ (j+1)*cols + i - 1 ]
           + 2 * g[ (j-1)*cols + i     ] - 2 * g[ (j+1)*cols + i     ]
           +     g[ (j-1)*cols + i + 1 ] -     g[ (j+1)*cols + i + 1 ];

    auto res = h*h + v*v;
    res = res > 65535 ? res = 65535 : res;

    s[ j * cols + i ] = sqrtf( res );
  }
}


/**
 * Kernel pour obtenir les contours à partir de l'image en niveaux de gris, en utilisant la mémoire shared
 * pour limiter les accès à la mémoire globale.
 */
__global__ void sobel_shared( unsigned char * g, unsigned char * s, std::size_t cols, std::size_t rows )
{
  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  extern __shared__ unsigned char sh[];

  if( i < cols && j < rows )
  {
    sh[ lj * w + li ] = g[ j * cols + i ];
  }

  __syncthreads();

  if( i < cols -1 && j < rows-1 && li > 0 && li < (w-1) && lj > 0 && lj < (h-1) )
  {
    auto h =     sh[ (lj-1)*w + li - 1 ] -     sh[ (lj-1)*w + li + 1 ]
           + 2 * sh[ (lj  )*w + li - 1 ] - 2 * sh[ (lj  )*w + li + 1 ]
           +     sh[ (lj+1)*w + li - 1 ] -     sh[ (lj+1)*w + li + 1 ];

    auto v =     sh[ (lj-1)*w + li - 1 ] -     sh[ (lj+1)*w + li - 1 ]
           + 2 * sh[ (lj-1)*w + li     ] - 2 * sh[ (lj+1)*w + li     ]
           +     sh[ (lj-1)*w + li + 1 ] -     sh[ (lj+1)*w + li + 1 ];

    auto res = h*h + v*v;
    res = res > 65535 ? res = 65535 : res;

    s[ j * cols + i ] = sqrtf( res );
  }
}


/**
 * Kernel fusionnant le passage en niveaux de gris et la détection de contours.
 */
__global__ void grayscale_sobel_shared( unsigned char * rgb, unsigned char * s, std::size_t cols, std::size_t rows ) {
  auto i = blockIdx.x * (blockDim.x-2) + threadIdx.x;
  auto j = blockIdx.y * (blockDim.y-2) + threadIdx.y;

  auto li = threadIdx.x;
  auto lj = threadIdx.y;

  auto w = blockDim.x;
  auto h = blockDim.y;

  extern __shared__ unsigned char sh[];

  if( i < cols && j < rows ) {
    sh[ lj * w + li ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) >> 10;
  }

  /**
   * Il faut synchroniser tous les warps (threads) du bloc pour être certain que le niveau de gris est calculé
   * par tous les threads du bloc avant de pouvoir accéder aux données des pixels voisins.
   */
  __syncthreads();

  if( i < cols -1 && j < rows-1 && li > 0 && li < (w-1) && lj > 0 && lj < (h-1) )
  {
    auto hr =     sh[ (lj-1)*w + li - 1 ] -     sh[ (lj-1)*w + li + 1 ]
           + 2 * sh[ (lj  )*w + li - 1 ] - 2 * sh[ (lj  )*w + li + 1 ]
           +     sh[ (lj+1)*w + li - 1 ] -     sh[ (lj+1)*w + li + 1 ];

    auto vr =     sh[ (lj-1)*w + li - 1 ] -     sh[ (lj+1)*w + li - 1 ]
           + 2 * sh[ (lj-1)*w + li     ] - 2 * sh[ (lj+1)*w + li     ]
           +     sh[ (lj-1)*w + li + 1 ] -     sh[ (lj+1)*w + li + 1 ];

    auto res = hr*hr + vr*vr;
    res = res > 65535 ? res = 65535 : res;

    s[ j * cols + i ] = sqrtf( res );
  }
}


int main()
{
  cv::Mat m_in = cv::imread("in2.jpg", cv::IMREAD_UNCHANGED );

  //auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;






  /*int blockSize, gridSize;
  cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, grayscale_sobel_shared, 0, size);
  dim3 block(blockSize, 1, 1);
  dim3 grid1((size + blockSize - 1) / blockSize, 1, 1);*/

  //dim3 block( 32, 4 );
  //dim3 grid0( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );
  /**
   * Pour la version shared il faut faire superposer les blocs de 2 pixels
   * pour ne pas avoir de bandes non calculées autour des blocs
   * on crée donc plus de blocs.
   */
  //dim3 grid1( ( cols - 1) / (block.x-2) + 1 , ( rows - 1 ) / (block.y-2) + 1 );

  int best_block = -1;
  int best_grid = -1;
  float bestTime;

  for (int block = 32; block<=256; block+=32){
    for (int grid = 1; grid<=4; grid++){

      //std::vector< unsigned char > g( rows * cols );
      // Allocation de l'image de sortie en RAM côté CPU.
      unsigned char * g = nullptr;
      cudaMallocHost( &g, rows * cols );
      cv::Mat m_out( rows, cols, CV_8UC1, g );

      // Copie de l'image en entrée dans une mémoire dite "pinned" de manière à accélérer les transferts.
      // OpenCV alloue la mémoire en interne lors de la décompression de l'image donc soit sans doute avec
      // un malloc standard.
      unsigned char * rgb = nullptr;
      cudaMallocHost( &rgb, 3 * rows * cols );

      std::memcpy( rgb, m_in.data, 3 * rows * cols );

      unsigned char * rgb_d;
      unsigned char * g_d;
      unsigned char * s_d;

      cudaMalloc( &rgb_d, 3 * rows * cols );
      cudaMalloc( &g_d, rows * cols );
      cudaMalloc( &s_d, rows * cols );

      auto size = 3 * rows * cols;

      cudaStream_t stream1, stream2, stream3, stream4;
      cudaStreamCreate(&stream1);
      cudaStreamCreate(&stream2);
      cudaStreamCreate(&stream3);
      cudaStreamCreate(&stream4);

      cudaMemcpyAsync( rgb_d, rgb, (size)/4, cudaMemcpyHostToDevice, stream1);
      cudaMemcpyAsync( rgb_d+size/4, rgb+size/4, (size)/4, cudaMemcpyHostToDevice, stream2);
      cudaMemcpyAsync( rgb_d+2*(size/4), rgb+2*(size/4), (size)/4, cudaMemcpyHostToDevice, stream3);
      cudaMemcpyAsync( rgb_d+3*(size/4), rgb+3*(size/4), (size)/4, cudaMemcpyHostToDevice, stream4);

      cudaEvent_t start, stop;

      cudaEventCreate( &start );
      cudaEventCreate( &stop );

      // Mesure du temps de calcul du kernel uniquement.
      cudaEventRecord( start );

      grayscale_sobel_shared<<< grid, block, 0, stream1 >>>( rgb_d, s_d, cols, rows );
      grayscale_sobel_shared<<< grid, block, 0, stream2 >>>( rgb_d, s_d, cols, rows );
      grayscale_sobel_shared<<< grid, block, 0, stream3 >>>( rgb_d, s_d, cols, rows );
      grayscale_sobel_shared<<< grid, block, 0, stream4 >>>( rgb_d, s_d, cols, rows );

      cudaStreamSynchronize(stream1);
      cudaStreamSynchronize(stream2);
      cudaStreamSynchronize(stream3);
      cudaStreamSynchronize(stream4);

      // Vérification des erreurs
      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess) {
          printf("Erreur CUDA: %s\n", cudaGetErrorString(error));
      }

      cudaEventRecord( stop );

      cudaMemcpy( g, s_d, rows * cols, cudaMemcpyDeviceToHost );

      cudaEventSynchronize( stop );
      float duration;
      cudaEventElapsedTime( &duration, start, stop );
      //std::cout << "time=" << duration << std::endl;
      if(grid == 1 && block == 32)
        bestTime=duration;
      else if(duration<bestTime){
        bestTime=duration;
        best_grid = grid;
        best_block = block;
      }


      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      cv::imwrite( "out.jpg", m_out );

      cudaFree( rgb_d);
      cudaFree( g_d);
      cudaFree( s_d);

      cudaStreamDestroy(stream1);
      cudaStreamDestroy(stream2);
      cudaStreamDestroy(stream3);
      cudaStreamDestroy(stream4);

      cudaFreeHost( g );
      cudaFreeHost( rgb );

    }
  }

  std::cout << "Best time : " << bestTime << "ms with block="<<best_block<<" and grid="<<best_grid<<std::endl;


  // Version en 2 étapes.
  /*grayscale<<< grid0, block, 0, stream1 >>>(rgb_d, g_d, cols, rows );
  grayscale<<< grid0, block, 0, stream2 >>>(rgb_d, g_d, cols, rows );
  grayscale<<< grid0, block, 0, stream3 >>>(rgb_d, g_d, cols, rows );
  grayscale<<< grid0, block, 0, stream4 >>>(rgb_d, g_d, cols, rows );

  sobel<<< grid0, block, 0, stream1 >>>( g_d, s_d, cols, rows );
  sobel<<< grid0, block, 0, stream2 >>>( g_d, s_d, cols, rows );
  sobel<<< grid0, block, 0, stream3 >>>( g_d, s_d, cols, rows );
  sobel<<< grid0, block, 0, stream4 >>>( g_d, s_d, cols, rows );*/




  // Version en 2 étapes, Sobel avec mémoire shared.
  /*grayscale<<< grid0, block, 0, stream1 >>>( rgb_d, g_d, cols, rows );
  grayscale<<< grid0, block, 0, stream2 >>>( rgb_d, g_d, cols, rows );
  grayscale<<< grid0, block, 0, stream3 >>>( rgb_d, g_d, cols, rows );
  grayscale<<< grid0, block, 0, stream4 >>>( rgb_d, g_d, cols, rows );


  sobel_shared<<< grid1, block, block.x * block.y, stream1 >>>( g_d, s_d, cols, rows );
  sobel_shared<<< grid1, block, block.x * block.y, stream2 >>>( g_d, s_d, cols, rows );
  sobel_shared<<< grid1, block, block.x * block.y, stream3 >>>( g_d, s_d, cols, rows );
  sobel_shared<<< grid1, block, block.x * block.y, stream4 >>>( g_d, s_d, cols, rows );
  */

  // Version fusionnée.
  /*grayscale_sobel_shared<<< grid1, block, block.x * block.y, stream1 >>>( rgb_d, s_d, cols, rows );
  grayscale_sobel_shared<<< grid1, block, block.x * block.y, stream2 >>>( rgb_d, s_d, cols, rows );
  grayscale_sobel_shared<<< grid1, block, block.x * block.y, stream3 >>>( rgb_d, s_d, cols, rows );
  grayscale_sobel_shared<<< grid1, block, block.x * block.y, stream4 >>>( rgb_d, s_d, cols, rows );
*/






  return 0;
}
