#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>


int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;

  // Traitement de l'image
  unsigned char* out_grey = new unsigned char[ m_in.rows * m_in.cols ];
  unsigned char* out_sobel = new unsigned char[m_in.rows * m_in.cols ];

  auto start = std::chrono::system_clock::now();

  int height = m_in.rows;
  int width = m_in.cols;

  unsigned int i, j, c;

  int h, v, res;

 #pragma omp parallel for
  for( std::size_t j = 0 ; j < m_in.rows ; ++j )
    {
      for( std::size_t i = 0 ; i < m_in.cols ; ++i )
	{
	  out_grey[ j * m_in.cols + i ] = (
			 307 * rgb[ 3 * ( j * m_in.cols + i ) ]
		       + 604 * rgb[ 3 * ( j * m_in.cols + i ) + 1 ]
		       + 113 * rgb[  3 * ( j * m_in.cols + i ) + 2 ]
		       ) / 1024;
	}
    }


    for(j = 1 ; j < height - 1 ; ++j) {

        for(i = 1 ; i < width - 1 ; ++i) {

            // Horizontal
            h = -1 * out_grey[((j - 1) * width + i - 1)] + out_grey[((j - 1) * width + i + 1)]
                - 2 * out_grey[(j * width + i - 1)] + 2 * out_grey[(j * width + i + 1)]
                - out_grey[((j + 1) * width + i - 1)] + out_grey[((j + 1) * width + i + 1)];

            // Vertical

            v = -1 * out_grey[((j - 1) * width + i - 1)] + out_grey[((j + 1) * width + i - 1)]
                - 2 * out_grey[((j - 1) * width + i)] + 2 * out_grey[((j + 1) * width + i)]
                - out_grey[((j - 1) * width + i + 1)] + out_grey[((j + 1) * width + i + 1)];

            //h = h > 255 ? 255 : h;
            //v = v > 255 ? 255 : v;

            res = h * h + v * v;
            res = res > 255 * 255 ? res = 255 * 255 : res;
            out_sobel[ j * width + i ] = sqrt(res);
        }

   }
    
  cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, out_sobel );
  
  auto stop = std::chrono::system_clock::now();

  auto duration = stop - start;
  auto ms = std::chrono::duration_cast< std::chrono::milliseconds >( duration ).count();

  std::cout << ms << " ms" << std::endl;
  
  cv::imwrite( "out_sobel.jpg", m_out );
  
  return 0;
}
