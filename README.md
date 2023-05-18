nvcc -o image image.cu $(pkg-configs --libs --cflags opencv) -std=c++11
eog out.png
