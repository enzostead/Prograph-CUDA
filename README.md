Un Makefile est à disposition pour faciliter la compilation

in.jpg est une image relativement petite. in2.jpg est une très grande image. Ces deux images ont été utilisées pour réaliser nos tests.

Commande pour la compilation fichier par fichier :
nvcc -o image image.cu $(pkg-configs --libs --cflags opencv) -std=c++11
eog out.png
