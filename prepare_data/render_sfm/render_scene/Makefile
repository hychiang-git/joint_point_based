CPPFLAGS= -Wall -O2 -std=c++11 -fopenmp -lpthread
RENDER_CPP=render.cpp
UTILS=tinyply.cpp triangleCube.cpp polygon_3darea.cpp
INCLUDES=/tmp3/hychiang/eigen-eigen-b3f3d4950030

ALL:
	g++ $(CPPFLAGS) $(RENDER_CPP) $(UTILS) -I $(INCLUDES) -o render
