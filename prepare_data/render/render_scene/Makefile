#CPPFLAGS=-Wall -O2 -std=c++14
#EIGEN_INC=/tmp3/hychiang/eigen-eigen-b3f3d4950030	## path to your eigen library
#INC=/home/master/05/hychiang/.local/include/   	## path to include, usually /usr/local/include
#LIB=/home/master/05/hychiang/.local/lib/		## path to include, usually /usr/local/lib

CPPFLAGS=-Wall -O2 -std=c++14
EIGEN_INC=/path/to/eigen
INC=/usr/local/include
LIB=/usr/local/lib


sfm_pose: sfm_pose.o render.o tinyply.o
	g++ $(CPPFLAGS) sfm_pose.o render.o tinyply.o -o $@ -L $(LIB) -fopenmp -lpthread -lz -lopenblas -llapack -lOpenImageIO
	rm *.o

sfm_pose.o: sfm_pose.cpp
	g++ $(CPPFLAGS) -I $(INC) -I $(EIGEN_INC) -c $< -o $@ -fopenmp

syn_pose: syn_pose.o render.o tinyply.o
	g++ $(CPPFLAGS) syn_pose.o render.o tinyply.o -o $@ -L $(LIB) -fopenmp -lpthread -lz -lopenblas -llapack -lOpenImageIO
	rm *.o

syn_pose.o: syn_pose.cpp
	g++ $(CPPFLAGS) -I $(INC) -I $(EIGEN_INC) -c $< -o $@ -fopenmp

render.o: render.cpp
	g++ $(CPPFLAGS) -I $(INC) -I $(EIGEN_INC) -c $< -o $@

tinyply.o: tinyply.cpp
	g++ $(CPPFLAGS) -c $< -o $@ 

clean:
	rm syn_pose sfm_pose
