/*
build:
  g++ -std=c++14 -O2 -lgmsh -o rectangle.exe rectangle.cpp
run:
  ./rectangle
 */
#include <gmsh.h>

int main() {
  gmsh::initialize();
  gmsh::open("rectangle.msh");
  gmsh::clear();
  gmsh::finalize();
}
