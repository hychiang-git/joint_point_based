#include <math.h>
#include <vector>
#ifndef triangleCube_h
#define triangleCube_h
#define DOT(A, B) ((A).x * (B).x + (A).y * (B).y + (A).z * (B).z)
#define CROSS( A, B, C ) { \
  (C).x =  (A).y * (B).z - (A).z * (B).y; \
  (C).y = -(A).x * (B).z + (A).z * (B).x; \
  (C).z =  (A).x * (B).y - (A).y * (B).x; \
   }
#define SUB( A, B, C ) { \
  (C).x =  (A).x - (B).x; \
  (C).y =  (A).y - (B).y; \
  (C).z =  (A).z - (B).z; \
   }

#define ADD( A, B, C ) { \
  (C).x =  (A).x + (B).x; \
  (C).y =  (A).y + (B).y; \
  (C).z =  (A).z + (B).z; \
   }
#define LERP( A, B, C) ((B)+(A)*((C)-(B)))
#define MIN3(a,b,c) ((((a)<(b))&&((a)<(c))) ? (a) : (((b)<(c)) ? (b) : (c)))
#define MAX3(a,b,c) ((((a)>(b))&&((a)>(c))) ? (a) : (((b)>(c)) ? (b) : (c)))
#define INSIDE 0
#define OUTSIDE 1

typedef struct Point3 {
    Point3 () {}
    Point3 (float _x, float _y, float _z): x(_x), y(_y), z(_z) {} 
    float           x;
    float           y;
    float           z;
} Point3;

typedef struct{
   Point3 v1;                 /* Vertex1 */
   Point3 v2;                 /* Vertex2 */
   Point3 v3;                 /* Vertex3 */
   } Triangle3;

long t_c_intersection(Triangle3); 
float area3D_Polygon(int, std::vector<Point3>&, Point3);
float triangle_area_inside_cube(Triangle3);
#endif
