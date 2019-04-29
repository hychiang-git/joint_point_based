#include "triangleCube.h"
#include <vector>
using namespace std;
// Copyright 2000 softSurfer, 2012 Dan Sunday
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// iSurfer.org makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.
 


// a Triangle is given by three points: Point3 V0, V1, V2 

// a Polygon is given by:
//       int n = number of vertex points
//       Point3* V[] = an array of n+1 vertex points with V[n]=V[0]
 

// Note: for efficiency low-level short functions are declared to be inline.


// isLeft(): test if a point is Left|On|Right of an infinite 2D line.
//    Input:  three points P0, P1, and P2
//    Return: >0 for P2 left of the line through P0 to P1
//          =0 for P2 on the line
//          <0 for P2 right of the line
inline int
isLeft( Point3 P0, Point3 P1, Point3 P2 )
{
    return ( (P1.x - P0.x) * (P2.y - P0.y)
           - (P2.x - P0.x) * (P1.y - P0.y) );
}
//===================================================================


// orientation2D_Triangle(): test the orientation of a 2D triangle
//  Input:  three vertex points V0, V1, V2
//  Return: >0 for counterclockwise 
//          =0 for none (degenerate)
//          <0 for clockwise
inline int
orientation2D_Triangle( Point3 V0, Point3 V1, Point3 V2 )
{
    return isLeft(V0, V1, V2);
}
//===================================================================


// area2D_Triangle(): compute the area of a 2D triangle
//  Input:  three vertex points V0, V1, V2
//  Return: the (float) area of triangle T
inline float
area2D_Triangle( Point3 V0, Point3 V1, Point3 V2 )
{
    return (float)isLeft(V0, V1, V2) / 2.0;
}
//===================================================================


// orientation2D_Polygon(): test the orientation of a simple 2D polygon
//  Input:  int n = the number of vertices in the polygon
//          Point3* V = an array of n+1 vertex points with V[n]=V[0]
//  Return: >0 for counterclockwise 
//          =0 for none (degenerate)
//          <0 for clockwise
//  Note: this algorithm is faster than computing the signed area.
int
orientation2D_Polygon( int n, vector<Point3>& V )
{
    // first find rightmost lowest vertex of the polygon
    int rmin = 0;
    int xmin = V[0].x;
    int ymin = V[0].y;

    for (int i=1; i<n; i++) {
        if (V[i].y > ymin)
            continue;
        if (V[i].y == ymin) {   // just as low
            if (V[i].x < xmin)  // and to left
                continue;
        }
        rmin = i;      // a new rightmost lowest vertex
        xmin = V[i].x;
        ymin = V[i].y;
    }

    // test orientation at the rmin vertex
    // ccw <=> the edge leaving V[rmin] is left of the entering edge
    if (rmin == 0)
        return isLeft( V[n-1], V[0], V[1] );
    else
        return isLeft( V[rmin-1], V[rmin], V[rmin+1] );
}
 //===================================================================


// area2D_Polygon(): compute the area of a 2D polygon
//  Input:  int n = the number of vertices in the polygon
//          Point3* V = an array of n+1 vertex points with V[n]=V[0]
//  Return: the (float) area of the polygon
float
area2D_Polygon( int n, vector<Point3>& V )
{
    float area = 0;
    int  i, j, k;   // indices

    if (n < 3) return 0;  // a degenerate polygon

    for (i=1, j=2, k=0; i<n; i++, j++, k++) {
        area += V[i].x * (V[j].y - V[k].y);
    }
    area += V[n].x * (V[1].y - V[n-1].y);  // wrap-around term
    return area / 2.0;
}
//===================================================================


// area3D_Polygon(): compute the area of a 3D planar polygon
//  Input:  int n = the number of vertices in the polygon
//          Point3* V = an array of n+1 points in a 2D plane with V[n]=V[0]
//          Point3 N = a normal vector of the polygon's plane
//  Return: the (float) area of the polygon
float
area3D_Polygon( int n, vector<Point3>& V, Point3 N )
{
    float area = 0;
    float an, ax, ay, az; // abs value of normal and its coords
    int  coord;           // coord to ignore: 1=x, 2=y, 3=z
    int  i, j, k;         // loop indices

    if (n < 3) return 0;  // a degenerate polygon

    // select largest abs coordinate to ignore for projection
    ax = (N.x>0 ? N.x : -N.x);    // abs x-coord
    ay = (N.y>0 ? N.y : -N.y);    // abs y-coord
    az = (N.z>0 ? N.z : -N.z);    // abs z-coord

    coord = 3;                    // ignore z-coord
    if (ax > ay) {
        if (ax > az) coord = 1;   // ignore x-coord
    }
    else if (ay > az) coord = 2;  // ignore y-coord

    // compute area of the 2D projection
    switch (coord) {
      case 1:
        for (i=1, j=2, k=0; i<n; i++, j++, k++)
            area += (V[i].y * (V[j].z - V[k].z));
        break;
      case 2:
        for (i=1, j=2, k=0; i<n; i++, j++, k++)
            area += (V[i].z * (V[j].x - V[k].x));
        break;
      case 3:
        for (i=1, j=2, k=0; i<n; i++, j++, k++)
            area += (V[i].x * (V[j].y - V[k].y));
        break;
    }
    switch (coord) {    // wrap-around term
      case 1:
        area += (V[n].y * (V[1].z - V[n-1].z));
        break;
      case 2:
        area += (V[n].z * (V[1].x - V[n-1].x));
        break;
      case 3:
        area += (V[n].x * (V[1].y - V[n-1].y));
        break;
    }

    // scale to get area before projection
    an = sqrt( ax*ax + ay*ay + az*az); // length of normal vector
    switch (coord) {
      case 1:
        area *= (an / (2 * N.x));
        break;
      case 2:
        area *= (an / (2 * N.y));
        break;
      case 3:
        area *= (an / (2 * N.z));
    }
    return area;
}
//===================================================================
