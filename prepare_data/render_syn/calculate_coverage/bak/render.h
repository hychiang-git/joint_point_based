#ifndef RENDER_H_
#define RENDER_H_

#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <set>
#include <cstring>
#include <Eigen/Dense>

#include "tinyply.h"
#include "render.h"

namespace render{
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

    typedef struct Point3 {
        Point3 () {}
        Point3 (float _x, float _y, float _z): x(_x), y(_y), z(_z) {} 
        float           x;
        float           y;
        float           z;
    } Point3;
    
    typedef struct Triangle3 {
       Point3 v1;                 /* Vertex1 */
       Point3 v2;                 /* Vertex2 */
       Point3 v3;                 /* Vertex3 */
       } Triangle3;

    typedef struct int3 {
        int3() {}
        int3(int a, int b, int c) {
            v[0] = a, v[1] = b, v[2] = c;
        }
        int v[3];
    } int3;
    
    typedef struct int4 {
        int4() {}
        int4(int a, int b, int c) {
            v[0] = a, v[1] = b, v[2] = c, v[3] = 255;
        }
        int4(int a, int b, int c, int d) {
            v[0] = a, v[1] = b, v[2] = c, v[3] = d;
        }
        int v[4];
    } int4;
    
    typedef struct {
        unsigned char v[4];
    } uchar4;
    
    
    
    typedef struct RenderParam{
        RenderParam() {}
        RenderParam(int h, int w):height(h), width(w){}
        std::vector<Eigen::Matrix4f> poses;
        std::vector<Point3> v;
        std::vector<int3> f;
        std::vector<uchar4> v_color;
        Eigen::Matrix4f cam_intrinsic;
        Eigen::Matrix4f cam_extrinsic;
        int height;
        int width;
    } RenderParam;
    
    
    typedef struct RenderBuffer{
        RenderBuffer() {}
        RenderBuffer(int w, int h):width(w), height(h) {
            is_valid = false;
            depth = (double*)malloc(width*height*sizeof(double));
            color = (unsigned char*)malloc(width*height*3*sizeof(unsigned char));
            normal = (double*)malloc(width*height*3*sizeof(double));
            pixcoord = (double*)malloc(width*height*3*sizeof(double));
            pixmeta = (double*)malloc(width*height*12*sizeof(double));
            mesh_vertex = (int*)malloc(width*height*3*sizeof(int));
            for(int h = 0; h < height; ++h){
                for(int w = 0; w < width; ++w){
                    // depth
                    int buf_depth_idx = width*h + w;
                    depth[buf_depth_idx] = 0 ;
                    // color 
                    int buf_color_idx = width*3*h + w*3;
                    for(int i=0; i<3; i++)
                        color[buf_color_idx+i]=0;
                    // normal 
                    int buf_normal_idx = width*3*h + w*3;
                    for(int i=0; i<3; i++)
                        normal[buf_normal_idx+i]=0;
                    // pixcoord
                    int buf_pc_idx = width*3*h + w*3;
                    for(int i=0; i<3; i++)
                        pixcoord[buf_pc_idx+i]=0;
                    // pixmeta 
                    int buf_pm_idx = width*9*h + w*9;
                    for(int i=0; i<12; i++)
                        pixmeta[buf_pm_idx+i]=0;
                    // mesh_vertex 
                    int buf_mv_idx = width*3*h + w*3;
                    for(int i=0; i<3; i++)
                        mesh_vertex[buf_mv_idx+i]=0;
                }
            }
        }
        int width;
        int height;
        int channel=22;  // 1+3+3+3+3+9
        double* depth;
        unsigned char* color;
        double* normal;
        double* pixcoord;
        double* pixmeta;
        int* mesh_vertex;
        bool is_valid;
        std::set<int> vertex_set;
    } RenderBuffer;
    
    
    int get_buf_index(const RenderBuffer& buf, const std::string& type, int h, int w);

    void read_ply(const std::string &filename, std::vector<Point3> &v, std::vector<int3> &f, std::vector<uchar4> &v_color);

    void scene_range(const std::vector<Point3> &v, std::vector<int> &scene_min, std::vector<int> &scene_max);

    Eigen::Matrix4f read_camera_parameters(const std::string &filename);

    Eigen::Matrix4f euler2mat(float deg_x, float deg_y, float deg_z);

    std::vector<Eigen::Matrix4f> pseudo_camera_poses(const std::vector<int>& scene_min, const std::vector<int>& scene_max, const int pose_mode);

    void render(const std::vector<Point3> &v, const std::vector<int3> &f, const std::vector<uchar4> &v_color, const Eigen::Matrix4f intrinsic, const Eigen::Matrix4f extrinsic, const int height, const int width, RenderBuffer& buf);
    
    int update_vset(std::vector<std::set<int>>& images_vset, std::set<int>& remove_set, std::set<int>& all_vertex);

    double cover_scene_images(const std::vector<Point3>& vertex, const std::vector<RenderBuffer>& valid_images, std::vector<RenderBuffer>& coverage_images, std::vector<RenderBuffer>& scene_images);

    void dump_buffer(render::RenderBuffer& buf, const std::string& scene_dir, const std::string& fname);
}

#endif
