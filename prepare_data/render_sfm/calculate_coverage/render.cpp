#include "triangleCube.h"
#include "tinyply.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <vector>
#include <set>
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <glob.h>
#include <Eigen/Dense>


using namespace Eigen;
using namespace tinyply;
using namespace std;


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

vector<string> globVector(const string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    vector<string> files;
    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}



typedef struct RenderBuffer {
    RenderBuffer() {}
    RenderBuffer(int w, int h):width(w), height(h) {
        depth = (double*)malloc(width*height*sizeof(double));
        color = (int*)malloc(width*height*3*sizeof(int));
        normal = (double*)malloc(width*height*3*sizeof(double));
        world_coord = (double*)malloc(width*height*3*sizeof(double));
        mesh_vertex = (int*)malloc(width*height*3*sizeof(int));
        mesh_vertex_coord = (double*)malloc(width*height*9*sizeof(double));
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
                // world_coord 
                int buf_wc_idx = width*3*h + w*3;
                for(int i=0; i<3; i++)
                    world_coord[buf_wc_idx+i]=0;
                // mesh_vertex 
                int buf_mv_idx = width*3*h + w*3;
                for(int i=0; i<3; i++)
                    mesh_vertex[buf_mv_idx+i]=0;
                // mesh_vertex 
                int buf_mvc_idx = width*9*h + w*9;
                for(int i=0; i<9; i++)
                    mesh_vertex_coord[buf_mvc_idx+i]=0;
            }
        }

        
    }
    int width;
    int height;
    int channel=22;  // 1+3+3+3+3+9
    double* depth;
    int* color;
    double* normal;
    double* world_coord;
    int* mesh_vertex;
    double* mesh_vertex_coord;
} RenderBuffer;


int get_buf_index(const RenderBuffer& buf, const string& type, int h, int w){
    if (type == "depth"){
        return buf.width*h + w;
    }
    else if (type == "color"){
        return buf.width*3*h + w*3;
    }
    else if (type == "normal"){
        return buf.width*3*h + w*3;
    }
    else if (type == "world_coord"){
        return buf.width*3*h + w*3;
    }
    else if (type == "mesh_vertex"){
        return buf.width*3*h + w*3;
    }
    else if (type == "mesh_vertex_coord"){
        return buf.width*9*h + w*9;
    }
    else { return 0; }
}

Matrix4f read_camera_parameters(const string &filename){
    Matrix4f M;
    string s;
    fstream file;
    file.open(filename, ios::in);
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            if(j==3)
                std::getline(file, s ,'\n'); 
            else
                std::getline(file, s ,' '); 
            M(i,j)=stof(s);
        }
    }
    return M;
}

void read_ply(const string &filename, vector<Point3> &v, vector<int3> &f, vector<uchar4> &v_color){
    try
    {
        // Read the file and create a std::istringstream suitable
        // for the lib -- tinyply does not perform any file i/o.
        std::ifstream ss(filename, std::ios::binary);

        if (ss.fail())
        {
            throw std::runtime_error("failed to open " + filename);
        }

        // Read ply file
        PlyFile file;
        shared_ptr<PlyData> faces, vertices, colors;
        file.parse_header(ss);
        faces = file.request_properties_from_element("face", { "vertex_indices" });
        vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
        colors = file.request_properties_from_element("vertex", {"red", "green", "blue", "alpha"});
        //colors = file.request_properties_from_element("vertex", {"red", "green", "blue"});
        
        file.read(ss);

        // Copy data from plyfile reader to vectors
        const size_t numVerticesBytes = vertices->buffer.size_bytes();
        const size_t numFacesBytes = faces->buffer.size_bytes();
        const size_t numColorBytes = colors->buffer.size_bytes();
        v.resize(vertices->count);
        f.resize(faces->count);
        v_color.resize(vertices->count);
        memcpy(v.data(), vertices->buffer.get(), numVerticesBytes);
        memcpy(f.data(), faces->buffer.get(), numFacesBytes);
        memcpy(v_color.data(), colors->buffer.get(), numColorBytes);
    }
    catch (const std::exception & e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }    
}


//void dump_buffer(const RenderBuffer& buf, const string& fname){
//    ofstream fout(fname, std::ios::binary);
//    fout.write((char*) &buf.height, sizeof(int));
//    fout.write((char*) &buf.width, sizeof(int));
//    // color
//    unsigned long long color_byte_size = buf.width * buf.height * 3 * sizeof(int);
//    fout.write((char*) &color_byte_size, sizeof(unsigned long long));
//    for(int h = 0; h < buf.height; ++h){
//        for(int w = 0; w < buf.width; ++w){
//            int buf_color_idx = get_buf_index(buf, "color", h, w);
//            for(int i=0; i<3; i++){
//                fout.write((char*) &buf.color[buf_color_idx+i], sizeof(int));
//            }
//        }
//    }
//    // depth 
//    unsigned long long depth_byte_size = buf.width * buf.height * 1 * sizeof(double);
//    fout.write((char*) &depth_byte_size, sizeof(unsigned long long));
//    for(int h = 0; h < buf.height; ++h){
//        for(int w = 0; w < buf.width; ++w){
//            int buf_depth_idx = get_buf_index(buf, "depth", h, w);
//            fout.write((char*) &buf.depth[buf_depth_idx], sizeof(double));
//        }
//    }
//    // normal 
//    unsigned long long normal_byte_size = buf.width * buf.height * 3 * sizeof(double);
//    fout.write((char*) &normal_byte_size, sizeof(unsigned long long));
//    for(int h = 0; h < buf.height; ++h){
//        for(int w = 0; w < buf.width; ++w){
//            int buf_normal_idx = get_buf_index(buf, "normal", h, w);
//            for(int i=0; i<3; i++){
//                fout.write((char*) &buf.normal[buf_normal_idx+i], sizeof(double));
//            }
//        }
//    }
//    // world coordinates
//    unsigned long long wc_byte_size = buf.width * buf.height * 3 * sizeof(double);
//    fout.write((char*) &wc_byte_size, sizeof(unsigned long long));
//    for(int h = 0; h < buf.height; ++h){
//        for(int w = 0; w < buf.width; ++w){
//            int buf_wc_idx = get_buf_index(buf, "world_coord", h, w);
//            for(int i=0; i<3; i++){
//                fout.write((char*) &buf.world_coord[buf_wc_idx+i], sizeof(double));
//            }
//        }
//    }
//    // mesh vertices 
//    unsigned long long mv_byte_size = buf.width * buf.height * 3 * sizeof(int);
//    fout.write((char*) &mv_byte_size, sizeof(unsigned long long));
//    for(int h = 0; h < buf.height; ++h){
//        for(int w = 0; w < buf.width; ++w){
//            int buf_mv_idx = get_buf_index(buf, "mesh_vertex", h, w);
//            for(int i=0; i<3; i++){
//                fout.write((char*) &buf.mesh_vertex[buf_mv_idx+i], sizeof(int));
//            }
//        }
//    }
//    // mesh vertices coordinates
//    unsigned long long mvc_byte_size = buf.width * buf.height * 9 * sizeof(double);
//    fout.write((char*) &mvc_byte_size, sizeof(unsigned long long));
//    for(int h = 0; h < buf.height; ++h){
//        for(int w = 0; w < buf.width; ++w){
//            // mesh_vertex 
//            int buf_mvc_idx = get_buf_index(buf, "mesh_vertex_coord", h, w);
//            for(int i=0; i<9; i++){
//                fout.write((char*) &buf.mesh_vertex_coord[buf_mvc_idx+i], sizeof(double));
//            }
//        }
//    }
//    fout.close();
//}

void render(vector<Point3> &v, set<int> &all_vset, vector<int3> &f, vector<uchar4> &v_color, Matrix4f intrinsic, Matrix4f extrinsic, const string& output_fname){
    //cout << "Scene vertex size:"<< v.size() << endl;
    //cout << "Scene face size:"<< f.size() << endl;
    //cout << "Scene vertex color size:"<< v_color.size() << endl;

    Matrix3f intrinsic3f = intrinsic.block(0,0,3,3);
    Matrix3f extrinsic3f = extrinsic.block(0,0,3,3);
    Vector3d translation(extrinsic(0,3),extrinsic(1,3),extrinsic(2,3));
    Matrix3f extrinsic3f_inv = extrinsic3f.inverse();

    int width = (int)ceil(intrinsic3f(0,2)*2); 
    int height = (int)ceil(intrinsic3f(1,2)*2);
    //cout << "Image width:"<< width <<", Image height:"<<height<<" from intrinsic."<<endl;
    if (width > 620 && width <660)
        width = 640;
    if (height > 460 && height <500)
        height = 480;
    //cout << "Normalized Image width:"<< width <<", Image height:"<<height<<endl;
    RenderBuffer buf(width, height);


    vector<Point3> v_cam(v.size());
    vector<Point3> v_pix(v.size());
    std::set<int> f_vset;
    //vector<int> visible_vidx;
    for(int i=0; i<v.size(); i++){
        Point3 p_t, p_rt, p_img;
        p_t.x = v[i].x - translation(0);
        p_t.y = v[i].y - translation(1);
        p_t.z = v[i].z - translation(2);
        p_rt.x = extrinsic3f_inv(0,0)*p_t.x + extrinsic3f_inv(0,1)*p_t.y + extrinsic3f_inv(0,2)*p_t.z; 
        p_rt.y = extrinsic3f_inv(1,0)*p_t.x + extrinsic3f_inv(1,1)*p_t.y + extrinsic3f_inv(1,2)*p_t.z; 
        p_rt.z = extrinsic3f_inv(2,0)*p_t.x + extrinsic3f_inv(2,1)*p_t.y + extrinsic3f_inv(2,2)*p_t.z; 
        // save vertex camera coordinate
        v_cam[i] = p_rt;

        p_img.x = intrinsic3f(0,0)*p_rt.x + intrinsic3f(0,1)*p_rt.y + intrinsic3f(0,2)*p_rt.z; 
        p_img.y = intrinsic3f(1,0)*p_rt.x + intrinsic3f(1,1)*p_rt.y + intrinsic3f(1,2)*p_rt.z; 
        p_img.z = intrinsic3f(2,0)*p_rt.x + intrinsic3f(2,1)*p_rt.y + intrinsic3f(2,2)*p_rt.z; 
        p_img.x = p_img.x / p_img.z;
        p_img.y = p_img.y / p_img.z;
        // save vertex pixel coordinate
        v_pix[i] = p_img;

    }


    for (int idx=0; idx<f.size(); idx++){
        Triangle3 tri_pix;
        tri_pix.v1 = v_pix[f[idx].v[0]];
        tri_pix.v2 = v_pix[f[idx].v[1]];
        tri_pix.v3 = v_pix[f[idx].v[2]];
        if( (tri_pix.v1.x<0 || tri_pix.v1.x>width || tri_pix.v1.y<0 || tri_pix.v1.y>height) &&
            (tri_pix.v2.x<0 || tri_pix.v2.x>width || tri_pix.v2.y<0 || tri_pix.v2.y>height) ) 
            continue;

        Triangle3 tri_cam;
        tri_cam.v1 = v_cam[f[idx].v[0]];
        tri_cam.v2 = v_cam[f[idx].v[1]];
        tri_cam.v3 = v_cam[f[idx].v[2]];
        // Calc normal for the face
        Point3 t1, t2, normal;
        SUB(tri_cam.v2, tri_cam.v1, t1);
        SUB(tri_cam.v3, tri_cam.v1, t2);
        CROSS(t1, t2, normal);
        // normal (a, b, c): ax+by+cz=k
        double k = DOT(normal, tri_cam.v1);
        
        
        // get pixel bounding box (max min)
        int max_x = (int)ceil(max(max(tri_pix.v1.x, tri_pix.v2.x), tri_pix.v3.x));
        int max_y = (int)ceil(max(max(tri_pix.v1.y, tri_pix.v2.y), tri_pix.v3.y));
        int min_x = (int)floor(min(min(tri_pix.v1.x, tri_pix.v2.x), tri_pix.v3.x));
        int min_y = (int)floor(min(min(tri_pix.v1.y, tri_pix.v2.y), tri_pix.v3.y));
        if (max_x >=width) max_x=width-1;
        if (max_y >=height) max_y=height-1;
        if (min_x <0) min_x=0;
        if (min_y <0) min_y=0;
        //#pragma omp parallel for collapse(2)
        for(int py=min_y; py<max_y; py++){
            for(int px=min_x; px<max_x; px++){
                // check the point in triangle 
                bool b1 = ((px - tri_pix.v1.x) * (tri_pix.v2.y - tri_pix.v1.y) - (tri_pix.v2.x - tri_pix.v1.x) * (py - tri_pix.v1.y)) > 0.0;
                bool b2 = ((px - tri_pix.v2.x) * (tri_pix.v3.y - tri_pix.v2.y) - (tri_pix.v3.x - tri_pix.v2.x) * (py - tri_pix.v2.y)) > 0.0;
                bool b3 = ((px - tri_pix.v3.x) * (tri_pix.v1.y - tri_pix.v3.y) - (tri_pix.v1.x - tri_pix.v3.x) * (py - tri_pix.v3.y)) > 0.0;
                if (((b1 == b2) && (b2 == b3))==false) continue;
                // emmit a ray from pixel
                double cam_vec_x = (px+0.5 - intrinsic3f(0,2)) / intrinsic3f(0,0);
                double cam_vec_y = (py+0.5 - intrinsic3f(1,2)) / intrinsic3f(1,1);
                Point3 cam_vec;
                cam_vec.x = cam_vec_x;
                cam_vec.y = cam_vec_y;
                cam_vec.z = 1;
                // get the intersect point of ray and triangle
                double numerator = k;
                double denominator = DOT(normal, cam_vec);
                if (fabs(denominator-0) < 1e-9&& ! fabs(numerator-0)<1e-9 ) continue;
                double t = numerator / denominator;
                if (t < 0) continue;
                Point3 intersec;
                intersec.x = cam_vec.x * t;
                intersec.y = cam_vec.y * t;
                intersec.z = cam_vec.z * t;
                // record the depth if the depth is closer
                int buf_depth_idx = get_buf_index(buf, "depth", py, px);
                if (intersec.z < buf.depth[buf_depth_idx] | buf.depth[buf_depth_idx]==0){
                    // assign depth
                    buf.depth[buf_depth_idx] = intersec.z;
                    // assign normal 
                    int buf_normal_idx = get_buf_index(buf, "normal", py, px);
                    double normal_length = sqrt(pow(normal.x, 2) +  pow(normal.y, 2) + pow(normal.z, 2));
                    double nx = normal.x / normal_length; 
                    double ny = normal.y / normal_length; 
                    double nz = normal.z / normal_length; 
                    buf.normal[buf_normal_idx+0] = nx;
                    buf.normal[buf_normal_idx+1] = ny;
                    buf.normal[buf_normal_idx+2] = nz;
                    // assign color 
                    int buf_color_idx = get_buf_index(buf, "color", py, px);
                    int r = (int)(v_color[f[idx].v[0]].v[0]+v_color[f[idx].v[1]].v[0]+v_color[f[idx].v[2]].v[0])/3;
                    int g = (int)(v_color[f[idx].v[0]].v[1]+v_color[f[idx].v[1]].v[1]+v_color[f[idx].v[2]].v[1])/3;
                    int b = (int)(v_color[f[idx].v[0]].v[2]+v_color[f[idx].v[1]].v[2]+v_color[f[idx].v[2]].v[2])/3;
                    buf.color[buf_color_idx+0] = r;
                    buf.color[buf_color_idx+1] = g;
                    buf.color[buf_color_idx+2] = b;
                    // assign world coord 
                    int buf_wc_idx = get_buf_index(buf, "world_coord", py, px);
                    Point3 world_inter_coord;
                    world_inter_coord.x = extrinsic3f(0,0)*intersec.x + extrinsic3f(0,1)*intersec.y + extrinsic3f(0,2)*intersec.z + translation(0); 
                    world_inter_coord.y = extrinsic3f(1,0)*intersec.x + extrinsic3f(1,1)*intersec.y + extrinsic3f(1,2)*intersec.z + translation(1); 
                    world_inter_coord.z = extrinsic3f(2,0)*intersec.x + extrinsic3f(2,1)*intersec.y + extrinsic3f(2,2)*intersec.z + translation(2); 
                    buf.world_coord[buf_wc_idx+0] = world_inter_coord.x;
                    buf.world_coord[buf_wc_idx+1] = world_inter_coord.y;
                    buf.world_coord[buf_wc_idx+2] = world_inter_coord.z;
                    // assign mesh_vertex
                    int buf_mv_idx = get_buf_index(buf, "mesh_vertex", py, px);
                    buf.mesh_vertex[buf_mv_idx+0] = f[idx].v[0];
                    buf.mesh_vertex[buf_mv_idx+1] = f[idx].v[1];
                    buf.mesh_vertex[buf_mv_idx+2] = f[idx].v[2];
                    f_vset.insert(f_vset.end(), f[idx].v[0]);
                    f_vset.insert(f_vset.end(), f[idx].v[1]);
                    f_vset.insert(f_vset.end(), f[idx].v[2]);
                    // assign mesh_vertex_coord
                    int buf_mvc_idx = get_buf_index(buf, "mesh_vertex_coord", py, px);
                    buf.mesh_vertex_coord[buf_mvc_idx+0] = v[f[idx].v[0]].x;
                    buf.mesh_vertex_coord[buf_mvc_idx+1] = v[f[idx].v[0]].y;
                    buf.mesh_vertex_coord[buf_mvc_idx+2] = v[f[idx].v[0]].z;
                    buf.mesh_vertex_coord[buf_mvc_idx+3] = v[f[idx].v[1]].x;
                    buf.mesh_vertex_coord[buf_mvc_idx+4] = v[f[idx].v[1]].y;
                    buf.mesh_vertex_coord[buf_mvc_idx+5] = v[f[idx].v[1]].z;
                    buf.mesh_vertex_coord[buf_mvc_idx+6] = v[f[idx].v[2]].x;
                    buf.mesh_vertex_coord[buf_mvc_idx+7] = v[f[idx].v[2]].y;
                    buf.mesh_vertex_coord[buf_mvc_idx+8] = v[f[idx].v[2]].z;
                }
                
            }
        }
    }
    std::set<int> tmp;
    //cout << "    all_vset.size(): " << all_vset.size() << endl;
    std::set_difference(all_vset.begin(), all_vset.end(), f_vset.begin(), f_vset.end(), inserter(tmp, tmp.end()));
    all_vset = tmp;
    //cout << "    f_vset.size(): " << f_vset.size() << endl;
    //cout << "    tmp.size(): " << tmp.size() << endl;
    //cout << "    all_vset.size(): " << all_vset.size() << endl;
    //cout << "    current coverage: " << (float)all_vset.size()/v.size() << endl;
    //dump_buffer(buf, output_fname);
}

int main(int argc, char *argv[]) {
    //printf("Reading scene ply file: %s\n", argv[1]);
    vector<Point3> vertex; 
    vector<int3> face;
    vector<uchar4> vertex_color;
    read_ply(argv[1], vertex, face, vertex_color);

    //printf("Reading depth intrinsic parameter txt: %s\n", argv[2]);
    //printf("Reading depth extrinsic parameter txt: %s\n", argv[3]);
    Matrix4f cam_intrinsic = read_camera_parameters(argv[2]);
    Matrix4f cam_extrinsic = read_camera_parameters(argv[3]);
    //printf("Reading camera pose dir: %s\n", argv[4]);
    string posedir = argv[4];
    vector<string> posefiles = globVector(posedir.append("/*.txt"));
    //printf("Get %d camera pose in directory: %s\n", posefiles.size(), argv[4]);
    //printf("Output directory: %s\n", argv[5]);
    string output_dir = argv[5];


    set<int> all_vertex_set;
    for (unsigned int i = 0; i < vertex.size(); ++i) 
        all_vertex_set.insert(all_vertex_set.end(), i);

    for(int i =0; i<posefiles.size(); i++){
        // extrinsic 
        //cout << posefiles[i] << endl;
        Matrix4f extrinsic = read_camera_parameters(posefiles[i]) * cam_extrinsic;
        //  render
        int len_posefile_name = posefiles[i].size();
        string output_fname = output_dir +"/"+posefiles[i].substr(len_posefile_name-10, 6)+".bin";
        //cout << output_fname << endl;
        render(vertex, all_vertex_set, face, vertex_color, cam_intrinsic, extrinsic, output_fname);
    }
    cout << "scene_non_coverage: " << (float)all_vertex_set.size()/vertex.size() << endl;

    std::ofstream logfile;
    logfile.open("val_scene_non_coverage.txt", std::ios_base::app);
    logfile << argv[1] <<", non_coverage: " << (float)all_vertex_set.size()/vertex.size() << endl;
    return 0;
}
