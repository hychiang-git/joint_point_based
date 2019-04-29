#include "render.h"
#include "tinyply.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cstring>
#include <stdexcept>
#include "xtensor/xio.hpp"
#include "xtensor-io/ximage.hpp"
#include "xtensor-io/xnpz.hpp"

int render::get_buf_index(const render::RenderBuffer& buf, const std::string& type, int h, int w){
    if (type == "depth"){
        return buf.width*h + w;
    }
    else if (type == "color"){
        return buf.width*3*h + w*3;
    }
    else if (type == "normal"){
        return buf.width*3*h + w*3;
    }
    else if (type == "pixcoord"){
        return buf.width*3*h + w*3;
    }
    else if (type == "pixmeta"){
        return buf.width*12*h + w*12;
    }
    else if (type == "mesh_vertex"){
        return buf.width*3*h + w*3;
    }
    else {
         throw std::invalid_argument( "received negative value" );
    }
}


void render::read_ply(
    const std::string &filename, 
    std::vector<render::Point3> &v, 
    std::vector<render::int3> &f, 
    std::vector<render::uchar4> &v_color)
{
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
        tinyply::PlyFile file;
        std::shared_ptr<tinyply::PlyData> faces, vertices, colors;
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

void render::scene_range(
    const std::vector<Point3> &v, 
    std::vector<int> &scene_min, 
    std::vector<int> &scene_max)
{
    float min_x = INT_MAX; float min_y = INT_MAX; float min_z = INT_MAX; 
    float max_x = INT_MIN; float max_y = INT_MIN; float max_z = INT_MIN; 
    for(unsigned int i=0; i<v.size(); i++){
        if (min_x > v[i].x)
            min_x = v[i].x;
        if (min_y > v[i].y)
            min_y = v[i].y;
        if (min_z > v[i].z)
            min_z = v[i].z;
        if (max_x < v[i].x)
            max_x = v[i].x;
        if (max_y < v[i].y)
            max_y = v[i].y;
        if (max_z < v[i].z)
            max_z = v[i].z;
    }
    scene_min[0] = (int)floor(min_x); scene_min[1] = (int)floor(min_y); scene_min[2] = (int)floor(min_z);
    scene_max[0] = (int)ceil(max_x); scene_max[1] = (int)ceil(max_y); scene_max[2] = (int)ceil(max_z);
}

Eigen::Matrix4f render::read_camera_parameters(const std::string &filename){
    Eigen::Matrix4f M;
    std::string s;
    std::fstream file;
    file.open(filename, std::ios::in);
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

Eigen::Matrix4f render::euler2mat(float deg_x, float deg_y, float deg_z){
    float rad_x = (deg_x / 180.0) * M_PI;
    float rad_y = (deg_y / 180.0) * M_PI;
    float rad_z = (deg_z / 180.0) * M_PI;
    Eigen::AngleAxisf pitchAngle(rad_x, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf yawAngle(rad_y, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rollAngle(rad_z, Eigen::Vector3f::UnitZ());
    Eigen::Quaternion<float> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3f mat3f = q.matrix();
    Eigen::Matrix4f mat4f = Eigen::Matrix4f::Identity();
    mat4f.block(0,0,3,3) = mat3f;

    return mat4f;
}

std::vector<Eigen::Matrix4f> render::pseudo_camera_poses(
    const std::vector<int>& scene_min, 
    const std::vector<int>& scene_max, 
    const int pose_mode)
{
    float x_axis[3] = {-30, 0, 30};
    float z_axis[12]; // 10 degrees a step
    for(int i=0; i<12; i++)
        z_axis[i] = i*30;

    std::vector<float> h_step, w_step, d_step;
    for(float h=scene_min[2]+1.5; h <= (scene_min[2]+2.5); h+=0.5){
        h_step.push_back(h);
    }
    float d_stride = 0.1*(scene_max[1] - scene_min[1]);
    for(float d=scene_min[1]; d<scene_max[1]; d+=d_stride){
        d_step.push_back(d);
    }
    float w_stride = 0.1*(scene_max[0] - scene_min[0]);
    for(float w=scene_min[0]; w<scene_max[0]; w+=w_stride){
        w_step.push_back(w);
    }
    //int num_poses = (int)h_step.size()*w_step.size()*d_step.size()*3*(12/pose_mode);
    std::vector<Eigen::Matrix4f> pseudo_poses;
    for(unsigned int ih=0; ih<h_step.size(); ih++){
        for(unsigned int id=0; id<d_step.size(); id++){
            for(unsigned int iw=0; iw<w_step.size(); iw++){
                for(int ix=0; ix<3; ix++){
                    for(int iz=0; iz<12; iz+=pose_mode){
                        Eigen::Matrix4f init_mat = euler2mat(-90, 0, 0); // inital camera pose
                        Eigen::Matrix4f rot_mat = euler2mat(x_axis[ix], 0.0, z_axis[iz]);
                        Eigen::Matrix4f pose_mat = rot_mat * init_mat;
                        pose_mat(0,3) = w_step[iw];
                        pose_mat(1,3) = d_step[id];
                        pose_mat(2,3) = h_step[ih];
                        pseudo_poses.push_back(pose_mat);
                    } 
                }
            }
        }
    }
    return pseudo_poses;
}



void render::render(
    const std::vector<render::Point3> &v, 
    const std::vector<render::int3> &f, 
    const std::vector<render::uchar4> &v_color, 
    const Eigen::Matrix4f intrinsic, 
    const Eigen::Matrix4f extrinsic, 
    const int height, 
    const int width,
    render::RenderBuffer& buf)
{
    Eigen::Matrix3f intrinsic3f = intrinsic.block(0,0,3,3);
    Eigen::Matrix3f extrinsic3f = extrinsic.block(0,0,3,3);
    Eigen::Vector3d translation(extrinsic(0,3),extrinsic(1,3),extrinsic(2,3));
    Eigen::Matrix3f extrinsic3f_inv = extrinsic3f.inverse();


    std::vector<render::Point3> v_cam(v.size());
    std::vector<render::Point3> v_pix(v.size());
    //vector<int> visible_vidx;
    for(unsigned int i=0; i<v.size(); i++){
        render::Point3 p_t, p_rt, p_img;
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
    double max_depth = 0.0;
    for (unsigned int idx=0; idx<f.size(); idx++){
        render::Triangle3 tri_pix;
        tri_pix.v1 = v_pix[f[idx].v[0]];
        tri_pix.v2 = v_pix[f[idx].v[1]];
        tri_pix.v3 = v_pix[f[idx].v[2]];
        if( (tri_pix.v1.x<0 || tri_pix.v1.x>width || tri_pix.v1.y<0 || tri_pix.v1.y>height) &&
            (tri_pix.v2.x<0 || tri_pix.v2.x>width || tri_pix.v2.y<0 || tri_pix.v2.y>height) && 
            (tri_pix.v3.x<0 || tri_pix.v3.x>width || tri_pix.v3.y<0 || tri_pix.v3.y>height) ) 
            continue; // out of image range

        render::Triangle3 tri_cam;
        tri_cam.v1 = v_cam[f[idx].v[0]];
        tri_cam.v2 = v_cam[f[idx].v[1]];
        tri_cam.v3 = v_cam[f[idx].v[2]];
        // Calc normal for the face
        render::Point3 t1, t2, normal;
        SUB(tri_cam.v2, tri_cam.v1, t1);
        SUB(tri_cam.v3, tri_cam.v1, t2);
        CROSS(t1, t2, normal);
        // normal (a, b, c): ax+by+cz=k
        double k = DOT(normal, tri_cam.v1);

        render::Point3 view_dir(0, 0, 1);
        if(DOT(normal, view_dir)>0){
            //back to camera view direction, no need to render
            continue;
        }
        
        
        // get pixel bounding box (max min)
        int max_x = (int)ceil(std::max(std::max(tri_pix.v1.x, tri_pix.v2.x), tri_pix.v3.x));
        int max_y = (int)ceil(std::max(std::max(tri_pix.v1.y, tri_pix.v2.y), tri_pix.v3.y));
        int min_x = (int)floor(std::min(std::min(tri_pix.v1.x, tri_pix.v2.x), tri_pix.v3.x));
        int min_y = (int)floor(std::min(std::min(tri_pix.v1.y, tri_pix.v2.y), tri_pix.v3.y));
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
                render::Point3 cam_vec;
                cam_vec.x = cam_vec_x;
                cam_vec.y = cam_vec_y;
                cam_vec.z = 1;
                // get the intersect point of ray and triangle
                double numerator = k;
                double denominator = DOT(normal, cam_vec);
                if ( (fabs(denominator-0) < 1e-9) && !(fabs(numerator-0)<1e-9) ) continue;
                double t = numerator / denominator;
                if (t < 0) continue;
                render::Point3 intersec;
                intersec.x = cam_vec.x * t;
                intersec.y = cam_vec.y * t;
                intersec.z = cam_vec.z * t;
                // record the depth if the depth is closer
                int buf_depth_idx = get_buf_index(buf, "depth", py, px);
                if ((intersec.z < buf.depth[buf_depth_idx]) | (buf.depth[buf_depth_idx]==0)){
                    // record max depth
                    if(intersec.z > max_depth) max_depth = intersec.z;
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
                    buf.normal_vis[buf_normal_idx+0] = (unsigned char)(255*(nx + 1) * 0.5);
                    buf.normal_vis[buf_normal_idx+1] = (unsigned char)(255*(ny + 1) * 0.5);
                    buf.normal_vis[buf_normal_idx+2] = (unsigned char)(255*(nz + 1) * 0.5);
                    // assign color 
                    int buf_color_idx = get_buf_index(buf, "color", py, px);
                    unsigned char r = (unsigned char)((v_color[f[idx].v[0]].v[0]+v_color[f[idx].v[1]].v[0]+v_color[f[idx].v[2]].v[0])/3);
                    unsigned char g = (unsigned char)((v_color[f[idx].v[0]].v[1]+v_color[f[idx].v[1]].v[1]+v_color[f[idx].v[2]].v[1])/3);
                    unsigned char b = (unsigned char)((v_color[f[idx].v[0]].v[2]+v_color[f[idx].v[1]].v[2]+v_color[f[idx].v[2]].v[2])/3);
                    buf.color[buf_color_idx+0] = r;
                    buf.color[buf_color_idx+1] = g;
                    buf.color[buf_color_idx+2] = b;
                    // assign world coord 
                    int buf_pc_idx = get_buf_index(buf, "pixcoord", py, px);
                    render::Point3 world_inter_coord;
                    world_inter_coord.x = extrinsic3f(0,0)*intersec.x + extrinsic3f(0,1)*intersec.y + extrinsic3f(0,2)*intersec.z + translation(0); 
                    world_inter_coord.y = extrinsic3f(1,0)*intersec.x + extrinsic3f(1,1)*intersec.y + extrinsic3f(1,2)*intersec.z + translation(1); 
                    world_inter_coord.z = extrinsic3f(2,0)*intersec.x + extrinsic3f(2,1)*intersec.y + extrinsic3f(2,2)*intersec.z + translation(2); 
                    buf.pixcoord[buf_pc_idx+0] = world_inter_coord.x;
                    buf.pixcoord[buf_pc_idx+1] = world_inter_coord.y;
                    buf.pixcoord[buf_pc_idx+2] = world_inter_coord.z;
                    // assign mesh_vertex_coord
                    int buf_pm_idx = get_buf_index(buf, "pixmeta", py, px);
                    buf.pixmeta[buf_pm_idx+0] = f[idx].v[0];
                    buf.pixmeta[buf_pm_idx+1] = f[idx].v[1];
                    buf.pixmeta[buf_pm_idx+2] = f[idx].v[2];
                    buf.pixmeta[buf_pm_idx+3] = v[f[idx].v[0]].x;
                    buf.pixmeta[buf_pm_idx+4] = v[f[idx].v[0]].y;
                    buf.pixmeta[buf_pm_idx+5] = v[f[idx].v[0]].z;
                    buf.pixmeta[buf_pm_idx+6] = v[f[idx].v[1]].x;
                    buf.pixmeta[buf_pm_idx+7] = v[f[idx].v[1]].y;
                    buf.pixmeta[buf_pm_idx+8] = v[f[idx].v[1]].z;
                    buf.pixmeta[buf_pm_idx+9] = v[f[idx].v[2]].x;
                    buf.pixmeta[buf_pm_idx+10] = v[f[idx].v[2]].y;
                    buf.pixmeta[buf_pm_idx+11] = v[f[idx].v[2]].z;
                    // assign mesh_vertex
                    int buf_mv_idx = get_buf_index(buf, "mesh_vertex", py, px);
                    buf.mesh_vertex[buf_mv_idx+0] = f[idx].v[0];
                    buf.mesh_vertex[buf_mv_idx+1] = f[idx].v[1];
                    buf.mesh_vertex[buf_mv_idx+2] = f[idx].v[2];
                }
                
            } // for px
        } // for py
    } // for all faces
    // check frame
    buf.is_valid = false;
    float valid_pix = 0;
    for(int py=0; py<height; py++){
        for(int px=0; px<width; px++){
            int buf_depth_idx = get_buf_index(buf, "depth", py, px);
            buf.depth_vis[buf_depth_idx] = (unsigned char)(255 * (buf.depth[buf_depth_idx] / max_depth));
            if(buf.depth[buf_depth_idx]>0.5) valid_pix++;
        }
    }
    if(valid_pix/(height*width) > 0.5)
        buf.is_valid = true;
    // vertex set in a frame
    int buf_mv_start = get_buf_index(buf, "mesh_vertex", 0, 0);
    int buf_mv_end = get_buf_index(buf, "mesh_vertex", height-1, width-1);
    std::set<int> mv_set(&buf.mesh_vertex[buf_mv_start], &buf.mesh_vertex[buf_mv_end]);
    buf.vertex_set = mv_set;
}

int render::update_vset(
    std::vector<std::set<int>>& images_vset, 
    std::set<int>& remove_set, 
    std::set<int>& all_vertex)
{
    std::set<int> tmp;
    std::set_difference(all_vertex.begin(), all_vertex.end(), remove_set.begin(), remove_set.end(), inserter(tmp, tmp.end()));
    all_vertex = tmp;
    //cout << "all_vertex size:"<<all_vertex.size()<<endl;
    for(unsigned int i =0; i<images_vset.size(); i++){
        std::set<int> tmp;
        std::set_intersection(images_vset[i].begin(), images_vset[i].end(), all_vertex.begin(), all_vertex.end(), inserter(tmp, tmp.end()));
        images_vset[i] = tmp;
    }

    std::vector<std::set<int>>::iterator max_it = max_element(images_vset.begin(), images_vset.end(), 
                 [](const std::set<int> & a, const std::set<int>& b) -> bool{ 
                     return a.size() < b.size(); 
                 });

    int max_index = distance(images_vset.begin(), max_it);
    return max_index;
}

double render::cover_scene_images(
    const std::vector<render::Point3>& vertex, 
    const std::vector<render::RenderBuffer>& valid_images, 
    std::vector<render::RenderBuffer>& coverage_images, 
    std::vector<render::RenderBuffer>& scene_images)
{
    std::set<int> all_vertex;
    for (unsigned int i = 0; i < vertex.size(); ++i) 
        all_vertex.insert(all_vertex.end(), i);

    std::vector<std::set<int>> image_vset(valid_images.size());
    for (unsigned int i = 0; i < image_vset.size(); ++i){
        image_vset[i] = valid_images[i].vertex_set;
        //cout << image_vset[i].size()<<endl;
    }

    std::vector<bool> inside_cover_imgs(valid_images.size(), false);
    int max_index = 0;
    for(unsigned int i=0; i<valid_images.size(); i++){
        //cout<<"max_index:"<<max_index<<", max image vset size:"<<image_vset[max_index].size()<<", all_vertex size:"<<all_vertex.size()<<", vertex number:"<<vertex.size()<<endl;
        if(i==0){
            inside_cover_imgs[0] = true;
            coverage_images.push_back(valid_images[0]);
            max_index = render::update_vset(image_vset, image_vset[0], all_vertex);
        }
        else if (all_vertex.size()>0 && image_vset[max_index].size()>0){
            inside_cover_imgs[max_index] = true;
            coverage_images.push_back(valid_images[max_index]);
            max_index = render::update_vset(image_vset, image_vset[max_index], all_vertex);
        }
        else{
            break;
        }
    }
    for(unsigned int i=0; i<valid_images.size(); i++){
        if(inside_cover_imgs[i]==false){
            scene_images.push_back(valid_images[i]);
        }
    }
    double non_cover = all_vertex.size()/(float)vertex.size();
    return non_cover;
}


void render::dump_buffer(render::RenderBuffer& buf, const std::string& scene_dir, const std::string& fname){
    // color
    std::string color_dir = scene_dir+"/pseudo_pose_color";
    system(("mkdir -p " + color_dir).c_str());
    xt::xarray<unsigned char> color_frame(std::vector<size_t>{buf.height, buf.width, 3});
    memcpy(color_frame.data(), &buf.color[0], buf.width*buf.height*3*sizeof(unsigned char));
    xt::dump_image(color_dir + "/" + fname+".jpg", color_frame);
    // depth    
    std::string depth_dir = scene_dir+"/pseudo_pose_depth";
    system(("mkdir -p " + depth_dir).c_str());
    xt::xarray<unsigned char> depth_frame_png(std::vector<size_t>{buf.height, buf.width});
    memcpy(depth_frame_png.data(), &buf.depth_vis[0], buf.width*buf.height*sizeof(unsigned char));
    xt::dump_image(depth_dir + "/" + fname+".vis.png", depth_frame_png);
    xt::xarray<double> depth_frame_npz(std::vector<size_t>{buf.height, buf.width});
    memcpy(depth_frame_npz.data(), &buf.depth[0], buf.width*buf.height*sizeof(double));
    xt::dump_npz(depth_dir + "/" + fname+".npz", "depth", depth_frame_npz, true, false);
    // normal
    std::string normal_dir = scene_dir+"/pseudo_pose_normal";
    system(("mkdir -p " + normal_dir).c_str());
    xt::xarray<unsigned char> normal_frame_png(std::vector<size_t>{buf.height, buf.width, 3});
    memcpy(normal_frame_png.data(), &buf.normal_vis[0], buf.width*buf.height*3*sizeof(unsigned char));
    xt::dump_image(normal_dir + "/" + fname+".vis.png", normal_frame_png);
    xt::xarray<double> normal_frame_npz(std::vector<size_t>{buf.height, buf.width, 3});
    memcpy(normal_frame_npz.data(), &buf.normal[0], buf.width*buf.height*3*sizeof(double));
    xt::dump_npz(normal_dir + "/" + fname+".npz", "normal", normal_frame_npz, true, false);
    // pixel coord 
    std::string pixcoord_dir = scene_dir+"/pseudo_pose_pixel_coord";
    system(("mkdir -p " + pixcoord_dir).c_str());
    xt::xarray<double> pixcoord_frame_npz(std::vector<size_t>{buf.height, buf.width, 3});
    memcpy(pixcoord_frame_npz.data(), &buf.pixcoord[0], buf.width*buf.height*3*sizeof(double));
    xt::dump_npz(pixcoord_dir + "/" + fname+".npz", "pixcoord", pixcoord_frame_npz, true, false);
    // pixel meta 
    std::string pixmeta_dir = scene_dir+"/pseudo_pose_pixel_meta";
    system(("mkdir -p " + pixmeta_dir).c_str());
    xt::xarray<double> pixmeta_frame_npz(std::vector<size_t>{buf.height, buf.width, 12});
    memcpy(pixmeta_frame_npz.data(), &buf.pixmeta[0], buf.width*buf.height*12*sizeof(double));
    xt::dump_npz(pixmeta_dir + "/" + fname+".npz", "meta", pixmeta_frame_npz, true, false);
}


