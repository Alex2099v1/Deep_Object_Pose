#!/usr/bin/env -S blenderproc run

import blenderproc as bp  # must be first!
from blenderproc.python.utility.Utility import Utility
import bpy

import argparse
import cv2
import glob
import json
from math import acos, atan, cos, pi, sin, sqrt
import numpy as np
import os
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
import random
import sys

# --- HELPER FUNCTIONS FOR ROTATION ---

def Rx(angle):
    return np.array([[1, 0, 0],
                     [0, cos(angle), -sin(angle)],
                     [0, sin(angle),  cos(angle)]])

def Ry(angle):
    return np.array([[cos(angle), 0, sin(angle)],
                     [0, 1, 0],
                     [-sin(angle), 0, cos(angle)]])

def Rz(angle):
    return np.array([[cos(angle), -sin(angle), 0],
                     [sin(angle),  cos(angle), 0],
                     [0, 0, 1]])

def get_constrained_rotation_matrix(x_max_deg, y_max_deg, z_max_deg):
    """
    Generates a rotation matrix with constrained intervals per axis.
    """

    def get_random_angle(limit_deg):
        # Case 1: Full rotation allowed (Non-symmetric axis)
        if limit_deg >= 360:
            return random.random() * 2 * pi
        
        # Case 2: Axis is locked (e.g. Cylindrical symmetry)
        if limit_deg <= 0:
            return 0.0
            
        # Case 3: Constrained Symmetry (e.g. 90 or 180)
        limit_rad = np.deg2rad(limit_deg)
        return random.random() * limit_rad

    # Sample angles
    ax = get_random_angle(x_max_deg)
    ay = get_random_angle(y_max_deg)
    az = get_random_angle(z_max_deg)

    # Combine rotations (Order: Z * Y * X is standard, but independent sampling makes order less critical)
    # This creates the rotation matrix
    R = Rz(az) @ Ry(ay) @ Rx(ax)
    return R

# --- END HELPER FUNCTIONS ---


def random_object_position(near=5.0, far=40.0):
    # Specialized function to randomly place the objects in a visible location
    x = 20 - 40*random.random()
    y = near + (far-near)*random.random()
    z = 20 - 40*random.random()
    return np.array([x, y, z])


def random_depth_in_frustrum(tw, th, bw, bh, depth):
    A = (tw - bw) * (th - bw)/(depth * depth)
    B = (bw * (tw - bw) + bw * (th - bw))/depth
    C = bw * bw
    area = depth * (C + depth * (0.5 * B + depth * A / 3.0))
    r = random.random() * area
    det = B * B - 4 * A * C
    part1 = B * (B * B - 6 * A * C) - 12 * A * A * r
    part2 = sqrt(part1 * part1 - det * det * det)
    part3 = pow(part1 + part2, 1./3.)
    return (-(B + det / part3 + part3) / (2 * A))


def point_in_frustrum(camera, near=10, far=20):
    fov_w, fov_h = camera.get_fov()

    tw = sin(fov_w)*near # top (nearest to camera) width of frustrum
    th = sin(fov_h)*near # top (nearest to camera) height of frustrum
    bw = sin(fov_w)*far  # bottom width
    bh = sin(fov_h)*far  # bottom height

    # calculate random inverse depth: 0 at the 'far' plane and 1 at the 'near' plane
    inv_depth = random_depth_in_frustrum(tw, th, bw, bh, far-near)
    depth = far-inv_depth

    nd = depth/(far-near) # normalized depth
    w = nd*(bw-tw)
    h = nd*(bh-th)

    # construct points so that we are looking down -Z, +Y is up, +X to right
    x = (0.5 - random.random())*w
    y = (0.5 - random.random())*h
    z = depth

    # orient them along the camera's view direction
    xform = camera.get_camera_pose()
    return (np.array([x,y,z,1]) @ xform)[0:3]


def rotated_rectangle_extents(w, h, angle):
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    sin_a, cos_a = abs(sin(angle)), abs(cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr


def crop_around_center(image, width, height):
    size = image.size
    center = (int(size[0] * 0.5), int(size[1] * 0.5))

    if(width > size[0]):
        width = size[0]

    if(height > size[1]):
        height = size[1]

    x1 = int(center[0] - width * 0.5)
    x2 = int(center[0] + width * 0.5)
    y1 = int(center[1] - height * 0.5)
    y2 = int(center[1] + height * 0.5)

    return image.crop((x1, y1, x2, y2))


def crop_to_rotation(img, angle):
    angle_rad = angle*pi/180.0
    width, height = img.size

    img = img.rotate(angle)
    wr, hr = rotated_rectangle_extents(width, height, angle_rad)
    return crop_around_center(img, wr, hr)


def scale_to_original_shape(img, o_width, o_height):
    c_width, c_height = img.size
    o_ar = o_width/o_height
    c_ar = c_width/c_height
    if o_ar > c_ar:
        cropped = crop_around_center(img, c_width, c_width/o_ar)
    else:
        cropped = crop_around_center(img, c_height*o_ar, c_height)

    return cropped.resize((o_width, o_height))


def get_cuboid_image_space(mesh, camera):
    bbox = mesh.get_bound_box()
    centroid = np.array([0.,0.,0.])
    for ii in range(8):
        centroid += bbox[ii]
    centroid = centroid / 8

    cam_pose = np.linalg.inv(camera.get_camera_pose()) 
    tvec = -cam_pose[0:3,3]
    rvec = -cv2.Rodrigues(cam_pose[0:3,0:3])[0]
    K = camera.get_intrinsics_as_K_matrix()

    # DOPE order re-mapping
    dope_order = [5, 1, 2, 6, 4, 0, 3, 7]

    cuboid = [None for ii in range(9)]
    for ii in range(8):
        cuboid[dope_order[ii]] = cv2.projectPoints(bbox[ii], rvec, tvec, K, np.array([]))[0][0][0]
    cuboid[8] = cv2.projectPoints(centroid, rvec, tvec, K, np.array([]))[0][0][0]

    return np.array(cuboid, dtype=float).tolist()


def write_json(outf, args, camera, objects, objects_data, seg_map):
    cam_xform = camera.get_camera_pose()
    eye = -cam_xform[0:3,3]
    at = -cam_xform[0:3,2]
    up = cam_xform[0:3,0]

    K = camera.get_intrinsics_as_K_matrix()

    data = {
        "camera_data" : {
            "width" : args.width,
            'height' : args.height,
            'camera_look_at':
            {
                'at': [at[0], at[1], at[2]],
                'eye': [eye[0], eye[1], eye[2]],
                'up': [up[0], up[1], up[2]]
            },
            'intrinsics':{
                'fx':K[0][0],
                'fy':K[1][1],
                'cx':K[0][2],
                'cy':K[1][2]
            }
        },
        "objects" : []
    }

    for ii, oo in enumerate(objects):
        idx = ii+1 
        num_pixels = int(np.sum((seg_map == idx)))

        if num_pixels < args.min_pixels:
            continue
        projected_keypoints = get_cuboid_image_space(oo, camera)

        data['objects'].append({
            'class': objects_data[ii]['class'],
            'name': objects_data[ii]['name'],
            'visibility': num_pixels,
            'projected_cuboid': projected_keypoints,
            'location': objects_data[ii]['location'],
            'quaternion_xyzw': objects_data[ii]['quaternion_xyzw']
        })

    with open(outf, "w") as write_file:
        json.dump(data, write_file, indent=4)

    return data


def draw_cuboid_markers(objects, camera, im):
    colors = ['yellow', 'magenta', 'blue', 'red', 'green', 'orange', 'brown', 'cyan', 'white']
    R = 2 
    draw = ImageDraw.Draw(im)
    for oo in objects:
        projected_keypoints = get_cuboid_image_space(oo, camera)
        for idx, pp in enumerate(projected_keypoints):
            x = int(pp[0])
            y = int(pp[1])
            draw.ellipse((x-R, y-R, x+R, y+R), fill=colors[idx])

    return im


def randomize_background(path, width, height):
    img = Image.open(path)
    angle = 45.0 - random.random()*90.0
    img = crop_to_rotation(img, angle)
    img = scale_to_original_shape(img, width, height)

    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img


def set_world_background_hdr(filename, strength=1.0, rotation_euler=None):
    if rotation_euler is None:
        rotation_euler = [0.0, 0.0, 0.0]

    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links

    texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    texture_node.image = bpy.data.images.load(filename, check_existing=True)

    background_node = Utility.get_the_one_node_with_type(nodes, "Background")
    links.new(texture_node.outputs["Color"], background_node.inputs["Color"])
    background_node.inputs["Strength"].default_value = strength

    mapping_node = nodes.new("ShaderNodeMapping")
    tex_coords_node = nodes.new("ShaderNodeTexCoord")

    links.new(tex_coords_node.outputs["Generated"], mapping_node.inputs["Vector"])
    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])

    mapping_node.inputs["Rotation"].default_value = rotation_euler


def main(args):
    SEG_DISTRACT = 0
    out_directory = os.path.join(args.outf, str(args.run_id))
    os.makedirs(out_directory, exist_ok=True)

    # Background loading
    image_types = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.hdr', '*.HDR')
    backdrop_images = []
    if args.backgrounds_folder is not None:
        for ext in image_types:
            backdrop_images.extend(glob.glob(os.path.join(args.backgrounds_folder,
                                                      os.path.join('**', ext)),
                                             recursive=True))

    # Model loading
    object_models = []
    if args.path_single_obj:
        object_models.append(args.path_single_obj)
    else:
        object_models = glob.glob(args.objs_folder + "**/textured.obj", recursive=True)
    if len(object_models) == 0:
        print(f"Failed to find any loadable models.")
        exit(1)

    distractor_objs = glob.glob(args.distractors_folder + "**/model.obj", recursive=True)

    bp.init()

    # Camera setup
    cam_pose = bp.math.build_transformation_mat([0, -25, 0], [np.pi / 2, 0, 0])
    bp.camera.add_camera_pose(cam_pose)
    bp.camera.set_resolution(args.width, args.height)
    if args.focal_length:
        K = np.array([[args.focal_length, 0, args.width/2],
                      [0, args.focal_length, args.height/2],
                      [0,0,1]])
        bp.camera.set_intrinsics_from_K_matrix(K, args.width, args.height, clip_start=1.0, clip_end=1000.0)
    else:
        bp.camera.set_intrinsics_from_blender_params(lens=0.785398, lens_unit='FOV', clip_start=1.0, clip_end=1000.0)

    light = bp.lighting.add_intersecting_spot_lights_to_camera_poses(5.0, 50.0)

    bp.renderer.set_output_format('PNG')
    bp.renderer.set_render_devices(desired_gpu_ids=[0])

    # Create objects
    objects = []
    objects_data = []
    for idx in range(args.nb_objects):
        model_path =  object_models[random.randint(0, len(object_models) - 1)]
        obj = bp.loader.load_obj(model_path)[0]
        obj.set_cp("category_id", 1+idx)
        objects.append(obj)
        obj_class = args.object_class
        obj_name = obj_class + "_" + str(idx).zfill(3)
        objects_data.append({'class': obj_class,
                             'name': obj_name,
                             'id':1+idx
                             })

    # Create distractors
    distractors = []
    if len(distractor_objs) > 0:
        for idx_obj in range(int(args.nb_distractors)):
            distractor_fn = distractor_objs[random.randint(0,len(distractor_objs)-1)]
            distractor = bp.loader.load_obj(distractor_fn)[0]
            distractor.set_cp("category_id", SEG_DISTRACT)
            distractors.append(distractor)

    for frame in range(args.nb_frames):

        # Place object(s) with SYMMETRY CONSTRAINTS
        for idx, oo in enumerate(objects):
            xform = np.eye(4)
            xform[0:3,3] = random_object_position(near=20, far=100)
            
            # --- MODIFIED HERE: Use the new symmetric rotation function ---
            # Uses command line args for limits (defaults to 360)
            xform[0:3,0:3] = get_constrained_rotation_matrix(args.sym_x, args.sym_y, args.sym_z)
            # ------------------------------------------------------------
            
            oo.set_local2world_mat(xform)

            xform_in_cam = np.linalg.inv(bp.camera.get_camera_pose()) @ xform
            objects_data[idx]['location'] = xform_in_cam[0:3,3].tolist()
            tmp_wxyz = Quaternion(matrix=xform_in_cam[0:3,0:3]).elements
            q_xyzw = [tmp_wxyz[1], tmp_wxyz[2], tmp_wxyz[3], tmp_wxyz[0]]
            objects_data[idx]['quaternion_xyzw'] = q_xyzw

            oo.set_scale([args.scale, args.scale, args.scale])

        # Place distractors
        for dd in distractors:
            xform = np.eye(4)
            xform[0:3,3] = point_in_frustrum(bp.camera, near=5.0, far=100.)
            # Distractors don't need symmetry constraints, keep them random (360)
            xform[0:3,0:3] = get_constrained_rotation_matrix(360, 360, 360)
            dd.set_local2world_mat(xform)
            dd.set_scale([args.distractor_scale, args.distractor_scale, args.distractor_scale])

        # Render
        background_path = None
        if args.backgrounds_folder:
            background_path = backdrop_images[random.randint(0, len(backdrop_images) - 1)]
            if os.path.splitext(background_path)[1].lower() == ".hdr":
                strength = random.random()+0.5
                rotation = [random.random()*0.2-0.1, random.random()*0.2-0.1, random.random()*0.2-0.1]
                set_world_background_hdr(background_path, strength, rotation)
            else:
                bp.renderer.set_output_format(enable_transparency=True)

        # Logging redirection
        logfile = '/tmp/blender_render.log'
        open(logfile, 'a').close()
        old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        fd = os.open(logfile, os.O_WRONLY)

        segs = bp.renderer.render_segmap()
        data = bp.renderer.render()

        os.close(fd)
        os.dup(old)
        os.close(old)

        im = Image.fromarray(data['colors'][0])

        if args.backgrounds_folder:
            if os.path.splitext(background_path)[1].lower() != ".hdr":
                background = randomize_background(background_path, args.width, args.height)
                background = background.convert('RGB')
                background.paste(im, mask=im.convert('RGBA'))
                im = background

        if args.debug:
            im = draw_cuboid_markers(objects, bp.camera, im)

        filename = os.path.join(out_directory, str(frame).zfill(6) + ".png")
        im.save(filename)

        filename = os.path.join(out_directory, str(frame).zfill(6) + ".json")
        write_json(filename, args, bp.camera, objects, objects_data, segs['class_segmaps'][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Standard Arguments ---
    parser.add_argument('--nb_runs', default=1, type=int, help='Number of times the datagen script is run.')
    parser.add_argument('--run_id', default=0, type=int)
    parser.add_argument('--width', default=500, type=int)
    parser.add_argument('--height', default=500, type=int)
    parser.add_argument('--focal-length', default=None, type=float)
    parser.add_argument('--distractors_folder', default='google_scanned_models/')
    parser.add_argument('--objs_folder', default='models/')
    parser.add_argument('--path_single_obj', default=None)
    parser.add_argument('--object_class', required=True)
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--backgrounds_folder', default=None)
    parser.add_argument('--nb_objects', default=1, type=int)
    parser.add_argument('--nb_distractors', default=1)
    parser.add_argument('--distractor_scale', default=50, type=float)
    parser.add_argument('--nb_frames', type=int, default=2000)
    parser.add_argument('--min_pixels', type=int, default=1)
    parser.add_argument('--outf', default='output_example/')
    parser.add_argument('--debug', action='store_true', default=False)

    # --- NEW SYMMETRY ARGUMENTS ---
    parser.add_argument('--sym_x', type=float, default=360.0, 
                        help='Max rotation for X axis in degrees. Use 180 for top/bottom symmetry.')
    parser.add_argument('--sym_y', type=float, default=360.0, 
                        help='Max rotation for Y axis in degrees.')
    parser.add_argument('--sym_z', type=float, default=360.0, 
                        help='Max rotation for Z axis in degrees. Use 90 for square prism symmetry.')

    opt = parser.parse_args()
    main(opt)