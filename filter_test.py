from argparse import ArgumentParser
import json
import os, time

import cv2
import numpy as np

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses
from kalman import Kalmanfilter,Point_Kalman_process

#Setting Kalman filter Object
kalman=Kalmanfilter()
kalman_process=Point_Kalman_process()

#Array to save joint coordinate and Array to send
listed_array = []
converted_array = []
for t in range(17):
    converted_array.extend([float(t),0,0])

isfirst = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#TCP/IP Networking Setting IP Address, Port number
import socket
TCP_IP = '192.168.1.161'
TCP_PORT = 5005

def get_info(listed_array,num): # Function to read coordinate data of each joint from an array
    x_data = listed_array[0][num][0]
    y_data = listed_array[0][num][1]
    z_data = listed_array[0][num][2]
    return x_data,y_data,z_data

def print_info(arr,num): # Function to print coordinate data of each joint from an array
    point_x, point_y, point_z = get_info(arr,num)
    print("joint number : {}".format(num))
    print('x :',end='')
    print(point_x)
    print('y :',end='')
    print(point_y)
    print('z :',end='')
    print(point_z)

def change_array(orig_num,changed_num): # function to adapt number between poses_3d and array for sending
    if not listed_array[0][orig_num]:
        if isfirst[changed_num] == 0 : 
            listed_array[0][orig_num]=[0,0,0]
        elif isfirst[changed_num] != 0 :
            listed_array[0][orig_num]=[converted_array[3*changed_num+0],converted_array[3*changed_num+1],converted_array[3*changed_num+2]]

    converted_array[3*changed_num+0] = listed_array[0][orig_num][0]
    converted_array[3*changed_num+1] = listed_array[0][orig_num][1]
    converted_array[3*changed_num+2] = listed_array[0][orig_num][2]
    isfirst[changed_num] = 1

def write_array(arr,arr_num,x,y,z): # function to assist for creating joint.
    arr[3*arr_num+0] = x
    arr[3*arr_num+1] = y
    arr[3*arr_num+2] = z

def create_joint(num1,num2,num3) : #num1, num2 : Input two point, num3 : Output middle points
    XL,YL,ZL = get_info(listed_array,num1)
    XR,YR,ZR = get_info(listed_array,num2)
    XM = (XL+XR)/2
    YM = (YL+YR)/2
    ZM = (ZL+ZR)/2
    write_array(converted_array,num3,XM,YM,ZM)

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


if __name__ == '__main__':
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-m', '--model',
                        help='Required. Path to checkpoint with a trained model '
                             '(or an .xml file in case of OpenVINO inference).',
                        type=str, required=True)
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default='')
    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                             'The demo will look for a suitable plugin for device specified '
                             '(by default, it is GPU).',
                        type=str, default='GPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                             'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true', default=False)
    parser.add_argument('--use-tensorrt',
                        help='Optional. Run network with TensorRT as inference engine. '
                             'Only Nvidia GPU devices are supported.',
                        action='store_true', default=False)
    parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default='')
    parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    stride = 8
    if args.use_openvino:
        from modules.inference_engine_openvino import InferenceEngineOpenVINO
        net = InferenceEngineOpenVINO(args.model, args.device)
    elif args.use_tensorrt:
        from modules.inference_engine_tensorrt import InferenceEngineTensorRT
        net = InferenceEngineTensorRT(args.model, args.device)
    else:
        from modules.inference_engine_pytorch import InferenceEnginePyTorch
        net = InferenceEnginePyTorch(args.model, args.device)

    # canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    # plotter = Plotter3d(canvas_3d.shape[:2])
    # canvas_3d_window_name = 'Canvas 3D'
    # cv2.namedWindow(canvas_3d_window_name)
    # cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    frame_provider = ImageReader(args.images)
    is_video = False
    if args.video != '':
        frame_provider = VideoReader(args.video)
        is_video = True
    base_height = args.height_size
    fx = args.fx

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    for frame in frame_provider:
        current_time = cv2.getTickCount()
        if frame is None:
            break

        #reset X(P,Q), Y(P,Q), Z(P,Q)
        kalman_process.reset_kalman_filter_X
        kalman_process.reset_kalman_filter_Y
        kalman_process.reset_kalman_filter_Z
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])
        t0 = time.time()
        inference_result = net.infer(scaled_img)
        print('Infer: {:1.3f}'.format(time.time()-t0))
        # print(type(inference_result))
        # for ar in inference_result:
        #     print(ar.shape)
        # break
        t0 = time.time()
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        print('Extract: {:1.3f}'.format(time.time()-t0))
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            # print("array : {}".format(poses_3d))
            # print("x : {}".format(poses_3d[0,0,0]))
            # print("y : {}".format(poses_3d[0,0,1]))
            # print("z : {}".format(poses_3d[0,0,2]))
            #edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            
        #plotter.plot(canvas_3d, poses_3d, edges)
        # continue
        #cv2.imshow(canvas_3d_window_name, canvas_3d)
        #change number of the pose_3d adapting for the array to send
        listed_array = poses_3d.tolist()
        #print(listed_array)
        #print("len(listed_array) : {}".format(len(listed_array)))
        if len(listed_array) >= 1 :
            maxlength_person = [listed_array[0]]
            for i in listed_array :
                #print("len(maxlength_person) : {}".format(maxlength_person))
                #print("len(i) : {}".format(i))
                if len(i) > len(maxlength_person) :
                    maxlength_person = [i]
            listed_array = maxlength_person
            change_array(2,0)
            change_array(12,1)
            change_array(13,2)
            change_array(14,3)
            change_array(6,4)
            change_array(7,5)
            change_array(8,6)
            change_array(0,8)
            change_array(1,9)
            change_array(3,11)
            change_array(4,12)
            change_array(5,13)
            change_array(9,14)
            change_array(10,15)
            change_array(11,16)

            create_joint(0,2,7)
            create_joint(15,16,10)
            
            for i,poses in enumerate(converted_array):
                if i%3 == 0 :
                    id = i/3
                    if id == 0:
                        label = 'hip'
                    elif id == 1:
                        label = 'left hip'
                    elif id == 2:
                        label = 'left knee'
                    elif id == 3:
                        label = 'left ankle'
                    elif id == 4:
                        label = 'right hip'
                    elif id == 5:
                        label = 'right knee'
                    elif id == 6:
                        label = 'right ankle'
                    elif id == 7:
                        label = 'stomach'
                    elif id == 8:
                        label = 'heart'
                    elif id == 9:
                        label = 'neck'
                    elif id == 10:
                        label = 'nose'
                    elif id == 11:
                        label = 'right shoulder'
                    elif id == 12:
                        label = 'right elbow'
                    elif id == 13:
                        label = 'right wrist'
                    elif id == 14:
                        label = 'left shoulder'
                    elif id == 15:
                        label = 'left elbow'
                    elif id == 16:
                        label = 'left wrist'
                    converted_array[i],converted_array[i+1],converted_array[i+2]=kalman_process.do_kalman_filter(converted_array[i],converted_array[i+1],converted_array[i+2],label)
                    # converted_array[i] = out_kalman_x
                    # converted_array[i+1] = out_kalman_y
                    # converted_array[i+2] = out_kalman_z
            print(converted_array)
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP/IP 초기세팅
        # s.connect((TCP_IP, TCP_PORT))
        # send_array = " ".join(str(x) for x in converted_array)
        # s.sendall(bytes(str(send_array),encoding = 'utf-8'))
        # s.close()
        # print('Success Sending')
        # draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        print('FPS: {}'.format(int(1 / mean_time * 10) / 10))
        # cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
        #             (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        # cv2.putText(frame, 'Time: {:1.3f}'.format(current_time),
        #             (40, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        # cv2.imshow('ICV 3D Human Pose Estimation', frame)

        # key = cv2.waitKey(delay)
        # if key == esc_code:
        #     break
        # if key == p_code:
        #     if delay == 1:
        #         delay = 0
        #     else:
        #         delay = 1
        # if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
        #     key = 0
        #     while (key != p_code
        #            and key != esc_code
        #            and key != space_code):
        #         #plotter.plot(canvas_3d, poses_3d, edges)
        #         #cv2.imshow(canvas_3d_window_name, canvas_3d)
        #         key = cv2.waitKey(33)
        #     if key == esc_code:
        #         break
        #     else:
        #         delay = 1




