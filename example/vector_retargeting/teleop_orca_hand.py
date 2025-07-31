import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
import sapien
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector

# Import OrcaHand class
#from orca_core import OrcaHand

import pyrealsense2 as rs


class SapienVisualizer:
    def __init__(self, retargeting, config_path):
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")

        config = RetargetingConfig.load_from_file(config_path)

        # Setup
        scene = sapien.Scene()
        render_mat = sapien.render.RenderMaterial()
        render_mat.base_color = [0.06, 0.08, 0.12, 1]
        render_mat.metallic = 0.0
        render_mat.roughness = 0.9
        render_mat.specular = 0.8
        scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

        # Lighting
        scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
        scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
        scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
        scene.add_area_light_for_ray_tracing(
            sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
        )

        # Camera
        cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
        cam.set_local_pose(sapien.Pose([0.5, 0, 0.0], [0, 0, 0, -1]))

        self.viewer = Viewer()
        self.viewer.set_scene(scene)
        self.viewer.control_window.show_origin_frame = False
        self.viewer.control_window.move_speed = 0.01
        self.viewer.control_window.toggle_camera_lines(False)
        self.viewer.set_camera_pose(cam.get_local_pose())

        # Load robot and set it to a good pose to take picture
        loader = scene.create_urdf_loader()
        filepath = Path(config.urdf_path)
        robot_name = filepath.stem
        loader.load_multiple_collisions_from_file = True
        if "ability" in robot_name:
            loader.scale = 1.5
        elif "dclaw" in robot_name:
            loader.scale = 1.25
        elif "allegro" in robot_name:
            loader.scale = 1.4
        elif "shadow" in robot_name:
            loader.scale = 0.9
        elif "bhand" in robot_name:
            loader.scale = 1.5
        elif "leap" in robot_name:
            loader.scale = 1.4
        elif "svh" in robot_name:
            loader.scale = 1.5

        if "glb" not in robot_name:
            filepath = str(filepath).replace(".urdf", "_glb.urdf")
        else:
            filepath = str(filepath)

        robot = loader.load(filepath)
        self.robot = robot

        if "ability" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "shadow" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.2]))
        elif "dclaw" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "allegro" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.05]))
        elif "bhand" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.2]))
        elif "leap" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.15]))
        elif "svh" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.13]))
        elif "orca" in robot_name:
            robot.set_pose(sapien.Pose([0, 0, -0.13]))
        # Different robot loader may have different orders for joints
        sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        retargeting_joint_names = retargeting.joint_names
        self.retargeting_to_sapien = np.array(
            [retargeting_joint_names.index(name) for name in sapien_joint_names]
        ).astype(int)

    def update(self, qpos):
        self.robot.set_qpos(qpos[self.retargeting_to_sapien])
    def render(self):
        self.viewer.render()
        self.viewer.render()

def start_retargeting(
    queue: multiprocessing.Queue, robot_dir: str, config_path: str, orca_model_path: str, hand_type_str: str, use_visualizer: bool
):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    visualizer = None
    if use_visualizer:
        visualizer = SapienVisualizer(retargeting, config_path)

    # ================== OrcaHand Integration (START) ==================
    logger.info(f"Connecting to OrcaHand with model path: {orca_model_path}")
    #hand = OrcaHand(model_path=orca_model_path)
    #status, msg = hand.connect()
    #if not status:
    #    logger.error(f"Failed to connect to OrcaHand: {msg}")
    #    return
    
    # Set the control mode to a multi-turn mode to match calibration
    #hand.control_mode = 'current_based_position'
    #hand.init_joints()
    #logger.success("Successfully connected to and initialized OrcaHand.")

    # Create a mapping from the retargeting joint order to the OrcaHand joint name order
    retargeting_joint_names = retargeting.joint_names
    #orca_joint_names = hand.joint_ids
    
    # Ensure all OrcaHand joints can be found in the retargeting model by adding a prefix
    #expected_retargeting_joints = [f"{hand_type_str}_{name}" for name in orca_joint_names]
    #missing_joints = [name for name in expected_retargeting_joints if name not in retargeting_joint_names]
    #if missing_joints:
    #    logger.error(f"Mismatch between retargeting model and OrcaHand joints. Missing in retargeting (URDF): {missing_joints}")
    #    return
    # ================== OrcaHand Integration (END) ==================
    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)
    while True:
        loop_start_time = time.time()
        try:
            # First, a blocking get to wait for at least one frame.
            bgr = queue.get(timeout=5)

            # Then, drain the queue of any older frames to get the most recent one
            while not queue.empty():
                try:
                    bgr = queue.get_nowait()
                except Empty:
                    break  # The queue is empty, bgr is the latest frame
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            break

        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        if visualizer:
            visualizer.render()
        
        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
            continue
        
        retargeting_type = retargeting.optimizer.retargeting_type
        logger.info(f"Retargeting type: {retargeting_type}")
        indices = retargeting.optimizer.target_link_human_indices
        if retargeting_type == "POSITION":
            indices = indices
            ref_value = joint_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        qpos = retargeting.retarget(ref_value)

        if visualizer:
            visualizer.update(qpos)

        # send joint angles to OrcaHand
        joint_pos_dict = {}
        #for name in orca_joint_names:
            #prefixed_name = f"{hand_type_str}_{name}"
            # The check at the beginning ensures that the prefixed_name exists in retargeting_joint_names
            #rad_pos = qpos[retargeting_joint_names.index(prefixed_name)]
            #joint_pos_dict[name] = np.rad2deg(rad_pos)
        
        #hand.set_joint_pos(joint_pos_dict)

        # time recording
        loop_end_time = time.time()
        processing_time = (loop_end_time - loop_start_time) * 1000
        logger.info(f"Loop time: {processing_time:.2f} ms. Sent joint angles to OrcaHand.")
        
        # Read and log servo status
        #torque_status = hand.get_torque_status()
        #error_status = hand.get_hardware_error_status()
        #current_pos = hand.get_joint_pos(as_list=False) # Get current positions as dict
        #servo_goal_pos = hand.get_goal_joint_pos(as_dict=True) # Get goal positions from servo
        
        # Round all float values in dicts for cleaner logging
        #joint_pos_dict_rounded = {k: round(v, 4) for k, v in joint_pos_dict.items()}
        #current_pos_rounded = {k: (round(v, 4) if v is not None else None) for k, v in current_pos.items()}
        #servo_goal_pos_rounded = {k: (round(v, 4) if v is not None else None) for k, v in servo_goal_pos.items()}
        
        # Re-order all dictionaries based on the official joint order for consistent logging
        #ordered_cmd_goal = {k: joint_pos_dict_rounded.get(k) for k in orca_joint_names}
        #ordered_servo_goal = {k: servo_goal_pos_rounded.get(k) for k in orca_joint_names}
        #ordered_current_pos = {k: current_pos_rounded.get(k) for k in orca_joint_names}
        #ordered_torque_status = {k: torque_status.get(k) for k in orca_joint_names}
        #ordered_error_status = {k: error_status.get(k) for k in orca_joint_names}

        #logger.info(f"CMD Goal:    {ordered_cmd_goal}")
        #logger.info(f"Servo Goal:  {ordered_servo_goal}")
        #logger.info(f"Current Pos: {ordered_current_pos}")
        #logger.info(f"Torque Status: {ordered_torque_status}")
        #logger.info(f"Error Status: {ordered_error_status}")


    # Disconnect from the hand when the loop is broken
    #if 'hand' in locals() and hand.is_connected():
    #    hand.disconnect()
    #    logger.info("Disconnected from OrcaHand.")


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    logger.info("Starting RealSense camera pipeline...")
    try:
        pipeline.start(config)
    except Exception as e:
        logger.error(f"Failed to start RealSense pipeline: {e}")
        return

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            queue.put(image)
            # time.sleep(1 / 30.0)  # This sleep is redundant as wait_for_frames is blocking
    finally:
        pipeline.stop()
        logger.info("RealSense camera pipeline stopped.")


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    orca_model_path: str,
    camera_path: Optional[str] = None,
    use_visualizer: bool = False,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
        orca_model_path: The path to the OrcaHand model directory.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
        use_visualizer: Whether to show the SAPIEN visualizer.
    """
    # config that the path to urdf and yml which store the 
    # joints corresponding relationships between humanhand and robohand
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=2)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path), orca_model_path, hand_type.name, use_visualizer)
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
