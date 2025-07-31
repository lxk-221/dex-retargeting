import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
import pyrealsense2 as rs
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector

try:
    from dexh13_control_interface import DexH13Control, ControlMode
except ImportError as e:
    logger.warning(f"Could not import dexh13_control_interface: {e}. Real hand control will be unavailable.")
    DexH13Control = None
    ControlMode = None


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
        self.viewer.set_camera_pose(cam.get_local_pose())

        # Load robot and set it to a good pose to take picture
        loader = scene.create_urdf_loader()
        filepath = Path(config.urdf_path)
        robot_name = filepath.stem
        loader.load_multiple_collisions_from_file = True
        if "dexh13" in robot_name:
            loader.scale = 1.0
        
        if "glb" not in robot_name:
            filepath = str(filepath).replace(".urdf", "_glb.urdf")
        else:
            filepath = str(filepath)

        robot = loader.load(filepath)
        self.robot = robot
        robot.set_qpos(np.zeros(robot.dof))

        #if "dexh13" in robot_name:
        #    robot.set_pose(sapien.Pose([0, 0, -0.1], [0.7071068, 0, -0.7071068, 0]))

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


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str, use_filter: bool, use_visualizer: bool, use_real_hand: bool, port: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    visualizer = None
    if use_visualizer:
        logger.info("Initializing SAPIEN visualizer.")
        visualizer = SapienVisualizer(retargeting, config_path)

    hand_controller = None
    if use_real_hand:
        if DexH13Control is None:
            logger.error("DexH13 control library not imported. Cannot use real hand.")
        else:
            try:
                logger.info(f"Connecting to DexH13 hand on port {port}")
                hand_controller = DexH13Control()
                hand_controller.active_handy(port)
                if not hand_controller.is_connect_handy():
                    raise RuntimeError("Failed to connect to DexH13 hand.")

                if hand_controller.is_fault():
                    faults = hand_controller.get_fault_code()
                    fault_msgs = []
                    for fault in faults:
                        fault_msgs.append(f"finger: {fault.finger_name}, motor_id: {fault.motor_ID}, error: {fault.error_code.value}")
                    logger.error(f"DexH13 hand has faults: {'; '.join(fault_msgs)}")
                    raise RuntimeError("DexH13 hand has a fault.")

                logger.info("Setting POSITION control mode for DexH13 hand.")
                hand_controller.disable_motor()
                time.sleep(0.1)
                hand_controller.set_motor_control_mode(ControlMode.POSITION_CONTROL_MODE.value)
                time.sleep(2)  # Wait for mode to be set correctly
                current_mode = hand_controller.get_motor_control_mode()
                if current_mode != ControlMode.POSITION_CONTROL_MODE.value:
                    raise RuntimeError(f"Failed to set position control mode. Current mode: {current_mode}")

                hand_controller.enable_motor()
                time.sleep(0.1)
                if not hand_controller.is_motor_enabled():
                    raise RuntimeError("Failed to enable DexH13 motors.")

                logger.success("Successfully connected to and initialized DexH13 hand.")
            except Exception as e:
                logger.error(f"Failed to initialize DexH13 hand: {e}")
                if hand_controller and hand_controller.is_connect_handy():
                    hand_controller.disconnect_handy()
                hand_controller = None  # Ensure it's None on failure, so we don't try to use it.


    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    try:
        while True:
            loop_start_time = time.time()
            try:
                bgr = queue.get(timeout=5)
                while not queue.empty():
                    try:
                        bgr = queue.get_nowait()
                    except Empty:
                        break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Empty:
                logger.error("Fail to fetch image from camera in 5 secs. Please check your web camera device.")
                break

            _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            cv2.imshow("realtime_retargeting_demo", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if use_visualizer:
                visualizer.render()

            if joint_pos is None:
                logger.warning(f"{hand_type} hand is not detected.")
                continue
            
            retargeting_type = retargeting.optimizer.retargeting_type
            logger.info(f"Retargeting type: {retargeting_type}")
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            logger.info(f"human_hand_task_indices: {task_indices}")
            logger.info(f"human_hand_origin_indices: {origin_indices}")
            logger.info(f"human_hand_vector: {ref_value}")
            qpos = retargeting.retarget(ref_value)

            if use_visualizer:
                visualizer.update(qpos)

            if hand_controller:
                try:
                    hand_controller.set_joint_positions_radian(qpos.tolist())
                    logger.debug("Sent joint angles to real DexH13 hand.")
                except Exception as e:
                    logger.error(f"Error sending commands to real hand: {e}")

            retargeting.optimizer.robot.compute_forward_kinematics(qpos)

            def get_link_positions(optimizer, link_indices):
                """A helper function to get link positions."""
                link_poses = [optimizer.robot.get_link_pose(index) for index in optimizer.computed_link_indices]
                body_pos = np.array([pose[:3, 3] for pose in link_poses])
                return body_pos[link_indices]

            dexter_hand_joint_pos_origin = get_link_positions(retargeting.optimizer, retargeting.optimizer.origin_link_indices)
            dexter_hand_joint_pos_task = get_link_positions(retargeting.optimizer, retargeting.optimizer.task_link_indices)
            dexter_hand_vector = dexter_hand_joint_pos_task - dexter_hand_joint_pos_origin
            logger.info(f"dexter_hand_task_indices: {retargeting.optimizer.task_link_indices}")
            logger.info(f"dexter_hand_origin_indices: {retargeting.optimizer.origin_link_indices}")
            logger.info(f"dexter_hand_vector: {dexter_hand_vector}")

            logger.info(f"qpos: {qpos}") 
            loop_end_time = time.time()
            processing_time = (loop_end_time - loop_start_time) * 1000
            logger.info(f"Loop time: {processing_time:.2f} ms.")
    finally:
        if hand_controller and hand_controller.is_connect_handy():
            logger.info("Disabling motors and disconnecting from DexH13 hand.")
            hand_controller.disable_motor()
            hand_controller.disconnect_handy()


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
    finally:
        pipeline.stop()
        logger.info("RealSense camera pipeline stopped.")


def main(
    config_path: str = "../../src/dex_retargeting/configs/teleop/dexh13_right_dexpilot.yml",
    use_filter: bool = False,
    camera_path: Optional[str] = None,
    use_visualizer: bool = True,
    use_real_hand: bool = False,
    port: str = "/dev/ttyUSB0",
):
    """
    Detects human hand pose from a camera and retargets it to a DexH13 hand,
    with options to control a simulated and/or a real hand.
    Args:
        config_path: Path to the retargeting configuration file.
        use_filter: Whether to use a filter on the retargeting output.
        camera_path: Path to the camera device.
        use_visualizer: Whether to show the SAPIEN visualizer for the simulated hand.
        use_real_hand: Whether to control the real DexH13 hand.
        port: The serial port for the real DexH13 hand (e.g., /dev/ttyUSB0).
    """
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"

    queue = multiprocessing.Queue(maxsize=2)
    producer_process = multiprocessing.Process(target=produce_frame, args=(queue, camera_path))
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), config_path, use_filter, use_visualizer, use_real_hand, port)
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()


if __name__ == "__main__":
    tyro.cli(main)
