
import argparse
import cv2 as cv
import numpy as np
import open3d as o3d
from os import path, system
from lib.accl import AsyncAccl

###############################################################################
###############################################################################
###############################################################################

class PointCloudFromDepth:
    """
    A class that handles real-time depth estimation and generates a point cloud 
    from depth data using the camera and an AI model.
    
    Attributes:
        model_path (str): Path to the model file (TFLite or compiled DFP).
        cam_src (str): Path to the camera source.
        input_width (int): Width of the camera input frame.
        input_height (int): Height of the camera input frame.
        current_point_cloud (open3d.geometry.PointCloud): Point cloud object updated from depth.
        vis (open3d.visualization.VisualizerWithKeyCallback): Visualizer for rendering the point cloud.
        intrinsic (open3d.camera.PinholeCameraIntrinsic): Camera intrinsic parameters.
        accl (memryx.AsyncAccl): Async accelerator for model inference.
        depth_map (np.ndarray): Latest depth map from inference.
    """

    def __init__(self, model_path, dfp, cam_src=0):
        """
        Initializes the PointCloudFromDepth object, sets up the camera, and defines 
        camera intrinsics for generating the point cloud.
        
        Args:
            model_path (str): Path to the depth estimation model.
            dfp (str): Path to the compiled DFP file.
            cam_src (str): Path to the camera source (e.g., video0 for Linux).
        """
        self.model_path = model_path
        self.dfp = dfp
        self.cam_src = cam_src
        self.cam = cv.VideoCapture(self.cam_src)
        self.input_height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.current_point_cloud = None
        self.vis = None
        self.view_control = None
        self.accl = None
        self.depth_map = None
        
        # Set up intrinsic camera parameters based on actual camera properties'
        # Adjust based on your camera properties
        fx = self.input_width
        fy = self.input_height
        cx = self.input_width/2
        cy = self.input_height/2

        # Create a PinholeCameraIntrinsic object
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.input_width, 
            height=self.input_height, 
            fx=fx, 
            fy=fy, 
            cx=cx, 
            cy=cy
        )

###############################################################################

    def check_model(self):
        """
        Checks if the model is downloaded or compiled and downloads or compiles 
        it if necessary.
        """
        model_tar_path = "./midas_v2_small.tar.gz"
        extracted_file = "./1.tflite"
        
        # Use self.model_path and self.dfp
        target_model_path = self.model_path
        dfp_file = self.dfp

        if path.isfile(dfp_file):
            print("\033[93mCompiled model found. Skipping the download step.\033[0m")
        elif path.isfile(target_model_path):
            print("\033[93mModel found. Skipping the download step.\033[0m")
        else:
            print("\033[93mDownloading the model for the first time.\033[0m")
            system(f"curl -L -o {model_tar_path} https://www.kaggle.com/api/v1/models/intel/midas/tfLite/v2-1-small-lite/1/download")
            
            print("\033[93mModel downloaded. Extracting the model.\033[0m")
            
            # Use system to extract the tar.gz file
            system(f"tar -xzf {model_tar_path} -C ./")

            # Rename the extracted file (assuming it's named '1.tflite')
            if path.isfile(extracted_file):
                system(f"mv {extracted_file} {target_model_path}")
                print(f"\033[93mModel extraction completed and renamed to '{target_model_path}'.\033[0m")
            else:
                print("\033[91mError: Extracted file '1.tflite' not found.\033[0m")

###############################################################################

    def compile_model(self):
        """
        Compiles the model into a DFP format if necessary.
        """
        if path.isfile(self.dfp):
            print("\033[93mCompiled model found. Skipping the compilation step.\033[0m")
        else:
            raise RuntimeError("\033[91mDFP not found\033[0m")

###############################################################################

    def get_frame_and_preprocess(self):
        """
        Captures a new frame from the camera and preprocesses it for inference.
        
        Returns:
            np.ndarray: Preprocessed frame ready for model inference.
        """
        got_frame, frame = self.cam.read()
        if not got_frame:
            return None
        
        # Preprocess the frame
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) / 255.0
        frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_CUBIC)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        frame = (frame - mean) / std

        return frame.astype("float32")

###############################################################################

    def get_output_and_postprocess(self, *accl_output):
        """
        Postprocesses the depth data and updates the point cloud.
        
        Args:
            accl_output (tuple): Output from the model inference (depth map).
        """
        prediction = accl_output[0]
        self.depth_map = cv.resize(prediction, (self.input_width, self.input_height))


###############################################################################

    def setup_visualizer(self):
        """
        Sets up the Open3D visualizer and initializes a point cloud object.
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='Real-time Point Cloud', width=self.input_width, height=self.input_height)
        self.view_control = self.vis.get_view_control()

        # Add a key callback to exit the loop when 'q' is pressed
        self.vis.register_key_callback(ord("Q"), self.exit_callback)

        # Create an empty point cloud object and add it to the scene
        self.current_point_cloud = o3d.geometry.PointCloud()

###############################################################################

    def exit_callback(self, vis):
        """
        Callback function to close the visualizer when 'q' is pressed.
        
        Args:
            vis (open3d.visualization.Visualizer): Visualizer object.
        """
        print("Exiting...")
        vis.close()

###############################################################################

    def run_inference(self):
        """
        Runs the asynchronous inference on MXA and visualizes the point cloud.
        """
        self.accl = AsyncAccl(self.dfp)
        self.accl.connect_input(self.get_frame_and_preprocess)
        self.accl.connect_output(self.get_output_and_postprocess)
        print("\033[93mRunning Real-Time Inference\033[0m")

        self.render_point_cloud()

###############################################################################

    def render_point_cloud(self):
        """
        Renders the updated point cloud from depth data in real-time.
        """
        first_render = True

        while self.vis.poll_events():
            if self.depth_map is not None:
                # Set a range for depth values
                min_depth = 0     # Adjust as needed
                max_depth = 2000  # Adjust as needed
                
                # Ensure depth values don't go below min_depth, then invert and scale
                clipped_depth = np.clip(self.depth_map, min_depth, max_depth)
                
                # Invert depth map (closer objects get smaller values, farther objects get larger)
                inverted_depth = max_depth - clipped_depth
                
                # Scale the inverted depth map to the range [100, 255]
                scaled_depth = ((inverted_depth - min_depth) / (max_depth - min_depth)) * 255

                # Flip for correct orientation
                flipped_depth = cv.flip(scaled_depth, 0)

                # Convert the depth colormap to Open3D format
                depth_image_o3d = o3d.geometry.Image(flipped_depth.astype(np.float32))

                points = o3d.geometry.PointCloud.create_from_depth_image(
                    depth=depth_image_o3d,
                    intrinsic=self.intrinsic,
                    depth_scale=1000.0,
                    depth_trunc=5)

                # Apply transformation to fix camera orientation (rotate 180 degrees around Y-axis)
                transformation_matrix = [[-1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]]
                points.transform(transformation_matrix)

                if first_render:
                    # Add the point cloud for the first time
                    self.current_point_cloud = points
                    first_render = False
                    self.vis.add_geometry(self.current_point_cloud)
                else:
                    # Update the existing point cloud with new points and colors
                    self.current_point_cloud.points = points.points
                    self.current_point_cloud.colors = points.colors
                    self.vis.update_geometry(self.current_point_cloud)
                
                self.vis.poll_events()
                self.vis.update_renderer()

###############################################################################

    def cleanup(self):
        """
        Cleans up resources such as the camera and visualizer.
        """
        self.cam.release()
        self.vis.destroy_window()

###############################################################################

def main():
    """
    Main function to initialize, run, and clean up the point cloud depth estimator.
    """
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Run MX3 real-time inference with options for model path and DFP file.")
    parser.add_argument('-m', '--model', type=str, default="MiDaS_256_256_3_tflite.tflite", help="Specify the path to the model. Default is 'MiDaS_256_256_3_tflite.tflite'.")
    parser.add_argument('-d', '--dfp', type=str, default="models/MiDaS_256_256_3_tflite.dfp", help="Specify the path to the compiled DFP file. Default is 'models/MiDaS_256_256_3_tflite.dfp'.")
    args = parser.parse_args()

    # Initialize the PointCloudFromDepth object with the provided arguments
    point_cloud_from_depth = PointCloudFromDepth(model_path=args.model, dfp=args.dfp)
    point_cloud_from_depth.setup_visualizer()
    point_cloud_from_depth.run_inference()
    point_cloud_from_depth.cleanup()

if __name__ == "__main__":
    main()

# eof
