import torch
import numpy as np
import cv2
from PIL import Image
from realcugan_ncnn_py import Realcugan # đúng theo lib bạn dùng
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"

model2x = Realcugan(gpuid=1,scale=2,noise=0,model="models-se")
model3x = Realcugan(gpuid=1,scale=3,noise=0,model="models-se")
model4x = Realcugan(gpuid=1,scale=4,noise=0,model="models-se")

def upscale_image2x(img_path):
    img_pil = Image.open(img_path).convert("RGB")
    
    img_rgb = np.array(img_pil)

    sr_rgb = model2x.process_cv2(img_rgb)

    sr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
    return sr

def upscale_image3x(input_path):
    img_pil = Image.open(input_path).convert("RGB")

    img_rgb = np.array(img_pil)

    sr_rgb = model2x.process_cv2(img_rgb)

    sr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)

    return sr


    

def upscale_image4x(img_path):
    img_pil = Image.open(img_path).convert("RGB")

    
    img_rgb = np.array(img_pil)

    sr_rgb = model4x.process_cv2(img_rgb)

    sr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
    return sr


def upscale_video(input_video_path, output_video_path):
        realcugan = Realcugan(gpuid=0, scale = 2, noise = 0, model = "models-se")

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate new dimensions
        

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # Codec for .mp4 files
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Error: Could not create video writer for {output_video_path}")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame with Real-CUGAN
            upscaled_frame = realcugan.process_cv2(frame)
            upscaled_frame = cv2.resize(upscaled_frame,(width,height))
            

            # Write the upscaled frame to the output video
            out.write(upscaled_frame)

            frame_count += 1
            print(f"Processed frame {frame_count}")

        # Release resources
        cap.release()
        out.release()

def upscale_video4x(input_video_path, output_video_path):
        realcugan = Realcugan(gpuid=0, scale = 4, noise = 1, model = "models-se")

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate new dimensions
        

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # Codec for .mp4 files
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height*2))

        if not out.isOpened():
            print(f"Error: Could not create video writer for {output_video_path}")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame with Real-CUGAN
            upscaled_frame = realcugan.process_cv2(frame)
            

            # Write the upscaled frame to the output video
            out.write(upscaled_frame)

            frame_count += 1
            print(f"Processed frame {frame_count}")

        # Release resources
        cap.release()
        out.release()
        
        gc.collect()
        with torch.no_grad():
             ...
             torch.cuda.synchronize()
             torch.cuda.empty_cache()
