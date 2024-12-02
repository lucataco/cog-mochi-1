# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

MODEL_ID = "genmo/mochi-1-preview"
MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/genmo/mochi-1-preview/bf16.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.pipe = MochiPipeline.from_pretrained(
            MODEL_CACHE,
            variant="bf16",
            torch_dtype=torch.bfloat16
        )
        # Enable memory savings
        self.pipe.enable_model_cpu_offload()

    def predict(
        self,
        prompt: str = Input(
            description="Focus on a single, central subject. Structure the prompt from coarse to fine details. Start with 'a close shot' or 'a medium shot' if applicable. Append 'high resolution 4k' to reduce warping",
            default="Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
        ),
        num_frames: int = Input(description="Number of frames to generate", default=163, ge=30, le=170),
        num_inference_steps: int = Input(description="Number of inference steps", default=64, ge=10, le=200),
        guidance_scale: float = Input(description="The guidance scale for the model", default=6.0, ge=0.1, le=10.0),
        fps: int = Input(description="Frames per second", default=30, ge=10, le=60),
        seed: int = Input(description="Random seed", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        pipeline_args = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": 480,
            "width": 848,
            "max_sequence_length": 256,
            "output_type": "np",
            "generator": generator,
            "num_frames": num_frames,
        }
        video = self.pipe(**pipeline_args).frames[0]
        
        # Save the video to a file
        output_path = "/tmp/output.mp4"
        export_to_video(
            video,
            output_path,
            fps=fps
        )
        return Path(output_path)
