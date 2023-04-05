import base64
import torch
from io import BytesIO
from random import randint

from diffusers import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,DDIMScheduler

# need add more sampler
samplers = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "eular": EulerDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpm2": KDPM2DiscreteScheduler,
    "dpm2_a": KDPM2AncestralDiscreteScheduler,
    "ddim": DDIMScheduler
}

def model_fn(model_dir):
    # Load stable diffusion and move it to the GPU
    pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    return pipe


def predict_fn(data, pipe):
    # get prompt & parameters
    prompt = data.pop("prompt", data)
    # set valid HP for stable diffusion
    height = data.pop("height", 512)
    width = data.pop("width", 512)
    num_inference_steps = data.pop("num_inference_steps", 50)
    negative_prompt = data.pop("negative_prompt", "")
    guidance_scale = data.pop("guidance_scale", 7.5)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)

    seed = data.pop("seed", randint(0, 100000))
    generator = torch.Generator(
                device='cuda').manual_seed(seed)
    
    sampler = data.pop("sampler", None)
    if sampler is not None:
        sampler = samplers[sampler] if sampler in samplers else samplers["euler_a"]
    
    # set sampler
    pipe.scheduler = sampler.from_config(
                pipe.scheduler.config)
    
    # run generation with parameters
    generated_images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator
    )["images"]

    # create response
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # create response
    return {"generated_images": encoded_images}
