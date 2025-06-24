import dotenv

dotenv.load_dotenv(override=True)

import gradio as gr

import os
import argparse
import random
from datetime import datetime
from PIL import Image
import json

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

import piexif
import piexif.helper

from accelerate import Accelerator

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from omnigen2.utils.img_util import create_collage

NEGATIVE_PROMPT = "" #"(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model_cache_dir = f"{ROOT_DIR}/weights"

pipeline = None
accelerator = None
save_images = False
preview_image = None
interrupt_generation = False
last_seed = -1

def load_pipeline(accelerator, weight_dtype, args):
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
        cache_dir=model_cache_dir,
        local_files_only=True
    )
    pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        cache_dir=model_cache_dir,
        local_files_only=True
    )
    pipeline.transformer = pipeline.quantize_transformer(8)
    #pipeline.mllm = pipeline.quantize_mllm(8)
    #pipeline.vae.to(torch.float32)
    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(accelerator.device)
    return pipeline


def handle_interrupt_btn_click():
    global interrupt_generation
    interrupt_generation = not interrupt_generation
    if interrupt_generation:
        return gr.HTML(f'<span style="color:red"> Stopping generation... </span>',
                                                visible=True)
    else:
        return gr.HTML("", visible=False)


def preview_to_image(latent_image):
        latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            ).to(device="cpu", dtype=torch.uint8)

        return Image.fromarray(latents_ubyte.numpy())


class Latent2RGBPreviewer():
    def __init__(self, latent_rgb_factors):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu")

    def decode_latent_to_preview_simple(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        latent_image = x0[0].permute(1, 2, 0) @ self.latent_rgb_factors
        return preview_to_image(latent_image)


def preview_callback(iteration, latents):
    global preview_image
    if interrupt_generation:
        pipeline.interrupt = True
    if iteration % 3 != 0:
        return
    # convert latents to image
    # https://github.com/comfyanonymous/ComfyUI/blob/38c22e631ad090a4841e4a0f015a30c565a9f7fc/comfy/latent_formats.py
    latent_rgb_factors =[
            [-0.0404,  0.0159,  0.0609],
            [ 0.0043,  0.0298,  0.0850],
            [ 0.0328, -0.0749, -0.0503],
            [-0.0245,  0.0085,  0.0549],
            [ 0.0966,  0.0894,  0.0530],
            [ 0.0035,  0.0399,  0.0123],
            [ 0.0583,  0.1184,  0.1262],
            [-0.0191, -0.0206, -0.0306],
            [-0.0324,  0.0055,  0.1001],
            [ 0.0955,  0.0659, -0.0545],
            [-0.0504,  0.0231, -0.0013],
            [ 0.0500, -0.0008, -0.0088],
            [ 0.0982,  0.0941,  0.0976],
            [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020],
            [-0.1273, -0.0932, -0.0680]
        ]
    previewer = Latent2RGBPreviewer(latent_rgb_factors)
    preview_image = previewer.decode_latent_to_preview_simple(latents)


def run(
    instruction,
    width_input,
    height_input,
    scheduler,
    num_inference_steps,
    image_input_1,
    image_input_2,
    image_input_3,
    negative_prompt,
    guidance_scale_input,
    img_guidance_scale_input,
    cfg_range_start,
    cfg_range_end,
    num_images_per_prompt,
    max_input_image_side_length,
    max_pixels,
    seed_input,
    system_prompt,
    progress=gr.Progress(),
):
    global interrupt_generation, last_seed
    interrupt_generation = False
    pipeline.interrupt = False

    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]

    if len(input_images) == 0:
        input_images = None

    if seed_input == -1:
        seed_input = random.randint(0, 2**16 - 1)
    last_seed = seed_input

    generator = torch.Generator(device=accelerator.device).manual_seed(seed_input)

    def progress_callback(cur_step, timesteps):
        frac = (cur_step + 1) / float(timesteps)
        progress(frac)

    if scheduler == 'euler':
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    elif scheduler == 'dpmsolver':
        pipeline.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width_input,
        height=height_input,
        max_input_image_side_length=max_input_image_side_length,
        max_pixels=max_pixels,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=guidance_scale_input,
        image_guidance_scale=img_guidance_scale_input,
        cfg_range=(cfg_range_start, cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
        step_func=progress_callback,
        system_prompt=system_prompt,
        preview_callback=preview_callback
    )

    progress(1.0)

    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
    output_image = create_collage(vis_images)

    if save_images:
        # Create outputs directory if it doesn't exist
        output_dir = os.path.join(ROOT_DIR, "outputs_gradio")
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Generate unique filename with timestamp
        output_path = os.path.join(output_dir, f"{timestamp}.jpg")
        # Save the image
        output_image.save(output_path)

        geninfo = json.dumps({"prompt": instruction,
                   "negative_prompt": negative_prompt,
                   "num_inference_steps": num_inference_steps,
                   "text_guidance_scale": guidance_scale_input,
                   "image_guidance_scale": img_guidance_scale_input,
                   "seed": last_seed,
                   "cfg_range": [cfg_range_start, cfg_range_end],
                   "system_prompt": system_prompt})
        exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo or "", encoding="unicode")
                },
                })
        piexif.insert(exif_bytes, output_path)

        # Save All Generated Images
        if len(results.images) > 1:
            for i, image in enumerate(results.images):
                image_name, ext = os.path.splitext(output_path)
                image.save(f"{image_name}_{i}{ext}")

                piexif.insert(exif_bytes, f"{image_name}_{i}{ext}")
    return output_image


def get_example():
    case = [
        [
            "The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens.",
            1024,
            1024,
            "euler",
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            3.5,
            1.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "A snow maiden with pale translucent skin, frosty white lashes, and a soft expression of longing",
            1024,
            1024,
            "euler",
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            3.5,
            1.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Add a fisherman hat to the woman's head",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/flux5.jpg"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            " replace the sword with a hammer.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/d8f8f44c64106e7715c61b5dfa9d9ca0974314c5d4a4a50418acf7ff373432bb.jpg",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Extract the character from the picture and fill the rest of the background with white.",
            # "Transform the sculpture into jade",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR, "example_images/46e79704-c88e-4e68-97b4-b4c40cd29826.jpg"
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Make he smile",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR, "example_images/vicky-hladynets-C8Ta0gwPbQg-unsplash.jpg"
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Change the background to classroom",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/ComfyUI_temp_mllvz_00071_.jpg"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Raise his hand",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/289089159-a6d7abc142419e63cab0a566eb38e0fb6acb217b340f054c6172139b316f6596.jpg",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Generate a photo of an anime-style figurine placed on a desk. The figurine model should be based on the character photo provided in the attachment, accurately replicating the full-body pose, facial expression, and clothing style of the character in the photo, ensuring the entire figurine is fully presented. The overall design should be exquisite and detailed, soft gradient colors and a delicate texture, leaning towards a Japanese anime style, rich in details, with a realistic quality and beautiful visual appeal.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/RAL_0315.JPG"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Change the dress to blue.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/1.jpg"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Remove the cat",
            1024,
            1024,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/386724677-589d19050d4ea0603aee6831459aede29a24f4d8668c62c049f413db31508a54.jpg",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "In a cozy café, the anime figure is sitting in front of a laptop, smiling confidently.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/ComfyUI_00254_.jpg"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Create a wedding figure based on the girl in the first image and the man in the second image. Set the background as a wedding hall, with the man dressed in a suit and the girl in a white wedding dress. Ensure that the original faces remain unchanged and are accurately preserved. The man should adopt a realistic style, whereas the girl should maintain their classic anime style.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/1_20241127203215.jpg"),
            os.path.join(ROOT_DIR, "example_images/000050281.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            3.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Let the girl  and the boy get married in the church. ",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/8FtFUxRzXqaguVRGzkHvN.jpg"),
            os.path.join(ROOT_DIR, "example_images/01194-20240127001056_1024x1536.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            3.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Let the man from image1 and the woman from image2 kiss and hug",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/1280X1280.jpg"),
            os.path.join(ROOT_DIR, "example_images/000077066.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Please let the person in image 2 hold the toy from the first image in a parking lot.",
            1024,
            1024,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/04.jpg"),
            os.path.join(ROOT_DIR, "example_images/000365954.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Make the girl pray in the second image.",
            1024,
            682,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/000440817.jpg"),
            os.path.join(ROOT_DIR, "example_images/000119733.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Add the bird from image 1 to the desk in image 2",
            1024,
            682,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/996e2cf6-daa5-48c4-9ad7-0719af640c17_1748848108409.jpg",
            ),
            os.path.join(ROOT_DIR, "example_images/00066-10350085.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Replace the apple in the first image with the cat from the second image",
            1024,
            780,
            "euler",
            50,
            os.path.join(ROOT_DIR, "example_images/apple.jpg"),
            os.path.join(
                ROOT_DIR,
                "example_images/468404374-d52ec1a44aa7e0dc9c2807ce09d303a111c78f34da3da2401b83ce10815ff872.jpg",
            ),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Replace the woman in the second image with the woman from the first image",
            1024,
            747,
            "euler",
            50,
            os.path.join(
                ROOT_DIR, "example_images/byward-outfitters-B97YFrsITyo-unsplash.jpg"
            ),
            os.path.join(
                ROOT_DIR, "example_images/6652baf6-4096-40ef-a475-425e4c072daf.jpg"
            ),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "The cat is sitting on the table. The bird is perching on the edge of the table.",
            800,
            512,
            "euler",
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/996e2cf6-daa5-48c4-9ad7-0719af640c17_1748848108409.jpg",
            ),
            os.path.join(
                ROOT_DIR,
                "example_images/468404374-d52ec1a44aa7e0dc9c2807ce09d303a111c78f34da3da2401b83ce10815ff872.jpg",
            ),
            os.path.join(ROOT_DIR, "example_images/00066-10350085.jpg"),
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
    ]
    return case


def run_for_examples(
    instruction,
    width_input,
    height_input,
    scheduler,
    num_inference_steps,
    image_input_1,
    image_input_2,
    image_input_3,
    negative_prompt,
    text_guidance_scale_input,
    image_guidance_scale_input,
    cfg_range_start,
    cfg_range_end,
    num_images_per_prompt,
    max_input_image_side_length,
    max_pixels,
    seed_input,
    system_prompt
):
    return run(
        instruction,
        width_input,
        height_input,
        scheduler,
        num_inference_steps,
        image_input_1,
        image_input_2,
        image_input_3,
        negative_prompt,
        text_guidance_scale_input,
        image_guidance_scale_input,
        cfg_range_start,
        cfg_range_end,
        num_images_per_prompt,
        max_input_image_side_length,
        max_pixels,
        seed_input,
        system_prompt
    )

description = """
#### 💡 Quick Tips for Best Results (see our [github](https://github.com/VectorSpaceLab/OmniGen2?tab=readme-ov-file#-usage-tips) for more details)
- Image Quality: Use high-resolution images (at least 512x512 recommended).
- Be Specific: Instead of "Add bird to desk", try "Add the bird from image 1 to the desk in image 2".
- Use English: English prompts currently yield better results.
- Adjust image_guidance_scale for better consistency with the reference image:
    - Image Editing: 1.3 - 2.0
    - In-context Generation: 2.0 - 3.0
- `cfg_range_start`, `cfg_range_end`:
  Define the timestep range where CFG is applied. Per [this paper](https://arxiv.org/abs/2404.07724), reducing `cfg_range_end` can significantly decrease inference time with a negligible impact on quality.
"""

article = """
[Technical Report](https://arxiv.org/abs/2506.18871)
"""

def main(args):
    # Gradio
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(
                "# OmniGen2: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen2)"
            )
            gr.Markdown(description)
            preview_image_element = gr.Image(value=lambda: preview_image, every=1, width=256, height=256, interactive=False, label="Preview")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Edit system prompt", open=False):
                    system_prompt = gr.Textbox(show_label=False,
                                               value="You are a helpful assistant that generates high-quality images based on user instructions.")
                # text prompt
                instruction = gr.Textbox(
                    label='Enter your prompt. Use "first/second image" or “第一张图/第二张图” as reference.',
                    placeholder="Type your prompt here...",
                )

                with gr.Row(equal_height=True):
                    # input images
                    image_input_1 = gr.Image(label="First Image", type="pil")
                    image_input_2 = gr.Image(label="Second Image", type="pil")
                    image_input_3 = gr.Image(label="Third Image", type="pil")

                with gr.Row():
                    interrupt_btn = gr.Button("Stop", variant="stop")
                    generate_button = gr.Button("Generate Image")
                interrupt_label = gr.HTML("", visible=False)

                negative_prompt = gr.Textbox(
                    label="Enter your negative prompt (May cause nan error when editing images with quantization enabled)",
                    placeholder="Type your negative prompt here...",
                    value=NEGATIVE_PROMPT,
                )

                # slider
                with gr.Row(equal_height=True):
                    height_input = gr.Slider(
                        label="Height", minimum=256, maximum=1024, value=1024, step=128
                    )
                    width_input = gr.Slider(
                        label="Width", minimum=256, maximum=1024, value=1024, step=128
                    )
                with gr.Row(equal_height=True):
                    text_guidance_scale_input = gr.Slider(
                        label="Text Guidance Scale (CFG)",
                        minimum=1.0,
                        maximum=8.0,
                        value=5.0,
                        step=0.1,
                    )

                    image_guidance_scale_input = gr.Slider(
                        label="Image Guidance Scale",
                        minimum=1.0,
                        maximum=6.0,
                        value=2.0,
                        step=0.1,
                    )
                with gr.Row(equal_height=True):
                    cfg_range_start = gr.Slider(
                        label="CFG Range Start",
                        info="Best quality: 0.2 | Best trade-off when image editing: 0.05",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.05,
                        step=0.05,
                    )

                    cfg_range_end = gr.Slider(
                        label="CFG Range End",

                        minimum=0.0,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                    )
                
                def adjust_end_slider(start_val, end_val):
                    return max(start_val, end_val)

                def adjust_start_slider(end_val, start_val):
                    return min(end_val, start_val)
                
                cfg_range_start.input(
                    fn=adjust_end_slider,
                    inputs=[cfg_range_start, cfg_range_end],
                    outputs=[cfg_range_end]
                )

                cfg_range_end.input(
                    fn=adjust_start_slider,
                    inputs=[cfg_range_end, cfg_range_start],
                    outputs=[cfg_range_start]
                )

                with gr.Row(equal_height=True):
                    scheduler_input = gr.Dropdown(
                        label="Scheduler",
                        choices=["euler", "dpmsolver"],
                        value="euler",
                        info="The scheduler to use for the model.",
                    )

                    num_inference_steps = gr.Slider(
                        label="Inference Steps", minimum=5, maximum=100, value=35, step=1
                    )
                with gr.Row(equal_height=True):
                    num_images_per_prompt = gr.Slider(
                        label="Number of images per prompt",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                    )
                    with gr.Column():
                        seed_input = gr.Slider(
                            label="Seed", minimum=-1, maximum=2147483647, value=-1, step=1
                        )
                        reuse_seed = gr.Button("Re-use seed")
                with gr.Row(equal_height=True):
                    max_input_image_side_length = gr.Slider(
                        label="max_input_image_side_length",
                        minimum=256,
                        maximum=2048,
                        value=2048,
                        step=256,
                    )
                    max_pixels = gr.Slider(
                        label="max_pixels",
                        minimum=256 * 256,
                        maximum=1536 * 1536,
                        value=1024 * 1024,
                        step=256 * 256,
                    )

            with gr.Column():
                with gr.Column():
                    # output image
                    output_image = gr.Image(label="Output Image")
                    global save_images
                    save_images = gr.Checkbox(label="Save generated images", value=True)
                    gr.Textbox(label="Default negative prompt", value="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar", interactive=False)

        global accelerator
        global pipeline

        bf16 = False
        accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
        weight_dtype = torch.bfloat16 if bf16 else torch.float16

        pipeline = load_pipeline(accelerator, weight_dtype, args)


        interrupt_btn.click(fn=handle_interrupt_btn_click, inputs=[], outputs=[interrupt_label], show_progress="hidden")
        reuse_seed.click(lambda: last_seed, None, seed_input)

        # click
        generate_button.click(
            run,
            inputs=[
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                image_input_1,
                image_input_2,
                image_input_3,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels,
                seed_input,
                system_prompt
            ],
            outputs=output_image,
        ).then(lambda: gr.HTML("", visible=False), None, interrupt_label)

        gr.Examples(
            examples=get_example(),
            fn=run_for_examples,
            inputs=[
                instruction,
                width_input,
                height_input,
                scheduler_input,
                num_inference_steps,
                image_input_1,
                image_input_2,
                image_input_3,
                negative_prompt,
                text_guidance_scale_input,
                image_guidance_scale_input,
                cfg_range_start,
                cfg_range_end,
                num_images_per_prompt,
                max_input_image_side_length,
                max_pixels,
                seed_input,
                system_prompt
            ],
            outputs=output_image,
        )

        gr.Markdown(article)
    # launch
    demo.launch(share=args.share, server_port=args.port, allowed_paths=[ROOT_DIR], inbrowser=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the OmniGen2")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app")
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to use for the Gradio app"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="OmniGen2/OmniGen2",
        help="Path or HuggingFace name of the model to load."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
