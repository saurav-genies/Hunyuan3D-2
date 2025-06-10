# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError
import mimetypes
import time
from urllib.parse import urlparse

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

S3_BUCKET_NAME = "genies-ml-rnd"
S3_REGION = "us-west-2"
S3_KEY = "hunyuan3d/api_runs"


def download_s3_to_image(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        boto3.client("s3").download_fileobj(bucket, key, f)
        return Image.open(f.name)


def download_s3_to_bytesio(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    buf = BytesIO()
    boto3.client("s3").download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf

def upload_to_s3(file_path: str,) -> str:
    s3_client = boto3.client("s3")
    try:
        with open(file_path, "rb") as f:
            data = f.read()

        content_type, _ = mimetypes.guess_type(file_path)
        content_type = content_type or "application/octet-stream"

        filename = os.path.basename(file_path)
        unique_id = str(uuid.uuid4())
        key = f"{S3_KEY}/{unique_id}/{filename}"

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=data,
            ContentType=content_type
        )

        return f"s3://{S3_BUCKET_NAME}/{key}"

    except (BotoCoreError, NoCredentialsError, FileNotFoundError) as e:
        raise RuntimeError(f"S3 upload failed: {e}")


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2mini',
                 tex_model_path='tencent/Hunyuan3D-2',
                 subfolder='hunyuan3d-dit-v2-mini-turbo',
                 device='cuda',
                 enable_tex=True,
                 enable_mv=True,):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
        )
        if enable_mv:
            self.pipeline_mv = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mv",
                subfolder="hunyuan3d-dit-v2-mv-turbo",
                variant='fp16',)

        self.pipeline.enable_flashvdm(mc_algo='mc')

        if enable_tex:
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):


        multiview_images = None

        if 'image' in params:
            if isinstance(params['image'], str) and params['image'].startswith("s3://"):
                image = download_s3_to_image(params["image"])
            else:
                raise ValueError("Image must be a valid S3 URI starting with 's3://'")
            image = self.rembg(image)
            params['image'] = image

        elif 'images' in params:
            multiview_images = params['images']
            for key in multiview_images:
                img_val = multiview_images[key]
                if img_val.startswith("s3://"):
                    image = download_s3_to_image(img_val)
                else:
                    raise ValueError(f"Image path for view '{key}' must be a valid S3 URI starting with 's3://'")

                image = image.convert("RGBA")
                if image.mode == 'RGB':
                    image = self.rembg(image)
                multiview_images[key] = image

            params['image'] = multiview_images
            del params['images']
            image = multiview_images.get("front")

        elif 'text' in params:
            text = params["text"]
            image = self.pipeline_t2i(text)
            image = self.rembg(image)
            params['image'] = image

        else:
            raise ValueError("No input image(s) or text provided")

        if 'mesh' in params:
            mesh_data = params["mesh"]
            if isinstance(mesh_data, str) and mesh_data.startswith("s3://"):
                buf = download_s3_to_bytesio(mesh_data)
            else:
                raise ValueError("Mesh Path must be a valid S3 URI starting with 's3://'")
            mesh = trimesh.load(buf, file_type='glb')
            logger.info("Mesh provided — running texture-only pipeline.")
        else:
            seed = params.get("seed", 12345)
            params['generator'] = torch.Generator(self.device).manual_seed(seed)
            params['octree_resolution'] = params.get("octree_resolution", 380)
            params['num_inference_steps'] = params.get("num_inference_steps", 50)
            params['guidance_scale'] = params.get('guidance_scale', 5.0)
            params['num_chunks'] = params.get('num_chunks', 20000)
            import time
            start_time = time.time()
            print(f"Generating mesh with params: {params}")
            if multiview_images:
                logger.info("Using multiview images for mesh generation.")
                mesh = self.pipeline_mv(
                    image=multiview_images,
                    num_inference_steps=50,
                    octree_resolution=380,
                    num_chunks=20000,
                    generator=torch.manual_seed(12345),
                    output_type='trimesh'
                )[0]
            else:
                logger.info("Using single image for mesh generation.")
                mesh = self.pipeline(**params)[0]
            logger.info("--- %s seconds ---" % (time.time() - start_time))

        if params.get('texture', False):
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
            mesh = self.pipeline_tex(mesh, image)

        type = params.get('type', 'glb')
        with tempfile.NamedTemporaryFile(suffix=f'.{type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.{type}')
            mesh.export(save_path)

        torch.cuda.empty_cache()
        return save_path, uid


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你可以指定允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        s3_url = upload_to_s3(file_path,)
        return JSONResponse({"s3_url": s3_url}, status_code=200)
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        print("Caught Unknown Error", e)
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)



@app.post("/send")
async def generate(request: Request):
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    threading.Thread(target=worker.generate, args=(uid, params,)).start()
    ret = {"uid": str(uid)}
    return JSONResponse(ret, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    print(save_file_path, os.path.exists(save_file_path))
    if not os.path.exists(save_file_path):

        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        base64_str = base64.b64encode(open(save_file_path, 'rb').read()).decode()
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_tex', action='store_true')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(model_path=args.model_path, device=args.device, enable_tex=True,
                         tex_model_path=args.tex_model_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
