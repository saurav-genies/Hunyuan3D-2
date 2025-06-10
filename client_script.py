import requests
import boto3
from urllib.parse import urlparse

IP_ADDRESS = "35.83.205.38:8000"

from typing import Union, Optional, Dict
import base64
import os

def load_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def is_s3_uri(value: str) -> bool:
    return isinstance(value, str) and value.startswith("s3://")

def build_payload(
    image_path: Union[str, Dict[str, str]],
    mesh_path: Optional[str] = None,
    seed: int = 12345,
    resolution: int = 380,
    steps: int = 50,
    guidance: float = 5.0,
    output_type: str = "glb",
    texture: bool = True,
    num_chunks=20000,
    face_count: int = 40000
) -> dict:
    payload = {
        "seed": seed,
        "octree_resolution": resolution,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "type": output_type,
        "texture": texture,
        "face_count": face_count,
        "num_chunks": num_chunks,
    }

    # Handle mesh
    if mesh_path:
        if not is_s3_uri(mesh_path):
            raise ValueError("Mesh path must be an S3 URI starting with 's3://'")
        payload["mesh"] = mesh_path

    # Handle image(s)
    if isinstance(image_path, str):
        if not is_s3_uri(image_path):
            raise ValueError("Image path must be an S3 URI starting with 's3://'")
        payload["image"] = image_path

    elif isinstance(image_path, dict):
        for key, value in image_path.items():
            if not is_s3_uri(value):
                raise ValueError(f"Image path for '{key}' must be an S3 URI starting with 's3://'")
        payload["images"] = image_path

    else:
        raise ValueError("Invalid format for image_path")

    return payload


def download_file_from_s3_uri(s3_uri: str, output_path: str):
    parsed = urlparse(s3_uri)
    if parsed.scheme != 's3':
        raise ValueError("Invalid S3 URI. Must start with 's3://'")

    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, key, output_path)
        print(f"File downloaded from {s3_uri} and saved to {output_path}")
    except Exception as e:
        print(f"Failed to download from S3: {s3_uri}")
        print("Error:", e)

def generate_3d_model(
    payload: dict,
    server_url: str = f"http://{IP_ADDRESS}/generate",
    output_path: str = "output/test.glb"
) -> Optional[str]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    headers = {"Content-Type": "application/json"}

    print(f"Sending request to {server_url}...")
    response = requests.post(server_url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            response_data = response.json()
            s3_url = response_data.get("s3_url") or response_data.get("s3_uri")
            if s3_url:
                print(f"S3 URL received: {s3_url}")
                download_file_from_s3_uri(s3_url, output_path)
                return output_path
            else:
                print("No 's3_url' found in response.")
                return None
        except Exception as e:
            print(f"Failed to parse response JSON: {e}")
            print("Response content:", response.text)
            return None
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response:", response.text)
        return None


if __name__ == "__main__":
    # Run with a single image without texture
    image_file = "s3://genies-ml-rnd/hunyuan3d/test_data/genies/front.png"
    payload = build_payload(image_path=image_file, output_type="glb", texture=False)

    # Run with a single image with texture
    # image_file = "s3://genies-ml-rnd/hunyuan3d/test_data/genies/front.png"
    # payload = build_payload(image_path=image_file, output_type="glb", texture=True)

    # Run with multiple images without texture
    # images = {
    #     "front": "s3://genies-ml-rnd/hunyuan3d/test_data/genies/front.png",
    #     "left": "s3://genies-ml-rnd/hunyuan3d/test_data/genies/left.png",
    #     "back": "s3://genies-ml-rnd/hunyuan3d/test_data/genies/back.png",
    #     "right": "s3://genies-ml-rnd/hunyuan3d/test_data/genies/right.png",
    # }
    # payload = build_payload(image_path=images, output_type="glb", texture=False)

    # Run with mesh and image with texture
    # image_file = "s3://genies-ml-rnd/hunyuan3d/test_data/genies/front.png"
    # mesh_path = "s3://genies-ml-rnd/hunyuan3d/test_data/genies/genies.glb"
    # payload = build_payload(image_path=image_file, mesh_path=mesh_path, output_type="glb", texture=True)
