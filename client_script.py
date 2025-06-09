import requests
import base64
import json
import os
from typing import Optional
import boto3
from urllib.parse import urlparse

IP_ADDRESS = "35.89.38.118:8000"


def load_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def build_payload(
    image_path: str,
    seed: int = 1234,
    resolution: int = 128,
    steps: int = 5,
    guidance: float = 5.0,
    output_type: str = "glb",
    texture: bool = True,
    face_count: int = 40000
) -> dict:
    return {
        "image": load_image_as_base64(image_path),
        "seed": seed,
        "octree_resolution": resolution,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "type": output_type,
        "texture": texture,
        "face_count": face_count
    }

def download_file_from_s3_uri(s3_uri: str, output_path: str):
    parsed = urlparse(s3_uri)
    if parsed.scheme != 's3':
        raise ValueError("Invalid S3 URI. Must start with 's3://'")

    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, key, output_path)
        print(f"✅ File downloaded from {s3_uri} and saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to download from S3: {s3_uri}")
        print("Error:", e)

def generate_3d_model(
    payload: dict,
    server_url: str = f"http://{IP_ADDRESS}/generate",
    output_path: str = "output/test.glb"
) -> Optional[str]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    headers = {"Content-Type": "application/json"}

    print(f"📡 Sending request to {server_url}...")
    response = requests.post(server_url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            response_data = response.json()
            s3_url = response_data.get("s3_url") or response_data.get("s3_uri")
            if s3_url:
                print(f"🌐 S3 URL received: {s3_url}")
                download_file_from_s3_uri(s3_url, output_path)
                return output_path
            else:
                print("❌ No 's3_url' found in response.")
                return None
        except Exception as e:
            print(f"❌ Failed to parse response JSON: {e}")
            print("Response content:", response.text)
            return None
    else:
        print(f"❌ Request failed with status code {response.status_code}")
        print("Response:", response.text)
        return None

# 🧪 Example usage:
if __name__ == "__main__":
    image_file = "assets/demo.png"
    output_file = "output/test.glb"

    payload = build_payload(image_file, output_type="glb")
    generate_3d_model(payload, server_url="http://35.89.38.118:8000/generate", output_path=output_file)
