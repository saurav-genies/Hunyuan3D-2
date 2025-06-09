# 🚀 Launching Hunyuan API Server
Follow these steps to start the Hunyuan3D-2 backend service:

✅ Step-by-Step Instructions

## 1. Start the EC2 instance
From AWS Console, start the instance named: `recon-internal-rnd-copy`

And then SSH into the instance. You will require the private key file associated with the instance.
Ask @saurav or @jingwen for the key file if you don't have it.

## 2. Navigate to the project directory

```bash
cd /home/ubuntu/genies/Hunyuan3D-2
conda activate Hunyuan3D-2
```

## 3. Start the API server

```bash
nohup python api_server.py --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```
The server will now be listening on port 8000.

You can check the server logs by running:

```bash
tail -f server.log
```


## 4. Sending API Requests from Client
Once the server is running, you can use the `client_script.py` to send image-to-3D generation requests. You can copy this script to your local machine or run it directly on the server.

📄 Example:
```bash
python client_script.py
```
🔁 Note: The public IP is not static. You must update the IP address in the client script each time the instance is restarted.

📂 Output
Generated 3D models ( .glb) will be  uploaded to S3 (`s3://genies-ml-rnd/hunyuan3d/api_runs/`) and downloaded and saved locally, you can change the client script to fit your needs.


🛠 Requirements for local API call
```bash
Python 3.8+
boto3, requests
AWS credentials
```

