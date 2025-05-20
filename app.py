import os
import io
import json
import uuid
import argparse
from pathlib import Path

import boto3
import magic
from PIL import Image
from flask import Flask, render_template, request, abort, flash, redirect, url_for
from werkzeug.utils import secure_filename

MAX_SIDE = 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

def parse_arguments():
    parser = argparse.ArgumentParser(description="Flask application for pose estimation")
    
    # Flask app settings
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the Flask application on")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the Flask application on")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Run Flask in debug mode")
    # Pose runner settings
    parser.add_argument("--bucket", type=str, default="ziad-pose-estimation",
                        help="S3 bucket name for storing results")
    parser.add_argument("--checkpoint-path", type=str, default="models/human-pose.pth",
                        help="Path to model checkpoint file")
    parser.add_argument("--result-prefix", type=str, default="results",
                        help="Prefix for result files in S3")
    parser.add_argument("--height", type=int, default=256,
                        help="Network input layer height")
    parser.add_argument("--stride", type=int, default=8,
                        help="Model stride")
    parser.add_argument("--upsample-ratio", type=int, default=4,
                        help="Model upsample ratio")
    parser.add_argument("--use-cpu", action="store_true", default=True,
                        help="Use CPU for inference")
    
    return parser.parse_args()

args = parse_arguments()

from pose_runner import initialize_runner, run_pose

initialize_runner(
    bucket=args.bucket,
    checkpoint_path=args.checkpoint_path,
    result_prefix=args.result_prefix,
    height=args.height,
    stride=args.stride,
    upsample_ratio=args.upsample_ratio,
    use_cpu=args.use_cpu
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key-replace-in-production")

s3 = boto3.client("s3")

mime = magic.Magic(mime=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_image(image, max_side=MAX_SIDE):
    width, height = image.size
    if max(width, height) > max_side:
        ratio = max_side / float(max(width, height))
        new_width, new_height = int(width * ratio), int(height * ratio)
        return image.resize((new_width, new_height))
    return image


def save_image_to_s3(image_bytes, content_type="image/jpeg"):
    key = f"uploads/{uuid.uuid4().hex}.jpg"
    s3.put_object(
        Bucket=args.bucket,  
        Key=key,
        Body=image_bytes,
        ContentType=content_type
    )
    return key


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        
        if not file or not allowed_file(file.filename):
            flash("Please upload a JPG or PNG file")
            return redirect(request.url)

        try:
            img_bytes = file.read()
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            pil_image = resize_image(pil_image)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=90)
            img_bytes = buffer.getvalue()
            
            key = save_image_to_s3(img_bytes)
            
            img_key, json_key, data_url = run_pose(key)
            
            json_obj = s3.get_object(Bucket=args.bucket, Key=json_key)
            json_data = json.loads(json_obj["Body"].read())
            
            return render_template(
                "result.html",
                data_url=data_url,
                json=json.dumps(json_data, indent=2)
            )
            
        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            flash(f"Error processing image: {str(e)}")
            return redirect(request.url)

    return render_template("upload.html")


@app.errorhandler(413)
def too_large(e):
    flash(f"File is too large. Maximum size is {MAX_CONTENT_LENGTH/(1024*1024)}MB")
    return redirect(url_for("index"))


@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {str(e)}")
    return render_template("error.html", error=str(e)), 500


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=args.debug)
n(host="0.0.0.0", port=5000, debug=False)

