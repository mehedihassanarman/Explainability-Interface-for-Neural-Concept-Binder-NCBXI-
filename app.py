import logging
import os
import re
import torch

# NEW/CHANGED: Additional imports for feedback handling
from flask import Flask, request, render_template, send_from_directory, jsonify, session
import pandas as pd  # For writing/reading feedback in Excel
from datetime import datetime

from NCBXI_api import (
    Args, load_model, preprocess_image, run_inference, 
    visualize_block, implicit_inspection, comparative_inspection, 
    conceptual_inspection, create_block_concepts, preprocess_image_paths
)

app = Flask(__name__, static_folder="static")

# NEW/CHANGED: Set a secret key so we can use session
app.secret_key = "YOUR_SECRET_KEY_HERE"  # Replace with a secure random key in production

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # Go up one level
IMAGE_FOLDER = os.path.join(BASE_DIR, "data", "CLEVR-4-1", "test", "images")
PLOT_FOLDER = "static/images/plots/"

# NEW/CHANGED (Task 1): Feedback folder in same directory as app.py
FEEDBACK_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "User Feedback")
FEEDBACK_FILE = os.path.join(FEEDBACK_FOLDER, "feedback.xlsx")

# 🔹 Ensure feedback folder exists
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

# Function to check if the image exists
def check_image_exists(image_path):
    full_image_path = os.path.join(IMAGE_FOLDER, image_path)
    if not os.path.exists(full_image_path):
        logging.error(f"❌ ERROR: Image file not found at {full_image_path}")
        return False
    return True

# 🔹 Ensure image folder exists
if not os.path.exists(IMAGE_FOLDER):
    print(f"⚠️ Warning: Image folder '{IMAGE_FOLDER}' does not exist! Creating it now.")
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Load available images safely
def get_sorted_images(image_path=None):
    """Fetch available images from the specified folder, prioritize selected image."""
    if not os.path.exists(IMAGE_FOLDER):
        print("⚠️ Warning: Image folder does not exist! No images available.")
        return ["default.png"]

    try:
        images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        images.sort()
        
        if image_path and image_path in images:
            images.remove(image_path)
            images.insert(0, image_path)
        
        return images if images else ["default.png"]
    except FileNotFoundError:
        return ["default.png"]

def preprocess_and_infer(image_path):
    """Preprocess image and run inference, returning codes, model_path, model, args, image_path."""
    full_image_path = os.path.join(IMAGE_FOLDER, image_path)
    
    if not os.path.exists(full_image_path):
        print(f"⚠️ Warning: Image '{full_image_path}' not found! Skipping inference.")
        return None, None, None, None, None  

    model_path = "../model/CLEVR-4/retbind_seed_2/"
    args = Args(model_path=model_path)
    torch.manual_seed(args.seed)
    device = args.device
    
    # Load model
    model = load_model(args)
    
    # Image preprocessing
    image_tensor, _ = preprocess_image(full_image_path, args.image_size)
    
    # Run inference
    codes, _ = run_inference(image_tensor, model, device)
    
    return codes, model_path, model, args, image_path

def codes_to_string(codes):
    """Utility to turn torch codes into a string of integers."""
    if codes is None:
        return None
    raw_str = str(codes.cpu().numpy())
    activated_concepts = re.findall(r"[-+]?\d*\.\d+|\d+", raw_str)
    cleaned_activated_concepts = [int(num) for num in activated_concepts]
    return str(cleaned_activated_concepts)

@app.route("/", methods=["GET", "POST"])
def home():
    selected_image = request.form.get("image_path") if request.method == "POST" else None
    image_files = get_sorted_images(selected_image)

    return render_template(
        "index.html",
        image_list=image_files,
        first_image=image_files[0] if image_files else "default.png",
        image_path=selected_image,
        plot_paths=[]
    )

@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route("/run_model", methods=["POST"])
def run_model():
    from NCBXI_api import Args
    image_path = request.form["image_path"]
    print(f"📸 Received request to process image: {image_path}")
    codes, model_path, model, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        print("⚠️ Model inference failed or no output received!")
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    print("✅ Model inference successful, preparing response...")

    codes_str = codes_to_string(codes)

    # NEW/CHANGED (Task 1):
    #  1) Check existing feedback.xlsx to find max case number
    #  2) For each run_model, we define a new unique "case_id"
    if os.path.exists(FEEDBACK_FILE):
        df_existing = pd.read_excel(FEEDBACK_FILE)
        if not df_existing.empty:
            max_case = df_existing["Case"].max()
            current_case = int(max_case + 1)
        else:
            current_case = 1
    else:
        current_case = 1

    # Store session data
    session["case_id"] = current_case
    session["image_id"] = image_path
    session["codes_str"] = codes_str

    # Removed "plot_path" => ensures we don't show old plots after run_model
    return jsonify({
        "success": True,
        "message": "Model run successful!",
        "device": args.device,
        "codes_str": codes_str
    })

# ---------- Visualization Endpoints with device/codes added ---------- #

@app.route("/visualization", methods=["POST"])
def get_visualization():
    image_path = request.form["image_path"]
    block_id = int(request.form["block_id"])

    if not check_image_exists(image_path):
        return jsonify({"success": False, "message": "Image file not found!"}), 404

    codes, model_path, _, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    visualize_block(block_id, codes, model_path)
    codes_str = codes_to_string(codes)

    return jsonify({
        "success": True,
        "message": "Visualization completed!",
        "plot_path": os.path.join(PLOT_FOLDER, "Visualize_Concept_Block", "Concept_Block.png"),
        "device": args.device,
        "codes_str": codes_str
    })

@app.route("/implicit_inspection", methods=["POST"])
def get_implicit_inspection():
    from NCBXI_api import create_block_concepts, preprocess_image_paths
    image_path = request.form["image_path"]
    block_id = int(request.form["block_id"])
    cluster_id = int(request.form["cluster_id"])

    if not check_image_exists(image_path):
        return jsonify({"success": False, "message": "Image file not found!"}), 404

    codes, model_path, _, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    codes_str = codes_to_string(codes)

    block_concepts = create_block_concepts(args.retrieval_corpus_path)
    if block_id not in block_concepts:
        msg = (f"Block ID {block_id} is out of range. "
               f"Available blocks: {list(block_concepts.keys())}.")
        return jsonify({"success": False, "message": msg}), 200

    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
        msg = (f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
               f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        return jsonify({"success": False, "message": msg}), 200

    all_img_locs = preprocess_image_paths([
        os.path.join(args.data_dir, f"train/images/CLEVR_4_classid_0_{i:06}.png") for i in range(5000)
    ])

    implicit_inspection(block_concepts, all_img_locs, block_id=block_id, cluster_id=cluster_id)
    return jsonify({
        "success": True,
        "message": "Implicit Inspection completed!",
        "plot_path": os.path.join(PLOT_FOLDER, "Implicit_Inspection", "Implicit_Inspection.png"),
        "device": args.device,
        "codes_str": codes_str
    })

@app.route("/comparative_inspection", methods=["POST"])
def get_comparative_inspection():
    from NCBXI_api import create_block_concepts, preprocess_image_paths
    image_path = request.form["image_path"]
    block_id = int(request.form["block_id"])

    if not check_image_exists(image_path):
        return jsonify({"success": False, "message": "Image file not found!"}), 404

    codes, model_path, _, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    codes_str = codes_to_string(codes)

    block_concepts = create_block_concepts(args.retrieval_corpus_path)
    if block_id not in block_concepts:
        msg = (f"Block ID {block_id} is out of range. "
               f"Available blocks: {list(block_concepts.keys())}.")
        return jsonify({"success": False, "message": msg}), 200

    all_img_locs = preprocess_image_paths([
        os.path.join(args.data_dir, f"train/images/CLEVR_4_classid_0_{i:06}.png") for i in range(5000)
    ])

    comparative_inspection(
        block_concepts,
        all_img_locs,
        example_path=os.path.join(IMAGE_FOLDER, image_path),
        block_id=block_id,
        max_exemplars=5
    )
    return jsonify({
        "success": True,
        "message": "Comparative Inspection completed!",
        "plot_path": os.path.join(PLOT_FOLDER, "Comparative_Inspection", "Comparative_Inspection.png"),
        "device": args.device,
        "codes_str": codes_str
    })

@app.route("/conceptual_inspection", methods=["POST"])
def get_conceptual_inspection():
    from NCBXI_api import create_block_concepts
    image_path = request.form["image_path"]
    block_id = int(request.form["block_id"])
    cluster_id = int(request.form["cluster_id"])

    if not check_image_exists(image_path):
        return jsonify({"success": False, "message": "Image file not found!"}), 404

    codes, model_path, model, args, _ = preprocess_and_infer(image_path)
    if codes is None:
        return jsonify({"success": False, "message": "Model inference failed!"}), 500

    codes_str = codes_to_string(codes)

    block_concepts = create_block_concepts(args.retrieval_corpus_path)
    if block_id not in block_concepts:
        msg = (f"Block ID {block_id} is out of range. "
               f"Available blocks: {list(block_concepts.keys())}.")
        return jsonify({"success": False, "message": msg}), 200

    if cluster_id >= len(block_concepts[block_id]['prototypes']['ids']):
        msg = (f"Cluster ID {cluster_id} is out of range for Block {block_id}. "
               f"Available clusters: {len(block_concepts[block_id]['prototypes']['ids']) - 1}.")
        return jsonify({"success": False, "message": msg}), 200

    conceptual_inspection(
        block_concepts,
        model,
        example_path=os.path.join(IMAGE_FOLDER, image_path),
        block_id=block_id,
        cluster_id=cluster_id,
        args=args
    )
    return jsonify({
        "success": True,
        "message": "Conceptual Inspection completed!",
        "plot_path": os.path.join(PLOT_FOLDER, "Conceptual_Inspection", "Conceptual_Inspection.png"),
        "device": args.device,
        "codes_str": codes_str
    })

# ------------ Save Feedback ------------- #
@app.route("/save_feedback", methods=["POST"])
def save_feedback():
    """
    Saves/updates user feedback in feedback.xlsx in 'User Feedback' folder.

    Columns:
      'Case' 'Image ID' 'Activated Concepts for Each Block'
      'Block 0' ... 'Block 15'
    If row for (Case == case_id) & (Image ID == image_id) exists,
    we update the block column with the new label (overwriting).
    """
    block_id = request.form.get("block_id", None)
    feedback_label = request.form.get("feedback_label", "")

    # Session data
    case_id = session.get("case_id", 0)
    image_id = session.get("image_id", "")
    codes_str = session.get("codes_str", "")

    # Basic validation
    if block_id is None:
        return jsonify({"success": False, "message": "No block_id specified."}), 400
    try:
        block_id = int(block_id)
    except ValueError:
        return jsonify({"success": False, "message": "block_id must be an integer."}), 400

    # Prepare columns
    columns = (
        ["Case", "Image ID", "Activated Concepts for Each Block"]
        + [f"Block {i}" for i in range(16)]
    )

    # If feedback file doesn't exist, create a blank with correct columns
    if not os.path.exists(FEEDBACK_FILE):
        df_blank = pd.DataFrame(columns=columns)
        df_blank.to_excel(FEEDBACK_FILE, index=False)

    df_existing = pd.read_excel(FEEDBACK_FILE)

    # If the DataFrame is empty, just append a new row
    if df_existing.empty:
        # Create brand-new row
        row_data = {
            "Case": [case_id],
            "Image ID": [image_id],
            "Activated Concepts for Each Block": [codes_str],
        }
        for i in range(16):
            row_data[f"Block {i}"] = [""]
        # Overwrite the block we care about
        row_data[f"Block {block_id}"] = [feedback_label]

        df_new = pd.DataFrame(row_data, columns=columns)
        df_result = pd.concat([df_existing, df_new], ignore_index=True)

    else:
        # Make a boolean Series for the matching row
        match_mask = (df_existing["Case"] == case_id) & (df_existing["Image ID"] == image_id)

        if match_mask.any():
            # Overwrite the column for the matching row
            df_existing.loc[match_mask, f"Block {block_id}"] = feedback_label
            df_result = df_existing
        else:
            # Create a new row if no match
            row_data = {
                "Case": [case_id],
                "Image ID": [image_id],
                "Activated Concepts for Each Block": [codes_str],
            }
            for i in range(16):
                row_data[f"Block {i}"] = [""]
            row_data[f"Block {block_id}"] = [feedback_label]

            df_new = pd.DataFrame(row_data, columns=columns)
            df_result = pd.concat([df_existing, df_new], ignore_index=True)

    # Finally, write back to Excel
    df_result.to_excel(FEEDBACK_FILE, index=False)

    return jsonify({"success": True, "message": f"Feedback saved for block id {block_id}!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
