import os
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from NCBXI_api import Args, load_model, preprocess_image, run_inference, visualize_block, implicit_inspection, conceptual_inspection, create_block_concepts, preprocess_image_paths

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = "static/images/uploads"
OUTPUT_FOLDER = "static/images/plots/Visualize_Concept_Block"
IMPLICIT_INSPECTION_FOLDER = "static/images/plots/Implicit_Inspection"
CONCEPTUAL_INSPECTION_FOLDER = "static/images/plots/Conceptual_Inspection"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['IMPLICIT_INSPECTION_FOLDER'] = IMPLICIT_INSPECTION_FOLDER
app.config['CONCEPTUAL_INSPECTION_FOLDER'] = CONCEPTUAL_INSPECTION_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMPLICIT_INSPECTION_FOLDER, exist_ok=True)
os.makedirs(CONCEPTUAL_INSPECTION_FOLDER, exist_ok=True)

# Model setup
MODEL_PATH = "../model/CLEVR-4/retbind_seed_2/"
args = Args(model_path=MODEL_PATH)
torch.manual_seed(args.seed)
device = args.device
model = load_model(args)
codes = None  # Store inference result globally
uploaded_image_path = None  # Store the path of the uploaded image

@app.route('/')
def home():
    """Render the home page."""
    return render_template("home.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles image upload."""
    global uploaded_image_path
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        uploaded_image_path = file_path
        return jsonify({"image_path": f"/{file_path}"})

    return jsonify({"error": "Invalid file format"}), 400

@app.route('/run_model', methods=['POST'])
def run_model():
    """Handles model execution."""
    global codes  # Store codes for visualization
    data = request.get_json()
    image_path = data.get('image_path', '').lstrip('/')  

    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 400

    # Preprocess and run inference
    image_tensor, _ = preprocess_image(image_path, args.image_size)
    codes, probs = run_inference(image_tensor, model, device)  # Store inference result

    # Flatten codes to remove unnecessary brackets
    flattened_codes = [int(num) for sublist in codes.tolist() for subsublist in sublist for num in subsublist]

    formatted_output = (
        f"Device on: {args.device}<br><br>"
        f"Shape of the output: {codes.shape}<br><br>"
        f"Activated Concepts for Each Block:{flattened_codes}"
    )

    return jsonify({"output": formatted_output, "show_furhat_options": True})

@app.route('/visualize_block', methods=['POST'])
def visualize_block_route():
    """Handles block visualization request."""
    global codes  # Use the codes from model inference
    data = request.get_json()
    block_idx = data.get('block_idx')

    if block_idx is None:
        return jsonify({"error": "Block index not provided"}), 400
    if codes is None:
        return jsonify({"error": "Run the model first!"}), 400

    # Generate visualization using visualize_block
    visualize_block(block_idx, codes, MODEL_PATH)

    # Ensure Flask serves the correct path
    plot_path = "static/images/plots/Visualize_Concept_Block/Concept_Block.png"

    if not os.path.exists(plot_path):
        return jsonify({"error": "Visualization image not found!"}), 404

    return jsonify({
        "plot_url": f"/{plot_path}?t={int(os.path.getmtime(plot_path))}",  # Cache busting
        "show_visualization": True
    })

@app.route('/implicit_inspection', methods=['POST'])
def implicit_inspection_route():
    """Handles Implicit Inspection request and returns the plot."""
    data = request.get_json()
    block_id = data.get('block_id')
    cluster_id = data.get('cluster_id')

    # Create block concepts
    block_concepts = create_block_concepts(args.retrieval_corpus_path)
    
    # Preprocess image paths
    all_img_locs = [args.data_dir+f"train/images/CLEVR_4_classid_0_{i:06}.png" for i in range(5000)]
    all_img_locs = preprocess_image_paths(all_img_locs)

    if block_id is None or cluster_id is None:
        return jsonify({"error": "Both Block ID and Cluster ID are required"}), 400

    try:
        # Call the implicit inspection function
        implicit_inspection(block_concepts, all_img_locs, block_id=int(block_id), cluster_id=int(cluster_id))

        # Path of the saved plot
        plot_path = "static/images/plots/Implicit_Inspection/Implicit_Inspection.png"

        if not os.path.exists(plot_path):
            return jsonify({"error": "Visualization image not found!"}), 404

        return jsonify({
            "message": f"Implicit Inspection Completed for Block {block_id}, Cluster {cluster_id}",
            "plot_url": f"/{plot_path}?t={int(os.path.getmtime(plot_path))}"  # Cache busting
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/conceptual_inspection', methods=['POST'])
def conceptual_inspection_route():
    """Handles Conceptual Inspection request and returns the plot."""
    global uploaded_image_path
    data = request.get_json()
    block_id = data.get('block_id')
    cluster_id = data.get('cluster_id')

    if block_id is None or cluster_id is None or uploaded_image_path is None:
        return jsonify({"error": "Block ID, Cluster ID, and an uploaded image are required"}), 400

    # Create block concepts
    block_concepts = create_block_concepts(args.retrieval_corpus_path)

    try:
        # Call the conceptual inspection function
        conceptual_inspection(block_concepts, model, example_path=uploaded_image_path, block_id=int(block_id), cluster_id=int(cluster_id), args=args)

        # Path of the saved plot
        plot_path = "static/images/plots/Conceptual_Inspection/Conceptual_Inspection.png"

        if not os.path.exists(plot_path):
            return jsonify({"error": "Visualization image not found!"}), 404

        return jsonify({
            "message": f"Conceptual Inspection Completed for Block {block_id}, Cluster {cluster_id}",
            "plot_url": f"/{plot_path}?t={int(os.path.getmtime(plot_path))}"  # Cache busting
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Resets the output and UI."""
    global codes, uploaded_image_path
    codes = None  # Clear stored codes
    uploaded_image_path = None  # Clear uploaded image path
    return jsonify({
        "output": "Model output will appear here...",
        "show_furhat_options": False,
        "hide_middle_section": True,
        "hide_right_section": True
    })

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
