import logging
import os

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, send_file
from segment_anything import SamPredictor, sam_model_registry
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = r'C:\Users\anany\Desktop\house-ass\uploads'
app.config['PROCESSED_FOLDER'] = r'C:\Users\anany\Desktop\house-ass\processed'
app.config['SAM_CHECKPOINT'] = r'C:\Users\anany\Desktop\house-ass\sam_vit_l_0b3195.pth'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# Initialize SAM model
def initialize_sam_model(checkpoint_path):
    logging.info("Initializing SAM model...")
    sam_model = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
    sam_model.eval()  # Set the model to evaluation mode
    return SamPredictor(sam_model)

# Tiling the replacement image to fit the masked area with smaller tiles and high quality
def tile_image(replacement_img, target_shape, mask, scale_factor=0.05):
    """Tile the replacement image over the target shape, respecting the mask, with a smaller tile size."""
    # Resize the replacement image to make the tiles smaller
    small_tile = cv2.resize(replacement_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    tile_h, tile_w, _ = small_tile.shape
    target_h, target_w = target_shape[:2]

    # Create an empty target image of the same size as the room image
    tiled_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Iterate over the target area and tile the replacement image
    for i in range(0, target_h, tile_h):
        for j in range(0, target_w, tile_w):
            # Determine where to place the current tile
            end_i = min(i + tile_h, target_h)
            end_j = min(j + tile_w, target_w)

            # Place the tile in the correct location, within the boundaries
            tiled_img[i:end_i, j:end_j] = small_tile[:end_i - i, :end_j - j]

    # Only apply the tiled image to the masked area
    tiled_img = np.where(mask[:, :, None] > 0, tiled_img, 0)

    return tiled_img

# Route for the main page
@app.route('/')
def index():
    return render_template('index03.html')

# Route for processing the images
@app.route('/upload', methods=['POST'])
def upload_images():
    if 'room_image' not in request.files or 'replacement_image' not in request.files:
        return "Please upload both the room image and the replacement image.", 400

    # Get the uploaded files
    room_image_file = request.files['room_image']
    replacement_image_file = request.files['replacement_image']

    # Get the chosen option (wall or blinds)
    option = request.form.get('option')

    if not option:
        return "No option selected. Please choose whether to change the wall or blinds.", 400

    # Save the uploaded files
    room_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(room_image_file.filename))
    replacement_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(replacement_image_file.filename))
    room_image_file.save(room_image_path)
    replacement_image_file.save(replacement_image_path)

    # Read the images using OpenCV
    room_img = cv2.imread(room_image_path)
    replacement_img = cv2.imread(replacement_image_path)

    if room_img is None or replacement_img is None:
        return "Error loading one or both images.", 400

    # Initialize SAM model
    predictor = initialize_sam_model(app.config['SAM_CHECKPOINT'])
    predictor.set_image(room_img)

    # Coordinates for segmentation (adjusted for finer control)
    if option == 'wall':
        point_coords = np.array([[100, 100], [200, 100], [300, 100]])  # Refined points
        point_labels = np.array([1, 1, 1])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)

        if masks is not None and len(masks) > 0:
            # Choose the best mask
            mask = masks[0]

            # Apply morphological erosion to refine the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

            # Tile the replacement image to cover the masked area with smaller tiles
            tiled_replacement = tile_image(replacement_img, room_img.shape, mask, scale_factor=0.185)

            # Combine the tiled image with the original room image
            room_img[mask > 0] = tiled_replacement[mask > 0]

    elif option == 'Blinds':
        point_coords = np.array([[100, 100], [200, 100], [300, 100]])  # Refined points
        point_labels = np.array([1, 1, 1])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)

        if masks is not None and len(masks) > 0:
            # Choose the best mask
            mask = masks[0]

            # Apply morphological erosion to refine the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

            # Tile the replacement image to cover the masked area with smaller tiles
            tiled_replacement = tile_image(replacement_img, room_img.shape, mask, scale_factor=0.05)

            # Combine the tiled image with the original room image
            room_img[mask > 0] = tiled_replacement[mask > 0]

    else:
        return "Invalid option selected. Please choose 'wall' or 'blinds'.", 400

    # Save the processed image
    output_filename = f'room_with_{option}.jpg'
    output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    cv2.imwrite(output_filepath, room_img)

    # Return the processed image
    return send_file(output_filepath, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)