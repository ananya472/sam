import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
import requests
from typing import List, Tuple, Union, Optional, Dict, Any
import cv2
from dataclasses import dataclass
import json
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax): 
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def load_image(image_str: str) -> Image.Image:
    try:
        if image_str.startswith("http"):
            image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_str).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Error loading image from {image_str}: {e}")

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask

def detect(image: Image.Image, labels: List[str], threshold: float = 0.3, detector_id: Optional[str] = None) -> List[Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(image: Image.Image, detection_results: List[Dict[str, Any]], polygon_refinement: bool = False, segmenter_id: Optional[str] = None) -> List[DetectionResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(image: Union[Image.Image, str], labels: List[str], threshold: float = 0.3, polygon_refinement: bool = False, detector_id: Optional[str] = None, segmenter_id: Optional[str] = None) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections

def get_boxes(results: DetectionResult) -> List[List[float]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]

def numpy_to_base64(image_array) -> str:
    # Convert the image array to BGR format (if necessary)
    if image_array.shape[-1] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    _, encoded_image = cv2.imencode('.jpg', image_array)

    # Convert the encoded image to Base64
    base64_string = base64.b64encode(encoded_image).decode('utf-8')

    return base64_string

def process_images_handler():
    json_data = request.get_json()
    if json_data is None:
        return jsonify({'error': 'No JSON data provided'}), 400

    categoryname = json_data.get('categoryname')
    if categoryname is None:
        return jsonify({'error': 'Category name is not provided'}), 400
    categoryname = categoryname.capitalize()

    image_urls = [image.get('url') for image in json_data.get('images', []) if image.get('url')]

    # Determine labels based on the category name
    if categoryname == "Wallpaper":
        labels = ["Wall."]
    elif categoryname == "Flooring":
        labels = ["Floor."]
    else:
        labels = [f"{categoryname}."]

    threshold = 0.4
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"
    processed_images = []

    for i in image_urls:
        try:
            # Perform grounded segmentation
            image_array, detections = grounded_segmentation(
                image=i, labels=labels, threshold=threshold, polygon_refinement=True,
                detector_id=detector_id, segmenter_id=segmenter_id
            )

            img = image_array

            # Process each detection
            for detection in detections:
                m = detection.mask
                try:
                    # Fetch the product image from external API
                    url = f"https://newbackend.ayatrio.com/api/fetchProductsByCategory/{categoryname}"
                    response = requests.get(url)
                    response.raise_for_status()
                    data = json.loads(response.text)
                    product_image_url = data[0]['productImages'][0]['images'][0]

                    product_response = requests.get(product_image_url, stream=True)
                    product_response.raise_for_status()
                    product_image = np.asarray(bytearray(product_response.content), dtype="uint8")
                    product_image = cv2.imdecode(product_image, cv2.IMREAD_COLOR)
                    new_image = cv2.cvtColor(product_image, cv2.COLOR_BGR2RGB)

                    # If the category is "Flooring", rotate the image by 90 degrees
                    if categoryname == "Flooring":
                        new_image = cv2.rotate(new_image, cv2.ROTATE_90_CLOCKWISE)

                    # Resize and overlay the product image
                    x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
                    resized_new_image = cv2.resize(new_image, (w, h))
                    img[y:y+h, x:x+w] = np.where(m[y:y+h, x:x+w, np.newaxis], resized_new_image, img[y:y+h, x:x+w])

                except Exception as e:
                    print(f"Error fetching or overlaying product image: {e}")

            # Encode final processed image in base64 and add to the result list
            base64_encoded_image = numpy_to_base64(img)
            processed_images.append(base64_encoded_image)

        except Exception as e:
            print(f"Error processing image at URL {i}: {e}")

    # Return all processed images in the response
    return jsonify({'processed_images': processed_images})


@app.route('/ping', methods=['GET'])
def ping():
    return 'Healthy', 200
    
@app.route('/invocations', methods=['POST'])
def invocations():
    return process_images_handler()

if __name__ == '__main__':
    app.run(debug=True, port=8080)