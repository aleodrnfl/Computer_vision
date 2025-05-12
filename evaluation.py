import torch
import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from models import ActionModel
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_IMG_DIR = "CV_Test/Images"  # Path to test images
TEST_LABEL_DIR = "CV_Test/Labels"  # Path to ground truth JSON labels
DEFAULT_MODEL_PATH = "weights/best.pt"  # Default model path
CHECKPOINT_DIR = "Computer_vision/checkpoints"  # Directory containing checkpoint models
OUTPUT_DIR = "Computer_vision/test_outputs"
NUM_CLASSES = 4  # standing, sitting, lying, throwing
CLASS_NAMES = {0: "standing", 1: "lying", 2: "throwing", 3: "sitting"}
CLASS_LIST = ["standing", "lying", "throwing", "sitting"]
NUM_WORKERS = 4  # Number of workers for data loading

# Image transformation should match what was used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path, num_classes):
    """Load model from file - handles both full checkpoints and state_dict only files"""
    model = ActionModel(num_classes=num_classes)

    try:
        # First try loading as a complete checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=DEVICE)

        # If the checkpoint is a dict with 'model_state_dict' key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(
                f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        # If the file contains just the state_dict directly
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights directly")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def predict(model, image_path):
    """Make a prediction on a single image and measure inference time"""
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None

    # Get original image for visualization
    orig_img = img.copy()

    # Transform for model input
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    # Measure inference time
    start = time.time()
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    elapsed = (time.time() - start) * 1000

    # Get prediction and probability
    prob, pred = probs.max(1)
    class_name = CLASS_NAMES[int(pred)]
    confidence = float(prob)

    return {
        'label': class_name,
        'confidence': confidence,
        'time_ms': elapsed,
        'pred_idx': int(pred),
        'probs': probs.cpu().numpy()[0],
        'orig_img': orig_img
    }


def visualize_prediction(img, prediction, gt_label=None):
    """Create a visualization of the prediction with confidence scores"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot image with prediction
    ax1.imshow(img)
    title = f"Prediction: {prediction['label']} ({prediction['confidence']:.2%})"
    if gt_label:
        correct = prediction['label'] == gt_label
        title += f"\nGround Truth: {gt_label} ({'✓' if correct else '✗'})"
    ax1.set_title(title)
    ax1.axis('off')

    # Plot confidence bars for each class
    probs = prediction['probs']
    y_pos = np.arange(len(CLASS_LIST))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(CLASS_LIST)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Class Probabilities')

    # Highlight the predicted class
    pred_idx = prediction['pred_idx']
    ax2.get_children()[pred_idx].set_color('red')

    plt.tight_layout()
    return fig


def save_results(predictions, metrics, output_dir):
    """Save results to files and create visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    # Save textual results
    results_file = os.path.join(output_dir, "results.txt")
    with open(results_file, 'w') as f:
        # Write overall metrics
        if metrics:
            f.write("=== EVALUATION METRICS ===\n")
            f.write(f"Total images: {metrics['total']}\n")
            if 'accuracy' in metrics:
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write("\n=== CLASSIFICATION REPORT ===\n")
            if 'report' in metrics:
                f.write(metrics['report'])
            f.write("\n\n")

        # Write individual predictions
        f.write("=== INDIVIDUAL PREDICTIONS ===\n")
        for img_name, pred in predictions.items():
            f.write(
                f"{img_name}: {pred['label']} ({pred['confidence']:.2%}, {pred['time_ms']:.1f} ms)")
            if 'gt_label' in pred:
                f.write(f" | GT: {pred['gt_label']}")
            f.write("\n")

    # Create and save confusion matrix if we have ground truth
    if metrics and 'confusion_matrix' in metrics:
        plt.figure(figsize=(10, 8))
        cm = metrics['confusion_matrix']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(CLASS_LIST))
        plt.xticks(tick_marks, CLASS_LIST, rotation=45)
        plt.yticks(tick_marks, CLASS_LIST)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

    # Save visualizations for sample predictions (limit to 20)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    sample_count = min(20, len(predictions))
    sample_keys = list(predictions.keys())[:sample_count]

    for img_name in sample_keys:
        pred = predictions[img_name]
        gt_label = pred.get('gt_label', None)
        fig = visualize_prediction(pred['orig_img'], pred, gt_label)
        fig.savefig(os.path.join(
            vis_dir, f"{os.path.splitext(img_name)[0]}_pred.png"))
        plt.close(fig)

    print(f"Results saved to {output_dir}")
    print(f"- Text results: {results_file}")
    print(f"- Visualizations: {vis_dir}")
    if 'confusion_matrix' in metrics:
        print(
            f"- Confusion matrix: {os.path.join(output_dir, 'confusion_matrix.png')}")


def list_available_checkpoints():
    """List all available model checkpoints"""
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Checkpoint directory {CHECKPOINT_DIR} not found")
        return []

    checkpoints = []
    for file in os.listdir(CHECKPOINT_DIR):
        if file.endswith('.pth'):
            checkpoints.append(os.path.join(CHECKPOINT_DIR, file))

    if os.path.exists(DEFAULT_MODEL_PATH):
        checkpoints.append(DEFAULT_MODEL_PATH)

    return checkpoints


def detect_and_classify_people(image_path, detector):
    img = Image.open(image_path).convert("RGB")
    results = detector(img)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    person_results = []
    
    for box in results[0].boxes:
        # 소수점 유지하면서 좌표 추출
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # float 값 그대로 유지
        cls = int(box.cls)
        conf = float(box.conf)
        label = CLASS_NAMES[cls]
        
        # 바운딩박스와 분류 결과 시각화 (시각화는 정수 좌표 사용)
        draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline='green', width=3)
        draw.text((int(x1), int(y1)-10), f"{label} ({conf:.2f})", fill='black', font=font)
        person_results.append({
            'bbox': [x1, y1, x2, y2],  # 소수점이 있는 원본 좌표 저장
            'label': label,
            'confidence': conf
        })
    return img, person_results


def save_labelme_json(image_path, person_results, output_dir):
    """YOLO 결과를 labelme 형식의 JSON으로 저장"""
    # 이미지 정보 가져오기
    img = Image.open(image_path)
    width, height = img.size
    
    # labelme 형식의 JSON 구조 생성
    labelme_data = {
        "version": "5.8.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    
    # 각 감지된 객체에 대해 shape 정보 추가
    for result in person_results:
        x1, y1, x2, y2 = result['bbox']
        shape = {
            "label": result['label'],
            "points": [
                [x1, y1],
                [x2, y2]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }
        labelme_data["shapes"].append(shape)
    
    # JSON 파일 저장
    json_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, indent=4, ensure_ascii=False)
    
    return json_path


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to YOLO model weights')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help=f'Directory to save results (default: {OUTPUT_DIR})')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results in labelme JSON format')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # Verify test images directory
    if not os.path.isdir(TEST_IMG_DIR):
        print(f"Error: Test image directory not found at {TEST_IMG_DIR}")
        return

    # Get test images
    test_images = sorted([f for f in os.listdir(TEST_IMG_DIR)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not test_images:
        print(f"No images found in {TEST_IMG_DIR}")
        return

    print(f"Found {len(test_images)} test images")

    # Load YOLO model
    detector = YOLO(args.model)

    # Create output directories
    vis_dir = os.path.join(args.output, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    if args.save_json:
        json_dir = os.path.join(args.output, 'labels')
        os.makedirs(json_dir, exist_ok=True)

    # Process all images
    for img_name in test_images:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        vis_img, person_results = detect_and_classify_people(img_path, detector)
        
        # 시각화 이미지 저장
        vis_img.save(os.path.join(vis_dir, f"{os.path.splitext(img_name)[0]}_detected.png"))
        
        # JSON 파일 저장 (요청된 경우)
        if args.save_json:
            json_path = save_labelme_json(img_path, person_results, json_dir)
            print(f"{img_name}: {len(person_results)}분류 완료, JSON 저장됨: {json_path}")
        else:
            print(f"{img_name}: {len(person_results)}분류 완료")


if __name__ == '__main__':
    main()
