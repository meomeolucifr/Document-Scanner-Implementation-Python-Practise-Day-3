#!/usr/bin/env python3
"""
scan_improved_v8.py — Document scanner optimized for low contrast backgrounds

Key improvements for v8:
- Adaptive contrast enhancement based on image statistics
- Multi-method edge detection for low contrast scenarios  
- Enhanced preprocessing for grey/white backgrounds
- Better corner detection for subtle boundaries

Usage:
  python scan_improved_v8.py -i path/to/photo.jpg -o scanned.png
  python scan_improved_v8.py -i path/to/photo.jpg --preview

Requires: Python 3.8+, OpenCV (pip install opencv-python)
"""
import argparse
import os
import sys
from typing import Tuple, Optional
import numpy as np
import cv2

def resize_to_height(image: np.ndarray, height: int = 500) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if h == 0:
        raise ValueError("Invalid image with height=0")
    ratio = h / float(height)
    new_w = int(round(w / ratio))
    resized = cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)
    return resized, ratio

def analyze_image_contrast(gray: np.ndarray) -> dict:
    """Analyze image contrast characteristics"""
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)
    
    # Calculate contrast metrics
    rms_contrast = std_val
    michelson_contrast = (max_val - min_val) / (max_val + min_val + 1e-6)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val, 
        'max': max_val,
        'rms_contrast': rms_contrast,
        'michelson_contrast': michelson_contrast,
        'is_low_contrast': rms_contrast < 30 or michelson_contrast < 0.3,
        'is_bright_background': mean_val > 180
    }

def adaptive_contrast_enhancement(gray: np.ndarray, stats: dict) -> np.ndarray:
    """Apply contrast enhancement based on image statistics"""
    
    if stats['is_low_contrast']:
        # Aggressive enhancement for low contrast images
        if stats['is_bright_background']:
            # Bright, low contrast (grey/white background case)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Additional histogram stretching
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=-30)
        else:
            # Dark, low contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
            enhanced = clahe.apply(gray)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=20)
    else:
        # Normal contrast - mild enhancement
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    return enhanced

def multi_scale_edge_detection_v2(gray: np.ndarray, stats: dict) -> np.ndarray:
    """Enhanced edge detection for different contrast scenarios"""
    
    edges_list = []
    
    if stats['is_low_contrast']:
        # Method 1: Fine-scale edge detection (less blur)
        fine_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges1 = cv2.Canny(fine_blur, 30, 80)
        edges_list.append(edges1)
        
        # Method 2: Medium-scale (moderate blur)
        med_blur = cv2.GaussianBlur(gray, (7, 7), 0)  
        edges2 = cv2.Canny(med_blur, 20, 60)
        edges_list.append(edges2)
        
        # Method 3: Laplacian for subtle edges
        laplacian = cv2.Laplacian(fine_blur, cv2.CV_64F, ksize=3)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(np.clip(laplacian, 0, 255))
        _, lap_thresh = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
        edges_list.append(lap_thresh)
        
        # Method 4: Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        _, grad_thresh = cv2.threshold(magnitude, 25, 255, cv2.THRESH_BINARY)
        edges_list.append(grad_thresh)
        
    else:
        # Normal contrast - use standard approach
        heavy_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        edges1 = cv2.Canny(heavy_blur, 50, 150)
        edges_list.append(edges1)
        
        # Add gradient method
        grad_x = cv2.Sobel(heavy_blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(heavy_blur, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        _, grad_thresh = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
        edges_list.append(grad_thresh)
    
    # Combine all edge methods
    combined = edges_list[0].copy()
    for edge in edges_list[1:]:
        combined = cv2.bitwise_or(combined, edge)
    
    return combined

def preprocess_image_v2(gray: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Enhanced preprocessing with image analysis"""
    
    # Step 1: Analyze image characteristics
    stats = analyze_image_contrast(gray)
    
    # Step 2: Adaptive contrast enhancement
    enhanced = adaptive_contrast_enhancement(gray, stats)
    
    # Step 3: Noise reduction
    if stats['is_low_contrast']:
        # Gentle noise reduction for low contrast
        denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    else:
        # Standard noise reduction
        denoised = cv2.bilateralFilter(enhanced, 9, 80, 80)
    
    return denoised, stats

def order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2) float32
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(round(max(heightA, heightB)))
    if maxWidth < 1 or maxHeight < 1:
        raise ValueError("Computed warped size is invalid")
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_document_corners_v2(edges: np.ndarray, stats: dict) -> Optional[np.ndarray]:
    """Enhanced corner detection for low contrast scenarios"""
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Adjust minimum area threshold based on contrast
    if stats['is_low_contrast']:
        min_area_ratio = 0.05  # Lower threshold for low contrast
    else:
        min_area_ratio = 0.1
    
    min_area = edges.shape[0] * edges.shape[1] * min_area_ratio
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not large_contours:
        large_contours = contours[:5]  # Take more contours for low contrast
    
    # Try to find rectangular approximation with more epsilon values
    epsilon_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    if stats['is_low_contrast']:
        epsilon_values.extend([0.045, 0.05, 0.06])  # More aggressive approximation
    
    for c in large_contours:
        for epsilon_factor in epsilon_values:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon_factor * peri, True)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                if is_reasonable_rectangle(corners, edges.shape):
                    return corners
            elif len(approx) > 4 and len(approx) <= 8:
                # For low contrast, try to reduce polygon to 4 points
                hull = cv2.convexHull(approx)
                if len(hull) >= 4:
                    peri_hull = cv2.arcLength(hull, True)
                    approx_hull = cv2.approxPolyDP(hull, 0.03 * peri_hull, True)
                    if len(approx_hull) == 4:
                        corners = approx_hull.reshape(4, 2)
                        if is_reasonable_rectangle(corners, edges.shape):
                            return corners
    
    # Enhanced fallback methods for low contrast
    if large_contours:
        largest = large_contours[0]
        
        # Method 1: Douglas-Peucker with multiple epsilon values
        peri = cv2.arcLength(largest, True)
        for eps in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(largest, eps * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                if is_reasonable_rectangle(corners, edges.shape):
                    return corners
        
        # Method 2: Convex hull approach
        hull = cv2.convexHull(largest)
        peri_hull = cv2.arcLength(hull, True)
        approx_hull = cv2.approxPolyDP(hull, 0.04 * peri_hull, True)
        if len(approx_hull) == 4:
            corners = approx_hull.reshape(4, 2)
            if is_reasonable_rectangle(corners, edges.shape):
                return corners
        
        # Method 3: Minimum area rectangle (final fallback)
        box = cv2.boxPoints(cv2.minAreaRect(largest))
        return box.reshape(4, 2)
    
    return None

def is_reasonable_rectangle(corners: np.ndarray, image_shape: tuple) -> bool:
    """Check if detected corners form a reasonable rectangle"""
    h, w = image_shape[:2]
    
    # Check if corners are within bounds with some tolerance
    margin = 5
    for corner in corners:
        x, y = corner
        if x < -margin or x >= w + margin or y < -margin or y >= h + margin:
            return False
    
    # Calculate dimensions
    ordered = order_points(corners)
    (tl, tr, br, bl) = ordered
    
    width1 = np.linalg.norm(tr - tl)
    width2 = np.linalg.norm(br - bl)
    height1 = np.linalg.norm(bl - tl)
    height2 = np.linalg.norm(br - tr)
    
    avg_width = (width1 + width2) / 2
    avg_height = (height1 + height2) / 2
    
    # More lenient size requirements
    if avg_width < 30 or avg_height < 30:
        return False
    
    # More lenient aspect ratio
    aspect_ratio = max(avg_width/avg_height, avg_height/avg_width)
    if aspect_ratio > 15:  # Allow more extreme ratios
        return False
    
    return True

def adaptive_threshold_improved(gray: np.ndarray) -> np.ndarray:
    """Improved adaptive thresholding for final scan"""
    h, w = gray.shape[:2]
    
    # Determine block size
    if min(h, w) < 200:
        blockSize = 11
    elif min(h, w) < 500:
        blockSize = 15
    elif min(h, w) < 1000:
        blockSize = 21
    else:
        blockSize = 31
    
    # Multiple thresholding methods
    thresh1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, 10
    )
    
    thresh2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, 8
    )
    
    # Otsu's method
    _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine using majority voting
    combined = np.where((thresh1 == 255) & (thresh2 == 255), 255, 
                       np.where((thresh1 == 255) | (thresh2 == 255) | (thresh3 == 255), 255, 0))
    
    return combined.astype(np.uint8)

def scan_image(image_path: str, show_preview: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Resize for processing
    resized, ratio = resize_to_height(image, height=500)
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing with analysis
    processed, stats = preprocess_image_v2(gray)
    
    # Enhanced edge detection
    edges = multi_scale_edge_detection_v2(processed, stats)
    
    # Morphological operations (adaptive based on image type)
    if stats['is_low_contrast']:
        # Gentler morphology for low contrast
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        # Standard morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find document corners
    corners = find_document_corners_v2(closed, stats)
    
    if corners is None:
        # Last resort: use the whole resized frame
        h, w = resized.shape[:2]
        corners = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype="float32")
        print("[!] Could not detect document corners, using full image")
    
    # Map corners back to original image coordinates
    corners_original = corners.astype("float32") * ratio
    
    # Perspective transform
    warped_color = four_point_transform(image, corners_original)
    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    
    # Apply improved adaptive thresholding
    scanned = adaptive_threshold_improved(warped_gray)
    
    if show_preview:
        vis = resized.copy()
        cv2.drawContours(vis, [corners.astype(int)], -1, (0,255,0), 2)
        
        # Print image analysis
        print(f"\n=== IMAGE ANALYSIS ===")
        print(f"Mean brightness: {stats['mean']:.1f}")
        print(f"Contrast (std): {stats['std']:.1f}")
        print(f"RMS contrast: {stats['rms_contrast']:.1f}")
        print(f"Low contrast: {stats['is_low_contrast']}")
        print(f"Bright background: {stats['is_bright_background']}")
        
        # Show processing steps
        cv2.imshow("Step 1 - Original", resized)
        cv2.imshow("Step 2 - Enhanced", processed)
        cv2.imshow("Step 3 - Edges", edges)
        cv2.imshow("Step 4 - Closed Edges", closed)
        cv2.imshow("Step 5 - Detected Corners", vis)
        cv2.imshow("Step 6 - Warped Color", warped_color)
        cv2.imshow("Step 7 - Final Scan", scanned)
        
        print("\nPress any key (with any image window focused) to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return warped_color, scanned

def main():
    ap = argparse.ArgumentParser(description="Document scanner v8 - Enhanced low contrast detection")
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    ap.add_argument("-o", "--output", default=None, help="Path to save the B/W scanned output (PNG recommended)")
    ap.add_argument("--save-color", default=None, help="Optional path to save the color-warped image")
    ap.add_argument("--preview", action="store_true", help="Show interactive preview windows")
    args = ap.parse_args()

    try:
        color, bw = scan_image(args.image, show_preview=args.preview)

        # Derive output paths if not provided
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_bw = args.output or f"{base}_scanned_v8.png"
        out_color = args.save_color

        # Ensure PNG for crisp binary
        if not out_bw.lower().endswith(".png"):
            out_bw += ".png"
        
        ok1 = cv2.imwrite(out_bw, bw)
        if not ok1:
            print(f"[!] Failed to write {out_bw}", file=sys.stderr)
            return 1
            
        if out_color:
            ok2 = cv2.imwrite(out_color, color)
            if not ok2:
                print(f"[!] Failed to write {out_color}", file=sys.stderr)
                return 1
        
        print(f"[✓] Saved B/W scan to {out_bw}")
        if out_color:
            print(f"[✓] Saved color warp to {out_color}")
            
        return 0
        
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())