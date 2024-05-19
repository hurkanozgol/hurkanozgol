import gradio as gr
import cv2
import numpy as np
from circle_fit import hyperLSQ, taubinSVD

def process_image(image_path, threshold, diameter, sigmaColor, sigmaSpace):
    # Read the image from the given path
    img = cv2.imread(image_path)
    
    if img is None:
        return None, None, 0, None
    
    img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, diameter, sigmaColor, sigmaSpace)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    closed_contours = []
    total_pixels = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            closed_contours.append(cnt)
            total_pixels += cv2.contourArea(cnt)

    closed_contours = [cnt for cnt in closed_contours if cv2.pointPolygonTest(cnt, (20, 20), False) != 1 and cv2.pointPolygonTest(cnt, (img.shape[1]/2, img.shape[0]/2), False) == 1]

    lines = cv2.drawContours(img_rbg, closed_contours, -1, (0, 255, 0), 3)
    
    if closed_contours:
        circle_points = [pt[0] for cnt in closed_contours for pt in cnt]
        circle_points = np.array(circle_points)
        xc, yc, R, sigma = hyperLSQ(circle_points)
        circle_img = cv2.circle(img_rbg, (int(xc), int(yc)), int(R), (255, 0, 0), 2)
        diameter_mm = (R / 317) * 8
    else:
        circle_img = img_rbg
        diameter_mm = 0
    
    return circle_img, closed_contours, total_pixels, diameter_mm

def compare_images(image1, image2, threshold, diameter, sigmaColor, sigmaSpace):
    lines1, contours1, pixels1, diameter1 = process_image(image1.name, threshold, diameter, sigmaColor, sigmaSpace)
    lines2, contours2, pixels2, diameter2 = process_image(image2.name, threshold, diameter, sigmaColor, sigmaSpace)
    
    pixel_diff = abs(pixels1 - pixels2)
    diameter_diff = abs(diameter1 - diameter2)
    
    pixel_percent_diff = (pixel_diff / ((pixels1 + pixels2) / 2)) * 100 if (pixels1 + pixels2) != 0 else 0
    diameter_percent_diff = (diameter_diff / ((diameter1 + diameter2) / 2)) * 100 if (diameter1 + diameter2) != 0 else 0
    
    comparison_text = (
        f"Image 1 Pixels: {pixels1}\n"
        f"Image 1 Diameter: {diameter1:.2f} mm\n\n"
        f"Image 2 Pixels: {pixels2}\n"
        f"Image 2 Diameter: {diameter2:.2f} mm\n\n"
        f"Difference in Pixels: {pixel_diff}\n"
        f"Pixel Percentage Difference: {pixel_percent_diff:.2f}%\n\n"
        f"Difference in Diameter: {diameter_diff:.2f} mm\n"
        f"Diameter Percentage Difference: {diameter_percent_diff:.2f}%"
    )
    
    return lines1, lines2, comparison_text


with gr.Blocks() as hurkan_goruntuleme:
    with gr.Column():
        image1 = gr.File(label="Upload Image 1")
        image2 = gr.File(label="Upload Image 2")
        threshold = gr.Slider(1, 255, step=1, label="Threshold", value=90)
        with gr.Accordion("Advanced Settings", open=False):
            diameter = gr.Slider(1, 100, step=2, label="Diameter", value=23)
            sigmaColor = gr.Slider(1, 100, step=2, label="Sigma Color", value=61)
            sigmaSpace = gr.Slider(1, 100, step=2, label="Sigma Space", value=11)
        button = gr.Button("Process Images")
    
    with gr.Row():
        output1 = gr.Image(interactive=False, width="38vw", height="28vw", label="Processed Image 1")
        output2 = gr.Image(interactive=False, width="38vw", height="28vw", label="Processed Image 2")
        comparison = gr.TextArea(label="Comparison Result", interactive=False)
    
    button.click(fn=compare_images, inputs=[image1, image2, threshold, diameter, sigmaColor, sigmaSpace], outputs=[output1, output2, comparison])

hurkan_goruntuleme.launch(share=True)
