import customtkinter as ctk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import time
import numpy as np
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import tkinter as tk

# Load YOLOv8 model
try:
    model = YOLO("best.pt")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define all classes
CLASS_NAMES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule-Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis'
]

# Separate heart and lung diseases
HEART_DISEASES = ['Aortic enlargement', 'Calcification', 'Cardiomegaly']
LUNG_DISEASES = [cls for cls in CLASS_NAMES if cls not in HEART_DISEASES]

# Disease Classification Information
CLASSIFICATION_INFO = """
DISEASE CLASSIFICATION METHODOLOGY

This system classifies chest X-ray abnormalities based on:

1. ANATOMICAL LOCATION:
   - Lung parenchyma (Atelectasis, Consolidation, Nodule-Mass)
   - Pleural space (Pleural effusion, Pneumothorax, Pleural thickening)
   - Interstitial tissue (ILD, Pulmonary fibrosis)
   - Mediastinum (Cardiomegaly, Aortic enlargement)

2. RADIOLOGICAL APPEARANCE:
   - Opacities: Consolidation, Infiltration, Lung Opacity
   - Lucencies: Pneumothorax
   - Linear patterns: Pulmonary fibrosis, ILD
   - Calcifications: Calcification
   - Volume changes: Atelectasis, Cardiomegaly

3. PATHOPHYSIOLOGY:
   - Inflammatory: Consolidation, Infiltration
   - Fibrotic: Pulmonary fibrosis, ILD
   - Neoplastic: Nodule-Mass
   - Mechanical: Atelectasis, Pneumothorax
   - Degenerative: Calcification, Pleural thickening

4. SEVERITY ASSESSMENT:
   - Mild: Confidence < 50% (Subtle findings)
   - Moderate: Confidence 50-80% (Clear abnormalities)
   - Severe: Confidence > 80% (Advanced disease)

The YOLOv8 model was trained on annotated chest X-ray datasets with expert radiologist validation.
"""

# Causes dictionary
CAUSES = {
    "Atelectasis": "Causes: Airway blockage, surgery, lung pressure, mucus plugs, tumors, chest injuries",
    "Consolidation": "Causes: Pneumonia, tuberculosis, pulmonary edema, lung cancer, fungal infections",
    "ILD": "Causes: Autoimmune diseases, environmental exposures, medications, radiation therapy, genetics",
    "Infiltration": "Causes: Infections, inflammation, cancer cells, fluid accumulation, inflammatory conditions",
    "Lung Opacity": "Causes: Infections, inflammation, fluid, bleeding, tumors, fibrosis, atelectasis",
    "Nodule-Mass": "Causes: Benign tumors, cancer, infections, inflammation, congenital conditions",
    "Other lesion": "Causes: Various benign/malignant growths, infections, congenital abnormalities, trauma",
    "Pleural effusion": "Causes: Heart failure, pneumonia, cancer, pulmonary embolism, kidney/liver disease",
    "Pleural thickening": "Causes: Asbestos exposure, infections, trauma, autoimmune diseases, radiation",
    "Pneumothorax": "Causes: Chest trauma, lung disease, mechanical ventilation, smoking, tall thin build",
    "Pulmonary fibrosis": "Causes: Environmental exposures, autoimmune diseases, medications, radiation, genetics"
}

# Advice dictionary
ADVICE = {
    "Atelectasis": "Treatment: Breathing exercises, chest physiotherapy.\nPrevention: Stay active, avoid prolonged bed rest.",
    "Consolidation": "Treatment: Antibiotics for infection.\nPrevention: Vaccination, good hygiene.",
    "ILD": "Treatment: Steroids, antifibrotic drugs.\nPrevention: Avoid smoking, occupational hazards.",
    "Infiltration": "Treatment: Depends on underlying cause.\nPrevention: Early treatment of infections.",
    "Lung Opacity": "Treatment: Based on diagnosis.\nPrevention: Avoid exposure to pollutants.",
    "Nodule-Mass": "Treatment: Biopsy, surgery, chemo/radiotherapy.\nPrevention: Early detection, avoid smoking.",
    "Other lesion": "Treatment: Depends on nature.\nPrevention: Regular medical screening.",
    "Pleural effusion": "Treatment: Drainage, antibiotics.\nPrevention: Treat underlying diseases early.",
    "Pleural thickening": "Treatment: Symptom management.\nPrevention: Avoid asbestos exposure.",
    "Pneumothorax": "Treatment: Chest tube, surgery if recurrent.\nPrevention: Avoid smoking, lung health checkups.",
    "Pulmonary fibrosis": "Treatment: Antifibrotics, lung transplant.\nPrevention: Avoid lung irritants."
}

# Class colors
CLASS_COLORS = {
    'Atelectasis': (0, 255, 0),  # Green
    'Consolidation': (255, 0, 255),  # Magenta
    'ILD': (255, 255, 0),  # Cyan
    'Infiltration': (0, 128, 255),  # Light Blue
    'Lung Opacity': (255, 0, 128),  # Pink
    'Nodule-Mass': (128, 0, 255),  # Purple
    'Other lesion': (128, 128, 128),  # Gray
    'Pleural effusion': (0, 255, 128),  # Teal
    'Pleural thickening': (255, 128, 0),  # Orange
    'Pneumothorax': (128, 255, 0),  # Lime
    'Pulmonary fibrosis': (0, 128, 128)  # Dark Cyan
}


# Severity stage
def get_severity_stage(conf):
    if conf < 0.5:
        return "Mild"
    elif conf < 0.8:
        return "Moderate"
    else:
        return "Severe"


def draw_square_highlight(img, x1, y1, x2, y2, color, label=None, thickness=2):
    """
    Draw a square/rectangle highlight with semi-transparent fill
    """
    # Create overlay for semi-transparent fill
    overlay = img.copy()
    alpha = 0.3

    # Draw filled rectangle
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Add label above the square if provided
    if label:
        # Calculate text size for proper positioning
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 20 else y2 + 20

        # Draw text background for better readability
        cv2.rectangle(img,
                      (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 10, text_y + 5),
                      color, -1)

        # Draw text
        cv2.putText(img, label, (text_x + 5, text_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def add_disease_panel(img, diseases):
    """
    Adds a semi-transparent explanation panel showing detected diseases
    Auto-adjusts size based on number of diseases
    """
    if not diseases:
        return

    # Panel dimensions (auto-adjust based on content)
    panel_margin = 20
    panel_width = 350
    line_height = 25
    header_height = 40
    panel_height = header_height + (len(diseases) * line_height * 2) + 20

    # Get image dimensions
    img_height, img_width = img.shape[:2]

    # Panel position (top-left corner)
    x, y = panel_margin, panel_margin

    # Ensure panel doesn't go off-screen
    if x + panel_width > img_width:
        x = img_width - panel_width - panel_margin
    if y + panel_height > img_height:
        y = img_height - panel_height - panel_margin

    # Create overlay for semi-transparent panel
    overlay = img.copy()

    # Draw panel background
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                  (0, 0, 0), -1)

    # Blend panel with image
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Draw panel border
    cv2.rectangle(img, (x, y), (x + panel_width, y + panel_height),
                  (255, 255, 255), 2)

    # Panel title
    title = "Detected Diseases & Causes"
    cv2.putText(img, title, (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw separator line
    cv2.line(img, (x, y + header_height),
             (x + panel_width, y + header_height),
             (255, 255, 255), 1)

    # List detected diseases with causes
    for i, (disease, color, causes) in enumerate(diseases):
        y_pos_disease = y + header_height + 10 + (i * line_height * 2)
        y_pos_causes = y_pos_disease + line_height

        # Draw color indicator
        cv2.circle(img, (x + 15, y_pos_disease - 5), 5, color, -1)
        cv2.circle(img, (x + 15, y_pos_disease - 5), 5, (255, 255, 255), 1)

        # Draw disease name
        cv2.putText(img, disease, (x + 30, y_pos_disease),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw causes (truncated if too long)
        causes_text = causes[:45] + "..." if len(causes) > 45 else causes
        cv2.putText(img, causes_text, (x + 10, y_pos_causes),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def resize_image_with_aspect_ratio(image, target_size):
    """
    Resize image while maintaining aspect ratio
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    if target_width / target_height > aspect_ratio:
        # Fit to height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        # Fit to width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create black background
    result_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate position to center the image
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # Place resized image in center
    result_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return result_image, (x_offset, y_offset, new_width, new_height)


def scale_coordinates(bbox, original_size, display_size, offset):
    """
    Scale bounding box coordinates from original image to display size
    """
    x1, y1, x2, y2 = bbox
    orig_width, orig_height = original_size
    x_offset, y_offset, disp_width, disp_height = offset

    # Calculate scaling factors
    scale_x = disp_width / orig_width
    scale_y = disp_height / orig_height

    # Scale coordinates
    x1_scaled = int(x1 * scale_x) + x_offset
    y1_scaled = int(y1 * scale_y) + y_offset
    x2_scaled = int(x2 * scale_x) + x_offset
    y2_scaled = int(y2 * scale_y) + y_offset

    return x1_scaled, y1_scaled, x2_scaled, y2_scaled


# --- App class ---
class XRayApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🩻 Chest X-ray Disease Detection (YOLOv8)")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Sidebar
        sidebar = ctk.CTkFrame(self, width=250, corner_radius=15)
        sidebar.pack(side="left", fill="y", padx=15, pady=15)

        title = ctk.CTkLabel(sidebar, text="X-ray Analyzer", font=("Arial", 26, "bold"))
        title.pack(pady=20)

        self.upload_btn = ctk.CTkButton(sidebar, text="📂 Upload X-ray",
                                        fg_color="#1f6aa5", hover_color="#144870",
                                        command=self.upload_image, width=200, height=40, corner_radius=10)
        self.upload_btn.pack(pady=15)

        self.report_btn = ctk.CTkButton(sidebar, text="💾 Save Report",
                                        fg_color="#4caf50", hover_color="#357a38",
                                        command=self.generate_pdf_report, width=200, height=40, corner_radius=10)
        self.report_btn.pack(pady=15)

        # NEW: About Classification button
        self.about_btn = ctk.CTkButton(sidebar, text="🔍 About Classification",
                                       fg_color="#ff9800", hover_color="#e68900",
                                       command=self.show_classification_info, width=200, height=40, corner_radius=10)
        self.about_btn.pack(pady=15)

        self.reset_btn = ctk.CTkButton(sidebar, text="🔄 Reset",
                                       fg_color="#e53935", hover_color="#a02725",
                                       command=self.reset_app, width=200, height=40, corner_radius=10)
        self.reset_btn.pack(pady=15)

        self.status_label = ctk.CTkLabel(sidebar, text="Status: Waiting for input...", wraplength=200)
        self.status_label.pack(pady=20)

        self.progress = ctk.CTkProgressBar(sidebar, width=200)
        self.progress.set(0)
        self.progress.pack(pady=10)

        results_frame = ctk.CTkFrame(sidebar)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.result_list = tk.Listbox(results_frame, width=30, height=20, font=("Arial", 10),
                                      bg="#2b2b2b", fg="white", selectbackground="#1f6aa5")
        self.result_list.pack(fill="both", expand=True, padx=5, pady=5)

        main_panel = ctk.CTkFrame(self, corner_radius=15)
        main_panel.pack(side="right", expand=True, fill="both", padx=15, pady=15)

        self.canvas = ctk.CTkLabel(main_panel, text="")
        self.canvas.pack(pady=10)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#2b2b2b",
                        foreground="white",
                        rowheight=30,
                        fieldbackground="#2b2b2b",
                        font=("Arial", 12))
        style.map("Treeview", background=[("selected", "#1f6aa5")])

        # Treeview to show Disease, Stage, and Causes
        self.tree = ttk.Treeview(main_panel, columns=("Disease", "Stage", "Causes"), show="headings", height=12)
        self.tree.heading("Disease", text="Disease")
        self.tree.heading("Stage", text="Stage")
        self.tree.heading("Causes", text="Main Causes")
        self.tree.column("Disease", width=250, anchor="center")
        self.tree.column("Stage", width=100, anchor="center")
        self.tree.column("Causes", width=350, anchor="w")
        self.tree.pack(pady=10, fill="x")

        # Add scrollbar to treeview for causes column
        tree_scroll = ttk.Scrollbar(main_panel, orient="vertical", command=self.tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.image_path = None
        self.detected_results = []
        self.annotated_path = None
        self.original_image = None
        self.display_image = None

    def show_classification_info(self):
        """Show classification methodology in a new window"""
        info_window = ctk.CTkToplevel(self)
        info_window.title("Disease Classification Methodology")
        info_window.geometry("800x600")
        info_window.transient(self)
        info_window.grab_set()

        # Create text widget with scrollbar
        frame = ctk.CTkFrame(info_window)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        text_widget = tk.Text(frame, wrap="word", bg="#2b2b2b", fg="white",
                              font=("Arial", 12), padx=15, pady=15)
        text_widget.pack(side="left", fill="both", expand=True)

        scrollbar = ctk.CTkScrollbar(frame, command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.configure(yscrollcommand=scrollbar.set)

        # Insert classification information
        text_widget.insert("1.0", CLASSIFICATION_INFO)
        text_widget.configure(state="disabled")

        # Close button
        close_btn = ctk.CTkButton(info_window, text="Close",
                                  command=info_window.destroy, width=100)
        close_btn.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path = file_path
            threading.Thread(target=self.run_detection, args=(file_path,), daemon=True).start()

    def run_detection(self, img_path):
        self.status_label.configure(text="Processing image... ⏳")
        self.progress.set(0)

        # Simulate progress
        for i in range(1, 101, 5):
            self.progress.set(i / 100)
            time.sleep(0.01)

        try:
            if model is None:
                raise Exception("Model not loaded properly")

            # Clear previous results
            self.detected_results.clear()
            self.result_list.delete(0, tk.END)
            for row in self.tree.get_children():
                self.tree.delete(row)

            # Load and store original image
            self.original_image = cv2.imread(img_path)
            if self.original_image is None:
                raise Exception("Failed to load image")

            # Run YOLO detection on original image
            results = model(self.original_image)

            # Create a copy for display with proper aspect ratio
            display_size = (700, 700)
            self.display_image, offset = resize_image_with_aspect_ratio(
                self.original_image.copy(), display_size
            )

            disease_best = {}
            detected_diseases = []  # For the panel

            # Process detection results
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = CLASS_NAMES[cls_id]

                    # Skip heart diseases
                    if label in HEART_DISEASES:
                        continue

                    # Keep only the highest confidence detection per disease
                    if label not in disease_best or conf > disease_best[label]['conf']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        disease_best[label] = {
                            'conf': conf,
                            'bbox': (x1, y1, x2, y2)
                        }

            # Draw square annotations on display image
            original_size = (self.original_image.shape[1], self.original_image.shape[0])

            for label, info in disease_best.items():
                conf = info['conf']
                stage = get_severity_stage(conf)
                color = CLASS_COLORS.get(label, (255, 255, 255))
                causes = CAUSES.get(label, "Causes: Various factors")

                # Scale coordinates to display image
                scaled_bbox = scale_coordinates(
                    info['bbox'], original_size, display_size, offset
                )

                # Draw square highlight on display image
                draw_square_highlight(self.display_image, *scaled_bbox, color, label)

                # Store results (disease, stage, causes)
                self.detected_results.append((label, stage, causes))
                detected_diseases.append((label, color, causes))

                # Update UI - show disease, stage, and causes
                self.result_list.insert(tk.END, f"{label} - {stage}")
                hex_color = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
                self.result_list.itemconfig(tk.END, {'fg': hex_color})
                self.tree.insert("", "end", values=(label, stage, causes))

            # Add disease panel to display image (only if diseases detected)
            if detected_diseases:
                add_disease_panel(self.display_image, detected_diseases)

            # Save annotated image
            self.annotated_path = "annotated_xray.jpg"
            cv2.imwrite(self.annotated_path, self.display_image)

            # Convert and display image
            display_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(display_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.canvas.configure(image=img_tk)
            self.canvas.image = img_tk

            self.status_label.configure(text="✅ Detection Completed")
            self.progress.set(1.0)

        except Exception as e:
            self.status_label.configure(text=f"❌ Error: {str(e)}")
            self.progress.set(0)

    def generate_pdf_report(self):
        if not self.detected_results:
            messagebox.showwarning("No Data", "Please upload and detect an X-ray first!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"Xray_Report_{timestamp}.pdf"

        try:
            doc = SimpleDocTemplate(report_path)
            styles = getSampleStyleSheet()
            story = []

            # Title and classification info
            story.append(Paragraph("Chest X-ray Lung Disease Detection Report", styles["Title"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("<b>Classification Methodology:</b>", styles["Heading2"]))
            classification_lines = CLASSIFICATION_INFO.split('\n')
            for line in classification_lines:
                if line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 12))

            story.append(
                Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            story.append(Paragraph(f"<b>Total Findings:</b> {len(self.detected_results)}", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Disease findings
            story.append(Paragraph("<b>Detected Abnormalities:</b>", styles["Heading2"]))
            for disease, stage, causes in self.detected_results:
                story.append(Paragraph(f"<b>Disease:</b> {disease}", styles["Normal"]))
                story.append(Paragraph(f"<b>Stage:</b> {stage}", styles["Normal"]))
                story.append(Paragraph(f"<b>Main Causes:</b> {causes}", styles["Normal"]))
                story.append(Paragraph(f"<b>Treatment & Prevention:</b>", styles["Normal"]))

                advice = ADVICE.get(disease,
                                    "Consult with a healthcare professional for proper diagnosis and treatment.")
                advice_lines = advice.split('\n')
                for line in advice_lines:
                    story.append(Paragraph(f"• {line}", styles["Normal"]))

                story.append(Spacer(1, 12))

            if self.annotated_path and os.path.exists(self.annotated_path):
                story.append(Paragraph("<b>Annotated X-ray:</b>", styles["Normal"]))
                story.append(RLImage(self.annotated_path, width=400, height=400))

            story.append(Spacer(1, 12))
            story.append(Paragraph(
                "<i>This report is generated by AI and should be reviewed by a qualified medical professional.</i>",
                styles["Italic"]))

            doc.build(story)

            self.status_label.configure(text=f"💾 PDF Report Generated: {os.path.basename(report_path)}")
            messagebox.showinfo("Report Generated", f"Report saved as:\n{os.path.abspath(report_path)}")

        except Exception as e:
            self.status_label.configure(text=f"❌ Error generating PDF: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate PDF report:\n{str(e)}")

    def reset_app(self):
        self.canvas.configure(image="", text="")
        self.canvas.image = None

        self.result_list.delete(0, tk.END)
        for row in self.tree.get_children():
            self.tree.delete(row)

        self.status_label.configure(text="🔄 Reset done. Ready for new X-ray.")
        self.progress.set(0)

        self.detected_results.clear()
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.annotated_path = None

        # Clean up temporary annotated image
        if self.annotated_path and os.path.exists(self.annotated_path):
            try:
                os.remove(self.annotated_path)
            except:
                pass


if __name__ == "__main__":
    app = XRayApp()
    app.mainloop()