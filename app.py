import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile

# YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# TFLite
import tensorflow as tf
try:
    from tflite_support.metadata.python import metadata as _metadata
except ImportError:
    _metadata = None

st.set_page_config(page_title="Universal Object Detection App", layout="wide")
st.title("Universal Object Detection App (Custom Class Names Supported)")

def get_class_colors(num_classes):
    return {i: tuple(np.random.randint(0, 256, 3).tolist()) for i in range(num_classes)}

# ----------------- Model Selection -----------------
model_family = st.selectbox(
    "Select Model Family",
    ["YOLO", "SSD", "EfficientDet"]
)

file_format = None
if model_family:
    if model_family == "YOLO":
        file_format = st.selectbox("Select YOLO Format", [".pt", ".tflite"])
    else:
        file_format = ".tflite"

# ----------------- Upload Model -----------------
if file_format:
    model_file = st.file_uploader(f"Upload your model ({file_format})", type=[file_format.replace(".", "")])
    if model_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_format) as tmp:
            tmp.write(model_file.read())
            model_path = tmp.name
        st.success("✅ Model uploaded successfully!")

        # ----------------- Upload Image -----------------
        uploaded_image = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
        if uploaded_image:
            img = Image.open(uploaded_image).convert("RGB")
            img_np = np.array(img)
            st.image(img_np, caption="Uploaded Image", use_column_width=True)
            st.subheader("🔎 Detection Result")

            # ----------------- YOLO .pt -----------------
            if file_format == ".pt":
                if YOLO is None:
                    st.error("YOLO not installed. Run: pip install ultralytics")
                else:
                    model = YOLO(model_path)
                    class_names = model.names
                    class_colors = get_class_colors(len(class_names))
                    results = model(img_np)
                    img_out = results[0].plot()
                    st.image(img_out, caption="YOLO Detection", use_column_width=True)

            # ----------------- TFLite -----------------
            elif file_format == ".tflite":
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Extract class names from metadata
                if _metadata is not None:
                    try:
                        displayer = _metadata.MetadataDisplayer.with_model_file(model_path)
                        files = displayer.get_packed_associated_file_list()
                        if files:
                            labels_text = displayer.get_associated_file_buffer(files[0]).decode('utf-8').splitlines()
                            class_names = labels_text
                        else:
                            class_names = ["Class_"+str(i) for i in range(100)]
                    except:
                        class_names = ["Class_"+str(i) for i in range(100)]
                else:
                    class_names = ["Class_"+str(i) for i in range(100)]

                class_colors = get_class_colors(len(class_names))

                # Prepare input
                input_shape = input_details[0]['shape']
                img_resized = cv2.resize(img_np, (input_shape[2], input_shape[1]))
                input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Get outputs
                boxes = interpreter.get_tensor(output_details[0]['index'])[0]
                classes = interpreter.get_tensor(output_details[1]['index'])[0]
                scores = interpreter.get_tensor(output_details[2]['index'])[0]

                # Draw boxes
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                for i, score in enumerate(scores):
                    if score > 0.5:
                        y1, x1, y2, x2 = boxes[i]
                        h, w, _ = img_np.shape
                        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                        class_id = int(classes[i])
                        color = class_colors.get(class_id, (0,255,0))
                        label_name = class_names[class_id] if class_id < len(class_names) else f"ID {class_id}"
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img_bgr, f"{label_name}: {score:.2f}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                img_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_out, caption="TFLite Detection", use_column_width=True)
