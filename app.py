import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import json

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("skin_model.keras")

with open("classes.json", "r") as f:
    classes = json.load(f)

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = cv2.resize(img, (224,224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.reshape(img, (1,224,224,3))

# ---------------- OPENCV FACE DETECTION ----------------
def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]

    return img  # fallback

# ---------------- DRYNESS DETECTION ----------------
def detect_dryness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)
    texture = np.std(gray)

    if brightness < 110 and texture > 50:
        return True
    return False

# ---------------- SMART PREDICT ----------------
def predict(img):
    face = extract_face(img)

    processed = preprocess(face)
    pred = model.predict(processed)[0]

    top_indices = np.argsort(pred)[::-1]

    main = classes[top_indices[0]]
    second = classes[top_indices[1]]

    conf_main = pred[top_indices[0]]
    conf_second = pred[top_indices[1]]

    # Fix dry vs combination issue
    if main == "combination" and detect_dryness(face):
        return "dry", main, second, conf_main, face

    # Smart logic
    if conf_main < 0.45:
        result = "uncertain"
    elif abs(conf_main - conf_second) < 0.15:
        result = f"{main} / {second}"
    else:
        result = main

    return result, main, second, conf_main, face

# ---------------- PRODUCTS ----------------
PRODUCTS = {
    "oily": [
        {
            "name": "Cetaphil Oily Skin Cleanser",
            "image": "https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcTTpZuRhqCPOaiaJ18cSBcPlruhH-jj0wAFgOVcpWWxPmqRf-yPGOfdi6qBTmfZ-MZEk2XmBgChc-zwPXqUtQOixb5NNy4omjvuG5TZGih-_mph9sjMwlrK0ylb66fjI0AON7Eijg&usqp=CAc",
            "amazon": "https://www.amazon.in/s?k=cetaphil+oily+skin+cleanser",
            "flipkart": "https://www.flipkart.com/search?q=cetaphil+oily+skin+cleanser"
        },
        {
            "name": "Minimalist Niacinamide Serum",
            "image": "https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcSlVJd1VF3eCgNKUvW8NOurWRdSJohVbEDs7AD6CUwC5QwjRgNn2AH4ZQFEyuiuaDLKfYejcAYPMW6APR6TDkC8C9w5goZXn0bH41ZMNe4HkDT37V04t8naZpeOUXNY_Q2qjeFk8w&usqp=CAc",
            "amazon": "https://www.amazon.in/s?k=niacinamide+serum",
            "flipkart": "https://www.flipkart.com/search?q=niacinamide+serum"
        }
    ],

    "dry": [
        {
            "name": "Cetaphil Moisturizing Cream",
            "image": "https://encrypted-tbn2.gstatic.com/shopping?q=tbn:ANd9GcRlWmxLIRwx1YDtRrtBG0UJys9PYE6qqSP2XmzZmOcEcPdRrClMgJ-toxl-kO46m7yMYMmRoZZt5Q6TyJTtKPbLFv4ehZq2cjILu89I1nr3lHxqX3TA9cKcfGcp5khqm8OyXapxbg&usqp=CAc",
            "amazon": "https://www.amazon.in/s?k=cetaphil+moisturizing+cream",
            "flipkart": "https://www.flipkart.com/search?q=cetaphil+moisturizing+cream"
        },
        {
            "name": "Hyaluronic Acid Serum",
            "image": "https://encrypted-tbn0.gstatic.com/shopping?q=tbn:ANd9GcQgkcaufBHL491HcR7FjFBVffZOZ3WjDwMdnbalNYyIKuaKbi7tOR1da-0QgbBEhMSwcdTHAZDWF64rNjocv7b4HzyDPoDt_uw8V-x0JbznJSzlCOs7hdz4Vmf6DOLj2_ERg6P73g&usqp=CAc",
            "amazon": "https://www.amazon.in/s?k=hyaluronic+acid+serum",
            "flipkart": "https://www.flipkart.com/search?q=hyaluronic+acid+serum"
        }
    ],

    "sensitive": [
        {
            "name": "Aveeno Moisturizer",
            "image": "https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcRUETfQgsWaKB3zkE6Bc-LO5ETLfLJIwgApDx6I59obwuKHBZqlTM4Hd775lZMKXR8FLh3IjP3WocpWiHFNyI-2v2JJ_iM1CzPPIzA1hZul3h1nscPJEuz5BX8zGcrXWQmLpw&usqp=CAc",
            "amazon": "https://www.amazon.in/s?k=aveeno+moisturizer",
            "flipkart": "https://www.flipkart.com/search?q=aveeno+moisturizer"
        },
        {
            "name": "Aloe Vera Gel",
            "image": "https://www.jiomart.com/images/product/original/rvvckozwcl/life-aveda-pure-organic-aloe-vera-gel-deep-moisturizer-for-face-hair-and-body-soothes-hydrates-and-softens-skin-hair-relief-from-sunburn-tanning-skin-irritation-acne-rashes-130-ml-product-images-orvvckozwcl-p610628154-0-202411061120.jpg?im=Resize=(420,420)",
            "amazon": "https://www.amazon.in/s?k=aloe+vera+gel",
            "flipkart": "https://www.flipkart.com/search?q=aloe+vera+gel"
        }
    ],

    "combination": [
        {
            "name": "Simple Face Wash",
            "image": "https://encrypted-tbn2.gstatic.com/shopping?q=tbn:ANd9GcQUNvPKM9CUJ6MPnvh6H0XATmibx2YQiZRaa6M6kTpaLF4qwdLRrkTivV5RzyBxy481AVau2OZuTUkqJRzczaZGzFLZ6Sho472r1USdJ-KDpOg1iCDTDHsSBTewApJgs6RNcGxV5pzC&usqp=CAc",
            "amazon": "https://www.amazon.in/s?k=simple+face+wash",
            "flipkart": "https://www.flipkart.com/search?q=simple+face+wash"
        },
        {
            "name": "Plum Green Tea Moisturizer",
            "image": "https://encrypted-tbn2.gstatic.com/shopping?q=tbn:ANd9GcRv3mUMlt5P8exiUFuaY5g08muj6bIyfbc8fF3hRLRwKQsgVEIuzHMylwzOv3bMBD-ymDuBdOBYPAuB-8Lwnt9O6fAiRvERTV0SHNUgSnBLDOb0EPyNwVlHGWKScUmwksr2U3kKtTNCqA&usqp=CAc",
            "amazon": "https://www.amazon.in/s?k=plum+green+tea+moisturizer",
            "flipkart": "https://www.flipkart.com/search?q=plum+green+tea+moisturizer"
        }
    ]
}

# ---------------- UI ----------------
st.set_page_config(page_title="DermScan AI Pro", layout="centered")

st.title("🔬 DermScan AI Pro")
st.subheader("Smart Skin Type Detection & Product Recommendation")

# Input method
option = st.radio("Choose Input Method", ["Upload Image", "Open Camera"])

img = None

# Upload
if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file)
        img = np.array(image)
        st.image(image, caption="Uploaded Image", width="stretch")

# Camera
elif option == "Open Camera":
    camera = st.camera_input("Capture Image")
    if camera:
        image = Image.open(camera)
        img = np.array(image)
        st.image(image, caption="Captured Image", width="stretch")

# Prediction
if img is not None:
    if st.button("🔍 Analyze Skin"):
        with st.spinner("Analyzing..."):
            result, main, second, conf, face = predict(img)

        st.image(face, caption="Detected Face Region", width="stretch")

        if result == "uncertain":
            st.warning("⚠️ Unable to confidently detect skin type")
            st.info(f"Closest match: {main.capitalize()}")
        elif "/" in result:
            st.success(f"🧴 Skin Type: {result.capitalize()} (Combination-like)")
        else:
            st.success(f"🧴 Skin Type: {result.capitalize()}")

        st.markdown("### 🛍️ Recommended Products")

        for item in PRODUCTS.get(main, []):
            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(item["image"], width=150)

            with col2:
                st.markdown(f"### {item['name']}")
                st.markdown(f"[🛒 Buy on Amazon]({item['amazon']})")
                st.markdown(f"[🛒 Buy on Flipkart]({item['flipkart']})")

            st.markdown("---")

        st.caption("⚠️ AI-based suggestion (limited dataset)")

else:
    st.info("Upload or capture an image to begin")