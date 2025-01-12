import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image) -> int:
    'function for predicting disease'
    model = tf.keras.models.load_model("trained_model.keras", compile=False)

    # image to PIL format
    image = tf.keras.utils.load_img(test_image, target_size=(128, 128))
    # PIL Image to array
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.array([img_array]) # to covert single image to a batch

    prediction = model.predict(img_array)
    result_index = np.argmax(prediction)
    return result_index

def get_disease_name(index) -> str:
    'function for getting the disease name'
    class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
    ]
    disease_name_split = class_names[index].split('_')
    disease_name_list = []
    for i in disease_name_split:
        if i != "" and i not in disease_name_list:
            disease_name_list.append(i)
    disease_name = " ".join(disease_name_list)
    return disease_name

def app():
    'the streamlit app'
    # sidebar
    st.sidebar.title('Dashboard')
    app_mode = st.sidebar.selectbox("Select Page", ["Homepage", "Disease Recognition", "About"])

    # Home Page
    if app_mode == "Homepage":
        st.header("UTILIZING AI IN CROP DISEASE DETECTION")
        st.markdown(
            """
            write your introduction here
            """
        )
    elif app_mode == "Disease Recognition":
        st.header("Disease Recognition", divider=True)
        st.subheader(
            """
            This is the sub heading for this page.\n
            it should contain an explaination on how this page works.
            """
        )
        input_img = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jfif'])
        # show image button
        if st.button("Show Image"):
            if input_img:
                st.image(input_img, use_column_width=True)
            else:
                st.warning("No input image.", icon="🚨")
        # predict button
        if st.button("Predict"):
            if input_img:
                with st.spinner("processing image..."):
                    result = model_prediction(input_img)
                    name = get_disease_name(result)
                    if 'healthy' in name:
                        i = "🔥"
                    else:
                        i = "🚨"
                    st.write("Our Prediction")
                    st.success(f"Model is predicting it's ***{name}***.", icon=i)
                    st.balloons()
            else:
                st.warning("No input image.", icon="🚨")

    elif app_mode == "About":
        st.header("About", divider=True, anchor="about")
        st.markdown(
            """
            About should go here **JJ**.\n
            Typically should be info about the project members,
            refrences, etc.
            """
        )

if __name__ == "__main__":
    app() # run app
