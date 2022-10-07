from img_classification import teachable_machine_classification
import keras
import streamlit as st
st.title("Image Classification with Google's Teachable Machine")
st.header("Fruit Classification  Example")
st.text("Upload a Image for image classification as Fruit")
class_label = ['apple_6',
 'apple_braeburn_1',
 'apple_crimson_snow_1',
 'apple_golden_1',
 'apple_golden_2',
 'apple_golden_3',
 'apple_granny_smith_1',
 'apple_hit_1',
 'apple_pink_lady_1',
 'apple_red_1',
 'apple_red_2',
 'apple_red_3',
 'apple_red_delicios_1',
 'apple_red_yellow_1',
 'apple_rotten_1',
 'cabbage_white_1',
 'carrot_1',
 'cucumber_1',
 'cucumber_3',
 'eggplant_violet_1',
 'pear_1',
 'pear_3',
 'zucchini_1',
 'zucchini_dark_1']

uploaded_file = st.file_uploader("Choose a Image ...", type="jpg")
if uploaded_file is not None:
    image = keras.utils.load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'Fruit.h5')
    st.write("This image likely belongs to {}".format(class_label[label]))
