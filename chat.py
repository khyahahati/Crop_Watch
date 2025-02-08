import streamlit as st
import os
import shelve
import pandas as pd
import tensorflow as tf
from tf_sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import tempfile
import numpy as np

# Cached loading for both models
@st.cache_resource
def load_image_model():
    return tf.keras.models.load_model(r"/Users/aniketarora/Downloads/RESUME PROJECTS/PlantDisease/my_fmodel.keras")

@st.cache_resource
def load_embedder():
    MODEL_DIR = "/Users/aniketarora/Downloads/RESUME PROJECTS/PlantDisease/mod/universal-sentence-encoder-tensorflow2-universal-sentence-encoder-v2/"
    return tf.saved_model.load(MODEL_DIR)

model_d = load_image_model()
embed = load_embedder()

st.title("Crop Watch Chat")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Helper functions for managing chat history
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Load datasets (cached since they don't change during session)
@st.cache_resource
def load_datasets():
    questions_df = pd.read_csv('questions.csv')
    answers_df = pd.read_csv('answer.csv')
    questions = questions_df['question'].tolist()
    return questions, answers_df

questions, answers_df = load_datasets()

# Image Prediction
def model_prediction(test_image):
    print("Predicting for test image")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model_d.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Get sentence embeddings
def get_embeddings(text_list):
    return embed(text_list)

# Identify the most similar question
def get_most_similar_question_index(user_input, questions):
    input_embedding = get_embeddings([user_input])
    question_embeddings = get_embeddings(questions)
    similarities = cosine_similarity(input_embedding, question_embeddings)
    return similarities.argmax()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Sidebar with image uploader
with st.sidebar:
    if st.button("Delete Chat History"):
        print("Chat Cleared!")
        st.session_state.messages = []
        save_chat_history([])
    
    test_image = st.file_uploader("Upload Plant Image", type=["png", "jpg", "jpeg"])
    
    if test_image:
        temp_dir = tempfile.mkdtemp()
        test_image_path = os.path.join(temp_dir, test_image.name)
        with open(test_image_path, "wb") as f:
            f.write(test_image.getvalue())

        # Update the session state with the new image path
        st.session_state["test_image_path"] = test_image_path
        st.image(test_image, use_column_width=True)
            
        
    # Prediction only done once per session unless reset
    if st.button("Predict") and "test_image_path" in st.session_state:
        print("Sending prediction call to function")
        result_index = model_prediction(st.session_state["test_image_path"])
        
        class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
            'Potato___Late_blight',
            'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
            'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
        ]
        print(f"Result index: ",result_index,len(class_names))
        class_var = class_names[result_index]
        print("predicted og: ", class_var)

        st.session_state["class_var"] = class_var
        st.success(f"Model predicts: {class_var}")
        st.session_state.messages.append({"role": "bot", "content": f"Ask me anything about {class_var}"})

        print("predicted sess state: ", st.session_state.class_var)
        

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interaction
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    class_var = st.session_state.get("class_var", None)

    print(class_var)
    # Retrieve bot's response
    if "class_var" in st.session_state and st.session_state["class_var"] is not None:
        index = get_most_similar_question_index(prompt, questions)
        print("Most similar ques index: ",index)
        answer_row = answers_df[(answers_df['class'] == class_var) & (answers_df['question_index'] == index)]
        print("Answer: ",answer_row)

        answer = answer_row['answer'].values[0] if not answer_row.empty else "No matching answer found for this question and class."
    
    else:
        answer = "Please upload and predict an image first."

    # Display bot's response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(answer)
    
    # Save response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Save chat history at the end of interaction
save_chat_history(st.session_state.messages)
