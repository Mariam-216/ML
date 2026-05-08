import streamlit as st
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(bin_file):
    bin_str = get_base64_of_bin_file(bin_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    set_png_as_page_bg('bgg.png') 
except FileNotFoundError:
    st.error("لم يتم العثور على ملف الصورة. تأكد من وضعها في نفس مجلد الكود.")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
<style>
/* Dashboard*/
[data-testid="stSidebar"] h1 {
    color: #C9CDCD !important; 
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploader"] label {
    color: #36454F !important;
    font-family: "Times New Roman", serif !important;
    font-size: 40px !important;
    font-weight: 700 !important;
}
            h1, h2, h3, h4 {
        font-family: 'Times New Roman', Times, serif !important;
        color: #FAF9F6 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6) !important; 
    }
            
[data-testid="stSelectbox"] div {
    color: #C9CDCD !important;
    font-weight: bold !important;
}
[data-testid="stWidgetLabel"] p {
    color: #36454F !important; 
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 18px !important;
    font-weight: bold !important;
}
ul[role="listbox"] {
    background-color: skyblue !important;
}
ul[role="listbox"] li {
    color: #ffffff !important;
}
ul[role="listbox"] li:hover {
    background-color: #36454F !important;
    color: #36454F !important;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="ML", layout="wide")

if "data" not in st.session_state:
    st.session_state["data"] = None
if "data_processed" not in st.session_state:
    st.session_state["data_processed"] = None

pages_dict = {
    "Uploading": st.Page("upload.py", title="Uploading", default=True),
    "Preprocessing": st.Page("preprocess.py", title="Preprocessing"),
    "Visualization": st.Page("visual.py", title="Visualization"),
    "Model Selection": st.Page("models.py", title="Model Selection"),
    "Evaluation": st.Page("evaluation.py", title="Evaluation")
}

st.sidebar.title("Dashboard")
selection = st.sidebar.selectbox("Go to:", list(pages_dict.keys()))
pg = st.navigation([pages_dict[selection]], position="hidden")
pg.run()