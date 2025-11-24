import joblib
import streamlit as st
import pandas as pd 
# import google.generativeai as genai
import os
from google import genai

# # load the dataset globally for rule-based chatbot
# def load_data():
#     df=pd.read_csv('salary_prediction_data.csv')
#     return df

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Salary Prediction AI",
    page_icon="💼",
    layout="wide"
)
api_key = os.getenv("GEMINI_API_KEY")
@st.cache_resource
def load_data():
    try:
        model = joblib.load('salary_pipeline.pkl')
        data = pd.read_csv('cleaned_salary.csv')
        return model,data
    except Exception as e:
        return None,None
# ---------- GLOBAL STYLING ----------
st.markdown("""
<style>
/* ---------- GLOBAL PAGE STYLING ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fbeaff 0%, #ffffff 100%);
    color: #2c003e;
    font-family: 'Poppins', sans-serif;
}

/* ---------- SIDEBAR STYLING ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom right, #5f2c82, #9c27b0, #ff9a9e);
    padding-top: 1rem;
    color: white;
    border-right: 2px solid rgba(255,255,255,0.2);
}

[data-testid="stSidebar"] * {
    color: black !important;
    font-weight: 500;
}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

/* ---------- FORM SECTIONS ---------- */
.section {
    background: #ffffff;
    border-radius: 20px;
    padding: 28px 30px;
    box-shadow: 0px 6px 18px rgba(95, 44, 130, 0.15);
    margin-bottom: 25px;
    transition: transform 0.2s ease, box-shadow 0.3s ease;
}
.section:hover {
    transform: translateY(-3px);
    box-shadow: 0px 8px 20px rgba(156, 39, 176, 0.25);
}

/* ---------- LABELS & INPUT FIELDS ---------- */
label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    font-weight: 700 !important;
    font-size: 16px !important;
    color: #4a0072 !important;
    margin-bottom: 6px !important;
    display: block !important;
}

input, textarea, select, div[data-baseweb="input"] input {
    border: 2px solid #9c27b0 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
    color: #333 !important;
    transition: all 0.2s ease-in-out;
}
input:focus, select:focus {
    outline: none !important; 
    border-color: #7b1fa2 !important;
    box-shadow: 0px 0px 5px rgba(155, 39, 176, 0.4);
}

/* ---------- BUTTON ---------- */
div.stButton > button {
    background: linear-gradient(to right, #9c27b0, #ff9a9e);
    color: white;
    padding: 12px 35px;
    border-radius: 10px;
    border: none;
    font-size: 17px;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
    letter-spacing: 0.5px;
}
div.stButton > button:hover {
    background: linear-gradient(to right, #7b1fa2, #ff8a80);
    transform: scale(1.05);
}

/* ---------- HEADINGS ---------- */
h1, h2, h3 {
    color: #5f2c82;
    text-align: center;
    font-weight: 800;
}
h2 {
    border-bottom: 3px solid #d1a7ff;
    padding-bottom: 8px;
    display: inline-block;
}

/* ---------- SUCCESS & ERROR BOX ---------- */
.stSuccess {
    background-color: #f3e5f5 !important;
    color: #4a0072 !important;
    border-left: 6px solid #9c27b0 !important;
    font-weight: 600 !important;
}
.stError {
    background-color: #ffebee !important;
    color: #b71c1c !important;
    border-left: 6px solid #f44336 !important;
}

/* ---------- IMAGE ALIGNMENT ---------- */
[data-testid="stImage"] img {
    border-radius: 15px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.1);
}
/* Fix for select box arcs / Streamlit dropdown corners */
div[data-baseweb="select"] > div {
    border: 2px solid #9c27b0 !important;
    border-radius: 10px !important;
    background-color: #fff !important;
    box-shadow: none !important;
}

/* Remove extra inner arcs caused by nested divs */
div[data-baseweb="select"] > div > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Dropdown text and caret alignment */
div[data-baseweb="select"] span {
    color: #4a0072 !important;
    font-weight: 600 !important;
}

/* Hover and focus visual feedback */
div[data-baseweb="select"]:hover > div {
    border-color: #7b1fa2 !important;
}
div[data-baseweb="select"]:focus-within > div {
    border-color: #7b1fa2 !important;
    box-shadow: 0px 0px 5px rgba(155, 39, 176, 0.4);
}

</style>
""", unsafe_allow_html=True)

# data = load_data()
model,data = load_data()

# def convert_to_lpa(usd_amount):
#     """ Converts the USD into Lpa"""
#     inr_amount = usd_amount*USD_TO_INR
#     lpa = inr_amount/100000
#     return lpa
# ---------- APP SECTIONS ----------
def salary_prediction_ui():
    st.markdown("<h1>💼 Salary Prediction</h1>", unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model files not found! Please run 'train_model.py' first.")
        return
    
    st.markdown("### Fill the input details below to predict the expected salary.")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("👤 Personal Details")
        emp_name = st.text_input("Employee Name")
        age= st.number_input("Age", min_value=0, max_value=80, value=25)
        gender = st.selectbox("Gender",sorted(data['Gender'].unique()))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("🎓 Education Details")
        education = st.selectbox("Education", sorted(data["Education Level"].unique()))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("💼 Job Details")
        job_options = sorted(data["Job Title"].unique().tolist())
        job_options.append("Other(Type Manually)")
        job_title= st.selectbox("Job Title", job_options)
        custom_job_title = None
        if job_title == "Other (Type Manually)":
            custom_job_title = st.text_input("Enter your Job Title")
        
        final_job_title = custom_job_title if job_title == "Other (Type Manually)" and custom_job_title else job_title
        if job_title == "Other (Type Manually)" and not custom_job_title:
             final_job_title = "Software Engineer"
        experience_year = st.number_input("Years of Experience",0.0,50.0,2.0,step=0.5)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Predicting..."):
            if st.button("🔮 Predict Salary"):
                input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education],
            'Job Title': [final_job_title],
            'Years of Experience': [experience_year]
        })

                try:
                    pred_usd = model.predict(input_data)[0]
            
            
            # Show Result
                    st.markdown(f"""
                    <div class="metric-container">
                    <h2 style="color: #2e7d32;">💰 Estimated Package:${pred_usd:,.2f} USD)</h2>
                    </div>
                    """, unsafe_allow_html=True)

            # Stakeholder Insight: Student ROI
                    if education == "Bachelor's Degree":
                        input_data['Education Level'] = "Master's Degree"
                        masters_pred_usd = model.predict(input_data)[0]
                        diff_lpa = masters_pred_usd - pred_usd
                
                        if diff_lpa > 0:
                            st.markdown(f"""
                            <br>
                            <div class="insight-box">
                                <b>🎓 Career Insight:</b><br>
                                With a <b>Master's Degree</b>, this role could potentially pay 
                                <b>${masters_pred_usd:.2f} </b> 
                                (an increase of <b>${diff_lpa:.2f} </b>).
                            </div>
                            """, unsafe_allow_html=True)
            
                    if job_title == "Other (Type Manually)":
                        st.info(f"ℹ️ **Note:** Since '{final_job_title}' wasn't in our original training list, this prediction is based on general market trends for your experience and education level.")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    with right:
        st.image(
            "C:/Users/jiyad/OneDrive/Desktop/Salary project/Salary project/employee_image-removebg-preview (3).png",
            use_container_width=True
        )
def get_ai_response(user_query, data_context):
    """
    Placeholder for AI Integration (e.g., OpenAI or Gemini).
    If you have an API Key, uncomment the code below.
    """
    # --- UNCOMMENT & ADD YOUR KEY TO ENABLE REAL AI ---
    # import google.generativeai as genai
    full_prompt = f"Context: {data_context}\nUser Question: {user_query}\nAnswer:"
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # Use a current model version
        contents=full_prompt
    )
    # response = model.generate_content(full_prompt)
    return response.text
    
    # --- DEFAULT RULE-BASED FALLBACK ---
    return None

def chatbot_ui():
    st.markdown("<h1>🤖 Chatbot Assistant</h1>", unsafe_allow_html=True)
    st.info("Ask about market trends, salary comparisons, or career advice.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant. Ask me anything about salaries!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # 1. Try getting a real AI response
        # Create a context string with summary stats to feed the AI
        context = ""
        if data is not None:
            context = f"""{data}"""
        
        ai_response = get_ai_response(prompt, context)

        
        if ai_response:
            response = ai_response
        

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
# ---------- MAIN ----------
def main():
    st.sidebar.markdown("<h2 style='text-align:center;'>🧭 Navigation</h2>", unsafe_allow_html=True)
    menu = st.sidebar.selectbox("Choose Page", ["Salary Prediction", "AI Chatbot"])

    if menu == "Salary Prediction":
        salary_prediction_ui()
    elif menu == "AI Chatbot":
        chatbot_ui()


if __name__ == "__main__":
    main()