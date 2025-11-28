import joblib
import streamlit as st
import pandas as pd 
import requests
import json
import time
import random
import altair as alt
# ---------- CONFIGURATION & SECRETS ----------

try:
    # IMPORTANT: Streamlit environment will handle the API key
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback for local testing if secrets.toml is not configured
    API_KEY = ""

# API Setup
GEMINI_MODEL_NAME = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={API_KEY}"   # <<< FIXED
MAX_RETRIES = 5
BASE_DELAY = 1.0

# FIX 2: Replaced local image path with a web-based placeholder URL
PLACEHOLDER_IMAGE_URL = "C:/Users/jiyad/OneDrive/Desktop/Salary project/Salary project/employee_image-removebg-preview (3).png"

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Salary Prediction AI",
    page_icon="💼",
    layout="wide"
)

# ---------- CACHING & DATA LOADING ----------
@st.cache_resource
def load_data():
    """Loads the ML model and data for dropdowns/context."""
    try:
        # NOTE: Ensure these files are in the same directory as app.py
        model = joblib.load('salary_pipeline.pkl')
        data = pd.read_csv('cleaned_salary.csv')
        return model, data
    except FileNotFoundError:
        st.error("⚠️ Required ML files (salary_pipeline.pkl or cleaned_salary.csv) not found.")
        return None, None
    except Exception as e:
        st.error(f"Failed to load ML assets. Error: {e}")
        return None, None

model, data = load_data()

# ---------- LLM API INTERACTION WITH EXPONENTIAL BACKOFF (FIXED LOGIC) ----------

def call_gemini_api(payload):
    """ Handles the API request to Gemini with exponential backoff for resilience."""
    if API_KEY == "NO_KEY":
        return None

    for attempt in range(MAX_RETRIES):
        try:
            # Calculate wait time with exponential backoff and jitter
            jitter = random.uniform(0, 1.0)
            wait_time = BASE_DELAY * (2 ** attempt) + jitter
            
            if attempt > 0:
                time.sleep(wait_time)
                
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=20
            )
            response.raise_for_status() 
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                # Silently retry on transient errors
                continue 
            else:
                error_message = f"Failed after {attempt + 1} attempts. Final error: {e}. Check if the API key is valid and has quota."
                if response.status_code == 400:
                    st.error(f"LLM Error (HTTP 400): {error_message} - API key or request format issue.")
                elif response.status_code == 429:
                    st.error(f"LLM Error (HTTP 429 - Rate Limit): {error_message} - You've hit the usage limit.")
                else:
                    st.error(error_message)
                return None
                
    return None

def get_llm_response(prompt, system_instruction):
    """Generates content from the Gemini API with search grounding."""
    
    if API_KEY == "NO_KEY":
        return "LLM Error: API Key is missing. Cannot provide analysis or chat functionality.", []
    
    payload = {
    "systemInstruction": {                                   # <<< FIXED
        "role": "system",
        "parts": [{"text": system_instruction}]
    },
    "contents": [
        {
            "role": "user",
            "parts": [{"text": prompt}]
        }
    ],
    "generationConfig": {
        "temperature": 0.5
    },
    "tools": [
        {"googleSearchRetrieval": {}}                        # <<< KEPT
    ]
    }

    response_data = call_gemini_api(payload)
    
    if response_data and response_data.get('candidates'):
        candidate = response_data['candidates'][0]
        if 'parts' in candidate['content'] and candidate['content']['parts'][0].get('text'):
            text = candidate['content']['parts'][0]['text']
            sources = candidate.get('groundingMetadata', {}).get('groundingAttributions', [])
            return text, sources
    
    return "Error: Could not get a response from the LLM. Please check your API key and connection.", []

# ---------- SESSION STATE INITIALIZATION ----------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant. Get a prediction first, then ask me anything about your career."}]
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_prediction_context_message' not in st.session_state:
    st.session_state.last_prediction_context_message = None
if 'new_prediction_made' not in st.session_state:
    st.session_state.new_prediction_made = True


# ---------- GLOBAL STYLING (User's original styles) ----------
st.markdown("""
<style>
/* Global & Sidebar */
[data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #fbeaff 0%, #ffffff 100%); color: #2c003e; font-family: 'Poppins', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(to bottom right, #5f2c82, #9c27b0, #ff9a9e); color: white; }
[data-testid="stSidebar"] * { color: black !important; font-weight: 500; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] label { color: white !important; font-weight: 700 !important; }

/* Form & Inputs */
.section { background: #ffffff; border-radius: 20px; padding: 28px 30px; box-shadow: 0px 6px 18px rgba(95, 44, 130, 0.15); margin-bottom: 25px; }
label { font-weight: 700 !important; color: #4a0072 !important; }
div.stButton > button { background: linear-gradient(to right, #9c27b0, #ff9a9e); color: white; border: none; border-radius: 10px; padding: 12px 35px; font-weight: 600; }
div.stButton > button:hover { transform: scale(1.05); }

/* Custom Chatbot Button Styling (For redirection) */
.redirect-button button { 
    background: linear-gradient(to right, #5f2c82, #9c27b0) !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px 35px;
    font-weight: 600;
    margin-top: 15px;
}
/* Metric & Insights */
.metric-container h2 { 
    color: white !important; 
    background: linear-gradient(45deg, #4a0072, #9c27b0); 
    padding: 20px; 
    border-radius: 15px; 
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.insight-box { background-color: #f3e5f5; border-left: 5px solid #9c27b0; padding: 15px; border-radius: 10px; margin-top: 15px; }

/* Chat Bubbles (For better response formatting) */
[data-testid="stChatMessage"] {
    background-color: #fff;
    padding: 10px 15px;
    border-radius: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stChatMessage"]:has(div.stMarkdown > p:first-child:contains("Initial Prediction Analysis")) {
    background-color: #e6e6fa; /* Light purple for initial analysis */
    border-left: 5px solid #4a0072;
}
</style>
""", unsafe_allow_html=True)


# ---------- UI FUNCTIONS ----------
def salary_prediction_ui():
    st.markdown("<h1>💼 Salary Prediction</h1>", unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Cannot run prediction. Please check model file status.")
        return
    
    st.markdown("### Fill the input details below to predict the expected salary.")

    left, right = st.columns([2, 1])

    with left:
        # Use st.form for a clean submission process
        with st.form("salary_prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.subheader("👤 Personal Details")
                emp_name = st.text_input("Name")
                age = st.number_input("Age", min_value=18, max_value=80, value=25)
                gender = st.selectbox("Gender", sorted(data['Gender'].unique()))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.subheader("🎓 Education Details")
                education = st.selectbox("Education Level", sorted(data["Education Level"].unique()))
                st.markdown("</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.subheader("💼 Job Details")
                job_options = sorted(data["Job Title"].unique().tolist())
                job_options.append("Other (Type Manually)")
                job_title = st.selectbox("Job Title", job_options)
                
                custom_job_title = st.text_input("Enter Job Title") if job_title == "Other (Type Manually)" else None
                final_job_title = custom_job_title if custom_job_title else (job_title if job_title != "Other (Type Manually)" else "Software Engineer")
                
                experience_year = st.number_input("Years of Experience", 0.0, 50.0, 2.0, step=0.5)
                st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("🔮 Predict Salary", type="primary")

    with right:
        st.image(PLACEHOLDER_IMAGE_URL, use_container_width=True, caption="AI Career Coach Analysis")

    if submitted:
        input_data = pd.DataFrame({
            'Age': [age], 'Gender': [gender], 'Education Level': [education],
            'Job Title': [final_job_title], 'Years of Experience': [experience_year]
        })

        with st.spinner("Predicting salary and generating initial analysis..."):
            try:
                pred_usd = model.predict(input_data)[0]
                formatted_salary = f"{pred_usd:,.2f}"
                
                # FIX 5: Save Context for Chatbot
                st.session_state['last_prediction'] = {
                    "Role": final_job_title,
                    "Experience": f"{experience_year} years",
                    "Education": education,
                    "Predicted Salary": f"${pred_usd:,.2f}"
                }
                
                # Show Result
                left.markdown(f"""
                <div class="metric-container">
                    <h2>💰 Estimated Package: ${formatted_salary} USD</h2>
                </div>
                """, unsafe_allow_html=True)

                # Generate LLM Analysis
                prediction_prompt = (
                    f"A user with the following profile received a predicted salary: "
                    f"Role: '{final_job_title}', Education: '{education}', Experience: {experience_year} years. "
                    f"Predicted Salary: ${formatted_salary} USD. "
                    "Provide a one-paragraph analysis of this salary's market alignment (use search grounding) and suggest one actionable next step for career growth."
                )
                llm_system_prompt = "You are an expert HR Analyst and Career Coach. Be professional and encouraging."

                analysis, sources = get_llm_response(prediction_prompt, llm_system_prompt)
                
                # Store analysis message to auto-populate chat history
                analysis_message = f"**Initial Prediction Analysis for {final_job_title}**\n\n{analysis}"
                st.session_state.new_prediction_made = True
                
                left.markdown("<hr>", unsafe_allow_html=True)
                left.info("### 💡 AI Career Coach Analysis")
                left.write(analysis)
                
                if sources:
                    left.caption("🔍 **Grounded Sources:**")
                    for s in sources:
                        left.caption(f"- [{s['title']}]({s['uri']})")
                
                left.markdown("<div class='redirect-button'>", unsafe_allow_html=True)
                if left.button("🚀 Continue to AI Chatbot for Career Advice", key="go_to_chat", use_container_width=True):
                    st.session_state.messages.append({"role": "assistant", "content": analysis_message})
                    st.switch_page("app.py") # Use switch_page if you have a multi-page app or re-run with state change
                    # Since the chatbot_ui is on the same page, we update the state and prompt the user
                    st.session_state.current_page = "AI Chatbot"
                    st.rerun()
                left.markdown("</div>", unsafe_allow_html=True)
                # Rerun to automatically update the chat history with the analysis
                # st.rerun()

            except Exception as e:
                st.error(f"Prediction or LLM Error: {e}")
                st.session_state.last_prediction = None # Clear context on failure


def chatbot_ui():
    st.markdown("<h1>🤖 AI Career Coach Chatbot</h1>", unsafe_allow_html=True)
    st.info("Ask about market trends, salary comparisons, negotiation tips, or career advice based on your last prediction.")

    if st.session_state.new_prediction_made and st.session_state.last_prediction:
        # Get the analysis content that was saved during the prediction phase
        analysis_content = next((msg['content'] for msg in st.session_state.messages if msg['content'].startswith('**Initial Prediction Analysis')), None)
        
        if analysis_content:
            st.session_state.messages.append({"role": "assistant", "content": analysis_content})
        
        st.session_state.new_prediction_made = False # Reset flag

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # FIX 6: Build Contextual Prompt for Chatbot
        context_string = ""
        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            context_string = (
                f"The user's last profile was: Role: {p['Role']}, Experience: {p['Experience']}, "
                f"Education: {p['Education']}. Predicted Salary: {p['Predicted Salary']}. "
                "Use this profile to give personalized, actionable career advice and salary comparisons."
            )
        else:
            context_string = "No salary prediction context is available. Answer generally about job market trends and career development."
        
        chat_system_prompt = (
            "You are an expert AI Career Coach, specializing in salary negotiation and job market trends. "
            "Always use the Google Search tool to ensure your answers are grounded in the latest market data. "
            "Be empathetic, professional, and provide clear next steps. "
            f"**Current User Context:** {context_string}"
        )
        
        with st.spinner("AI Coach is formulating a response..."):
            ai_response, sources = get_llm_response(prompt, chat_system_prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.chat_message("assistant").write(ai_response)
            
            if sources:
                st.caption("🔍 **Grounded Sources:**")
                for s in sources:
                    st.caption(f"- [{s['title']}]({s['uri']})")

def data_analysis_ui():
    """NEW PAGE: Displays charts and analysis of the training data."""
    st.markdown("<h1>📊 ML Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("#### **Project Component: ML Engineering (Transparency & Insights)**")
    st.info("These charts visualize the distribution and relationships in the training data used to build the salary prediction model.")

    if data is None:
        st.error("Cannot load analysis data. Please check 'cleaned_salary.csv'.")
        return

    # 1. Salary Distribution (Histogram)
    st.markdown("---")
    st.subheader("1. Distribution of Salaries")
    chart1 = alt.Chart(data).mark_bar(color='#9c27b0', opacity=0.8).encode(
        x=alt.X('Salary', bin=alt.Bin(maxbins=30), title='Annual Salary (USD)'), # Adjusted bins for better distribution view
        y=alt.Y('count()', title='Number of Employees'),
        tooltip=['Salary', 'count()']
    ).properties(
        title='Salary Distribution Across the Dataset'
    ).interactive()
    st.altair_chart(chart1, use_container_width=True)

    # 2. Salary vs. Years of Experience (Scatter Plot/Regression Line)
    st.markdown("---")
    st.subheader("2. Salary vs. Years of Experience")
    
    # Scatter plot
    scatter = alt.Chart(data).mark_circle(size=60, color='#ff9a9e').encode(
        x=alt.X('Years of Experience', title='Years of Experience'),
        y=alt.Y('Salary', title='Salary (USD)'),
        tooltip=['Years of Experience', 'Salary', 'Job Title']
    )
    
    # Regression line
    line = scatter.transform_regression('Years of Experience', 'Salary').mark_line(color='#5f2c82', strokeWidth=3)

    chart2 = (scatter + line).properties(
        title='Experience vs. Salary Trend'
    ).interactive()
    st.altair_chart(chart2, use_container_width=True)

    # 3. Average Salary by Education Level (Bar Chart)
    st.markdown("---")
    st.subheader("3. Average Salary by Education Level")
    
    # Calculate means for accurate sorting and display
    avg_salary_data = data.groupby('Education Level')['Salary'].mean().reset_index(name='Average Salary')
    
    chart3 = alt.Chart(avg_salary_data).mark_bar(color='#5f2c82').encode(
        x=alt.X('Average Salary', title='Average Salary (USD)'),
        y=alt.Y('Education Level', title='Education Level', sort='-x'), # Sort by salary descending
        tooltip=[alt.Tooltip('Education Level'), alt.Tooltip('Average Salary', format='$,.0f')]
    ).properties(
        title='Average Salary by Education Level'
    ).interactive()
    st.altair_chart(chart3, use_container_width=True)
    
# ---------- MAIN (Simplified Navigation) ----------
def main():
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Salary Prediction"
        
    st.sidebar.markdown("<h2 style='text-align:center;'>🧭 Navigation</h2>", unsafe_allow_html=True)
    
    # Use the session state variable for the selectbox
    st.session_state.current_page = st.sidebar.selectbox(
        "Choose Page", 
        ["Salary Prediction", "AI Chatbot", "Data Analysis"],
        index=["Salary Prediction", "AI Chatbot", "Data Analysis"].index(st.session_state.current_page)
    )

    # Sidebar button for clearing history
    if st.sidebar.button("🗑️ Clear Prediction & Chat History", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant. Get a prediction first, then ask me anything about your career."}]
        st.session_state.last_prediction = None
        st.session_state.new_prediction_made = False
        st.rerun()

    if st.session_state.current_page == "Salary Prediction":
        salary_prediction_ui()
    elif st.session_state.current_page == "AI Chatbot":
        chatbot_ui()
    elif st.session_state.current_page == "Data Analysis":
        data_analysis_ui()


if __name__ == "__main__":
    main()