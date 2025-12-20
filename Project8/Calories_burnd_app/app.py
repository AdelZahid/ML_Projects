import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Calorie Burn Predictor & Fitness Advisor",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .advice-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #f44336;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open('calorie_burnt_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    return model, scaler

# Calculate BMI
def calculate_bmi(weight, height):
    return weight / ((height/100) ** 2)

# Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor Equation
def calculate_bmr(weight, height, age, gender):
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:  # female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    return bmr

# Calculate recommended calorie burn based on user profile
def calculate_recommended_calories(bmr, activity_level, target):
    # Activity multipliers
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    
    # Target adjustments
    target_multipliers = {
        'maintain': 1.0,
        'lose_weight': 0.85,
        'gain_weight': 1.15
    }
    
    tdee = bmr * activity_multipliers.get(activity_level, 1.2)
    recommended_intake = tdee * target_multipliers.get(target, 1.0)
    
    # For weight loss, recommend burning 300-500 more calories
    if target == 'lose_weight':
        recommended_burn = 300 + (recommended_intake * 0.15)
    else:
        recommended_burn = recommended_intake * 0.2  # 20% of intake for maintenance/gain
    
    return recommended_burn, recommended_intake

# Generate fitness advice
def generate_fitness_advice(user_data, predicted_calories, recommended_calories):
    advice = []
    warnings = []
    
    # Calculate calorie deficit/surplus
    calorie_difference = recommended_calories - predicted_calories
    
    # Age-based advice
    age = user_data['Age']
    if age < 18:
        advice.append("🚸 **Youth Fitness**: Since you're under 18, focus on building healthy habits rather than intense calorie burning.")
    elif age > 50:
        advice.append("👴 **Senior Fitness**: Consider lower-impact exercises and focus on mobility and strength maintenance.")
    
    # BMI-based advice
    bmi = calculate_bmi(user_data['Weight'], user_data['Height'])
    if bmi < 18.5:
        warnings.append("⚠️ Your BMI indicates you're underweight. Consider consulting a healthcare provider before increasing exercise intensity.")
        advice.append("💪 **Strength Focus**: Consider adding strength training to build muscle mass rather than focusing solely on calorie burn.")
    elif bmi > 25:
        advice.append("🏃 **Weight Management**: Combine cardio with strength training for optimal results. Aim for 150-300 minutes of moderate exercise per week.")
    else:
        advice.append("🎯 **Maintenance Mode**: Your BMI is in the healthy range. Focus on maintaining your current fitness level.")
    
    # Body temperature advice
    if user_data['Body_Temp'] > 37.5:
        warnings.append("🌡️ Your body temperature is elevated. Ensure proper hydration and consider lighter exercise.")
    elif user_data['Body_Temp'] < 36.0:
        advice.append("🧣 Consider warming up more thoroughly before exercise to optimize performance.")
    
    # Heart rate advice
    if user_data['Heart_Rate'] > 100:
        warnings.append("💓 Your resting heart rate is high. Consider consulting a doctor if this is persistent.")
    
    # Duration advice based on predicted vs recommended
    if calorie_difference > 100:
        additional_minutes = (calorie_difference / predicted_calories) * user_data['Duration']
        advice.append(f"⏱️ **Increase Duration**: Try adding {int(additional_minutes)} minutes to your workout to reach your recommended calorie burn.")
    elif calorie_difference < -50:
        advice.append("🎉 **Great Job!**: You're burning more calories than recommended. Consider incorporating rest days for recovery.")
    
    # Exercise type recommendation
    if user_data['Duration'] > 60:
        advice.append("⏳ **Long Session Tips**: For workouts over 60 minutes, ensure adequate hydration and consider electrolyte replacement.")
    
    return advice, warnings

# Main application
def main():
    # Title
    st.markdown('<h1 class="main-header">🔥 Calorie Burn Predictor & Fitness Advisor</h1>', unsafe_allow_html=True)
    
    # Load model
    try:
        model, scaler = load_model()
        
        # Debug information (hidden by default)
        with st.sidebar:
            if st.checkbox("Show Debug Info"):
                st.write(f"Scaler expects: {scaler.n_features_in_} features")
                st.write(f"Model type: {type(model).__name__}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Input & Prediction", "📈 Analysis & Advice", "🎯 Recommendations"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<h3 class="sub-header">Personal Information</h3>', unsafe_allow_html=True)
            
            # User inputs
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            
            # Calculate and display BMI
            bmi = calculate_bmi(weight, height)
            st.metric("BMI", f"{bmi:.1f}", 
                    delta="Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese")
            
            # Activity level and goals
            st.markdown('<h3 class="sub-header">Fitness Goals</h3>', unsafe_allow_html=True)
            activity_level = st.selectbox(
                "Activity Level",
                ["sedentary", "light", "moderate", "active", "very_active"],
                help="Sedentary: Little or no exercise\nLight: Light exercise 1-3 days/week\nModerate: Moderate exercise 3-5 days/week\nActive: Hard exercise 6-7 days/week\nVery Active: Very hard exercise & physical job"
            )
            fitness_goal = st.selectbox(
                "Primary Goal",
                ["maintain", "lose_weight", "gain_weight"],
                help="Maintain: Keep current weight\nLose Weight: Reduce body weight\nGain Weight: Increase muscle mass"
            )
        
        with col2:
            st.markdown('<h3 class="sub-header">Exercise Session Details</h3>', unsafe_allow_html=True)
            
            # Exercise inputs - Only include the 6 features your model expects
            duration = st.slider("Duration (minutes)", min_value=1, max_value=300, value=30, 
                                help="Length of your exercise session")
            heart_rate = st.slider("Heart Rate (bpm)", min_value=40, max_value=200, value=120, 
                                  help="Average heart rate during exercise")
            body_temp = st.slider("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=36.5, step=0.1,
                                 help="Body temperature during/after exercise")
            
            # Display what features will be used
            st.markdown("---")
            st.info(f"**Model Information:** Using {scaler.n_features_in_} features for prediction")
            
            # Prediction button
            if st.button("🚀 Predict Calorie Burn", type="primary", use_container_width=True):
                # Prepare input data - ONLY 6 FEATURES based on your scaler
                # Based on typical calorie prediction models, these are likely:
                # 1. Duration
                # 2. Heart Rate
                # 3. Body Temperature
                # 4. Age
                # 5. Height
                # 6. Weight
                
                # Create a DataFrame with the correct number of features
                # IMPORTANT: The order of features must match the order used during training
                features_df = pd.DataFrame([[
        age,
        height,
        weight,
        duration,
        heart_rate,
        body_temp
    ]], columns=[
        'Age',
        'Height',
        'Weight',
        'Duration',
        'Heart_Rate',
        'Body_Temp'
    ])
                
                # Debug: Show what we're sending to the scaler
                with st.expander("View input features"):
                    st.write("Features being sent to model:")
                    st.dataframe(features_df)
                
                try:
                    # Scale the features
                    scaled_features = scaler.transform(features_df)
                    
                    # Make prediction
                    prediction = model.predict(scaled_features)[0]
                    
                    # Store in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['user_data'] = {
                        'Age': age,
                        'Height': height,
                        'Weight': weight,
                        'Duration': duration,
                        'Heart_Rate': heart_rate,
                        'Body_Temp': body_temp,
                        'Gender': gender
                    }
                    st.session_state['activity_level'] = activity_level
                    st.session_state['fitness_goal'] = fitness_goal
                    st.session_state['bmi'] = bmi
                    
                    # Display prediction
                    st.success(f"✅ Predicted Calorie Burn: **{prediction:.0f} calories**")
                    
                    # Show additional metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        calories_per_minute = prediction / duration
                        st.metric("Calories/Min", f"{calories_per_minute:.1f}")
                    with col_b:
                        st.metric("Duration", f"{duration} min")
                    with col_c:
                        intensity_level = "Light" if calories_per_minute < 5 else "Moderate" if calories_per_minute < 10 else "Vigorous"
                        st.metric("Intensity", intensity_level)
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("Please check that the input features match what the model was trained on.")
    
    with tab2:
        if 'prediction' in st.session_state:
            st.markdown('<h3 class="sub-header">Analysis & Personalized Advice</h3>', unsafe_allow_html=True)
            
            # Calculate BMR and recommended calories
            bmr = calculate_bmr(
                st.session_state['user_data']['Weight'],
                st.session_state['user_data']['Height'],
                st.session_state['user_data']['Age'],
                st.session_state['user_data']['Gender']
            )
            
            recommended_burn, recommended_intake = calculate_recommended_calories(
                bmr,
                st.session_state['activity_level'],
                st.session_state['fitness_goal']
            )
            
            # Generate advice
            advice, warnings = generate_fitness_advice(
                st.session_state['user_data'],
                st.session_state['prediction'],
                recommended_burn
            )
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Burn", f"{st.session_state['prediction']:.0f} cal")
            with col2:
                st.metric("Recommended Burn", f"{recommended_burn:.0f} cal")
            with col3:
                diff = st.session_state['prediction'] - recommended_burn
                st.metric("Difference", f"{diff:+.0f} cal", 
                        delta_color="inverse" if diff < 0 else "normal")
            
            # Additional metrics
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("BMR", f"{bmr:.0f} cal/day")
            with col5:
                st.metric("Daily Intake", f"{recommended_intake:.0f} cal")
            with col6:
                st.metric("BMI", f"{st.session_state['bmi']:.1f}")
            
            # Display warnings
            if warnings:
                st.markdown('<h4>⚠️ Important Notes</h4>', unsafe_allow_html=True)
                for warning in warnings:
                    st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)
            
            # Display advice
            st.markdown('<h4>💡 Personalized Advice</h4>', unsafe_allow_html=True)
            for item in advice:
                st.markdown(f'<div class="advice-box">{item}</div>', unsafe_allow_html=True)
            
            # Visualization
            st.markdown('<h4>📊 Burn Analysis</h4>', unsafe_allow_html=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Bar chart
            labels = ['Predicted', 'Recommended']
            values = [st.session_state['prediction'], recommended_burn]
            colors = ['#FF6B6B', '#4ECDC4']
            
            ax1.bar(labels, values, color=colors)
            ax1.set_ylabel('Calories')
            ax1.set_title('Calorie Burn Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax1.text(i, v + max(values)*0.02, f'{v:.0f}', ha='center', fontweight='bold')
            
            # Progress chart
            progress = min(st.session_state['prediction'] / recommended_burn, 1.0)
            remaining = max(0, 1.0 - progress)
            
            ax2.pie([progress, remaining], labels=['Achieved', 'Remaining'], 
                   autopct='%1.1f%%', colors=['#06D6A0', '#FFD166'])
            ax2.set_title('Goal Progress')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed breakdown
            st.markdown('<h4>📋 Session Details</h4>', unsafe_allow_html=True)
            col_details1, col_details2 = st.columns(2)
            
            with col_details1:
                st.write("**Input Parameters:**")
                for key, value in st.session_state['user_data'].items():
                    if key != 'Gender':
                        st.write(f"- {key}: {value}")
            
            with col_details2:
                st.write("**Calculated Metrics:**")
                st.write(f"- Calories per minute: {st.session_state['prediction']/st.session_state['user_data']['Duration']:.1f}")
                st.write(f"- Activity Level: {st.session_state['activity_level'].title()}")
                st.write(f"- Fitness Goal: {st.session_state['fitness_goal'].replace('_', ' ').title()}")
    
    with tab3:
        st.markdown('<h3 class="sub-header">Detailed Recommendations</h3>', unsafe_allow_html=True)
        
        if 'prediction' in st.session_state:
            # Calculate needed adjustments
            bmr = calculate_bmr(
                st.session_state['user_data']['Weight'],
                st.session_state['user_data']['Height'],
                st.session_state['user_data']['Age'],
                st.session_state['user_data']['Gender']
            )
            
            recommended_burn, recommended_intake = calculate_recommended_calories(
                bmr,
                st.session_state['activity_level'],
                st.session_state['fitness_goal']
            )
            
            calorie_gap = recommended_burn - st.session_state['prediction']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Duration recommendations
                st.markdown('<h4>⏱️ Duration Adjustments</h4>', unsafe_allow_html=True)
                
                current_duration = st.session_state['user_data']['Duration']
                current_prediction = st.session_state['prediction']
                
                if calorie_gap > 0:
                    # Calculate additional minutes needed
                    calories_per_minute = current_prediction / current_duration
                    additional_minutes_needed = calorie_gap / calories_per_minute
                    
                    st.write(f"**Current session:** {current_duration} minutes → {current_prediction:.0f} calories")
                    st.write(f"**Target:** {recommended_burn:.0f} calories")
                    st.write(f"**Calorie gap:** {calorie_gap:.0f} calories")
                    
                    st.info(f"""
                    **To reach your goal:**
                    - Add **{additional_minutes_needed:.0f} minutes** to your current session
                    - **OR** exercise for **{current_duration + additional_minutes_needed:.0f} minutes total**
                    - **OR** increase intensity by **{(calorie_gap/current_prediction)*100:.0f}%**
                    """)
                    
                    # Suggest specific exercises
                    st.markdown('<h4>💪 Exercise Suggestions</h4>', unsafe_allow_html=True)
                    
                    if st.session_state['bmi'] > 25:
                        st.write("**For weight loss focus on:**")
                        st.write("• 30 minutes of brisk walking")
                        st.write("• 20 minutes of cycling")
                        st.write("• 15 minutes of jump rope")
                    else:
                        st.write("**For fitness maintenance:**")
                        st.write("• 20 minutes of jogging")
                        st.write("• 15 minutes of swimming")
                        st.write("• 25 minutes of dancing")
                        
                else:
                    st.success("""
                    🎯 **Great job!** You're exceeding your calorie burn goal!
                    
                    **Consider:**
                    - Focus on exercise variety
                    - Incorporate strength training
                    - Add flexibility exercises
                    - Ensure adequate rest and recovery
                    """)
                
            with col2:
                st.markdown('<h4>📅 Weekly Plan</h4>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                weekly_plan = {
                    'Monday': 'Cardio (30 min)',
                    'Tuesday': 'Strength Training',
                    'Wednesday': 'Active Recovery',
                    'Thursday': 'Cardio (40 min)',
                    'Friday': 'Strength Training',
                    'Saturday': 'Mixed Workout',
                    'Sunday': 'Rest'
                }
                
                for day, activity in weekly_plan.items():
                    st.write(f"**{day}:** {activity}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Progress tracking
                st.markdown('<h4>🎯 Progress Tracking</h4>', unsafe_allow_html=True)
                st.write("Track these weekly metrics:")
                
                metrics_to_track = {
                    "Weight": st.checkbox("Weight", value=True),
                    "Workout Frequency": st.checkbox("Workout Frequency", value=True),
                    "Average Heart Rate": st.checkbox("Average Heart Rate"),
                    "Sleep Hours": st.checkbox("Sleep Hours"),
                    "Energy Levels": st.checkbox("Energy Levels", value=True)
                }
                
                if st.button("Save Tracking Preferences"):
                    st.session_state['tracking_metrics'] = [k for k, v in metrics_to_track.items() if v]
                    st.success("Tracking preferences saved!")
        
        else:
            st.info("👈 First make a prediction in the Input & Prediction tab to get personalized recommendations")

    # Sidebar with additional information
    with st.sidebar:
        st.markdown('<h3>ℹ️ About This Tool</h3>', unsafe_allow_html=True)
        st.write("""
        **Calorie Burn Predictor** uses machine learning to estimate calories burned based on:
        
        • Personal demographics (age, height, weight)
        • Exercise parameters (duration, intensity)
        • Physiological metrics (heart rate, body temp)
        
        **Note:** These are estimates. Consult professionals for medical advice.
        """)
        
        st.markdown("---")
        
        # Quick BMI Calculator
        st.markdown('<h3>⚡ Quick BMI Check</h3>', unsafe_allow_html=True)
        quick_weight = st.number_input("Your weight (kg)", min_value=30.0, max_value=200.0, value=70.0, key="quick_weight")
        quick_height = st.number_input("Your height (cm)", min_value=100.0, max_value=250.0, value=170.0, key="quick_height")
        
        if quick_weight and quick_height:
            quick_bmi = calculate_bmi(quick_weight, quick_height)
            st.metric("Your BMI", f"{quick_bmi:.1f}")
            
            if quick_bmi < 18.5:
                st.warning("Underweight")
            elif quick_bmi < 25:
                st.success("Normal weight")
            elif quick_bmi < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")
        
        st.markdown("---")
        st.markdown("### 📞 Need Help?")
        st.write("""
        If predictions seem inaccurate:
        1. Check your inputs are correct
        2. Ensure you're using the same units as during training
        3. Contact support if issues persist
        """)

if __name__ == "__main__":
    main()