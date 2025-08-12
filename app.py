from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session management

# Load trained ML model + scaler
model_path = r"Models"
with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(model_path, 'mindset_model.pkl'), 'rb') as f:
    model_mindset = pickle.load(f)

# Load the dataset
CSV_FILE = "FY25_Candidates database.csv"


def get_next_s_no():
    """Get the next available serial number"""
    if os.path.exists(CSV_FILE):
        try:
            existing_df = pd.read_csv(CSV_FILE)
            if not existing_df.empty and 'S.No' in existing_df.columns:
                return int(existing_df['S.No'].max()) + 1
        except Exception as e:
            print(f"Error reading existing file: {e}")
    return 1

def safe_write_to_csv(data):
    """Safely write data to CSV with proper error handling"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # Write to CSV
        if os.path.exists(CSV_FILE):
            # Read existing data to check for duplicates
            existing_df = pd.read_csv(CSV_FILE)
            if not existing_df.empty and 'Primary Email ID (College)' in existing_df.columns:
                if data['Primary Email ID (College)'] in existing_df['Primary Email ID (College)'].values:
                    return False, "Email already exists in database"
            
            # Append new data
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_FILE, index=False)
        return True, "Data submitted successfully!"
    except PermissionError:
        print(f"Permission denied for file: {CSV_FILE}")
        return False, "Permission error - could not write to file"
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        return False, f"Error writing data: {str(e)}"

def unified_student_search(search_terms):
    """
    Search students who match ALL terms across skills/sports/extracurriculars 
    
    Parameters:
    - search_terms (str): Comma-separated terms (e.g. "python,hackathon,football")
    
    Returns:
    - List of dictionaries containing matching students' data
    """
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    # Clean data
    df_clean = df.copy()
    df_clean['Technonlogies/Skills known'] = df_clean['Technonlogies/Skills known'].fillna('').str.lower()
    df_clean['Sports'] = df_clean['Sports'].fillna('no').str.lower()
    df_clean['ExtraCurriculum'] = df_clean['ExtraCurriculum'].fillna('no').str.lower()
    
    # Fill NaN values for points columns with 0
    for col in ['Tech_points', 'Sports_Points', 'ExxCur_Points']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # Process search terms
    terms = [term.strip().lower() for term in search_terms.split(",") if term.strip()]
    
    if not terms:
        return []
    
    # Filter students where ALL terms appear in any of the three columns
    def matches_all_terms(row):
        combined_text = f"{row['Technonlogies/Skills known']} {row['Sports']} {row['ExtraCurriculum']}"
        return all(term in combined_text for term in terms)
    
    results = df_clean[df_clean.apply(matches_all_terms, axis=1)]
    
    # Convert to list of dicts with highlighted matches
    students_list = []
    for _, student in results.iterrows():
        student_dict = {
            'Name': student.get('Candidate Name', ''),
            'CGPA': student.get('Graduation CGPA', ''),
            'Email': student.get('Primary Email ID (College)', ''),
            'Skills': student.get('Technonlogies/Skills known', ''),
            'Sports': student.get('Sports', ''),
            'Extracurriculars': student.get('ExtraCurriculum', ''),
            'Tech_points': student.get('Tech_points', 0),
            'Sports_Points': student.get('Sports_Points', 0),
            'ExxCur_Points': student.get('ExxCur_Points', 0),
            'Matches': {}
        }
        
        # Track which terms were found in which columns
        for term in terms:
            student_dict['Matches'][term] = {
                'Skills': term in student.get('Technonlogies/Skills known', ''),
                'Sports': term in student.get('Sports', ''),
                'Extracurriculars': term in student.get('ExtraCurriculum', '')
            }
        
        students_list.append(student_dict)
    
    return students_list

def generate_booster_tips(student):
    tips = []
    
    try:
        cgpa = float(student.get('CGPA', 0))
        if cgpa < 6.5:
            tips.append("Work on improving your CGPA to enhance academic strength.")
    except (ValueError, TypeError):
        tips.append("CGPA is missing or invalid.")

    if not student.get('Skills') or str(student['Skills']).strip().lower() in ['', 'no', 'none']:
        tips.append("Consider learning technical skills like Python, SQL, or Web Development.")

    if not student.get('Sports') or str(student['Sports']).strip().lower() in ['', 'no', 'none']:
        tips.append("Engaging in sports builds team spirit and leadership skills.")

    if not student.get('Extracurriculars') or str(student['Extracurriculars']).strip().lower() in ['', 'no', 'none']:
        tips.append("Participating in extracurriculars shows you're a well-rounded individual.")

    try:
        if float(student.get('Total Experience', 0)) == 0:
            tips.append("Try doing internships or real-world projects to build practical experience.")
    except (ValueError, TypeError):
        tips.append("Could not assess your experience level.")

    return tips

# Login routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # In production, use proper password hashing and database lookup
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Main routes
@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('recommend.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        # Calculate points for all list-based fields
        technologies_list = [tech.strip() for tech in request.form['technologies_skills'].split(',') if tech.strip()]
        tech_points = len(technologies_list) * 5
        
        sports_list = [sport.strip() for sport in request.form.get('sports', '').split(',') if sport.strip()]
        sports_points = len(sports_list) * 7
        
        extra_curriculum_list = [item.strip() for item in request.form.get('extra_curriculum', '').split(',') if item.strip()]
        extra_curriculum_points = len(extra_curriculum_list) * 8

        data = {
            'S.No': get_next_s_no(),
            'College Full Name': request.form['college_full_name'],
            'College City': request.form['college_city'],
            'Roll No / PRN': request.form['roll_no'],
            'Prefix': request.form['prefix'],
            'Candidate Name': request.form['candidate_name'],
            'Gender': request.form['gender'],
            'Candidate Mobile Number': request.form['candidate_mobile_number'],
            'Alternate Mobile Number': request.form.get('alternate_mobile_number', ''),
            'Primary Email ID (College)': request.form['primary_email'],
            'Alternate Email ID': request.form.get('alternate_email', ''),
            '10th Board%': request.form['tenth_board_percentage'],
            '12th Board%': request.form['twelfth_board_percentage'],
            'Graduation Degree': request.form['graduation_degree'],
            'Grad-Specialization': request.form['grad_specialization'],
            'Graduation CGPA': request.form['graduation_cgpa'],
            'Year of Graduation': request.form['year_of_graduation'],
            'Post Graduation Degree': request.form.get('post_graduation_degree', ''),
            'Post Grad- Specialization': request.form.get('post_grad_specialization', ''),
            'Post Graduation CGPA': request.form.get('post_graduation_cgpa', ''),
            'Year of Post Grad': request.form.get('year_of_post_grad', ''),
            'Foreign Language (Except English)': request.form.get('foreign_language', ''),
            'Proficiency in foreign language (Beginner / Advanced / Mastery)': request.form.get('foreign_language_proficiency', ''),
            'Permanent Home Address (Not Campus or Hostel Address)': request.form['permanent_address'],
            'Permanent City': request.form['permanent_city'],
            'Permanent State': request.form['permanent_state'],
            'Permanent Pin Code': request.form['permanent_pin_code'],
            'Technonlogies/Skills known': request.form['technologies_skills'],
            'Tech_points': tech_points,
            'Organization worked with (If Any) Eg: Accenture/KPMG': request.form.get('organization_worked_with', ''),
            'Prior Experienced (Brief Summary of the work)': request.form.get('prior_experience', ''),
            'Total Experienced (In Years)': request.form.get('total_experience', 0),
            'Legal pursuit': request.form.get('legal_pursuit', ''),
            'Sports': request.form.get('sports', ''),
            'Sports_Points': sports_points,
            'ExtraCurriculum': request.form.get('extra_curriculum', ''),
            'ExxCur_Points': extra_curriculum_points
        }

        # Attempt to write with retry logic
        success, message = safe_write_to_csv(data)
        if success:
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"status": "error", "message": message}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/skill_recommend', methods=['GET', 'POST'])
def skill_recommend():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        search_input = request.form.get('search_input', '')
        if not search_input:
            return render_template('skill_recommend.html', error="Please enter search terms")
        
        students = unified_student_search(search_input)
        for student in students:
            student['BoosterTips'] = generate_booster_tips(student)
        
        return render_template('skill_result.html',
                           students=students,
                           search_terms=search_input.split(","))
    
    return render_template('skill_recommend.html')

@app.route('/students')
def view_students():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    try:
        df = pd.read_csv(CSV_FILE)
        students = df.to_dict('records')
        return render_template('students.html', students=students)
    except Exception as e:
        return render_template('students.html', error=f"Error reading data: {str(e)}")

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'user' not in session:
#         return redirect(url_for('login'))
    
#     try:
#         # Load models
#         model_path = os.path.join('models')
#         with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
#             scaler = pickle.load(f)
#         with open(os.path.join(model_path, 'model_mindset.pkl'), 'rb') as f:
#             model_mindset = pickle.load(f)
#     except Exception as e:
#         print(f"Error loading models: {e}")
#         return render_template('predict_result.html', error="Prediction service unavailable")

#     try:
#         cgpa = float(request.form['graduation_cgpa'])
#         year_grad = int(request.form['year_of_graduation'])
#         experience = float(request.form.get('total_experience', 0))
#         skills = request.form.get('technologies_skills', '')
#         skill_count = len(skills.split(',')) if skills.strip() else 0

#         lang_level = request.form.get('foreign_language_proficiency', 'Beginner')
#         lang_map = {'Beginner': 0, 'Advanced': 1, 'Mastery': 2}
#         lang_encoded = lang_map.get(lang_level, 0)

#         features = [[cgpa, year_grad, experience, skill_count, lang_encoded]]
#         scaled_features = scaler.transform(features)
#         prediction = model_mindset.predict(scaled_features)[0]

#         return render_template('predict_result.html', mindset=prediction)
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return render_template('predict_result.html', error="Invalid input format")

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        # Get form values
        cgpa = float(request.form['graduation_cgpa'])
        year_grad = int(request.form['year_of_graduation'])
        experience = float(request.form['total_experience'])
        skills = request.form['technologies_skills']
        skill_count = len(skills.split(',')) if skills.strip() else 0
        lang_level = request.form.get('foreign_language_proficiency', 'Beginner')
        lang_map = {'Beginner': 0, 'Advanced': 1, 'Mastery': 2}
        lang_encoded = lang_map.get(lang_level, 0)

        features = [[cgpa, year_grad, experience, skill_count, lang_encoded]]
        scaled_features = scaler.transform(features)

        prediction = model_mindset.predict(scaled_features)[0]

        # 👇 Return to recommend.html with prediction
        # return render_template('recommend.html', mindset=prediction)
        return render_template("predict_result.html", mindset=prediction)


    except Exception as e:
        import traceback
        traceback.print_exc()  # 🔍 Print full traceback in the terminal
        return f"Prediction failed. Error: {str(e)}"

def predict_mindset(form):
    try:
        cgpa = float(form['graduation_cgpa'])
        year_grad = int(form['year_of_graduation'])
        experience = float(form['total_experience'])
        skills = form['technologies_skills']
        skill_count = len(skills.split(','))

        lang_level = form.get('foreign_language_proficiency', 'Beginner')
        lang_map = {'Beginner': 0, 'Advanced': 1, 'Mastery': 2}
        lang_encoded = lang_map.get(lang_level, 0)

        features = [[cgpa, year_grad, experience, skill_count, lang_encoded]]
        scaled = scaler.transform(features)

        prediction = model_mindset.predict(scaled)[0]
        return prediction

    except Exception as e:
        print("Prediction error:", e)
        return None


if __name__ == '__main__':
    app.run(debug=True, port=5500)