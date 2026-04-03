# Employee Attrition Prediction

A machine learning web application that predicts employee attrition risk using Random Forest classifier. Built with Flask and trained on the IBM HR Analytics dataset.

## Features

- **Predict Attrition**: Input employee details and get instant predictions
- **Visual Results**: See probability breakdown with stay/leave percentages
- **Model Metrics**: View model performance statistics and confusion matrix
- **REST API**: JSON endpoint for programmatic access

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Anupkumar23/Employee-Attrition.git
cd Employee-Attrition
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

Open `Employee_Attrition.ipynb` in Jupyter Notebook and run all cells. The notebook includes:
- Data exploration and visualization
- Model training and evaluation
- **Section 8**: Exports `model.pkl` and `metrics.pkl` for the Flask app

```bash
jupyter notebook Employee_Attrition.ipynb
```

Or run via command line:
```bash
jupyter nbconvert --to notebook --execute Employee_Attrition.ipynb
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

Navigate to: **http://127.0.0.1:5000**

## Project Structure

```
Employee-Attrition/
├── app.py                 # Flask web application
├── Employee_Attrition.ipynb  # Model training & analysis notebook
├── model.pkl              # Trained model (generated from notebook)
├── metrics.pkl            # Model metrics (generated from notebook)
├── requirements.txt       # Python dependencies
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset
├── static/
│   ├── style.css          # Application styles
│   └── confusion.png      # Confusion matrix visualization
└── templates/
    ├── index.html         # Prediction form
    ├── result.html        # Prediction results
    └── metrics.html       # Model performance page
```

## API Usage

### POST /api/predict

Send JSON payload with employee features:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 5,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Married",
    "MonthlyIncome": 5000,
    "MonthlyRate": 14000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 14,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3
  }'
```

Response:
```json
{
  "success": true,
  "prediction": "STAY",
  "stay_probability": 85.3,
  "leave_probability": 14.7
}
```

## Technologies

- **Backend**: Flask, Python
- **ML**: Random Forest, scikit-learn
- **Data**: pandas, NumPy
- **Visualization**: matplotlib, seaborn

## License

MIT License - see [LICENSE](LICENSE) file
