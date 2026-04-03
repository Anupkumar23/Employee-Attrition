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

**Option A: Using Jupyter Notebook** (recommended for exploration)
```bash
jupyter notebook Employee_Attrition.ipynb
```
Run all cells. Section 8 exports the model for Flask.

**Option B: Using the export script** (quick setup)
```bash
python export_model.py
```

Both options generate `model.pkl`, `metrics.pkl`, and `static/confusion.png`.

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
├── export_model.py        # Quick model export script
├── model.pkl              # Trained model (generated)
├── metrics.pkl            # Model metrics (generated)
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
