# MLOps Assignment 02

This folder contains the complete solution for **Assignment 02: Final Marks for Students Using Partial Assessment Activities**.

## Files

- `src/complete_assignment.py`: executable source code
- `requirements.txt`: Python dependencies
- `outputs/Assignment_02_Predictions.xlsx`: required Excel workbook with one sheet for each test-set course
- `outputs/Assignment_02_Report.pdf`: required PDF report

## How to Run

From this folder:

```bash
python -m pip install -r requirements.txt
python src/complete_assignment.py --data-dir . --output-dir outputs
```

On this machine, the already prepared environment can be used directly:

```bash
/home/shehroz/Documents/genai-course/.venv/bin/python src/complete_assignment.py --data-dir . --output-dir outputs
```

The script expects these two dataset files in the project root:

- `[Template] CC Result Data Set.xlsx`
- `[Template] ICT Result Data Set.xlsx`

## Method Summary

For each course, the morning sheet is used as training data and the afternoon sheet is used as testing data. The target is the final `Total` score out of 100. Separate models are trained after the 5th, 6th, 7th, and later activities until the second-last activity.

For every prediction horizon, the script predicts the remaining-score ratio and adds it to the marks already earned by the known activities. Cloud Computing uses a Random Forest remaining-ratio model, while ICT uses an Elastic Net remaining-ratio model. A conservative engagement fallback is applied when a student has extremely low earned marks and mostly missing/zero known activities.
