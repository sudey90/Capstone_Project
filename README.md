# Capstone_Project#

# Prerequisite Setup and Environment Configuration

## 1. Create the GitHub Repository
1. Create a new repository on [GitHub](https://github.com/).
2. Set the repository name to `Capstone_Project` and the owner to `sudey90`.
3. Enable **Add a README file** and create the repository.
4. Repository URL: `https://github.com/sudey90/Capstone_Project.git`
5. Default branch: `main`.

## 2. Install Visual Studio Code and Git
1. Download and install [Visual Studio Code](https://code.visualstudio.com/).
2. Install the **Python** and **WSL** extensions published by Microsoft.
3. Download and install [Git for Windows](https://git-scm.com/).
4. Verify the installations:
    code --version
    git --version
5. Configure Git:
    git config --global user.name "Supriyo Dey"
    git config --global user.email "sudey90@gmail.com"

## 3. Clone the Repository
1. Create the project parent directory in Windows:
   `D:\Masai\GitHub_Projects`
2. Open Command Prompt and execute:
    cmd
    cd /d D:\Masai\GitHub_Projects
    git clone https://github.com/sudey90/Capstone_Project.git
    cd Capstone_Project
    code .
3. Verify the repository:
    git status

   **Note:** Use the exact repository URL displayed on GitHub. The URL above assumes the repository name is `Capstone_Project`.

## 4. Install WSL and Ubuntu
The initial Windows Python execution of `generate_data.py` failed with a pandas DLL loading error because a Windows Application Control policy blocked a required component.

To use a Linux environment without replacing Windows:
1. Open PowerShell as Administrator and execute:
   wsl --install -d Ubuntu
2. Restart Windows if requested and complete the initial Ubuntu username and password setup.
3. Open VS Code and install the Microsoft **WSL** extension if it is not already installed.
4. Press `Ctrl + Shift + P`, select `WSL: Connect to WSL`, and connect to Ubuntu.
5. Open the project folder from the WSL-connected VS Code window.
   The project directory used for this assignment is:
   `/mnt/d/Masai/Capstone_Project`
   This corresponds to `D:\Masai\Capstone_Project` in Windows. If the cloned repository is still under `D:\Masai\GitHub_Projects`, use its actual location instead.

## 5. Configure Python in Ubuntu
Open the VS Code terminal connected to Ubuntu and navigate to the project root:
cd /mnt/d/Masai/Capstone_Project

If Python, pip or the virtual environment module is missing, install the required Ubuntu packages:
sudo apt update
sudo apt install python3 python3-pip python3-venv

Create and activate the virtual environment:
python3 -m venv .venv
source .venv/bin/activate

Install the required libraries:
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn joblib

Verify the installation:
python --version
python -c "import numpy, pandas, scipy, sklearn, joblib; print('All libraries imported successfully')"

Configure Git in Ubuntu if necessary:
git config --global user.name "Supriyo Dey"
git config --global user.email "sudey90@gmail.com"

Generate `requirements.txt` from the active virtual environment:
python -m pip freeze > requirements.txt

## 6. Generate the Initial Dataset
Ensure that the virtual environment is active and the terminal is in the project root.

Execute:
python generate_data.py

Expected console output:
24 1224 48

The script generates the following files inside the `data/` directory:
- `data/hubs.csv`
- `data/raw_shipments.csv`
- `data/new_shipments.csv`

The provided `generate_data.py` script must remain unchanged, and the generated CSV files must not be manually edited.

## 7. Commit and Push Changes to GitHub
1. Open the **Source Control** panel in VS Code.
2. Review the modified files and stage the files to be committed.
3. Enter a meaningful commit message.
4. Click **Commit**.
5. Select **Push** from the Source Control menu to upload the committed changes to GitHub.
6. Verify the changes in the GitHub repository.

git status
git add requirements.txt
git commit -m "Add project environment requirements"
git push

Replace the staged filename with the appropriate project files when committing other changes. Stage generated files only when required by the assignment instructions.

## 8. Environment Verification
Before starting the assignment, confirm that
- VS Code is connected to Ubuntu through WSL.
- The terminal is in the correct project root.
- The `.venv` virtual environment is active.
- All required Python libraries import successfully.
- `generate_data.py` prints `24 1224 48`.
- All three initial CSV files are present in `data/`.
- Git recognises the repository and can communicate with GitHub.

Once these checks pass, proceed with Part A (Data Foundation and Delay Evidence), followed by Parts B, C and D.

## Data Foundation and Delay Evidence
Based on the processed shipment dataset (1,200 observations), the analysis revealed the following key insights:

1. **Service Type Delay Rates:** **Express** shipments experience delays most frequently, recording a delay rate of **0.4897** (119 delayed out of 243 shipments), compared to Standard shipments at **0.2056** and Economy shipments at **0.0857**.
2. **Hub Tier Progression:** Delay rates increase sharply from lower to higher tiers, climbing from **0.0400** in Tier 1 hubs to **0.2150** in Tier 2, and peaking at **0.4400** in Tier 3 hubs.
3. **Distance Band Impact:** Delivery delays escalate significantly with transit distance, starting at **0.0219** for the 0–200 km band, rising through **0.1368** (201–400 km) and **0.4364** (401–600 km), and reaching **0.8523** for distances of 601+ km.
4. **Top Correlations with Delivery Hours:** Ranking the features by absolute correlation with `delivery_hours` (excluding itself), the top three columns are `promised_hours` (**0.8372**), `distance_km` (**0.4013**), and `n_stops` (**0.3247**).
5. **Declared Value Transformation:** `declared_value_inr` was log-transformed because its right-skewed raw distribution (mean of **1,533.05** vs. median of **1,121.50**) compresses after transformation into a more symmetric distribution (mean of **7.0280** vs. median of **7.0220**), reducing the undue leverage of high-value outliers in modeling.

## Delivery Hours Regression

### 1. Baseline performance
The baseline model predicted the average training delivery time for all validation shipments. It achieved an R² of **-0.021** and RMSE of **18.855 hours**, indicating poor predictive performance.

### 2. Decision Tree depth comparison
Decision Tree depth 6 achieved the lowest validation RMSE of **6.280 hours**. Depth 1 showed high bias (underfitting), while unlimited depth showed high variance (overfitting), with training RMSE of 0 and validation RMSE of 7.789 hours.

### 3. Model comparison and learning rate
Four regression models were evaluated. **Linear Regression** achieved the lowest validation RMSE of **4.891 hours** with R² of 0.931, making it the selected model.

For Gradient Boosting, learning rate **0.1** performed best (R² = 0.931). A rate of 0.01 learned too slowly, while 1.0 resulted in overly aggressive updates.

### 4. Cross-validation
Five-fold cross-validation was performed on the two best models.
- Linear Regression: Mean R² = 0.926, Std = 0.00475
- Gradient Boosting: Mean R² = 0.935, Std = 0.00442

Both models showed consistent performance across folds.

### 5. Final test performance
Linear Regression was selected based on its lowest validation RMSE, even though Gradient Boosting achieved a slightly higher cross-validation R².

Final test results:
- **MAE:** 3.567 hours
- **MSE:** 20.462
- **RMSE:** 4.524 hours
- **R²:** 0.949

The final model explains approximately 94.9% of the variation in delivery hours on unseen test data.

## Classification — Delayed Shipment Prediction

### 1. Baseline performance
The baseline model predicted every shipment as on-time. It achieved an accuracy of **76.67%**, but failed to identify any delayed shipments (Recall = 0), making it unsuitable for delay prediction.

### 2. Model comparison
Three classification models were evaluated. Logistic Regression achieved a validation F1-score of **0.805**, Decision Tree (depth 6) achieved **0.747**, and Random Forest achieved **0.785**. All three performed better than the baseline.

### 3. Feature selection
Random Forest feature importance and SelectKBest were used to identify the most useful features. The top seven features were distance_km, hub_tier, hub_daily_capacity, n_stops, promised_hours, rain_flag, and sale_event_flag. The five intentionally added noise features together contributed **22.44%** of the Random Forest importance.

### 4. Hyperparameter tuning
Grid Search evaluated 27 parameter combinations, while Randomized Search evaluated 9 combinations using 5-fold cross-validation. The default Random Forest with seven selected features achieved the highest validation F1-score of **0.833**, so it was selected as the final model.

### 5. Classification threshold
A threshold of **0.5** was selected because it was the highest tested threshold that maintained validation recall of at least 0.80. Increasing the threshold improves precision but can cause more delayed shipments to be missed.

### 6. Final test performance
The selected Random Forest model was evaluated once on the unseen test data.

- **Accuracy:** 86.67%
- **Precision:** 68.89%
- **Recall:** 75.61%
- **F1-score:** 0.721

The model achieved better performance than the baseline and successfully identified delayed shipments on unseen data.