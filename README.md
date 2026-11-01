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

## Data Foundation and Delay Evidence - Part A
Based on the processed shipment dataset (1,200 observations), the analysis revealed the following key insights:

1. **Service Type Delay Rates:** **Express** shipments experience delays most frequently, recording a delay rate of **0.4897** (119 delayed out of 243 shipments), compared to Standard shipments at **0.2056** and Economy shipments at **0.0857**.
2. **Hub Tier Progression:** Delay rates increase sharply from lower to higher tiers, climbing from **0.0400** in Tier 1 hubs to **0.2150** in Tier 2, and peaking at **0.4400** in Tier 3 hubs.
3. **Distance Band Impact:** Delivery delays escalate significantly with transit distance, starting at **0.0219** for the 0–200 km band, rising through **0.1368** (201–400 km) and **0.4364** (401–600 km), and reaching **0.8523** for distances of 601+ km.
4. **Top Correlations with Delivery Hours:** Ranking the features by absolute correlation with `delivery_hours` (excluding itself), the top three columns are `promised_hours` (**0.8372**), `distance_km` (**0.4013**), and `n_stops` (**0.3247**).
5. **Declared Value Transformation:** `declared_value_inr` was log-transformed because its right-skewed raw distribution (mean of **1,533.05** vs. median of **1,121.50**) compresses after transformation into a more symmetric distribution (mean of **7.0280** vs. median of **7.0220**), reducing the undue leverage of high-value outliers in modeling.

## Delivery Hours Regression - Part B

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

## Classification — Delayed Shipment Prediction - Part C

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

## How to run - Part D

## How to Run the Project :-
Follow these steps to run the project from a fresh GitHub clone.

### Step 1: Clone the repository
git clone https://github.com/sudey90/Capstone-Project.git
cd Capstone-Project

### Step 2: Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

### Step 3: Install the required libraries
pip install -r requirements.txt

This installs the required Python libraries, including pandas, NumPy, scikit-learn and joblib, using the versions specified in `requirements.txt`.

### Step 4: Generate the input data
python generate_data.py
This creates the three input CSV files inside the `data/` folder.

### Step 5: Run Part A – Data Cleaning and EDA
python part_a_clean_eda.py

This cleans the raw shipment data, removes duplicates, handles missing values, creates the modelling table and generates initial delay analysis.

### Step 6: Run Part B – Delay Prediction (Regression)
python part_b_regression.py

This builds and evaluates a regression model to estimate shipment delivery delays.

### Step 7: Run Part C – Delay Classification
python part_c_classification.py

This builds and evaluates a classification model to identify shipments that are likely to be delayed. It also determines the alert threshold.

### Step 8: Run Part D – Hub Segmentation
python part_d_segments.py

This groups the 24 hubs into three operational segments: Standard, Watch and Escalate.

### Step 9: Package the final prediction model
python package_model.py

This regenerates `models/delay_model.pkl` using the final classification model and its selected configuration.

The committed pickle file is version-dependent and loads reliably only with the scikit-learn version pinned in `requirements.txt`. If the installed scikit-learn version differs, the grader should run `package_model.py` to regenerate the model rather than relying on the committed pickle.

### Step 10: Score new shipments
python predict_delay.py

This uses the packaged classification model and hub segments to analyse the 48 new shipments and generates the final action recommendations.

The output is saved in:

`outputs/new_shipment_actions.csv`

**Important:** Run the scripts in the above order because each part uses the output generated by the previous part.

## 2. How to run

## Operations Recommendation
Based on the hub segmentation and shipment delay analysis, the following operational recommendations are suggested for the regional head.

### 1. Hub Performance and Escalation
The 24 hubs have been divided into three operational segments using K-Means clustering with K=3.

| Segment | Number of Hubs | Average Delay Rate | Recommended Action |
|---|---:|---:|---|
| Standard | 7 | 2.57% | Continue normal operations |
| Watch | 8 | 15.50% | Monitor performance regularly |
| Escalate | 9 | 46.00% | Immediate hub-manager attention |

The nine hubs in the Escalate segment should receive the highest operational priority because their average delay rate is 46%.

Hub managers should investigate possible reasons such as transportation delays, staffing problems, sorting capacity and operational bottlenecks.

### 2. Tier and Segment Mismatches
The following hubs have a mismatch between their original tier classification and their actual performance-based segment.

| Hub | Location | Original Tier | Actual Segment |
|---|---|---|---|
| HUB07 | Kolkata | Tier 1 | Watch |
| HUB13 | Nagpur | Tier 2 | Escalate |
| HUB15 | Kochi | Tier 2 | Escalate |
| HUB22 | Dehradun | Tier 3 | Watch |

These mismatches show that the original tier classification does not always represent the actual delay performance of a hub.

For example, HUB07 is classified as Tier 1 but belongs to the Watch segment. Similarly, HUB13 and HUB15 are Tier 2 hubs but belong to the Escalate segment.

Therefore, the original tier should be used as background information for understanding hub importance, while the performance-based segment should guide operational monitoring and escalation decisions.

### 3. Shipment Delay Alert Threshold
The final classification model uses an alert threshold of **0.50 (50%)**.

A shipment is flagged when its predicted probability of delay is equal to or greater than 0.50.

The previously reported test performance was:
- Precision: 68.9%
- Recall: 75.6%

This means approximately:
- Out of every 10 flagged shipments, 7 are actually delayed.
- Out of every 10 actually delayed shipments, approximately 8 are identified by the model.

This helps the operations team identify potentially delayed shipments before delivery problems occur.

### 4. Actions for the 48 New Shipments
The final prediction script analysed 48 new shipments and generated the following recommendations.

| Recommended Action | Number of Shipments |
|---|---:|
| Escalate to hub manager | 6 |
| Proactive customer calls | 2 |
| No action required | 40 |
| **Total** | **48** |

Out of the 48 shipments:
- 8 shipments were flagged by the delay prediction model.
- 6 flagged shipments require hub-manager escalation because of their associated hub segment.
- 2 flagged shipments require proactive customer communication.
- 40 shipments do not require immediate action.

This allows the operations team to focus its attention on shipments that need intervention instead of manually checking every shipment.

### 5. Why K=3 Was Selected
K=3 was selected because it provides three clear operational categories: Standard, Watch and Escalate.

The separation analysis produced the following results:

| Number of Segments (K) | Separation Score |
|---|---:|
| 2 | 2.894 |
| 3 | 5.393 |
| 4 | 6.305 |
| 5 | 6.887 |

Although K=4 and K=5 produced higher separation scores, they also created more operational groups.

K=3 provides a good balance between separating hubs based on their performance and keeping the recommendations simple enough for the operations team to follow.

Therefore, three segments were retained as the final operational solution.

## 3. Production Plan

### 1. Production Readiness
The project includes the following characteristics to support future production use.

| Characteristic | How It Is Addressed in This Project |
|---|---|
| Accuracy | The classification model is evaluated using precision, recall and F1-score on test data. |
| Reliability | The model is saved using `package_model.py` so that it can be loaded and used again for new shipment predictions. |
| Reproducibility | The same scripts can be executed again to regenerate the data, models and prediction outputs. |
| Performance (Speed) | The project uses a Random Forest model and processes 48 new shipments in a single execution. |
| Maintainability | The project is divided into separate Python scripts for Parts A, B, C and D, making it easier to update individual sections. |
| Consistency | The shared preprocessing functions in `src/shipment_prep.py` ensure that the same feature preparation logic is used during training and prediction. |
| Monitorability | The project generates CSV reports containing hub segments, delay predictions and recommended operational actions. These can be reviewed regularly by the operations team. |

### 2. Monitoring Plan
After deployment, the following business measurements should be monitored regularly.

| Measurement | What It Tells Us | Suggested Alert Trigger |
|---|---|---|
| Weekly delayed shipment percentage | Percentage of delivered shipments that were delayed | Alert if it exceeds 30% |
| Weekly flagged shipment percentage | Percentage of shipments identified by the model as potentially delayed | Alert if it exceeds 20% |
| Model recall | Percentage of actually delayed shipments correctly identified by the model | Alert if recall falls below 70% |
| Hub delay rate | Percentage of delayed shipments handled by each hub | Review any hub with a delay rate above 30% |

These are proposed monitoring thresholds for initial production use. They should be reviewed and adjusted after sufficient real-world shipment data becomes available.

The operations team should review these measurements every week and investigate any unusual increase or decrease.

### 3. Model Retraining Schedule
The classification model should be reviewed every month and retrained at least once every three months, using newly available shipment data.

Retraining should also be considered earlier if the model's recall falls below the agreed target or if shipment patterns change significantly.

### 4. Handling the September Sale Event
The generated shipment data contains a `sale_event_flag` that identifies shipments associated with the sale period from 15 September to 28 September.

During this period, shipment volume and delivery delays may increase because of higher customer demand.

Therefore, sale-event shipments should not be compared directly with ordinary business weeks.

Instead, the operations team should compare the September sale period with previous sale events of a similar type.

This will help determine whether the increase in shipment delays is caused by normal sale-related demand or by an actual operational problem.

### 5. Data Drift Monitoring
Data drift means that the characteristics of new shipment data start becoming different from the data used to train the original model.

For example:
- Average shipment distance may increase.
- Average parcel weight may change.
- The number of Express, Standard or Economy shipments may change.
- Some hubs may start handling significantly more shipments.
- The percentage of sale-event shipments may increase.

If these changes continue over time, the existing model may become less accurate.

Therefore, shipment characteristics and model performance should be monitored regularly. If significant changes are observed, the model should be evaluated and retrained using recent data.

## 4. Limitations
The project provides a useful starting point for shipment delay analysis and hub performance monitoring. However, there are some limitations.

### 1. Synthetic Data
The project uses synthetic shipment data generated through `generate_data.py`, rather than actual Delhivery business data.

Therefore, the model's performance may be different when applied to real shipment records.

### 2. Limited Test Data
The model is evaluated using a limited test dataset. In particular, the number of delayed shipments in the test set is relatively small.

As a result, precision and recall scores may change when the model is tested on a larger dataset.

### 3. Threshold Selection
The alert threshold of 0.50 was selected using the available validation results.

This threshold may not be the best choice for every business situation. It should be reviewed using additional real-world shipment data and the cost of missed delays versus unnecessary alerts.

### 4. Changing Business Conditions
Shipment delays can be affected by weather, traffic, festivals, seasonal demand, transportation problems and unexpected operational issues.

The current dataset may not capture all these real-world situations.

Therefore, the model cannot guarantee that every delayed shipment will be identified correctly.

### 5. Hub Segmentation Limitations
The hub segmentation is based on the available shipment performance features.

Hub performance can change over time because of changes in shipment volume, staffing, transportation capacity and operational conditions.

Therefore, hub segments should be reviewed periodically rather than treated as permanent classifications.

### 6. Sale Event Limitations
The September sale period is represented in the synthetic dataset using `sale_event_flag`.

However, one simulated sale period is not sufficient to understand every possible seasonal demand pattern.

Actual historical sale-event data would be required for more reliable comparisons.