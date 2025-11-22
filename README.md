

## Project Overview

The goal of this project is to navigate the full data engineering lifecycle using real-world datasets on **motor vehicle collisions in New York City**.  

We explored, cleaned, integrated, and visualized the data, then built a fully interactive website to help users discover insights about traffic crashes, contributing factors, and affected populations.

---
### Primary Datasets

- **NYC Motor Vehicle Collisions – Crashes**  
  https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95  

- **NYC Motor Vehicle Collisions – Persons**  
  https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu  

The datasets were integrated using the common field **`COLLISION_ID`**.

---

## Setup Instructions 

This project was developed using **Google Colab** for initial exploration and cleaning, and the interactive dashboard was built using **Streamlit** for local and cloud deployment.

### 1. Open the Notebook

Click the link to open the notebook in Google Colab:

[Add your Colab notebook link here]

### 2. Run the Notebook

- Click `Runtime` → `Run all` to execute all cells sequentially.
- The notebook will:
    - Load the NYC crash datasets
    - Perform pre-cleaning and integration
    - Produce the final cleaned dataset
    - Generate visualizations for exploration

---

## Deployment Instructions

The dashboard was deployed online for interactive use.  

### Steps:

1. The dashboard was built using **Streamlit** in Python, with **Plotly** used for interactive visualizations.
2. All dependencies are listed in `requirements.txt`.
3. The repository can be deployed to cloud services that support Python and Streamlit (e.g., Render, Streamlit Cloud, or Heroku).
4. The deploy platform will install dependencies from `requirements.txt` and run the Streamlit app.

### Live Dashboard

**URL:**  
`[Paste your live dashboard link here]`

> Users can interact with the dashboard in real time without installing anything locally.

---

## Related Repositories

- **Upstream / Reference Repo:** https://github.com/kevorkian-mano/Motor_Vehicle_Collisions_Project.git

---

## Team Contributions

| Team Member        | Contribution                                                        |
| ------------------ | ------------------------------------------------------------------- |
| **Osama Loay**     | Pre-cleaning of the Crashes dataset + Post-integration cleaning     |
| **Manuel Youssef** | Pre-cleaning of the Persons dataset + Integration                   |
| **Dareen Ahmed**   | Dashboard design & development, advanced visualizations, deployment |
| **Lama Hany**      | Dashboard development support & deployment assistance               |

### Detailed Team Contributions

#### **Manuel Youssef – Pre-Cleaning of Persons Dataset**

**Steps Taken:**

1. **Initial Data Exploration**
    
    - Checked all columns for unique values and missing data.
        
    - Identified columns with too many unique values and focused on sampling for efficiency.
        
2. **Datetime Standardization**
    
    - Converted `CRASH_DATE` and `CRASH_TIME` to proper datetime objects.
        
    - Combined into a single `CRASH_DATETIME` column.
        
    - Removed future dates and normalized invalid or missing times.
        
3. **PERSON_ID Cleaning**
    
    - Converted mixed-format IDs to strings.
        
    - Converted legacy numeric IDs to UUIDs for consistency.
        
    - Removed invalid single-digit IDs (~3.3% of the dataset).
        
4. **VEHICLE_ID**
    
    - Dropped rows with missing `VEHICLE_ID`.
        
5. **PERSON_AGE**
    
    - Detected outliers using boxplots and IQR method.
        
    - Applied domain rules (valid age 0–120).
        
    - Imputed invalid or missing ages with the mean age.
        
6. **Contributing Factors**
    
    - Columns `CONTRIBUTING_FACTOR_1` and `CONTRIBUTING_FACTOR_2` were mostly null (>80%) → dropped.
        
    - Ensured valid string formatting.
        
7. **PERSON_SEX**
    
    - Filled missing values with 'U' (Undefined).
        
8. **POSITION_IN_VEHICLE & PERSON_TYPE**
    
    - Corrected unlikely vehicle positions based on age and person type (e.g., infants cannot be in driver seat).
        
    - Set pedestrian-only columns (`PED_LOCATION`, `PED_ACTION`, `PED_ROLE`) correctly, marking invalid values as "Does Not Apply".
        
    - Adjusted positions for Bicyclists and Other Motorized vehicles based on domain logic.
        
9. **BODILY_INJURY, EJECTION, PERSON_INJURY, SAFETY_EQUIPMENT, EMOTIONAL_STATUS, COMPLAINT**
    
    - Dropped rows with all missing values in critical columns.
        
    - Standardized injury and ejection logic (e.g., ejected → injured if not killed).
        
    - Filled missing or invalid complaints based on most common complaint for each injury type.
        
    - Corrected safety equipment values based on person type and role.
        
10. **Final Cleaning Steps**
    
    - Standardized string formats (lowercased, stripped spaces, replaced 'nan' with 'unknown').
        
    - Removed duplicate rows.
        
    - Verified the cleaned dataset and saved to CSV for integration.
        

**Result:**  
The Persons dataset was fully cleaned, standardized, and ready for integration with the Crashes dataset.


#### **Osama Loay – Pre-Cleaning of Crashes Dataset and Post Cleaning**
**Steps Taken:**

1. **Full crash CSV cleaning:** Cleaned the original `crashes.csv` file by standardizing column names, trimming whitespace, normalizing case, and removing obvious duplicates and invalid rows.

2. **Latitude/Longitude → ZIP code / Borough imputation:** Imputed missing `ZIP_CODE` values using latitude and longitude when available. When `lat`/`long` were missing, constructed an address using the available `ON_STREET_NAME`, `CROSS_STREET_NAME`, and `OFF_STREET_NAME` (when present) to geocode approximate coordinates, then inferred ZIP and `BOROUGH` from the geocoded ZIP.

3. **Vehicle type normalization:** Reduced the noisy `VEHICLE_TYPE_CODE` domain from ~1,204 distinct raw values down to a consolidated set of ~12 normalized categories (e.g., `Car`, `Taxi`, `Truck`, `Bus`, `Motorcycle`, `Bicycle`, `Pedestrian`, `Other`, etc.) through mappings and rule-based grouping.

4. **Contributing factor normalization:** Reduced the contributing factor columns from about 56 noisy values to a smaller, cleaned set of canonical factors by grouping similar text labels and correcting common misspellings and variants.

5. **Missing lat/long imputation from intersection fields:** For rows missing coordinates, formed an address string from `CROSS_STREET_NAME`, `ON_STREET_NAME`, and `OFF_STREET_NAME` to approximate location, then used that to derive latitude/longitude and subsequently ZIP and `BOROUGH` when direct coordinates were not present.

6. **Dropped sparse columns and aggregated extras:** Removed very sparse columns such as `CONTRIBUTING_FACTOR_VEHICLE_3/4/5` and `VEHICLE_TYPE_CODE_3/4/5`. For multi-valued vehicle / factor columns that were sporadically populated, aggregated their non-null values into a single list column to preserve information while simplifying the schema.

7. **Dropped extremely sparse rows:** Removed rows with excessive missingness that could not be reliably imputed, focusing downstream analyses on higher-quality records.

8. **Post-integration consistency checks:** After joining with the `persons.csv` dataset using `COLLISION_ID`, removed rows where crash-level information remained too sparse. Recomputed and corrected inconsistent injury and killed counts where possible.

9. **Group-based imputation for multi-person crashes:** Where a crash had missing person-level values but multiple persons recorded in `persons.csv`, attempted to fill missing fields by grouping on `COLLISION_ID` and propagating the most plausible values across related person records (e.g., shared crash_time, location, or vehicle-level features) to reduce sparsity.

10. **General cleanup & verification:** Standardized string fields, collapsed redundant categories, removed duplicates, validated ranges (ages 0–120), and saved the cleaned crashes table for integration and dashboarding.

**Result:**

The crashes dataset was cleaned and enriched: missing ZIP/Borough and many missing coordinates were imputed, vehicle and contributing-factor values were normalized, sparse columns and rows were removed or consolidated, and post-integration repairs improved consistency across crash- and person-level data. The cleaned dataset was output for integration with the Persons table and for use by the dashboard and aggregation scripts.

#### **Dareen Ahmed & Lama Hany – Dashboard Development and Deployment**
**Steps Taken:**

1. **Product design & UX leadership**

   - Led the visual and interaction design for the dashboard, producing a clear, presentation-ready layout organized into purpose-driven tabs (Crash Geography, Vehicles & Factors, People & Injuries, Demographics, Advanced Analytics).

   - Implemented a dark navy/indigo theme and accessible color palette for strong contrast and professional presentation suitable for reports and demos.

2. **Engineering the interactive UI**

   - Built robust Streamlit pages and components with responsive sidebar controls, session-state callbacks, and an explicit "Generate" action to make updates deterministic and reproducible during demos.

   - Removed redundant widgets and consolidated filters (e.g., replacing a year slider with a multi-select) to simplify user workflows and avoid duplicate state.

3. **High-quality visualizations**

   - Implemented interactive Plotly charts (maps, line/bar trends, heatmaps, histograms) tuned for clarity: informative hover text, consistent color mappings, and helpful axis/legend defaults.

   - Extracted figure-building code into reusable functions so charts are consistent, testable, and can be used both in the UI and in offline exports.

4. **Performance & engineering optimizations**

   - Architected the app to use Parquet + DuckDB scanning for memory-efficient queries rather than loading full DataFrames into memory, which enables cloud-friendly deployment.

   - Added support for pre-aggregated Parquet artifacts generated by offline scripts; integrated a `use_preaggregates` fast path so person-level analytics render instantly when conditions allow.

   - Tuned queries and I/O to minimize repeated scans and to leverage cached/parquet-scan operations for common views.

5. **Testing, robustness & documentation**

   - Added smoke tests for figure builders to ensure each chart returns a Plotly figure object and to catch regressions during refactors.

   - Handled messy real-world schema variations by adding column-detection and safe identifier quoting so the app tolerates upstream schema drift.

   - Documented dashboard behavior, pre-aggregation workflow, and data storage paths in the README so teammates and reviewers can reproduce results.

6. **Deployment & polish**

   - Prepared the app for deployment on cloud hosts supporting Streamlit and Git LFS for large artifacts; ensured `requirements.txt` and startup instructions are complete.

   - Finalized UI polish and accessibility checks so the dashboard presents well for graders, stakeholders, and public demos.

**Result:**

The dashboard codebase now contains professionally designed UI/UX, production-oriented engineering (DuckDB + Parquet + pre-aggregates), tested and reusable visualization builders, and clear documentation — a comprehensive, polished deliverable that greatly improves the project's presentation quality and reliability.


## Tools & Technologies

- Python
- Pandas & NumPy
- Plotly & Plotly Express
- Streamlit (app framework)
- Scikit-learn (clustering)
- Git & GitHub
 - Git LFS (for large data/artifacts)
 - Parquet (columnar storage for datasets and pre-aggregations)
 - Pre-aggregation workflows (Parquet aggregates + DuckDB queries)




## Questions for Manuel Youssef

**Q1.**  Is there a relationship between **safety equipment used (seatbelt/airbag/none)** and the **severity of injury (PERSON_INJURY)** among vehicle occupants?

> Goal: See if safety equipment reduces injury severity.  
> Variables: `SAFETY_EQUIPMENT`, `PERSON_INJURY`, `PERSON_TYPE`

---

**Q2.**  At what **time of day** (CRASH_TIME) do the **highest number of injuries** occur?

> Goal: Identify dangerous time periods.  
> Variables: `CRASH_TIME`, `PERSON_INJURY`, `CRASH_DATE`

---

## Questions for Osama Loay

**Q3.**  Which **age group** is most frequently involved in crashes?  
(Example groups: 0–18, 19–30, 31–50, 51+)

> Goal: Identify high-risk age categories.  
> Variable: `PERSON_AGE`

---

**Q4.**  Do **males or females** tend to receive **more severe injuries** in accidents?

> Goal: Compare injury severity by gender.  
> Variables: `PERSON_SEX`, `PERSON_INJURY`

---

## Questions for Lama Hany

**Q5.**  Are **pedestrians and bicyclists** more likely to be **injured** than vehicle occupants?

> Goal: Compare risk among person types.  
> Variables: `PERSON_TYPE`, `PERSON_INJURY`

---

**Q6.**  
Does being **ejected from the vehicle** increase the chance of serious injury?

> Goal: Measure severity vs ejection.  
> Variables: `EJECTION`, `PERSON_INJURY`

---

## Questions for Dareen Ahmed

**Q7.**  Does the **position in the vehicle** (Driver vs Passenger) affect **injury severity**?

> Goal: Compare injuries for drivers and passengers.  
> Variables: `POSITION_IN_VEHICLE`, `PERSON_INJURY`

---

**Q8.**  Do **weekends or weekdays** have more crashes involving injured persons?

> Goal: Understand crash patterns by day.  
> Variables: `CRASH_DATE`, `PERSON_INJURY`

## Additional Questions 

**Q9.**  Are crashes involving motorcycles more likely to result in fatalities than those involving only cars?

> Goal: Compare danger level of motorcycles vs cars.  
> Variables: `VEHICLE_TYPE`, `NUMBER_OF_PERSONS_KILLED`


**Q10.**  Is there a relationship between crash location (borough) and the type of injury (minor vs severe)?

> Goal: Check if injury severity varies by borough.  
> Variables: `BOROUGH`, `PERSON_INJURY`

