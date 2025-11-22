- # Sample deployed Dasboard ( 🎀 girly pop 🎀)
  `[https://nyc-crashes.up.railway.app/]`

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
## Setup instruction 
1. Open the notebook `[https://colab.research.google.com/drive/1UK9vFCahXrsqw1JO0YTvsdzSSdIHK_tj?usp=sharing]`
2. run each cell
3. open ngrok link

## Deployment Instructions

The dashboard was deployed online for interactive use.  

### Steps:

1. The dashboard was built using **Streamlit** in Python, with **Plotly** used for interactive visualizations.
2. All dependencies are listed in `requirements.txt`.
3. The repository can be deployed to cloud services that support Python and Streamlit (e.g., Render, Streamlit Cloud, or Heroku).
4. The deploy platform will install dependencies from `requirements.txt` and run the Streamlit app.

### Live Dashboard

**URL:**  
`[https://nyc-crashes-dashboard-82lpeohclvrgupvgqth2j8.streamlit.app/]`
`[https://nyc-crashes-dashboard-82lpeohclvrgupvgqth2j8.streamlit.app/]`

> Users can interact with the dashboard in real time without installing anything locally.

---

## Related Repositories

- **Upstream / Reference Repo:** https://github.com/kevorkian-mano/Motor_Vehicle_Collisions_Project.git

---

## Related Links

- **Exploration Notebook (Colab):** https://colab.research.google.com/drive/1UK9vFCahXrsqw1JO0YTvsdzSSdIHK_tj?usp=sharing
- **Live demo (ngrok tunnel):** https://septimal-elene-tribrachial.ngrok-free.dev/
- **Supplementary repo:** https://github.com/lusgad/nyc-crash-data.git

---

## Team Contributions

| Team Member        | Contribution                                                        |
| ------------------ | ------------------------------------------------------------------- |
| **Osama Loay**     | Pre-cleaning of the Crashes dataset + Post-integration cleaning     |
| **Manuel Youssef** | Pre-cleaning of the Persons dataset + Integration                   |
| **Dareen Ahmed**   | Dashboard engineering, interactive visualizations, performance & deployment |
| **Lama Hany**      | UX / visual design lead, advanced chart design, presentation polish |

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

#### **Lama Hany – UX & Advanced Visual Design (Lead)**

**Highlights:**

- Led the visual and interaction design for the dashboard, producing a clear, presentation-ready layout organized into purpose-driven tabs (Crash Geography, Vehicles & Factors, People & Injuries, Demographics, Advanced Analytics).
- Drove the design system: dark navy/indigo theme, accessible color palette, and consistent typography and spacing to make the dashboard polished and demo-ready.
- Designed advanced, presentation-quality visualizations (complex heatmaps, layered map views, annotation-ready charts) that significantly raised the project's visual standards.
- Collaborated closely with engineering to ensure visual fidelity across responsive views and export-friendly layouts.

**Impact:** Lama's design leadership and advanced chart design were instrumental — the dashboard's clarity, visual polish, and storytelling quality would not have reached its current level without her work.

#### **Dareen Ahmed – Dashboard Engineering & Delivery (Lead)**

**Highlights:**

- Built the Streamlit pages, components, and control flows that power the interactive experience, including session-state management and a deterministic "Generate" workflow for reproducible demos.
- Implemented the Plotly visualizations and extracted reusable figure-builder functions to ensure consistency and simplify testing and reuse.
- Implemented performance-oriented engineering: Parquet + DuckDB scanning, pre-aggregated Parquet artifacts, and query tuning so large person-level analytics render quickly.
- Added smoke tests, schema-robust column-detection, and error handling to improve reliability during refactors and deployment.
- Prepared the app for cloud deployments and integration with Git LFS for large artifacts; ensured `requirements.txt` and deployment instructions are complete.

**Impact:** Dareen's engineering leadership turned the cleaned datasets and visual designs into a stable, high-performance dashboard—without this work the interactive product would not have been deliverable.

**Acknowledgement:**

Lama and Dareen together drove the dashboard's design and engineering end-to-end; their combined contributions were essential — the project would not be in a presentable, production-ready state without their work.


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

