

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

This project was developed using **Google Colab**

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

1. The dashboard was built using **Dash (Plotly)** in Python.
2. All dependencies are listed in `requirements.txt`.
3. The repository was pushed to **Render** for deployment.
4. Render automatically installed the dependencies and launched the app.

### Live Dashboard

**URL:**  
`[Paste your live dashboard link here]`

> Users can interact with the dashboard in real time without installing anything locally.

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

1. .
        
2. 
        
3. 
        
4. 
        
5. 
        
6. 
        
7. 
        
8. 
        
9. 
        
10. 
        

**Result:**  

#### **Dareen Ahmed & Lama Hany – Dashboard Development and Deployment**

**Steps Taken:**

1. **Dashboard Design and Layout**
    
    - Designed an interactive dashboard to visualize NYC traffic crash data.
        
    - Organized visualizations into five tabs for logical exploration:
        
        1. **Crash Geography** – Maps, trends over time, and borough comparison.
            
        2. **Vehicles & Factors** – Contributing factors, vehicle vs factor heatmaps, and vehicle type trends.
            
        3. **People & Injuries** – Safety equipment, injury types, emotional status, ejection, position in vehicle, person types over time, top complaints.
            
        4. **Demographics** – Age and gender distributions, real-time statistics.
            
        5. **Advanced Analytics** – Crash hotspot clustering, risk correlation matrix, temporal risk patterns, severity prediction, and spatial risk density.
            
2. **Data Integration into Dashboard**
    
    - Loaded the cleaned and integrated crash + persons dataset.
        
    - Used **Pandas** for data manipulation and **NumPy** for numeric calculations.
        
    - Ensured all visualizations respond dynamically to filters.
        
3. **Interactive Filters & Features**
    
    - Implemented **dropdown filters** for Borough, Year, Vehicle Type, Contributing Factor, and Injury Type.
        
    - Developed a **search mode** for text-based queries (e.g., “Brooklyn 2022 pedestrian crashes”).
        
    - Added a **Generate Report button** to update all visualizations dynamically.
        
4. **Visualizations**
    
    - Used **Plotly Express** and **Plotly Graph Objects** for interactive charts.
        
    - Types of visualizations included:
        
        - Maps for geographic hotspots.
            
        - Line and bar charts for temporal trends.
            
        - Heatmaps for correlations between factors and vehicle types.
            
        - Pie charts and histograms for demographics.
            
    - Incorporated interactivity: hover info, zoom, and real-time filter updates.
        
5. **Advanced Analytics**
    
    - Implemented machine learning clustering to identify crash hotspots.
        
    - Created risk correlation matrices and temporal risk patterns to guide safety interventions.
        
    - Analyzed factors contributing to severe injuries and mapped spatial risk densities.
        
6. **Deployment**
    
    - Prepared the dashboard for deployment using **streamline** .
        
    - Ensured all dependencies are included in `requirements.txt`.
        
    - Verified live functionality: all interactive filters, search mode, and charts update correctly in real time.
        
    - Provided a live URL for public access.
        

**Result:**  
A fully interactive, user-friendly dashboard enabling city planners, safety advocates, and the public to explore crash data, identify trends, and analyze risk factors for informed decision-making.


## Tools & Technologies

- Python
- Pandas & NumPy
- Plotly & Plotly Express
- Dash (Plotly)
- Dash Bootstrap Components
- Scikit-learn (clustering)
- Git & GitHub




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


**Q9.**  s there a relationship between crash location (borough) and the type of injury (minor vs severe)?

> Goal: Check if injury severity varies by borough.  
> Variables: `BOROUGH`, `PERSON_INJURY`

