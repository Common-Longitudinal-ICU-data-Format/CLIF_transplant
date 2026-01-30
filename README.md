# CLIF Heart Transplant Recipients

## CLIF VERSION 2.1

## Objective

Identify heart transplant recipient hospitalizations from CLIF databases using CPT procedure codes and analyze post-transplant medication patterns.

## Required CLIF tables and fields

Please refer to the online [CLIF data dictionary](https://clif-consortium.github.io/website/data-dictionary.html), [ETL tools](https://github.com/clif-consortium/CLIF/tree/main/etl-to-clif-resources), and [specific table contacts](https://github.com/clif-consortium/CLIF?tab=readme-ov-file#relational-clif) for more information on constructing the required tables and fields.

The following CLIF tables are required:

| Table | Required Columns |
|-------|------------------|
| **patient** | patient_id, race_category, ethnicity_category, sex_category, birth_date |
| **hospitalization** | patient_id, hospitalization_id, admission_dttm, discharge_dttm, admission_type_category, discharge_category |
| **vitals** | hospitalization_id, recorded_dttm, vital_category, vital_value (filtered to weight_kg) |
| **labs** | hospitalization_id, lab_result_dttm, lab_order_category, lab_category, lab_value_numeric (creatinine, bilirubin_total, albumin, sodium) |
| **medication_admin_continuous** | hospitalization_id, admin_dttm, med_category, med_dose, med_dose_unit (dobutamine, milrinone, dopamine, epinephrine, norepinephrine, isoproterenol, nitric_oxide) |
| **medication_admin_intermittent** | hospitalization_id, med_category, admin_dttm, med_dose, med_dose_unit (methylprednisolone) |
| **adt** | hospitalization_id, in_dttm, out_dttm, location_category |
| **patient_procedure** | hospitalization_id, procedure_code, procedure_code_format, procedure_billed_dttm |

## Cohort identification

The cohort is identified using heart transplant CPT codes:

| cpt_code | proc_name                                                     |
|----------|---------------------------------------------------------------|
| 33945    | PR HEART TRANSPLANT W/WO RECIPIENT CARDIECTOMY               |
| 33935    | PR HEART-LUNG TRNSPL W/RECIPIENT CARDIECTOMY-PNUMEC          |

### Transplant timing

The transplant cross-clamp time is estimated using methylprednisolone administration as a proxy:
- **Primary**: First methylprednisolone dose >500mg
- **Fallback**: First methylprednisolone dose >100mg (if no >500mg dose found)

Post-transplant ICU admission time is calculated as:
- If >500mg dose exists: cross-clamp time + 12 hours
- If only >100mg dose exists: use admin_dttm directly

## Expected Results

### Cohort file
`output/intermediate/{site_name}_cohort.csv` containing:
- `patient_id`
- `hospitalization_id`
- `transplant_cross_clamp`
- `post_transplant_ICU_in_dttm`

> **⚠️ Important**: Do not upload any files from `output/intermediate/`. These contain patient-level data and should remain local to your site.

### Summary outputs in `output/final/`
- `{site_name}_hourly_meds_summary.csv` - Hourly medication dose summaries (0-168 hours post-transplant)
- `{site_name}_aggregate_registry_comp.csv` - Registry comparison by year
- `{site_name}_tableone.csv` - Demographics and clinical characteristics
- `{site_name}_methylprednisolone_chart_daily_summary.csv` - Daily methylprednisolone summary (21 days)

### Figures in `output/final/figures/`
- Hourly dose and patient count charts for each vasoactive/inotrope
- Daily methylprednisolone dose chart

## Detailed Instructions for running the project

## 1. Update `config/config.json`
Follow instructions in the [config/README.md](config/README.md) file for detailed configuration steps.

Required configuration fields:
- `site_name`: Institution identifier
- `tables_path`: Path to CLIF tables
- `file_type`: Data format (csv/parquet/fst)
- `time_zone`: Site timezone for datetime conversions

## 2. Set up the project environment

### Option A: Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install dependencies:
```bash
uv sync
```

### Option B: Using traditional venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run code

The main analysis notebook is `code/01_hr_transplant_recipients.py` (a marimo notebook).

### Run as Python script (recommended)
```bash
# Using uv
uv run python code/01_hr_transplant_recipients.py

# Using venv
python code/01_hr_transplant_recipients.py
```

### Run interactively in browser (optional)
```bash
# Using uv
uv run marimo edit code/01_hr_transplant_recipients.py

# Using venv
marimo edit code/01_hr_transplant_recipients.py
```

The notebook performs:

1. **Data Loading**: Loads CLIF tables using `clifpy` with site-specific configuration
2. **Unit Conversion**: Converts medication doses to standardized units (mcg/kg/min for vasoactives)
3. **Cohort Identification**: Identifies heart transplant procedures and estimates transplant timing
4. **Registry Comparison**: Compares identified cases against national registry data
5. **Table One Generation**: Computes demographics, discharge disposition, and clinical characteristics
6. **Medication Analysis**: Generates hourly summaries for vasoactives/inotropes (7 days) and daily methylprednisolone (21 days)
7. **Export**: Saves all outputs to `output/final/` and `output/intermediate/`

Logs are written to `output/final/logs/` for debugging.

---


