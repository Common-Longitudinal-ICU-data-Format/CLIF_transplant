# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For all tasks, use the CLIF schema files in the references/mCIDE directory, the clif_2.1.0.txt file, and the CLIF data dictionary (clif-icu.com_data-dictionary_data-dictionary-2.0.0.md) to understand the data structure and column names.

In CLIF, *_category variables have a set of permissible values that are defined in the references/mCIDE directory. Read these when working with category variables to ensure you are using the correct permissible values.

## Beginning of a session

Start by pulling any changes from the main branch.


## Standard Workflow
0. Check the screenshots folder to see if any new screenshots have been added, if so analyze them before planning the task
1. First think through the problem, read the codebase for relevant files, and **consult the CLIF data dictionary** (clif-icu.com_data-dictionary_data-dictionary-2.0.0.md) for correct table and column names, then write a plan to todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. Make your code changes adhere to the CLIF schema and the marimo rules. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.



## Project Overview

This is a medical research project that identifies transplant recipient hospitalizations from the CLIF (Critical Care Literature Information Framework) database using CPT procedure codes. The project processes comprehensive adult inpatient data to extract transplant recipients across heart, lung, liver, and kidney transplants.

## Project Structure

- `config/` - Configuration files for site-specific settings
- `code/` - Analysis scripts and templates (Python/R)
- `utils/` - Utility functions including configuration loading
- `outlier-thresholds/` - CSV files defining outlier thresholds for data cleaning
- `output/` - Results directory with `final/` for deliverables and `intermediate/` for processing files

## Configuration

1. Copy `config/config_template.json` to `config/config.json`
2. Update `config.json` with site-specific settings:
   - `site_name`: Institution identifier
   - `tables_path`: Path to CLIF tables
   - `file_type`: Data format (csv/parquet/fst)
3. Use `utils/config.py` to load configuration in Python scripts

## Workflow

The project follows a three-step analysis workflow:

1. **Cohort Identification**: Apply inclusion/exclusion criteria using CPT codes for transplant procedures
2. **Quality Control**: Handle outliers using thresholds from `outlier-thresholds/` directory
3. **Analysis**: Generate final results and statistics

## Data Processing

- Input: CLIF 2.0 and 2.1 tables (patient, hospitalization, vitals, labs, medication_admin_*, respiratory_support, crrt_therapy, ecmo_mcs, patient_procedure)
- Output: Filtered parquet files in CLIF format plus transplant table with patient_id, transplant_type, recorded_dttm
- Cohort identified using specific CPT codes for different organ transplants

## Environment Setup

### Option A: Using uv (recommended)

```bash
uv sync
uv run marimo edit code/heart_transplant_report.py
```

### Option B: Using traditional venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## File Naming Convention

Results should follow the pattern: `[RESULT_NAME]_[SITE_NAME]_[SYSTEM_TIME].pdf`

Use the config object to get the site name for consistent file naming across the project.

## CLIF Data Structure and Keys

**Critical**: Understanding CLIF table relationships and primary keys:

1. **Patient Table**: Primary key is `patient_id` - contains demographic and static patient information
2. **All Other CLIF Tables**: Primary key is `hospitalization_id` - contains time-varying clinical data during hospitalizations
3. **Table Relationships**: 
   - `patient` table: `patient_id` (one-to-many with hospitalizations)
   - All clinical tables: `hospitalization_id` (vitals, labs, medications, respiratory_support, etc.)
   - To link patient demographics to clinical data: JOIN patient ON hospitalization.patient_id
4. **Data Filtering**: When filtering clinical data by patient, always use `hospitalization_id` from the hospitalization table, never filter clinical tables directly by `patient_id`

## CLIF DateTime Handling

**Important**: CLIF uses timezone-aware datetimes in UTC format (YYYY-MM-DD HH:MM:SS+00:00). When working with dates and times:

1. **Always handle timezone-awareness**: Use `pd.to_datetime(date_col, utc=True)` for datetime columns
2. **Convert birth_date to UTC**: Use `.tz_localize('UTC')` since birth_date is typically date-only
3. **Calculate ages correctly**: Use timezone-aware datetime arithmetic to avoid errors
4. **Store timezone info in config**: CLIF projects store timezone information in the configuration file

## Marimo Notebook Guidelines

This project is built using marimo notebooks. There are important coding rules to follow to avoid errors:

1. **Avoid Variable Redefinition**: Marimo tracks variable dependencies across cells. Never reuse common variable names like `df`, `table_name`, `data`, etc. across different cells as this causes "variable redefinition" errors.

2. **Use Descriptive Variable Names**: Instead of generic names, use specific names like:
   - `df` → `patient_df`, `vitals_df`, `labs_df`
   - `table_name` → `tbl_name`, `current_table`
   - `data` → `filtered_data`, `export_data`

3. **Check Dependencies**: Before creating new cells, ensure variable names don't conflict with existing cell outputs.

This prevents marimo from showing "This cell wasn't run because it has errors" due to variable conflicts.