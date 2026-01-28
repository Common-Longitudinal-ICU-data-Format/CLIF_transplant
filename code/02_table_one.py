import marimo

__generated_with = "0.19.6"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md("""
    # Setup and Load Data
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import sys
    from pathlib import Path
    from datetime import datetime
    import json

    import matplotlib.pyplot as plt
    import altair as alt

    # Add utils directory relative to current script location
    CURRENT_DIR = Path(__file__).resolve().parent
    UTILS_DIR = CURRENT_DIR.parent / 'utils'
    sys.path.append(str(UTILS_DIR))

    from load_config import load_config

    # Load config to get site_name
    config = load_config()
    site_name = config['site_name']

    # Load cohort file
    project_root = Path(__file__).resolve().parent.parent
    interm_dir = project_root / 'output' / 'intermediate'
    cohort_file = interm_dir / f'{site_name}_cohort.csv'
    HEART_TRANSPLANT_HOSPITALIZATIONS = pd.read_csv(cohort_file)
    HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'hospitalization_id']] = (
        HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'hospitalization_id']]
        .astype(str)
    )
    HEART_TRANSPLANT_HOSPITALIZATIONS['apprx_transplant_date'] = pd.to_datetime(
        HEART_TRANSPLANT_HOSPITALIZATIONS['apprx_transplant_date']
    )


    # Import clifpy table classes
    from clifpy.tables import (
        Hospitalization, PatientProcedures,
        Labs, MedicationAdminContinuous, MedicationAdminIntermittent,
        Patient, Adt
    )
    return (
        Adt,
        HEART_TRANSPLANT_HOSPITALIZATIONS,
        Hospitalization,
        Labs,
        MedicationAdminContinuous,
        MedicationAdminIntermittent,
        Path,
        Patient,
        alt,
        config,
        mo,
        pd,
        project_root,
        site_name,
    )


@app.cell
def _(project_root):
    import logging
    import psutil
    import os

    # Create logs directory under output/final/logs
    t1_dir = project_root / 'output' / 'final' / 'logs'
    t1_dir.mkdir(parents=True, exist_ok=True)

    # Main logger
    logger = logging.getLogger('t1')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler for main log
    main_log_path = t1_dir / 't1.log'
    file_handler = logging.FileHandler(main_log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Memory logger (separate file)
    mem_logger = logging.getLogger('t1_memory')
    mem_logger.setLevel(logging.INFO)
    mem_logger.handlers.clear()

    mem_log_path = t1_dir / 't1_memory_usage.log'
    mem_handler = logging.FileHandler(mem_log_path)
    mem_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    mem_logger.addHandler(mem_handler)

    def log_memory(label: str = ""):
        """Log memory usage to dedicated memory log file"""
        mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        msg = f"{label}: {mb:.1f} MB" if label else f"{mb:.1f} MB"
        mem_logger.info(msg)
        return mb

    logger.info(f"Logging initialized. Main log: {main_log_path}")
    logger.info(f"Memory log: {mem_log_path}")
    log_memory("Initial memory")
    return log_memory, logger


@app.cell
def _(
    Adt,
    HEART_TRANSPLANT_HOSPITALIZATIONS,
    Hospitalization,
    Labs,
    MedicationAdminContinuous,
    MedicationAdminIntermittent,
    Path,
    Patient,
    config,
    log_memory,
    logger,
):
    # Setup paths and load core tables using clifpy
    tables_path = Path(config['tables_path'])
    file_type = config['file_type']
    time_zone = config['time_zone']

    log_memory("Before loading core tables")

    try:
        hosp_table = Hospitalization.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone=time_zone,
            filters={'hospitalization_id': (HEART_TRANSPLANT_HOSPITALIZATIONS['hospitalization_id'].astype(str).unique().tolist())},
            columns=['patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm', 'admission_type_category', 'discharge_category']
        )
        logger.info(f"Loaded hospitalization: {hosp_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        log_memory("After loading hospitalization table")
    except Exception as e:
        logger.error(f"Error loading hospitalization table: {e}")



    try:
        patient_table = Patient.from_file(
        data_directory=str(tables_path),
        filetype=file_type,
        timezone=time_zone,
        columns=['patient_id', 'race_category', 'ethnicity_category', 'sex_category', 'birth_date']
        )
        logger.info(f"Loaded patients: {patient_table.df['patient_id'].nunique():,} unique patients")
        log_memory("After loading patients table")
    except Exception as e:
        logger.error(f"Error loading patients table: {e}")

    try:
        labs_table = Labs.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone=time_zone,
            filters={'lab_category':['creatinine', 'bilirubin_total', 'albumin', 'sodium']},
            columns=['hospitalization_id', 'lab_collect_dttm', 'lab_order_category', 'lab_category', 'lab_value_numeric']
        )
        logger.info(f"Loaded labs: {hosp_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        log_memory("After loading labs table")
    except Exception as e:
        logger.error(f"Error loading labs table: {e}")

    try:
        meds_interm_table = MedicationAdminIntermittent.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone=time_zone,
            columns=['hospitalization_id', 'med_category', 'admin_dttm', 'med_dose', 'med_dose_unit'],
            filters = {'med_category':['prednisone']}
        )
        logger.info(f"Loaded meds intermtable: {meds_interm_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        log_memory("After loading meds interm table table")
    except Exception as e:
        logger.error(f"Error loading meds interm table table: {e}")

    try:
        meds_table = MedicationAdminContinuous.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone=time_zone,
            filters={'med_category': ['dobutamine', 'milrinone', 'dopamine', 'epinephrine', 'norepinephrine', 'nitric_oxide']},
            columns=['hospitalization_id', 'admin_dttm', 'med_category', 'med_dose', 'med_dose_unit']
        )
        logger.info(f"Loaded meds: {meds_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        log_memory("After loading meds table")
    except Exception as e:
        logger.error(f"Error loading meds table: {e}")

    try:
        adt_table = Adt.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone=time_zone,
            columns=['hospitalization_id', 'in_dttm', 'out_dttm', 'location_category']
        )
        logger.info(f"Loaded adt: {hosp_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        log_memory("After loading adt table")
    except Exception as e:
        logger.error(f"Error loading adt table: {e}")
    return hosp_table, labs_table, meds_interm_table, meds_table, patient_table


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Table One: Cohort Characteristics
    """)
    return


@app.cell
def _(
    HEART_TRANSPLANT_HOSPITALIZATIONS,
    hosp_table,
    labs_table,
    meds_48hr,
    meds_interm_table,
    meds_table,
    patient_table,
    pd,
):
    n_total = HEART_TRANSPLANT_HOSPITALIZATIONS['hospitalization_id'].nunique()

    # Helper: format n (%)
    def n_pct(n, total):
        if total == 0:
            return "0 (0.0%)"
        return f"{n} ({100*n/total:.1f}%)"

    # Helper: format median [IQR]
    def median_iqr(series):
        if series.empty or series.dropna().empty:
            return "N/A"
        med = series.median()
        q1, q3 = series.quantile([0.25, 0.75])
        return f"{med:.1f} [{q1:.1f}-{q3:.1f}]"

    rows = []
    rows.append({"Characteristic": "N", "Value": str(n_total)})

    # Demographics: Join through hospitalization to get patient_id, then to patient_table
    cohort_patient_ids = HEART_TRANSPLANT_HOSPITALIZATIONS['patient_id'].unique()
    pt_df = patient_table.df[patient_table.df['patient_id'].isin(cohort_patient_ids)].drop_duplicates('patient_id')
    n_patients = len(pt_df)

    # Sex - only show Male
    rows.append({"Characteristic": "Sex", "Value": ""})
    n_male = (pt_df['sex_category'] == 'Male').sum()
    rows.append({"Characteristic": "  Male", "Value": n_pct(n_male, n_patients)})

    # Race - show Asian, Black, White, group rest as Other
    rows.append({"Characteristic": "Race", "Value": ""})
    for race in ['Asian', 'Black or African American', 'White']:
        n_race = (pt_df['race_category'] == race).sum()
        rows.append({"Characteristic": f"  {race}", "Value": n_pct(n_race, n_patients)})
    n_other_race = (~pt_df['race_category'].isin(['Asian', 'Black or African American', 'White'])).sum()
    rows.append({"Characteristic": "  Other", "Value": n_pct(n_other_race, n_patients)})

    # Ethnicity - only show Hispanic
    rows.append({"Characteristic": "Ethnicity", "Value": ""})
    n_hispanic = (pt_df['ethnicity_category'] == 'Hispanic').sum()
    rows.append({"Characteristic": "  Hispanic", "Value": n_pct(n_hispanic, n_patients)})

    # Age at transplant
    cohort_with_dates = HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'apprx_transplant_date']].drop_duplicates('patient_id')
    cohort_with_dates = cohort_with_dates.merge(pt_df[['patient_id', 'birth_date']], on='patient_id', how='left')
    # Strip timezone from both for age calculation
    tx_date = cohort_with_dates['apprx_transplant_date'].dt.tz_convert(None)
    birth_date = pd.to_datetime(cohort_with_dates['birth_date'])
    if birth_date.dt.tz is not None:
        birth_date = birth_date.dt.tz_convert(None)
    age_at_tx = (tx_date - birth_date).dt.days / 365.25
    rows.append({"Characteristic": "Age at transplant (years), median [IQR]", "Value": median_iqr(age_at_tx)})

    # Discharge disposition
    cohort_hosp_ids = HEART_TRANSPLANT_HOSPITALIZATIONS['hospitalization_id'].unique()
    hosp_df = hosp_table.df[hosp_table.df['hospitalization_id'].isin(cohort_hosp_ids)].copy()
    rows.append({"Characteristic": "Discharge disposition", "Value": ""})
    for val, cnt in hosp_df['discharge_category'].value_counts().items():
        rows.append({"Characteristic": f"  {val}", "Value": n_pct(cnt, n_total)})

    inotrope_list = ['dobutamine', 'milrinone', 'dopamine', 'epinephrine', 'norepinephrine']

    # Filter to inotropes only
    inotrope_meds = meds_48hr[meds_48hr['med_category'].isin(inotrope_list)]

    # Find first inotrope per hospitalization (earliest admin_dttm)
    first_inotrope = (
        inotrope_meds
        .sort_values('admin_dttm')
        .drop_duplicates('hospitalization_id', keep='first')
    )

    n_any_inotrope = first_inotrope['hospitalization_id'].nunique()
    rows.append({"Characteristic": "Any inotrope post-op (48hr)", "Value": n_pct(n_any_inotrope, n_total)})

    # Count by first inotrope given
    rows.append({"Characteristic": "First inotrope post-op (48hr)", "Value": ""})
    for med in inotrope_list:
        n_first = (first_inotrope['med_category'] == med).sum()
        rows.append({"Characteristic": f"  {med.capitalize()}", "Value": n_pct(n_first, n_any_inotrope)})

    # Prednisone after transplant
    # Merge prednisone data with transplant cohort
    prednisone_merged = meds_interm_table.df.merge(
        HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'hospitalization_id', 'apprx_transplant_date']],
        on='hospitalization_id',
        how='inner'
    )

    # Calculate days from transplant
    prednisone_merged['days_from_tx'] = (
        (prednisone_merged['admin_dttm'] - prednisone_merged['apprx_transplant_date'])
        .dt.total_seconds() / 86400
    ).astype(int)
    prednisone_post_tx = prednisone_merged[prednisone_merged['admin_dttm'] > prednisone_merged['apprx_transplant_date']]
    n_prednisone = prednisone_post_tx['patient_id'].nunique()
    rows.append({"Characteristic": "Prednisone post-transplant", "Value": n_pct(n_prednisone, n_patients)})

    # Nitric oxide after transplant
    nitric_oxide_df = meds_table.df[meds_table.df['med_category'] == 'nitric_oxide']
    nitric_merged = nitric_oxide_df.merge(
        HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'hospitalization_id', 'apprx_transplant_date']],
        on='hospitalization_id', how='inner'
    )
    nitric_post_tx = nitric_merged[nitric_merged['admin_dttm'] > nitric_merged['apprx_transplant_date']]
    n_nitric = nitric_post_tx['patient_id'].nunique()
    rows.append({"Characteristic": "Nitric oxide post-transplant", "Value": n_pct(n_nitric, n_patients)})

    # Pre-transplant labs: most recent value before transplant date
    labs_merged = labs_table.df.merge(
        HEART_TRANSPLANT_HOSPITALIZATIONS[['hospitalization_id', 'apprx_transplant_date']],
        on='hospitalization_id', how='inner'
    )
    # Filter to labs BEFORE transplant
    pre_tx_labs = labs_merged[labs_merged['lab_collect_dttm'] < labs_merged['apprx_transplant_date']]

    # Get most recent lab per hospitalization per category
    pre_tx_labs = pre_tx_labs.sort_values('lab_collect_dttm', ascending=False)
    most_recent_labs = pre_tx_labs.drop_duplicates(['hospitalization_id', 'lab_category'], keep='first')

    # Calculate median [IQR] for each lab category
    lab_categories = {'creatinine': 'Creatinine', 'bilirubin_total': 'Bilirubin',
                      'albumin': 'Albumin', 'sodium': 'Sodium'}
    for lab_cat, display_name in lab_categories.items():
        lab_values = most_recent_labs[most_recent_labs['lab_category'] == lab_cat]['lab_value_numeric']
        rows.append({"Characteristic": f"{display_name}, median [IQR]", "Value": median_iqr(lab_values)})

    table_one_df = pd.DataFrame(rows)
    table_one_df
    return prednisone_merged, table_one_df


@app.cell
def _(HEART_TRANSPLANT_HOSPITALIZATIONS, meds_table, pd):
    _revew = meds_table.df.merge(HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id','hospitalization_id', 'apprx_transplant_date']], on='hospitalization_id', how = 'left')
    _revew['hours_from_tx'] = (
          (_revew['admin_dttm'] - _revew['apprx_transplant_date'])
          .dt.total_seconds() / 3600
      )

    # Filter to within 48 hours AFTER transplant (0 to 48 hours)
    meds_48hr = _revew[
      (_revew['hours_from_tx'] >= 0) &
      (_revew['hours_from_tx'] <= 47)
    ].copy()
    meds_48hr['tx_hour'] = meds_48hr['hours_from_tx'].astype(int)

    _meds_7d = _revew[
      (_revew['hours_from_tx'] >= 23) &
      (_revew['hours_from_tx'] <= 191)
    ].copy()
    _meds_7d['tx_hour'] = _meds_7d['hours_from_tx'].astype(int)

    # Find first and last hour for each patient-medication combination
    _patient_med_bounds = (
        _meds_7d
        .groupby(['hospitalization_id', 'med_category'], as_index=False)
        .agg(first_hour=('tx_hour', 'min'), last_hour=('tx_hour', 'max'))
    )

    # Create grid: each patient × each med × hours from first_hour to 191
    # This ensures patients who stopped contribute 0 after their last dose
    _grid_rows = []
    for _, _row in _patient_med_bounds.iterrows():
        _hosp_id = _row['hospitalization_id']
        _med_cat = _row['med_category']
        for _hour in range(int(_row['first_hour']), 192):
            _grid_rows.append({'hospitalization_id': _hosp_id, 'med_category': _med_cat, 'tx_hour': _hour})

    _patient_med_grid = pd.DataFrame(_grid_rows)

    # Join actual doses to grid
    _meds_7d_filled = _patient_med_grid.merge(
        _meds_7d[['hospitalization_id', 'med_category', 'tx_hour', 'med_dose', 'med_dose_unit']],
        on=['hospitalization_id', 'med_category', 'tx_hour'],
        how='left'
    )

    # Fill missing doses with 0 (patient had drug before but not at this hour)
    _meds_7d_filled['med_dose'] = _meds_7d_filled['med_dose'].fillna(0)

    # Median dose per category/hour (now includes zeros for weaned patients)
    _median_dose_by_cat_hour = (
        _meds_7d_filled
        .groupby(['med_category', 'tx_hour'], as_index=False)['med_dose']
        .median()
    )

    # Patient counts: total in analysis vs actively receiving
    _patients_by_cat_hour = (
        _meds_7d_filled
        .groupby(['med_category', 'tx_hour'], as_index=False)
        .agg(
            n_patients=('hospitalization_id', 'nunique'),
            n_receiving=('med_dose', lambda x: (x > 0).sum())
        )
    )

    # Combine into single dataframe
    hourly_meds_summary = _median_dose_by_cat_hour.merge(
        _patients_by_cat_hour,
        on=['med_category', 'tx_hour'],
        how='outer'
    )
    hourly_meds_summary
    return hourly_meds_summary, meds_48hr


@app.cell
def _(alt, hourly_meds_summary):
    # Create separate dose and n_receiving charts for each med_category
    hourly_med_dose_charts = {}
    hourly_med_npatients_charts = {}
    for _med_cat in hourly_meds_summary['med_category'].unique():
        _med_data = hourly_meds_summary[hourly_meds_summary['med_category'] == _med_cat]

        # Chart for median dose
        _dose_chart = alt.Chart(_med_data).mark_line(point=True).encode(
            x=alt.X('tx_hour:Q', title='Hours Post-Transplant'),
            y=alt.Y('med_dose:Q', title='Median Dose'),
            tooltip=['tx_hour', 'med_dose', 'n_receiving']
        ).properties(
            title=f'{_med_cat.capitalize()} - Median Hourly Dose (Days 1-7 Post-Transplant)',
            width=500,
            height=300
        )
        hourly_med_dose_charts[_med_cat] = _dose_chart

        # Chart for n_receiving
        _npatients_chart = alt.Chart(_med_data).mark_line(point=True, color='coral').encode(
            x=alt.X('tx_hour:Q', title='Hours Post-Transplant'),
            y=alt.Y('n_receiving:Q', title='N Patients'),
            tooltip=['tx_hour', 'med_dose', 'n_receiving']
        ).properties(
            title=f'{_med_cat.capitalize()} - N Patients by Hour (Days 1-7 Post-Transplant)',
            width=500,
            height=300
        )
        hourly_med_npatients_charts[_med_cat] = _npatients_chart
    return hourly_med_dose_charts, hourly_med_npatients_charts


@app.cell
def _(alt, pd, prednisone_merged):
    # Filter to 21 days post-transplant (day 1 to day 21)
    prednisone_21d = prednisone_merged[
        (prednisone_merged['days_from_tx'] >= 1) &
        (prednisone_merged['days_from_tx'] <= 21)
    ].copy()

    # Sum doses per patient per day (for days they received it)
    daily_dose_per_patient = (
        prednisone_21d
        .groupby(['hospitalization_id', 'days_from_tx'], as_index=False)['med_dose']
        .sum()
        .rename(columns={'med_dose': 'daily_dose'})
    )

    # Find first and last day for each patient
    _patient_day_bounds = (
        daily_dose_per_patient
        .groupby('hospitalization_id', as_index=False)
        .agg(first_day=('days_from_tx', 'min'), last_day=('days_from_tx', 'max'))
    )

    # Create grid: each patient × days from first_day to 21
    # This ensures patients who stopped contribute 0 after their last dose
    _grid_rows = []
    for _, _row in _patient_day_bounds.iterrows():
        _hosp_id = _row['hospitalization_id']
        for _day in range(int(_row['first_day']), 22):  # days up to 21
            _grid_rows.append({'hospitalization_id': _hosp_id, 'days_from_tx': _day})

    _patient_day_grid = pd.DataFrame(_grid_rows)

    # Join actual doses to grid
    daily_dose_filled = _patient_day_grid.merge(
        daily_dose_per_patient,
        on=['hospitalization_id', 'days_from_tx'],
        how='left'
    )

    # Fill missing with 0 (patient had prednisone before but not this day)
    daily_dose_filled['daily_dose'] = daily_dose_filled['daily_dose'].fillna(0)

    # Average across patients (now includes zeros for stopped patients)
    avg_daily_prednisone = (
        daily_dose_filled
        .groupby('days_from_tx', as_index=False)
        .agg(
            avg_dose=('daily_dose', 'mean'),
            n_patients=('hospitalization_id', 'nunique'),
            n_receiving=('daily_dose', lambda x: (x > 0).sum())
        )
    )

    # Plot average daily prednisone dose
    prednisone_chart = alt.Chart(avg_daily_prednisone).mark_line(point=True).encode(
        x=alt.X('days_from_tx:Q', title='Days Post-Transplant'),
        y=alt.Y('avg_dose:Q', title='Average Daily Dose (mg)'),
        tooltip=['days_from_tx', 'avg_dose', 'n_patients']
    ).properties(
        title='Average Daily Prednisone Dose (21 Days Post-Transplant)',
        width=500,
        height=300
    )
    return avg_daily_prednisone, prednisone_chart


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Exports
    """)
    return


@app.cell
def _(
    avg_daily_prednisone,
    hourly_med_dose_charts,
    hourly_med_npatients_charts,
    hourly_meds_summary,
    logger,
    prednisone_chart,
    project_root,
    site_name,
    table_one_df,
):
    output_dir = project_root / 'output' / 'final'
    figures_dir = output_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    meds_file = output_dir / f'{site_name}_hourly_meds_summary.csv'
    hourly_meds_summary.to_csv(meds_file, index=False)
    logger.info(f"Saved hourly medication summary to {meds_file}")

    t1_file = output_dir / f'{site_name}_tableone.csv'
    table_one_df.to_csv(t1_file, index=False)
    logger.info(f"Saved tableone to {t1_file}")

    # Save prednisone daily chart
    prednisone_file = figures_dir / f'{site_name}_prednisone_daily.png'
    prednisone_chart.save(str(prednisone_file))
    logger.info(f"Saved prednisone daily chart to {prednisone_file}")

    # Save prednisone daily summary CSV
    prednisone_csv_file = output_dir / f'{site_name}_prednisone_daily_summary.csv'
    avg_daily_prednisone.to_csv(prednisone_csv_file, index=False)
    logger.info(f"Saved prednisone daily summary to {prednisone_csv_file}")

    # Save hourly medication dose charts
    for _med_cat, _chart in hourly_med_dose_charts.items():
        chart_file = figures_dir / f'{site_name}_{_med_cat}_hourly_dose.png'
        _chart.save(str(chart_file))
        logger.info(f"Saved {_med_cat} hourly dose chart to {chart_file}")

    # Save hourly medication n_patients charts
    for _med_cat, _chart in hourly_med_npatients_charts.items():
        chart_file = figures_dir / f'{site_name}_{_med_cat}_hourly_npatients.png'
        _chart.save(str(chart_file))
        logger.info(f"Saved {_med_cat} hourly n_patients chart to {chart_file}")
    return


if __name__ == "__main__":
    app.run()
