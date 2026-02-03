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

    # Import clifpy table classes
    from clifpy.tables import (
        Hospitalization, PatientProcedures,
        Labs, MedicationAdminContinuous, MedicationAdminIntermittent,
        Patient, Adt, Vitals
    )
    from clifpy.utils.unit_converter import convert_dose_units_by_med_category

    # Heart transplant CPT codes
    HEART_TRANSPLANT_CPTS = ['33945', '33935']
    return (
        Adt,
        HEART_TRANSPLANT_CPTS,
        Hospitalization,
        Labs,
        MedicationAdminContinuous,
        MedicationAdminIntermittent,
        Path,
        Patient,
        PatientProcedures,
        Vitals,
        alt,
        convert_dose_units_by_med_category,
        load_config,
        mo,
        pd,
    )


@app.cell
def _(Path):
    import logging
    import psutil
    import os

    # Create logs directory under output/final/logs
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / 'output' / 'final' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Main logger
    logger = logging.getLogger('cohort_identification')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler for main log (mode='w' to overwrite on each run)
    main_log_path = log_dir / 'cohort_identification.log'
    file_handler = logging.FileHandler(main_log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Memory logger (separate file)
    mem_logger = logging.getLogger('memory')
    mem_logger.setLevel(logging.INFO)
    mem_logger.handlers.clear()

    mem_log_path = log_dir / 'memory_usage.log'
    mem_handler = logging.FileHandler(mem_log_path, mode='w')
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
    return log_memory, logger, project_root


@app.cell
def _(
    Adt,
    HEART_TRANSPLANT_CPTS,
    Hospitalization,
    Labs,
    MedicationAdminContinuous,
    MedicationAdminIntermittent,
    Path,
    Patient,
    PatientProcedures,
    load_config,
    log_memory,
    logger,
):
    # Setup paths and load core tables using clifpy
    config = load_config()
    tables_path = Path(config['tables_path'])
    site_name = config['site_name']
    file_type = config['file_type']
    time_zone = config['time_zone']

    log_memory("Before loading core tables")

    try:
        hosp_table = Hospitalization.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone=time_zone,
            columns=['patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm', 'admission_type_category', 'discharge_category']
        )
        logger.info(f"Loaded hospitalization: {hosp_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        log_memory("After loading hospitalization table")
    except Exception as e:
        logger.error(f"Error loading hospitalization table: {e}")


    try:
        proc_table = PatientProcedures.from_file(
        data_directory=str(tables_path),
        filetype=file_type,
        timezone=time_zone,
        filters={'procedure_code': HEART_TRANSPLANT_CPTS}
        )
        logger.info(f"Loaded procedures: {proc_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        log_memory("After loading procedures table")
    except Exception as e:
        logger.error(f"Error loading procedures table: {e}")


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
            columns=['hospitalization_id', 'lab_result_dttm', 'lab_order_category', 'lab_category', 'lab_value_numeric']
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
            filters = {'med_category':['methylprednisolone']}
        )
        logger.info(f"Loaded meds intermtable: {meds_interm_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        # Convert med_dose to numeric (handles string values in source data)
        meds_interm_table.df["med_dose"] = pd.to_numeric(meds_interm_table.df["med_dose"], errors="coerce")
        log_memory("After loading meds interm table table")
    except Exception as e:
        logger.error(f"Error loading meds interm table table: {e}")

    try:
        meds_table = MedicationAdminContinuous.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone=time_zone,
            filters={'med_category': ['dobutamine', 'milrinone', 'dopamine', 'epinephrine', 'norepinephrine', 'isoproterenol', 'nitric_oxide']},
            columns=['hospitalization_id', 'admin_dttm', 'med_category', 'med_dose', 'med_dose_unit']
        )
        logger.info(f"Loaded meds: {meds_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        # Convert med_dose to numeric (handles string values in source data)
        meds_table.df["med_dose"] = pd.to_numeric(meds_table.df["med_dose"], errors="coerce")
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
    return (
        adt_table,
        config,
        hosp_table,
        labs_table,
        meds_interm_table,
        meds_table,
        patient_table,
        proc_table,
        site_name,
    )


@app.cell
def _(
    Path,
    Vitals,
    config,
    convert_dose_units_by_med_category,
    log_memory,
    logger,
    meds_interm_table,
    meds_table,
):
    # Load vitals table for patient weights (needed for weight-based unit conversion)
    log_memory("Before loading vitals for weight")
    try:
        _vitals_table = Vitals.from_file(
            data_directory=str(Path(config['tables_path'])),
            filetype=config['file_type'],
            timezone=config['time_zone'],
            filters={'vital_category': ['weight_kg']},
            columns=['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value']
        )
        logger.info(f"Loaded vitals (weight_kg): {_vitals_table.df['hospitalization_id'].nunique():,} unique hospitalizations")
        log_memory("After loading vitals for weight")
    except Exception as e:
        logger.error(f"Error loading vitals table: {e}")
        _vitals_table = None

    # Define preferred units for continuous medications (vasoactives/inotropes)
    _preferred_units_cont = {
        "norepinephrine": "mcg/kg/min",
        "epinephrine": "mcg/kg/min",
        "dopamine": "mcg/kg/min",
        "dobutamine": "mcg/kg/min",
        "isoproterenol":"mcg/min",
        "milrinone": "mcg/kg/min"
    }

    # Define preferred units for intermittent medications
    _preferred_units_interm = {
        "methylprednisolone": "mg"
    }

    # Convert continuous medication units
    log_memory("Before continuous meds unit conversion")
    try:
        meds_table.df, _meds_cont_counts = convert_dose_units_by_med_category(
            meds_table.df,
            vitals_df=_vitals_table.df if _vitals_table else None,
            preferred_units=_preferred_units_cont,
            override=True  # Continue even if some units can't be converted
        )
        logger.info(f"Converted continuous medication units. Conversion summary:")
        logger.info(f"\n{_meds_cont_counts.to_string()}")
        # Handle nitric_oxide specially: ppm is already the target unit, no conversion needed
        _nitric_ppm_mask = (meds_table.df['med_category'] == 'nitric_oxide') & (meds_table.df['med_dose_unit'] == 'ppm')
        meds_table.df.loc[_nitric_ppm_mask, '_convert_status'] = 'success'
        meds_table.df.loc[_nitric_ppm_mask, 'med_dose_converted'] = meds_table.df.loc[_nitric_ppm_mask, 'med_dose']
        logger.info(f"Marked {_nitric_ppm_mask.sum()} nitric_oxide ppm records as successful (no conversion needed)")

        # Log conversion status breakdown by medication category
        _status_counts = meds_table.df.groupby(['med_category', '_convert_status']).size().reset_index(name='count')
        logger.info(f"Unit conversion status by medication category:\n{_status_counts.to_string()}")
        log_memory("After continuous meds unit conversion")
    except Exception as e:
        logger.error(f"Error converting continuous medication units: {e}")

    # Convert intermittent medication units
    log_memory("Before intermittent meds unit conversion")
    try:
        meds_interm_table.df, _meds_interm_counts = convert_dose_units_by_med_category(
            meds_interm_table.df,
            vitals_df=_vitals_table.df if _vitals_table else None,
            preferred_units=_preferred_units_interm,
            override=True
        )
        logger.info(f"Converted intermittent medication units. Conversion summary:")
        logger.info(f"\n{_meds_interm_counts.to_string()}")
        log_memory("After intermittent meds unit conversion")
    except Exception as e:
        logger.error(f"Error converting intermittent medication units: {e}")
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md("""
    # Cohort Identification and Registry Comparison
    """)
    return


@app.cell
def _(HEART_TRANSPLANT_CPTS, log_memory, logger, proc_table):
    log_memory("Before filtering procedures")
    _heart_procedures_df = proc_table.df[
        proc_table.df['procedure_code'].astype(str).isin(HEART_TRANSPLANT_CPTS)
        & proc_table.df['procedure_code_format'].astype(str).str.lower().eq('cpt')
    ].copy()

    # Deduplicate by code, hospitalization_id, and procedure_billed_dttm
    _before = len(_heart_procedures_df)
    clif_tx_hospids  = _heart_procedures_df.drop_duplicates(
        subset=['procedure_code', 'hospitalization_id', 'procedure_billed_dttm']
    )

    logger.info(f"Unique heart transplant hospitalization procedures found: {_heart_procedures_df['hospitalization_id'].nunique():,} (removed {_before - _heart_procedures_df['hospitalization_id'].nunique():,} duplicates)")
    log_memory("After filtering procedures")
    clif_tx_hospids
    return (clif_tx_hospids,)


@app.cell
def _(clif_tx_hospids, log_memory, logger, meds_interm_table, pd):
    # Step 2a: Filter to Methylprednisolone 1g dose as proxy for transplant cross-clamp time
    log_memory("Before filtering methylprednisolone")

    # Get methylprednisolone medications (already filtered in table load)
    methylpred_steroids = meds_interm_table.df[
        meds_interm_table.df["med_category"] == "methylprednisolone"
    ]

    # Step 2a: Filter methylprednisolone by dose thresholds
    # Primary: >500mg, Fallback: >100mg
    # Note: Use case-insensitive comparison for med_dose_unit (some sites use "MG" vs "mg")
    xclamp_methylpred_gt500 = methylpred_steroids[
        (methylpred_steroids["med_dose"] > 500) &
        (methylpred_steroids["med_dose_unit"].str.lower() == "mg")
    ].copy()

    xclamp_methylpred_gt100 = methylpred_steroids[
        (methylpred_steroids["med_dose"] > 100) &
        (methylpred_steroids["med_dose_unit"].str.lower() == "mg")
    ].copy()

    # Get first >500mg dose per hospitalization
    xclamp_first_gt500 = (
        xclamp_methylpred_gt500
        .sort_values("admin_dttm")
        .groupby("hospitalization_id")
        .first()
        .reset_index()[["hospitalization_id", "admin_dttm"]]
        .rename(columns={"admin_dttm": "transplant_cross_clamp"})
    )

    # Get first >100mg dose per hospitalization (fallback)
    xclamp_first_gt100 = (
        xclamp_methylpred_gt100
        .sort_values("admin_dttm")
        .groupby("hospitalization_id")
        .first()
        .reset_index()[["hospitalization_id", "admin_dttm"]]
        .rename(columns={"admin_dttm": "transplant_cross_clamp"})
    )

    # Merge: prefer >500mg, fallback to >100mg
    transplant_cross_clamp_times = xclamp_first_gt500.copy()
    # Add hospitalizations that only have >100mg (not in >500mg)
    xclamp_fallback_hosps = xclamp_first_gt100[~xclamp_first_gt100["hospitalization_id"].isin(xclamp_first_gt500["hospitalization_id"])]
    transplant_cross_clamp_times = pd.concat([transplant_cross_clamp_times, xclamp_fallback_hosps], ignore_index=True)

    # Log counts
    n_xclamp_gt500 = xclamp_first_gt500["hospitalization_id"].nunique()
    n_xclamp_fallback = xclamp_fallback_hosps["hospitalization_id"].nunique()
    logger.info(f"Transplant cross-clamp times: {n_xclamp_gt500} using >500mg, {n_xclamp_fallback} fallback to >100mg, {n_xclamp_gt500 + n_xclamp_fallback} total")
    log_memory("After filtering methylprednisolone")
    clif_tx_patientids_xclamp = clif_tx_hospids.merge(transplant_cross_clamp_times, on='hospitalization_id', how='inner')
    clif_tx_patientids_xclamp
    return (clif_tx_patientids_xclamp,)


@app.cell
def _(clif_tx_patientids_xclamp, logger, meds_interm_table, pd):
    # Calculate post_transplant_ICU_in_dttm based on methylprednisolone doses

    # Get all methylprednisolone doses in mg (case-insensitive)
    methylpred_all = meds_interm_table.df[
        (meds_interm_table.df["med_category"] == "methylprednisolone") &
        (meds_interm_table.df["med_dose_unit"].str.lower() == "mg")
    ].copy()

    # Option 1: First dose > 500mg, then add 12 hours
    methylpred_gt500 = methylpred_all[methylpred_all["med_dose"] > 500]
    first_gt500 = (
        methylpred_gt500
        .sort_values("admin_dttm")
        .groupby("hospitalization_id")
        .first()
        .reset_index()[["hospitalization_id", "admin_dttm"]]
    )
    first_gt500["post_transplant_ICU_in_dttm"] = first_gt500["admin_dttm"] + pd.Timedelta(hours=12)
    first_gt500 = first_gt500[["hospitalization_id", "post_transplant_ICU_in_dttm"]]

    # Option 2 (fallback): First dose > 100mg, use admin_dttm directly
    methylpred_gt100 = methylpred_all[methylpred_all["med_dose"] > 100]
    first_gt100 = (
        methylpred_gt100
        .sort_values("admin_dttm")
        .groupby("hospitalization_id")
        .first()
        .reset_index()[["hospitalization_id", "admin_dttm"]]
        .rename(columns={"admin_dttm": "post_transplant_ICU_in_dttm"})
    )

    # Merge: prefer >500mg rule, fallback to >100mg rule
    transplant_cohort = clif_tx_patientids_xclamp.copy()
    transplant_cohort = transplant_cohort.merge(first_gt500, on="hospitalization_id", how="left")
    transplant_cohort = transplant_cohort.merge(
        first_gt100, on="hospitalization_id", how="left", suffixes=("", "_fallback")
    )
    transplant_cohort["post_transplant_ICU_in_dttm"] = transplant_cohort[
        "post_transplant_ICU_in_dttm"
    ].fillna(transplant_cohort["post_transplant_ICU_in_dttm_fallback"])
    transplant_cohort = transplant_cohort.drop(columns=["post_transplant_ICU_in_dttm_fallback"])

    logger.info(f"Transplant cohort: {transplant_cohort['hospitalization_id'].nunique()} hospitalizations")
    logger.info(f"  - Using >500mg + 12hr rule: {first_gt500['hospitalization_id'].nunique()}")
    logger.info(f"  - Using >100mg fallback: {(transplant_cohort['post_transplant_ICU_in_dttm'].notna() & ~transplant_cohort['hospitalization_id'].isin(first_gt500['hospitalization_id'])).sum()}")
    return (transplant_cohort,)


@app.cell
def _(hosp_table, transplant_cohort):
    w_patientid = transplant_cohort.merge(hosp_table.df, on='hospitalization_id', how = 'inner')
    w_patientid = w_patientid.drop_duplicates(['patient_id', 'hospitalization_id']).copy()
    w_patientid
    return (w_patientid,)


@app.cell
def _(w_patientid):
    # Filter to hospitalizations where transplant date falls within admission-discharge window
    date_matches_c = w_patientid[
        (w_patientid['transplant_cross_clamp'] >= w_patientid['admission_dttm']) &
        (w_patientid['transplant_cross_clamp'] <= w_patientid['discharge_dttm'])
    ]
    date_matches_c
    return (date_matches_c,)


@app.cell
def _(date_matches_c, pd):
    final_df = date_matches_c
    final_df['tx_year'] = (
        pd.to_datetime(
            final_df['transplant_cross_clamp'],
            errors='coerce'
        )
        .dt.year
    )

    tx_patients_by_year_c = (
        final_df
        .dropna(subset=['tx_year'])
        .sort_values('transplant_cross_clamp')
        .drop_duplicates('patient_id')
        .groupby('tx_year')['patient_id']
        .nunique()
        .rename('n_patients')
        .reset_index()
    )
    return final_df, tx_patients_by_year_c


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Registry Comparison
    """)
    return


@app.cell(hide_code=True)
def _(Path, logger, pd, site_name, tx_patients_by_year_c):
    # Load national registry data filtered by site_name
    _registry_path_c = Path(__file__).resolve().parent.parent / 'public' / 'data' / 'clif_hr_tx_counts.csv'
    _registry_df_c = pd.read_csv(_registry_path_c)

    # Check if site_name exists in registry (case-insensitive)
    _valid_sites = _registry_df_c['clif_site'].str.lower().unique()
    _site_found = site_name.lower() in _valid_sites

    if not _site_found:
        logger.warning(f"Site '{site_name}' not found in registry. Valid sites: {sorted(_registry_df_c['clif_site'].unique())}")
        logger.warning("Saving CLIF yearly aggregates only (no registry comparison)")
        # Fallback: just save CLIF yearly counts without registry merge
        registry_comparison = tx_patients_by_year_c.rename(
            columns={'n_patients': 'clif_hx_transplants'}
        ).copy()
    else:
        registry_ucmc_df_c = _registry_df_c[
            (_registry_df_c['clif_site'].str.lower() == site_name.lower()) &
            (_registry_df_c['ORG_TY'] == 'HR') &
            (_registry_df_c['peds'] == 0)
        ].copy()

        # Check for year overlap between CLIF data and registry
        _clif_years = set(tx_patients_by_year_c['tx_year'].dropna().astype(int))
        _registry_years = set(registry_ucmc_df_c['year'].unique()) if not registry_ucmc_df_c.empty else set()
        _overlapping_years = _clif_years.intersection(_registry_years)

        if not _overlapping_years:
            logger.warning(f"No overlapping years between CLIF data and registry. CLIF years: {sorted(_clif_years)}, Registry years (2018-2024): {sorted(_registry_years)}")
            logger.warning("Saving CLIF yearly aggregates only (no registry comparison)")
            # Fallback: just save CLIF yearly counts without registry merge
            registry_comparison = tx_patients_by_year_c.rename(
                columns={'n_patients': 'clif_hx_transplants'}
            ).copy()
        else:
            _missing_clif_years = _clif_years - _registry_years
            if _missing_clif_years:
                logger.warning(f"Some CLIF years not in registry (will be excluded from comparison): {sorted(_missing_clif_years)}")

            compare_c = tx_patients_by_year_c.merge(registry_ucmc_df_c[['year', 'transplants']], right_on = 'year', left_on = 'tx_year', how = 'inner')
            compare_c = compare_c.rename(
                columns={'transplants': 'srtr_hx_transplants',
                        'n_patients': 'clif_hx_transplants'}
            )
            registry_comparison = compare_c.drop(columns = 'year')

    registry_comparison
    return (registry_comparison,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Table One
    """)
    return


@app.cell
def _(
    HEART_TRANSPLANT_HOSPITALIZATIONS,
    adt_table,
    hosp_table,
    labs_table,
    meds_24hr,
    meds_interm_table,
    meds_table,
    patient_table,
    pd,
):
    n_total = HEART_TRANSPLANT_HOSPITALIZATIONS['patient_id'].nunique()

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
    n_male = (pt_df['sex_category'].str.lower() == 'male').sum()
    rows.append({"Characteristic": "  Male", "Value": n_pct(n_male, n_patients)})

    # Race - show Asian, Black, White, group rest as Other
    rows.append({"Characteristic": "Race", "Value": ""})
    for race in ['Asian', 'Black or African American', 'White']:
        n_race = (pt_df['race_category'].str.lower() == race.lower()).sum()
        rows.append({"Characteristic": f"  {race}", "Value": n_pct(n_race, n_patients)})
    n_other_race = (~pt_df['race_category'].str.lower().isin(['asian', 'black or african american', 'white'])).sum()
    rows.append({"Characteristic": "  Other", "Value": n_pct(n_other_race, n_patients)})

    # Ethnicity - only show Hispanic
    rows.append({"Characteristic": "Ethnicity", "Value": ""})
    n_hispanic = (pt_df['ethnicity_category'].str.lower() == 'hispanic').sum()
    rows.append({"Characteristic": "  Hispanic", "Value": n_pct(n_hispanic, n_patients)})

    # Age at transplant
    cohort_with_dates = HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'transplant_cross_clamp']].drop_duplicates('patient_id')
    cohort_with_dates = cohort_with_dates.merge(pt_df[['patient_id', 'birth_date']], on='patient_id', how='left')
    # Handle timezone awareness for birth_date
    if cohort_with_dates['birth_date'].dt.tz is None:
        cohort_with_dates['birth_date'] = pd.to_datetime(cohort_with_dates['birth_date']).dt.tz_localize(config['time_zone'])
    age_at_tx = (cohort_with_dates['transplant_cross_clamp'] - cohort_with_dates['birth_date']).dt.days / 365.25
    rows.append({"Characteristic": "Age at transplant (years), median [IQR]", "Value": median_iqr(age_at_tx)})

    # Discharge disposition - use latest hospitalization per patient
    _latest_hosp_for_discharge = (
        HEART_TRANSPLANT_HOSPITALIZATIONS
        .sort_values('transplant_cross_clamp', ascending=False)
        .drop_duplicates('patient_id', keep='first')
    )
    hosp_df = hosp_table.df[hosp_table.df['hospitalization_id'].isin(_latest_hosp_for_discharge['hospitalization_id'])].copy()
    rows.append({"Characteristic": "Discharge disposition", "Value": ""})
    for val, cnt in hosp_df['discharge_category'].value_counts().items():
        rows.append({"Characteristic": f"  {val}", "Value": n_pct(cnt, n_total)})

    # Location at transplant cross-clamp time
    _latest_hosp_per_patient = (
        HEART_TRANSPLANT_HOSPITALIZATIONS
        .sort_values('transplant_cross_clamp', ascending=False)
        .drop_duplicates('patient_id', keep='first')
    )
    _adt_merged = adt_table.df.merge(
        _latest_hosp_per_patient[['hospitalization_id', 'transplant_cross_clamp']],
        on='hospitalization_id', how='inner'
    )
    _location_at_xclamp = _adt_merged[
        (_adt_merged['in_dttm'] <= _adt_merged['transplant_cross_clamp']) &
        (_adt_merged['transplant_cross_clamp'] <= _adt_merged['out_dttm'])
    ].drop_duplicates('hospitalization_id', keep='first')

    rows.append({"Characteristic": "Location at first high dose methylpred admin", "Value": ""})
    for loc, cnt in _location_at_xclamp['location_category'].value_counts().items():
        rows.append({"Characteristic": f"  {loc}", "Value": n_pct(cnt, n_total)})

    inotrope_list = ['dobutamine', 'dopamine', 'epinephrine', 'isoproterenol', 'milrinone', 'norepinephrine']

    # Filter to inotropes only
    inotrope_meds = meds_24hr[meds_24hr['med_category'].isin(inotrope_list)]

    # Count unique patients that received any inotrope
    n_any_inotrope = inotrope_meds['patient_id'].nunique()
    rows.append({"Characteristic": "Any inotrope post-op (24hr)", "Value": n_pct(n_any_inotrope, n_total)})

    # Count of inotropes given in first 24hr (by hospitalization)
    rows.append({"Characteristic": "Inotropes received post-op (24hr)", "Value": ""})
    for med in inotrope_list:
        n_received = inotrope_meds[inotrope_meds['med_category'] == med]['hospitalization_id'].nunique()
        rows.append({"Characteristic": f"  {med.capitalize()}", "Value": n_pct(n_received, n_any_inotrope)})

    # methylprednisolone after transplant
    # Merge methylprednisolone data with transplant cohort
    methylprednisolone_merged = meds_interm_table.df.merge(
        HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'hospitalization_id', 'post_transplant_ICU_in_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    # Calculate days from transplant
    methylprednisolone_merged['days_from_tx'] = (
        (methylprednisolone_merged['admin_dttm'] - methylprednisolone_merged['post_transplant_ICU_in_dttm'])
        .dt.total_seconds() / 86400
    ).astype(int)
    methylprednisolone_post_tx = methylprednisolone_merged[methylprednisolone_merged['admin_dttm'] > methylprednisolone_merged['post_transplant_ICU_in_dttm']]
    n_methylprednisolone = methylprednisolone_post_tx['patient_id'].nunique()
    rows.append({"Characteristic": "Methylprednisolone post-transplant", "Value": n_pct(n_methylprednisolone, n_patients)})

    # Nitric oxide after transplant
    _nitric_oxide_df = meds_table.df[meds_table.df['med_category'] == 'nitric_oxide']
    nitric_merged = _nitric_oxide_df.merge(
        HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'hospitalization_id', 'post_transplant_ICU_in_dttm']],
        on='hospitalization_id', how='inner'
    )
    _nitric_post_tx = nitric_merged[nitric_merged['admin_dttm'] > nitric_merged['post_transplant_ICU_in_dttm']]
    n_nitric = _nitric_post_tx['patient_id'].nunique()
    rows.append({"Characteristic": "Nitric oxide post-transplant", "Value": n_pct(n_nitric, n_patients)})

    # Pre-transplant labs: most recent value before transplant date
    labs_merged = labs_table.df.merge(
        HEART_TRANSPLANT_HOSPITALIZATIONS[['hospitalization_id', 'transplant_cross_clamp']],
        on='hospitalization_id', how='inner'
    )
    # Filter to labs BEFORE transplant
    pre_tx_labs = labs_merged[labs_merged['lab_result_dttm'] < labs_merged['transplant_cross_clamp']]

    # Get most recent lab per hospitalization per category
    pre_tx_labs = pre_tx_labs.sort_values('lab_result_dttm', ascending=False)
    most_recent_labs = pre_tx_labs.drop_duplicates(['hospitalization_id', 'lab_category'], keep='first')

    # Calculate median [IQR] for each lab category
    lab_categories = {'creatinine': 'Creatinine', 'bilirubin_total': 'Bilirubin',
                      'albumin': 'Albumin', 'sodium': 'Sodium'}
    for lab_cat, display_name in lab_categories.items():
        lab_values = most_recent_labs[most_recent_labs['lab_category'] == lab_cat]['lab_value_numeric']
        rows.append({"Characteristic": f"{display_name}, median [IQR]", "Value": median_iqr(lab_values)})

    table_one_df = pd.DataFrame(rows)
    table_one_df
    return methylprednisolone_merged, table_one_df


@app.cell
def _(mo):
    mo.md(r"""
    # Other Summaries
    """)
    return


@app.cell
def _(final_df, logger):
    # Define HEART_TRANSPLANT_HOSPITALIZATIONS first so other cells can use it
    # Filter to only include years 2018-2024 (registry comparison years)
    _n_before = final_df['patient_id'].nunique()
    HEART_TRANSPLANT_HOSPITALIZATIONS = final_df[
        (final_df['tx_year'] >= 2018) & (final_df['tx_year'] <= 2024)
    ].copy()
    _n_after = HEART_TRANSPLANT_HOSPITALIZATIONS['patient_id'].nunique()

    if _n_before != _n_after:
        logger.info(f"Filtered to years 2018-2024: {_n_before} -> {_n_after} patients ({_n_before - _n_after} excluded)")
    else:
        logger.info(f"All {_n_after} patients within years 2018-2024")

    return (HEART_TRANSPLANT_HOSPITALIZATIONS,)


@app.cell
def _(HEART_TRANSPLANT_HOSPITALIZATIONS, logger, meds_table, pd):
    # Step 1: Merge medications with transplant cohort
    _meds_merged = meds_table.df.merge(
        HEART_TRANSPLANT_HOSPITALIZATIONS[['patient_id', 'hospitalization_id', 'post_transplant_ICU_in_dttm']],
        on='hospitalization_id', how='left'
    )
    _meds_merged['hours_from_tx'] = (
        (_meds_merged['admin_dttm'] - _meds_merged['post_transplant_ICU_in_dttm'])
        .dt.total_seconds() / 3600
    )

    # Step 2: Filter to 24hr and 7-day windows
    meds_24hr = _meds_merged[
        (_meds_merged['hours_from_tx'] >= 0) & (_meds_merged['hours_from_tx'] <= 23)
    ].copy()
    meds_24hr['tx_hour'] = meds_24hr['hours_from_tx'].astype(int)

    _meds_7d = _meds_merged[
        (_meds_merged['hours_from_tx'] >= 0) & (_meds_merged['hours_from_tx'] <= 167)
    ].copy()
    _meds_7d['tx_hour'] = _meds_7d['hours_from_tx'].astype(int)

    # Step 3: Create Complete Skeleton Grid (ALL patients × ALL hours × ALL medications)
    _transplant_hosp_ids = HEART_TRANSPLANT_HOSPITALIZATIONS['hospitalization_id'].unique()
    _med_categories = ['dobutamine', 'milrinone', 'dopamine', 'epinephrine', 'norepinephrine', 'nitric_oxide', 'isoproterenol']

    _skeleton = pd.DataFrame([
        {"hospitalization_id": h, "tx_hour": hr, "med_category": med}
        for h in _transplant_hosp_ids
        for hr in range(168)  # Hours 0-167 (7 days)
        for med in _med_categories
    ])

    # Step 4: Filter to only successfully converted records, then aggregate
    _meds_7d_success = _meds_7d[_meds_7d['_convert_status'] == 'success'].copy()
    logger.info(f"Hourly summary: {len(_meds_7d)} total records, {len(_meds_7d_success)} with successful unit conversion")

    _meds_7d_agg = (
        _meds_7d_success
        .groupby(['hospitalization_id', 'med_category', 'tx_hour'], as_index=False)['med_dose_converted']
        .sum()
    )

    # Join actual doses to skeleton
    _meds_filled = _skeleton.merge(
        _meds_7d_agg,
        on=['hospitalization_id', 'med_category', 'tx_hour'],
        how='left'
    )

    # Step 5: Apply fill logic (Backfill + LOCF - matches GitHub notebook approach)
    # This ensures early hours before first observation get backfilled from first dose
    _meds_filled = _meds_filled.sort_values(['hospitalization_id', 'med_category', 'tx_hour'])

    # Step 5a: Backfill (fills early NAs from first observation)
    _meds_filled['med_dose_imputed'] = (
        _meds_filled
        .groupby(['hospitalization_id', 'med_category'])['med_dose_converted']
        .bfill()
    )

    # Step 5b: Forward fill (LOCF for any remaining gaps)
    _meds_filled['med_dose_converted'] = (
        _meds_filled
        .groupby(['hospitalization_id', 'med_category'])['med_dose_imputed']
        .ffill()
    )

    # Step 5c: Fill remaining NaN with 0 (patient never received this medication)
    _meds_filled['med_dose_converted'] = _meds_filled['med_dose_converted'].fillna(0.0)
    _meds_filled = _meds_filled.drop(columns=['med_dose_imputed'])

    # Step 6: Calculate Summary Statistics
    hourly_meds_summary = (
        _meds_filled
        .groupby(['med_category', 'tx_hour'], as_index=False)
        .agg(
            med_dose_median=('med_dose_converted', 'median'),
            med_dose_mean=('med_dose_converted', 'mean'),
            n_receiving=('med_dose_converted', lambda x: (x > 0).sum())
        )
    )
    hourly_meds_summary
    return hourly_meds_summary, meds_24hr


@app.cell
def _(alt, hourly_meds_summary, pd):
    # Unit mapping for each medication category (matches preferred_units from conversion)
    _med_units = {
        "norepinephrine": "mcg/kg/min",
        "epinephrine": "mcg/kg/min",
        "dopamine": "mcg/kg/min",
        "dobutamine": "mcg/kg/min",
        "milrinone": "mcg/kg/min",
        "isoproterenol": "mcg/min",
        "nitric_oxide": "ppm"
    }

    # Create separate dose and n_receiving charts for each med_category
    hourly_med_dose_charts = {}
    hourly_med_npatients_charts = {}
    for _med_cat in hourly_meds_summary['med_category'].unique():
        _med_data = hourly_meds_summary[hourly_meds_summary['med_category'] == _med_cat].copy()
        _unit = _med_units.get(_med_cat, "units")

        # Reshape data for proper legend (melt median and mean into single column)
        _med_data_long = pd.melt(
            _med_data,
            id_vars=['med_category', 'tx_hour', 'n_receiving'],
            value_vars=['med_dose_median', 'med_dose_mean'],
            var_name='measure',
            value_name='dose'
        )
        _med_data_long['measure'] = _med_data_long['measure'].map({
            'med_dose_median': 'Median',
            'med_dose_mean': 'Mean'
        })

        # Chart for median and mean dose with proper legend inside chart area
        _dose_chart = alt.Chart(_med_data_long).mark_line().encode(
            x=alt.X('tx_hour:Q', title='Hours Post-Transplant'),
            y=alt.Y('dose:Q', title=f'Dose ({_unit})'),
            color=alt.Color('measure:N', scale=alt.Scale(
                domain=['Median', 'Mean'],
                range=['steelblue', 'coral']
            ), legend=alt.Legend(
                title=None,
                orient='none',
                legendX=400,
                legendY=10,
                direction='horizontal',
                fillColor='white',
                strokeColor='gray',
                padding=5
            )),
            tooltip=['tx_hour', 'measure', 'dose', 'n_receiving']
        ).properties(
            title=f'{_med_cat.capitalize()} - Hourly Dose ({_unit})',
            width=500,
            height=300
        ).configure_axis(grid=False).configure_view(strokeWidth=0)
        hourly_med_dose_charts[_med_cat] = _dose_chart

        # Chart for n_receiving
        _npatients_chart = alt.Chart(_med_data).mark_line(color='coral').encode(
            x=alt.X('tx_hour:Q', title='Hours Post-Transplant'),
            y=alt.Y('n_receiving:Q', title=f'N Patients Receiving ({_unit})'),
            tooltip=['tx_hour', 'med_dose_median', 'med_dose_mean', 'n_receiving']
        ).properties(
            title=f'{_med_cat.capitalize()} - N Patients Receiving ({_unit})',
            width=500,
            height=300
        ).configure_axis(grid=False).configure_view(strokeWidth=0)
        hourly_med_npatients_charts[_med_cat] = _npatients_chart
    return hourly_med_dose_charts, hourly_med_npatients_charts


@app.cell
def _(alt, methylprednisolone_merged, pd):
    # Filter to 21 days post-transplant (day 1 to day 21)
    methylprednisolone_merged_21d = methylprednisolone_merged[
        (methylprednisolone_merged['days_from_tx'] >= 0) &
        (methylprednisolone_merged['days_from_tx'] <= 20)
    ].copy()
    # Filter to 21 days post-transplant (day 1 to day 21)
    methylprednisolone_merged_21d = methylprednisolone_merged[
        (methylprednisolone_merged['days_from_tx'] >= 0) &
        (methylprednisolone_merged['days_from_tx'] <= 20)
    ].copy()

    # Sum doses per patient per day (for days they received it)
    daily_dose_per_patient = (
        methylprednisolone_merged_21d
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
        for _day in range(int(_row['first_day']), 21):  # days up to 21
            _grid_rows.append({'hospitalization_id': _hosp_id, 'days_from_tx': _day})

    _patient_day_grid = pd.DataFrame(_grid_rows)

    # Join actual doses to grid
    daily_dose_filled = _patient_day_grid.merge(
        daily_dose_per_patient,
        on=['hospitalization_id', 'days_from_tx'],
        how='left'
    )

    # Fill missing with 0 (patient had methylprednisolone before but not this day)
    daily_dose_filled['daily_dose'] = daily_dose_filled['daily_dose'].fillna(0)

    # Average across patients (now includes zeros for stopped patients)
    avg_daily_methylprednisolone = (
        daily_dose_filled
        .groupby('days_from_tx', as_index=False)
        .agg(
            avg_dose=('daily_dose', 'mean'),
            n_patients=('hospitalization_id', 'nunique'),
            n_receiving=('daily_dose', lambda x: (x > 0).sum())
        )
    )

    # Plot average daily methylprednisolone dose
    methylprednisolone_chart = alt.Chart(avg_daily_methylprednisolone).mark_line().encode(
        x=alt.X('days_from_tx:Q', title='Days Post-Transplant'),
        y=alt.Y('avg_dose:Q', title='Average Daily Dose (mg)'),
        tooltip=['days_from_tx', 'avg_dose', 'n_patients']
    ).properties(
        title='Average Daily Methylprednisolone Dose (21 Days Post Transplant ICU `in_dttm`)',
        width=500,
        height=300
    ).configure_axis(grid=False).configure_view(strokeWidth=0)
    # Sum doses per patient per day (for days they received it)
    daily_dose_per_patient = (
        methylprednisolone_merged_21d
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

    # Fill missing with 0 (patient had methylprednisolone before but not this day)
    daily_dose_filled['daily_dose'] = daily_dose_filled['daily_dose'].fillna(0)

    # Average across patients (now includes zeros for stopped patients)
    avg_daily_methylprednisolone = (
        daily_dose_filled
        .groupby('days_from_tx', as_index=False)
        .agg(
            avg_dose=('daily_dose', 'mean'),
            n_patients=('hospitalization_id', 'nunique'),
            n_receiving=('daily_dose', lambda x: (x > 0).sum())
        )
    )

    # Plot average daily methylprednisolone dose
    methylprednisolone_chart = alt.Chart(avg_daily_methylprednisolone).mark_line().encode(
        x=alt.X('days_from_tx:Q', title='Days Post-Transplant'),
        y=alt.Y('avg_dose:Q', title='Average Daily Dose (mg)'),
        tooltip=['days_from_tx', 'avg_dose', 'n_patients']
    ).properties(
        title='Average Daily Methylprednisolone Dose (21 Days Post-Transplant)',
        width=500,
        height=300
    ).configure_axis(grid=False).configure_view(strokeWidth=0)
    return avg_daily_methylprednisolone, methylprednisolone_chart


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # Exports
    """)
    return


@app.cell(hide_code=True)
def _(
    avg_daily_methylprednisolone,
    config,
    final_df,
    hourly_med_dose_charts,
    hourly_med_npatients_charts,
    hourly_meds_summary,
    logger,
    methylprednisolone_chart,
    project_root,
    registry_comparison,
    site_name,
    table_one_df,
):
    output_dir = project_root / 'output' / 'final'
    interm_dir = project_root / 'output' / 'intermediate'
    figures_dir = output_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Convert to site timezone before saving
    cohort_df = final_df[['patient_id', 'hospitalization_id', 'transplant_cross_clamp', 'post_transplant_ICU_in_dttm']].copy()
    cohort_df['transplant_cross_clamp'] = cohort_df['transplant_cross_clamp'].dt.tz_convert(config['time_zone'])

    cohort_file = interm_dir / f'{site_name}_cohort.csv'
    cohort_df.to_csv(cohort_file, index=False)
    logger.info(f"Saved cohort ids to {cohort_file}")

    meds_file = output_dir / f'{site_name}_hourly_meds_summary.csv'
    hourly_meds_summary.to_csv(meds_file, index=False)
    logger.info(f"Saved hourly medication summary to {meds_file}")

    registry_file = output_dir / f'{site_name}_aggregate_registry_comp.csv'
    registry_comparison.to_csv(registry_file, index=False)
    logger.info(f"Saved registry comparison by year to {registry_file}")

    t1_file = output_dir / f'{site_name}_tableone.csv'
    table_one_df.to_csv(t1_file, index=False)
    logger.info(f"Saved tableone to {t1_file}")

    # Save methylprednisolone_chart daily chart
    methylprednisolone_chart_file = figures_dir / f'{site_name}_methylprednisolone_chart_daily.png'
    methylprednisolone_chart.save(str(methylprednisolone_chart_file))
    logger.info(f"Saved methylprednisolone_chart daily chart to {methylprednisolone_chart_file}")

    # Save methylprednisolone daily summary CSV
    methylprednisolone_chart_csv_file = output_dir / f'{site_name}_methylprednisolone_chart_daily_summary.csv'
    avg_daily_methylprednisolone.to_csv(methylprednisolone_chart_csv_file, index=False)
    logger.info(f"Saved methylprednisolone_chart daily summary to {methylprednisolone_chart_csv_file}")

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
