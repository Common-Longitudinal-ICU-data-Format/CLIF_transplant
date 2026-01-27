import marimo

__generated_with = "0.19.4"
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
        Hospitalization, PatientProcedures
    )

    # Heart transplant CPT codes
    HEART_TRANSPLANT_CPTS = ['33945', '33935']
    return (
        HEART_TRANSPLANT_CPTS,
        Hospitalization,
        Path,
        PatientProcedures,
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

    # File handler for main log
    main_log_path = log_dir / 'cohort_identification.log'
    file_handler = logging.FileHandler(main_log_path)
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
    return log_memory, logger, project_root


@app.cell
def _(
    HEART_TRANSPLANT_CPTS,
    Hospitalization,
    Path,
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
            columns=['patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm']
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
    return config, hosp_table, proc_table, site_name


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md("""
    # Cohort Identification and Registry Comparison (C)
    """)
    return


@app.cell
def _(Path, log_memory, logger, pd):
    log_memory("Before loading review_2.csv")
    clif_tx_patientids = Path(__file__).resolve().parent.parent / 'review_hr_tx_ids_final.csv'
    clif_tx_patientids = pd.read_csv(clif_tx_patientids)
    cols = ['proc_patient_id', 'enc_hospitalization_id', 'proc_hospitalization_id']
    clif_tx_patientids[cols] = clif_tx_patientids[cols].astype(str)
    clif_HEART_tx_patientids = clif_tx_patientids['proc_patient_id'].nunique()
    logger.info(f"Unique patient IDs in review_2.csv: {clif_HEART_tx_patientids}")
    clif_tx_patientids = clif_tx_patientids.drop_duplicates()
    log_memory("After loading review_2.csv")
    return (clif_tx_patientids,)


@app.cell
def _(
    HEART_TRANSPLANT_CPTS,
    clif_tx_patientids,
    log_memory,
    logger,
    proc_table,
):
    log_memory("Before filtering procedures (C)")
    _heart_procedures_df_c = proc_table.df[
        proc_table.df['procedure_code'].astype(str).isin(HEART_TRANSPLANT_CPTS)
        & proc_table.df['procedure_code_format'].astype(str).str.lower().eq('cpt')
    ].copy()

    # Deduplicate by code, hospitalization_id, and procedure_billed_dttm
    _before_c = len(_heart_procedures_df_c)
    _heart_procedures_df_c = _heart_procedures_df_c.drop_duplicates(
        subset=['procedure_code', 'hospitalization_id', 'procedure_billed_dttm']
    )
    _heart_procedures_df_c = _heart_procedures_df_c.rename(
        columns={'procedure_billed_dttm': 'apprx_transplant_date'}
    )
    logger.info(f"[C] Unique heart transplant hospitalization procedures found: {_heart_procedures_df_c['hospitalization_id'].nunique():,} (removed {_before_c - _heart_procedures_df_c['hospitalization_id'].nunique():,} duplicates)")
    heart_procedures_w_patientid = (clif_tx_patientids.merge(_heart_procedures_df_c, left_on = 'proc_hospitalization_id', right_on='hospitalization_id', how = 'inner').drop(columns=['hospitalization_id']))
    logger.info(f"[C] Unique patient IDs after merge: {heart_procedures_w_patientid['proc_patient_id'].nunique()}")
    heart_procedures_w_patientid = heart_procedures_w_patientid.drop_duplicates(['proc_patient_id', 'enc_hospitalization_id', 'proc_hospitalization_id', 'apprx_transplant_date']).copy()
    log_memory("After filtering procedures (C)")
    return (heart_procedures_w_patientid,)


@app.cell
def _(heart_procedures_w_patientid, hosp_table, log_memory, logger):
    log_memory("Before merge with hospitalization (C)")
    hosp_for_heart_tx = heart_procedures_w_patientid.merge(hosp_table.df[["patient_id", "hospitalization_id", "admission_dttm", "discharge_dttm"]], right_on = ["patient_id", "hospitalization_id"], left_on=["proc_patient_id", "enc_hospitalization_id"], how = "inner").drop(columns=["patient_id", "hospitalization_id"])
    logger.info(f"[C] Unique patient IDs after hospitalization merge: {hosp_for_heart_tx['proc_patient_id'].nunique()}")
    log_memory("After merge with hospitalization (C)")
    return (hosp_for_heart_tx,)


@app.cell
def _(hosp_for_heart_tx, logger):
    # Filter to hospitalizations where transplant date falls within admission-discharge window
    date_matches_c = hosp_for_heart_tx[
        (hosp_for_heart_tx['apprx_transplant_date'].dt.date >= hosp_for_heart_tx['admission_dttm'].dt.date) &
        (hosp_for_heart_tx['apprx_transplant_date'].dt.date <= hosp_for_heart_tx['discharge_dttm'].dt.date)
    ]

    verify = date_matches_c.drop_duplicates(['proc_patient_id', 'enc_hospitalization_id', 'apprx_transplant_date']).copy()
    final_df = verify.drop_duplicates(["proc_hospitalization_id"])
    final_df['proc_patient_id'] = final_df['proc_patient_id'].astype('string')
    final_df = final_df.rename(columns={
        'proc_patient_id': 'patient_id',
        'enc_hospitalization_id': 'hospitalization_id'
    })


    logger.info(f"[C] Final unique patient IDs: {final_df['patient_id'].nunique()}")
    logger.info(f"[C] Final unique enc_hospitalization_ids: {final_df['hospitalization_id'].nunique()}")
    logger.info(f"[C] Final unique proc_hospitalization_ids: {final_df['proc_hospitalization_id'].nunique()}")
    return (final_df,)


@app.cell
def _(final_df, hosp_for_heart_tx):
    patient_not_in_final = hosp_for_heart_tx[
        ~hosp_for_heart_tx['proc_patient_id'].isin(final_df['patient_id'])
    ]
    patient_not_in_final['proc_patient_id'].unique()
    return


@app.cell
def _(final_df, hosp_for_heart_tx):
    final_df.groupby('patient_id')['proc_hospitalization_id'].nunique().gt(1).sum()
    patients_multi_hosp = (
        final_df
        .groupby('patient_id')['proc_hospitalization_id']
        .nunique()
        .reset_index(name='n_hospitalizations')
        .query('n_hospitalizations > 1')
    )
    multi_ids = patients_multi_hosp['patient_id']

    hosp_for_heart_tx_multi = hosp_for_heart_tx[
        hosp_for_heart_tx['proc_patient_id'].isin(multi_ids)
    ]
    hosp_for_heart_tx_multi = hosp_for_heart_tx_multi[['proc_patient_id', 'enc_hospitalization_id', 'apprx_transplant_date']].drop_duplicates()
    return


@app.cell
def _(final_df, pd):
    final_df['tx_year'] = (
        pd.to_datetime(
            final_df['apprx_transplant_date'],
            errors='coerce'
        )
        .dt.year
    )

    tx_patients_by_year_c = (
        final_df
        .dropna(subset=['tx_year'])
        .sort_values('apprx_transplant_date')
        .drop_duplicates('patient_id')
        .groupby('tx_year')['patient_id']
        .nunique()
        .rename('n_patients')
        .reset_index()
    )
    return (tx_patients_by_year_c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Registry Comparison (C)
    """)
    return


@app.cell(hide_code=True)
def _(Path, config, pd, tx_patients_by_year_c):
    # Load national registry data filtered by site_name
    _registry_path_c = Path(__file__).resolve().parent.parent / 'public' / 'data' / 'clif_hr_tx_counts.csv'
    _registry_df_c = pd.read_csv(_registry_path_c)

    _site_name_c = config['site_name']
    registry_ucmc_df_c = _registry_df_c[
        (_registry_df_c['clif_site'] == _site_name_c) &
        (_registry_df_c['ORG_TY'] == 'HR') &
        (_registry_df_c['peds'] == 0)

    ].copy()
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
    # Export Cohort to csv
    """)
    return


@app.cell(hide_code=True)
def _(final_df, logger, project_root, registry_comparison, site_name):
    output_dir = project_root / 'output' / 'final'
    interm_dir = project_root / 'output' / 'intermediate'
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort_file = interm_dir / f'{site_name}_cohort.csv'
    final_df[['patient_id', 'hospitalization_id']].to_csv(cohort_file, index=False)
    logger.info(f"Saved cohort ids to {cohort_file}")

    registry_file = output_dir / f'{site_name}_aggregate_registry_comp.csv'
    registry_comparison.to_csv(registry_file, index=False)
    logger.info(f"Saved registry comparison by year to {registry_file}")
    return


if __name__ == "__main__":
    app.run()
