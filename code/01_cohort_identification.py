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

    for old_log in log_dir.glob("*.log"):
        try:
            old_log.unlink()
            print(f"Deleted old log: {old_log}")  
        except Exception as e:
            print(f"Could not delete {old_log}: {e}")

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
    _before_b = len(_heart_procedures_df)
    heart_procedures_df = _heart_procedures_df.drop_duplicates(
        subset=['procedure_code', 'hospitalization_id', 'procedure_billed_dttm']
    )
    heart_procedures_df = heart_procedures_df.rename(
        columns={'procedure_billed_dttm': 'apprx_transplant_date'}
    )
    logger.info(f"Unique heart transplant hospitalization procedures found: {heart_procedures_df['hospitalization_id'].nunique():,} (removed {_before_b - _heart_procedures_df['hospitalization_id'].nunique():,} duplicates)")
    log_memory("After filtering procedures")
    return (heart_procedures_df,)


@app.cell
def _(heart_procedures_df, hosp_table, log_memory, logger):
    log_memory("Before merge procedures with hospitalizations")
    final_df = heart_procedures_df.merge(hosp_table.df[['patient_id', 'hospitalization_id']], on='hospitalization_id', how='inner')

    _heart_hosp_ids = set(heart_procedures_df['hospitalization_id'].unique())
    _heart_hosp_ids_merged = set(final_df['hospitalization_id'].unique())

    missing_hosp_ids = _heart_hosp_ids - _heart_hosp_ids_merged

    logger.info(
        f"Heart transplant hospitalization_ids from procedures NOT in hospitalization table: "
        f"{len(missing_hosp_ids):,}"
    )
    logger.info(
        f"Unique Heart transplant patient_ids: "
        f"{final_df['patient_id'].nunique():,}"
    )
    log_memory("After merge procedures with hospitalizations")
    return (final_df,)


@app.cell
def _(final_df, pd):
    final_df['tx_year'] = (
        pd.to_datetime(
            final_df['apprx_transplant_date'],
            errors='coerce'
        )
        .dt.year
    )

    tx_patients_by_year = (
        final_df
        .dropna(subset=['tx_year'])
        .sort_values('apprx_transplant_date')
        .drop_duplicates('patient_id')
        .groupby('tx_year')['patient_id']
        .nunique()
        .rename('n_patients')
        .reset_index()
    )
    return (tx_patients_by_year,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Registry Comparison 
    """)
    return


@app.cell(hide_code=True)
def _(Path, config, pd, tx_patients_by_year):
    # Load national registry data filtered by site_name
    _registry_path = Path(__file__).resolve().parent.parent / 'public' / 'data' / 'clif_hr_tx_counts.csv'
    _registry_df = pd.read_csv(_registry_path)

    _site_name = config['site_name']
    registry_ucmc_df = _registry_df[
        (_registry_df['clif_site'] == _site_name) &
        (_registry_df['ORG_TY'] == 'HR') &
        (_registry_df['peds'] == 0)

    ].copy()
    compare= tx_patients_by_year.merge(registry_ucmc_df[['year', 'transplants']], right_on = 'year', left_on = 'tx_year', how = 'inner')
    compare = compare.rename(
        columns={'transplants': 'srtr_hx_transplants',
                'n_patients': 'clif_hx_transplants'}
    )
    registry_comparison = compare.drop(columns = 'year')
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
    final_df[['patient_id', 'hospitalization_id', 'apprx_transplant_date']].to_csv(cohort_file, index=False)
    logger.info(f"Saved cohort ids to {cohort_file}")

    registry_file = output_dir / f'{site_name}_aggregate_registry_comp.csv'
    registry_comparison.to_csv(registry_file, index=False)
    logger.info(f"Saved registry comparison by year to {registry_file}")
    return


if __name__ == "__main__":
    app.run()
