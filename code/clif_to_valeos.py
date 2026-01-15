import marimo

__generated_with = "0.15.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Construct Valeos Inpatient Database from CLIF
    ## Filter CLIF tables to transplant patients identified by CPT codes

    **Transplant CPT Codes:**
    - Heart: 33945, 33935
    - Lung: 32851, 32852, 32853, 32854
    - Liver: 47135
    - Kidney: 50360, 50365
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import sys
    import logging
    from datetime import datetime as dt_now
    from pathlib import Path
    import marimo as mo

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('clif_to_valeos_processing.log')
        ]
    )
    logger = logging.getLogger(__name__)

    # Add utils directory relative to current script location
    CURRENT_DIR = Path(__file__).resolve().parent
    UTILS_DIR = CURRENT_DIR.parent / 'utils'
    sys.path.append(str(UTILS_DIR))

    from load_config import load_config

    # Import clifpy table classes
    from clifpy.tables import (
        Patient, Hospitalization,
        Vitals, Labs, RespiratorySupport,
        MedicationAdminContinuous, MedicationAdminIntermittent,
        CrrtTherapy, EcmoMcs, Adt,
        PatientProcedures, CodeStatus, HospitalDiagnosis,
        MicrobiologyCulture, MicrobiologyNonculture, MicrobiologySusceptibility
    )

    logger.info("=== CLIF TO VALEOS PROCESSING STARTED ===")
    logger.info(f"Processing started at: {dt_now.now()}")

    # Define transplant CPT codes
    TRANSPLANT_CPT_CODES = {
        'heart': ['33945', '33935'],
        'lung': ['32851', '32852', '32853', '32854'],
        'liver': ['47135'],
        'kidney': ['50360', '50365']
    }

    # Flatten to list of all CPT codes
    ALL_TRANSPLANT_CPTS = []
    for codes in TRANSPLANT_CPT_CODES.values():
        ALL_TRANSPLANT_CPTS.extend(codes)

    return (
        Adt,
        ALL_TRANSPLANT_CPTS,
        CodeStatus,
        CrrtTherapy,
        EcmoMcs,
        HospitalDiagnosis,
        Hospitalization,
        Labs,
        MedicationAdminContinuous,
        MedicationAdminIntermittent,
        MicrobiologyCulture,
        MicrobiologyNonculture,
        MicrobiologySusceptibility,
        Path,
        Patient,
        PatientProcedures,
        RespiratorySupport,
        TRANSPLANT_CPT_CODES,
        Vitals,
        dt_now,
        load_config,
        logger,
        mo,
        pd,
    )


@app.cell
def _(load_config, logger):
    # Load configuration
    logger.info("Step 1: Loading configuration...")
    config = load_config()
    logger.info(f"✓ Configuration loaded successfully")
    logger.info(f"  - Site: {config['site_name']}")
    logger.info(f"  - Tables path: {config['tables_path']}")
    logger.info(f"  - File type: {config['file_type']}")
    print(f"Site: {config['site_name']}")
    print(f"Tables path: {config['tables_path']}")
    print(f"File type: {config['file_type']}")
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 2: Identify Transplant Recipients from Procedure Table (CPT Codes)""")
    return


@app.cell
def _(
    ALL_TRANSPLANT_CPTS,
    PatientProcedures,
    TRANSPLANT_CPT_CODES,
    config,
    logger,
    pd,
):
    # Load procedure table using clifpy and identify transplant recipients by CPT codes
    logger.info("Step 2: Loading procedure table and identifying transplant recipients...")

    tables_path = config['tables_path']
    file_type = config['file_type']

    try:
        proc_table = PatientProcedures.from_file(
            data_directory=tables_path,
            filetype=file_type,
            timezone='UTC'
        )
        procedures_df = proc_table.df
        logger.info(f"✓ Loaded procedures table via clifpy: {len(procedures_df):,} records")
        print(f"Loaded procedures table: {len(procedures_df):,} records")
    except Exception as e:
        logger.error(f"Failed to load procedures table: {e}")
        procedures_df = None
        print(f"Error loading procedures: {e}")

    # Filter to transplant CPT codes
    if procedures_df is not None:
        # Ensure procedure_code is string for matching
        procedures_df['procedure_code'] = procedures_df['procedure_code'].astype(str)

        # Filter to transplant procedures
        transplant_procedures = procedures_df[
            procedures_df['procedure_code'].isin(ALL_TRANSPLANT_CPTS)
        ].copy()

        logger.info(f"✓ Found {len(transplant_procedures):,} transplant procedure records")
        print(f"Transplant procedure records: {len(transplant_procedures):,}")

        # Map CPT codes to organ type
        def get_organ_from_cpt(cpt_code):
            for organ, codes in TRANSPLANT_CPT_CODES.items():
                if str(cpt_code) in codes:
                    return organ
            return 'unknown'

        transplant_procedures['organ'] = transplant_procedures['procedure_code'].apply(get_organ_from_cpt)

        # Get unique hospitalization IDs with transplants
        transplant_hosp_ids = transplant_procedures['hospitalization_id'].unique()
        logger.info(f"✓ Identified {len(transplant_hosp_ids):,} hospitalizations with transplant procedures")
        print(f"Hospitalizations with transplants: {len(transplant_hosp_ids):,}")

        # Count by organ type
        organ_counts = transplant_procedures.groupby('organ')['hospitalization_id'].nunique()
        logger.info("Transplant counts by organ:")
        for organ, count in organ_counts.items():
            logger.info(f"  - {organ}: {count} hospitalizations")
            print(f"  {organ}: {count} hospitalizations")
    else:
        transplant_procedures = None
        transplant_hosp_ids = []

    return (
        file_type,
        procedures_df,
        tables_path,
        transplant_hosp_ids,
        transplant_procedures,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 3: Load Hospitalization Table and Get Patient IDs""")
    return


@app.cell
def _(Hospitalization, config, logger, pd, transplant_hosp_ids):
    # Load hospitalization table using clifpy
    logger.info("Step 3: Loading hospitalization table...")

    try:
        hosp_table = Hospitalization.from_file(
            data_directory=config['tables_path'],
            filetype=config['file_type'],
            timezone='UTC'
        )
        hospitalization_df = hosp_table.df
        logger.info(f"✓ Loaded hospitalization table via clifpy: {len(hospitalization_df):,} records")
        print(f"Loaded hospitalization table: {len(hospitalization_df):,} records")
    except Exception as e:
        logger.error(f"Failed to load hospitalization table: {e}")
        hospitalization_df = None

    # Filter to transplant hospitalizations and get patient IDs
    if hospitalization_df is not None and len(transplant_hosp_ids) > 0:
        # Filter to transplant hospitalizations
        transplant_hospitalization_df = hospitalization_df[
            hospitalization_df['hospitalization_id'].isin(transplant_hosp_ids)
        ].copy()

        # Convert datetime columns
        transplant_hospitalization_df['admission_dttm'] = pd.to_datetime(
            transplant_hospitalization_df['admission_dttm'], utc=True
        )
        transplant_hospitalization_df['discharge_dttm'] = pd.to_datetime(
            transplant_hospitalization_df['discharge_dttm'], utc=True
        )

        # Get unique patient IDs
        transplant_patient_ids = transplant_hospitalization_df['patient_id'].unique()
        logger.info(f"✓ Identified {len(transplant_patient_ids):,} unique transplant patients")
        print(f"Unique transplant patients: {len(transplant_patient_ids):,}")

        # Convert to string for consistent matching
        transplant_patient_ids_str = [str(pid) for pid in transplant_patient_ids]
    else:
        transplant_hospitalization_df = None
        transplant_patient_ids = []
        transplant_patient_ids_str = []

    return (
        hospitalization_df,
        transplant_hospitalization_df,
        transplant_patient_ids,
        transplant_patient_ids_str,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 4: Create Transplant Table from Procedure Data""")
    return


@app.cell
def _(
    TRANSPLANT_CPT_CODES,
    logger,
    pd,
    transplant_hospitalization_df,
    transplant_procedures,
):
    # Create transplant table with patient_id, transplant_date, organ
    logger.info("Step 4: Creating transplant table from procedure data...")

    if transplant_procedures is not None and transplant_hospitalization_df is not None:
        # Merge procedures with hospitalizations to get patient_id
        transplant_df = pd.merge(
            transplant_procedures[['hospitalization_id', 'procedure_code', 'recorded_dttm', 'organ']],
            transplant_hospitalization_df[['hospitalization_id', 'patient_id']],
            on='hospitalization_id',
            how='inner'
        )

        # Rename recorded_dttm to transplant_date
        transplant_df = transplant_df.rename(columns={'recorded_dttm': 'transplant_date'})

        # Convert transplant_date to datetime
        transplant_df['transplant_date'] = pd.to_datetime(transplant_df['transplant_date'], utc=True)

        # Sort by patient and date
        transplant_df = transplant_df.sort_values(['patient_id', 'transplant_date'])

        # Add transplant numbering
        transplant_df['transplant_number'] = transplant_df.groupby('patient_id').cumcount() + 1
        transplant_df['total_transplants'] = transplant_df.groupby('patient_id')['patient_id'].transform('count')

        # Select final columns
        transplant_df = transplant_df[['patient_id', 'transplant_date', 'organ', 'transplant_number', 'total_transplants']]

        logger.info(f"✓ Created transplant table: {len(transplant_df):,} transplant records")
        print(f"Transplant table created: {len(transplant_df):,} records")

        # Count by organ
        organ_counts = transplant_df['organ'].value_counts()
        for organ, count in organ_counts.items():
            logger.info(f"  - {organ}: {count} transplants")
            print(f"  {organ}: {count} transplants")
    else:
        transplant_df = None
        logger.warning("Could not create transplant table")

    return (transplant_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 5: Load Patient Table and Filter to Transplant Recipients""")
    return


@app.cell
def _(Patient, config, logger, transplant_patient_ids_str):
    # Load patient table using clifpy
    logger.info("Step 5: Loading and filtering patient table...")

    try:
        patient_table = Patient.from_file(
            data_directory=config['tables_path'],
            filetype=config['file_type'],
            timezone='UTC'
        )
        patient_df_full = patient_table.df
        logger.info(f"✓ Loaded patient table via clifpy: {len(patient_df_full):,} records")
    except Exception as e:
        logger.error(f"Failed to load patient table: {e}")
        patient_df_full = None

    # Filter to transplant patients
    if patient_df_full is not None and len(transplant_patient_ids_str) > 0:
        patient_df = patient_df_full[
            patient_df_full['patient_id'].astype(str).isin(transplant_patient_ids_str)
        ].copy()
        logger.info(f"✓ Filtered patient table: {len(patient_df_full):,} → {len(patient_df):,} records")
        print(f"Patient table: {len(patient_df_full):,} → {len(patient_df):,} (filtered)")
    else:
        patient_df = None

    return patient_df, patient_df_full


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 6: Load and Filter Clinical Tables Using clifpy""")
    return


@app.cell
def _(
    Adt,
    CrrtTherapy,
    EcmoMcs,
    Labs,
    MedicationAdminContinuous,
    MedicationAdminIntermittent,
    RespiratorySupport,
    Vitals,
    config,
    logger,
    transplant_hosp_ids,
):
    # Load clinical tables using clifpy with hospitalization_id filters
    logger.info("Step 6: Loading and filtering clinical tables via clifpy...")

    hosp_ids_list = list(transplant_hosp_ids) if len(transplant_hosp_ids) > 0 else []

    # Initialize variables
    vitals_df = None
    labs_df = None
    med_cont_df = None
    med_int_df = None
    resp_df = None
    adt_df = None
    crrt_df = None
    ecmo_df = None

    if len(hosp_ids_list) > 0:
        # Vitals
        try:
            vitals_table = Vitals.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            vitals_df = vitals_table.df
            logger.info(f"✓ Vitals: {len(vitals_df):,} records")
            print(f"Vitals: {len(vitals_df):,} records")
        except Exception as e:
            logger.warning(f"Vitals not loaded: {e}")
            print(f"Vitals not loaded: {e}")

        # Labs
        try:
            labs_table = Labs.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            labs_df = labs_table.df
            logger.info(f"✓ Labs: {len(labs_df):,} records")
            print(f"Labs: {len(labs_df):,} records")
        except Exception as e:
            logger.warning(f"Labs not loaded: {e}")
            print(f"Labs not loaded: {e}")

        # Continuous medications
        try:
            med_cont_table = MedicationAdminContinuous.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            med_cont_df = med_cont_table.df
            logger.info(f"✓ Continuous medications: {len(med_cont_df):,} records")
            print(f"Continuous medications: {len(med_cont_df):,} records")
        except Exception as e:
            logger.warning(f"Continuous medications not loaded: {e}")
            print(f"Continuous medications not loaded: {e}")

        # Intermittent medications
        try:
            med_int_table = MedicationAdminIntermittent.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            med_int_df = med_int_table.df
            logger.info(f"✓ Intermittent medications: {len(med_int_df):,} records")
            print(f"Intermittent medications: {len(med_int_df):,} records")
        except Exception as e:
            logger.warning(f"Intermittent medications not loaded: {e}")
            print(f"Intermittent medications not loaded: {e}")

        # Respiratory support
        try:
            resp_table = RespiratorySupport.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            resp_df = resp_table.df
            logger.info(f"✓ Respiratory support: {len(resp_df):,} records")
            print(f"Respiratory support: {len(resp_df):,} records")
        except Exception as e:
            logger.warning(f"Respiratory support not loaded: {e}")
            print(f"Respiratory support not loaded: {e}")

        # ADT
        try:
            adt_table = Adt.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            adt_df = adt_table.df
            logger.info(f"✓ ADT: {len(adt_df):,} records")
            print(f"ADT: {len(adt_df):,} records")
        except Exception as e:
            logger.warning(f"ADT not loaded: {e}")
            print(f"ADT not loaded: {e}")

        # CRRT
        try:
            crrt_table = CrrtTherapy.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            crrt_df = crrt_table.df
            logger.info(f"✓ CRRT: {len(crrt_df):,} records")
            print(f"CRRT: {len(crrt_df):,} records")
        except Exception as e:
            logger.warning(f"CRRT not loaded: {e}")
            print(f"CRRT not loaded: {e}")

        # ECMO/MCS
        try:
            ecmo_table = EcmoMcs.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            ecmo_df = ecmo_table.df
            logger.info(f"✓ ECMO/MCS: {len(ecmo_df):,} records")
            print(f"ECMO/MCS: {len(ecmo_df):,} records")
        except Exception as e:
            logger.warning(f"ECMO/MCS not loaded: {e}")
            print(f"ECMO/MCS not loaded: {e}")

    logger.info("✓ Step 6 completed - Clinical tables loaded")
    return (
        adt_df,
        crrt_df,
        ecmo_df,
        hosp_ids_list,
        labs_df,
        med_cont_df,
        med_int_df,
        resp_df,
        vitals_df,
    )


@app.cell
def _(
    CodeStatus,
    HospitalDiagnosis,
    MicrobiologyCulture,
    MicrobiologyNonculture,
    MicrobiologySusceptibility,
    config,
    hosp_ids_list,
    logger,
    transplant_patient_ids_str,
):
    # Load additional tables
    logger.info("Loading additional clinical tables...")

    code_status_df = None
    hosp_diag_df = None
    micro_culture_df = None
    micro_non_culture_df = None
    micro_susceptibility_df = None

    if len(hosp_ids_list) > 0:
        # Code status (filtered by patient_id)
        try:
            code_status_table = CodeStatus.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC'
            )
            code_status_df = code_status_table.df
            code_status_df = code_status_df[
                code_status_df['patient_id'].astype(str).isin(transplant_patient_ids_str)
            ]
            logger.info(f"✓ Code status: {len(code_status_df):,} records")
            print(f"Code status: {len(code_status_df):,} records")
        except Exception as e:
            logger.warning(f"Code status not loaded: {e}")
            print(f"Code status not loaded: {e}")

        # Hospital diagnosis
        try:
            hosp_diag_table = HospitalDiagnosis.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            hosp_diag_df = hosp_diag_table.df
            logger.info(f"✓ Hospital diagnosis: {len(hosp_diag_df):,} records")
            print(f"Hospital diagnosis: {len(hosp_diag_df):,} records")
        except Exception as e:
            logger.warning(f"Hospital diagnosis not loaded: {e}")
            print(f"Hospital diagnosis not loaded: {e}")

        # Microbiology culture
        try:
            micro_culture_table = MicrobiologyCulture.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            micro_culture_df = micro_culture_table.df
            logger.info(f"✓ Microbiology culture: {len(micro_culture_df):,} records")
            print(f"Microbiology culture: {len(micro_culture_df):,} records")
        except Exception as e:
            logger.warning(f"Microbiology culture not loaded: {e}")
            print(f"Microbiology culture not loaded: {e}")

        # Microbiology non-culture
        try:
            micro_non_culture_table = MicrobiologyNonculture.from_file(
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone='UTC',
                filters={'hospitalization_id': hosp_ids_list}
            )
            micro_non_culture_df = micro_non_culture_table.df
            logger.info(f"✓ Microbiology non-culture: {len(micro_non_culture_df):,} records")
            print(f"Microbiology non-culture: {len(micro_non_culture_df):,} records")
        except Exception as e:
            logger.warning(f"Microbiology non-culture not loaded: {e}")
            print(f"Microbiology non-culture not loaded: {e}")

        # Microbiology susceptibility
        if micro_culture_df is not None and len(micro_culture_df) > 0:
            try:
                micro_susceptibility_table = MicrobiologySusceptibility.from_file(
                    data_directory=config['tables_path'],
                    filetype=config['file_type'],
                    timezone='UTC'
                )
                micro_susceptibility_df = micro_susceptibility_table.df
                # Filter by organism_id from culture
                organism_ids = micro_culture_df['organism_id'].unique()
                micro_susceptibility_df = micro_susceptibility_df[
                    micro_susceptibility_df['organism_id'].isin(organism_ids)
                ]
                logger.info(f"✓ Microbiology susceptibility: {len(micro_susceptibility_df):,} records")
                print(f"Microbiology susceptibility: {len(micro_susceptibility_df):,} records")
            except Exception as e:
                logger.warning(f"Microbiology susceptibility not loaded: {e}")
                print(f"Microbiology susceptibility not loaded: {e}")

    return (
        code_status_df,
        hosp_diag_df,
        micro_culture_df,
        micro_non_culture_df,
        micro_susceptibility_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 7: Create Valeos Output Directory and Export Tables""")
    return


@app.cell
def _(Path, config, logger):
    # Create Valeos output directory
    logger.info("Step 7: Setting up Valeos output directory...")
    valeos_tables_path = Path(config['tables_path'])
    valeos_dir = valeos_tables_path / 'Valeos'
    valeos_dir.mkdir(exist_ok=True)

    logger.info(f"✓ Created/verified Valeos directory: {valeos_dir}")
    print(f"Created Valeos directory: {valeos_dir}")

    # Get site name for file naming
    site_name = config['site_name']
    logger.info(f"✓ Site name for file naming: {site_name}")

    return site_name, valeos_dir, valeos_tables_path


@app.cell
def _(
    adt_df,
    code_status_df,
    crrt_df,
    ecmo_df,
    hosp_diag_df,
    labs_df,
    logger,
    med_cont_df,
    med_int_df,
    micro_culture_df,
    micro_non_culture_df,
    micro_susceptibility_df,
    patient_df,
    procedures_df,
    resp_df,
    site_name,
    transplant_hosp_ids,
    transplant_hospitalization_df,
    valeos_dir,
    vitals_df,
):
    # Export all tables to Valeos directory
    logger.info("Step 8: Exporting tables to parquet format...")

    # Filter procedures to transplant hospitalizations
    if procedures_df is not None:
        proc_df_filtered = procedures_df[
            procedures_df['hospitalization_id'].isin(transplant_hosp_ids)
        ]
    else:
        proc_df_filtered = None

    # Dictionary of tables to export
    tables_to_export = {
        'patient': patient_df,
        'hospitalization': transplant_hospitalization_df,
        'vitals': vitals_df,
        'labs': labs_df,
        'medication_admin_continuous': med_cont_df,
        'medication_admin_intermittent': med_int_df,
        'respiratory_support': resp_df,
        'adt': adt_df,
        'crrt_therapy': crrt_df,
        'ecmo_mcs': ecmo_df,
        'patient_procedure': proc_df_filtered,
        'code_status': code_status_df,
        'hospital_diagnosis': hosp_diag_df,
        'microbiology_culture': micro_culture_df,
        'microbiology_non_culture': micro_non_culture_df,
        'microbiology_susceptibility': micro_susceptibility_df
    }

    # Export each table as parquet
    exported_files = []
    tables_with_data = 0
    tables_without_data = 0

    for export_tbl_name, export_tbl_df in tables_to_export.items():
        if export_tbl_df is not None and len(export_tbl_df) > 0:
            filename = f"{site_name}_valeos_inpatient_{export_tbl_name}.parquet"
            filepath = valeos_dir / filename
            try:
                export_tbl_df.to_parquet(filepath, index=False)
                exported_files.append(filename)
                tables_with_data += 1
                logger.info(f"✓ Exported {export_tbl_name}: {len(export_tbl_df):,} records")
                print(f"Exported {export_tbl_name}: {len(export_tbl_df):,} records")
            except Exception as e:
                logger.error(f"✗ Failed to export {export_tbl_name}: {str(e)}")
        else:
            tables_without_data += 1
            logger.info(f"- Skipped {export_tbl_name}: No data available")
            print(f"Skipped {export_tbl_name}: No data available")

    logger.info(f"✓ Export summary: {tables_with_data} tables exported, {tables_without_data} tables skipped")

    return exported_files, proc_df_filtered, tables_to_export, tables_with_data, tables_without_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 8: Create and Export Transplant Table""")
    return


@app.cell
def _(logger, site_name, transplant_df, valeos_dir):
    # Export transplant table
    logger.info("Step 9: Exporting transplant table...")

    if transplant_df is not None:
        # Rename 'organ' to 'transplant_type' for clarity
        transplant_table = transplant_df.copy()
        transplant_table = transplant_table.rename(columns={'organ': 'transplant_type'})

        # Ensure proper column order
        transplant_table = transplant_table[['patient_id', 'transplant_date', 'transplant_type', 'transplant_number', 'total_transplants']]

        # Export transplant table
        transplant_filename = f"{site_name}_valeos_inpatient_transplant.parquet"
        transplant_filepath = valeos_dir / transplant_filename

        try:
            transplant_table.to_parquet(transplant_filepath, index=False)
            logger.info(f"✓ Exported transplant table: {len(transplant_table):,} records")
            logger.info(f"✓ File location: {transplant_filepath}")
            print(f"Exported transplant table: {len(transplant_table):,} records")
            print(f"Columns: {list(transplant_table.columns)}")
        except Exception as e:
            logger.error(f"✗ Failed to export transplant table: {str(e)}")
            transplant_table = None
    else:
        transplant_table = None
        logger.warning("No transplant table to export")

    return (transplant_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 9: Summary Statistics""")
    return


@app.cell
def _(
    dt_now,
    exported_files,
    logger,
    site_name,
    tables_to_export,
    tables_with_data,
    tables_without_data,
    transplant_table,
    valeos_dir,
):
    # Final summary
    logger.info("Step 10: Generating final summary...")

    print("\n" + "=" * 50)
    print("VALEOS DATABASE EXPORT SUMMARY")
    print("=" * 50)
    print(f"Site: {site_name}")
    print(f"Output directory: {valeos_dir}")
    print(f"Total files exported: {len(exported_files) + 1}")  # +1 for transplant table
    print()

    print("Table Export Summary:")
    total_records = 0
    for tbl_name, tbl_df in tables_to_export.items():
        if tbl_df is not None and len(tbl_df) > 0:
            total_records += len(tbl_df)
            print(f"  ✓ {tbl_name}: {len(tbl_df):,} records")
        else:
            print(f"  ✗ {tbl_name}: No data")

    if transplant_table is not None:
        total_records += len(transplant_table)
        print(f"  ✓ transplant: {len(transplant_table):,} records")

        print("\nTransplant Summary by Organ:")
        transplant_counts = transplant_table['transplant_type'].value_counts()
        for organ_type, count in transplant_counts.items():
            print(f"  {organ_type}: {count:,} transplants")

        unique_patients = transplant_table['patient_id'].nunique()
        print(f"\nTotal unique transplant patients: {unique_patients:,}")
        print(f"Total transplant procedures: {len(transplant_table):,}")

        # Multi-transplant patients
        multi_transplant = transplant_table[transplant_table['total_transplants'] > 1]
        if len(multi_transplant) > 0:
            multi_patients = multi_transplant['patient_id'].nunique()
            print(f"Patients with multiple transplants: {multi_patients:,}")

    print(f"\nTotal clinical records exported: {total_records:,}")
    print(f"Tables with data: {tables_with_data + 1}")
    print(f"Tables skipped: {tables_without_data}")

    logger.info(f"Processing completed at: {dt_now.now()}")
    logger.info("=== CLIF TO VALEOS PROCESSING COMPLETED SUCCESSFULLY ===")

    print("\n" + "=" * 50)
    print("PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 50)
    return


if __name__ == "__main__":
    app.run()
