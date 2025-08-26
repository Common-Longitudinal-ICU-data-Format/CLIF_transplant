import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Construct Valeos Inpatient Database from CLIF
    ## Filter CLIF tables to transplant patients and their hospitalizations
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import sys
    import os
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
    
    logger.info("=== CLIF TO VALEOS PROCESSING STARTED ===")
    logger.info(f"Processing started at: {dt_now.now()}")
    
    return Path, dt_now, load_config, logger, mo, os, pd


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
    mo.md(r"""## Step 2: Load Transplant Patient IDs""")
    return


@app.cell
def _(logger, pd):
    # load in transplant_df.csv
    logger.info("Step 2: Loading transplant patient data...")
    transplant_df = pd.read_csv('/Users/dema/WD/CRI_2018_24/VALEOS/unique_enc_pat.csv')
    logger.info(f"✓ Loaded transplant data: {len(transplant_df)} records")

    #drop hospitalization_id column
    transplant_df = transplant_df.drop(columns=['hospitalization_id'])
    logger.info("✓ Dropped hospitalization_id column")

    #drop duplicate rows
    original_count = len(transplant_df)
    transplant_df = transplant_df.drop_duplicates()
    logger.info(f"✓ Removed {original_count - len(transplant_df)} duplicate rows")

    # Get unique transplant patient IDs from the CSV
    transplant_patient_ids = transplant_df['patient_id'].unique()
    logger.info(f"✓ Identified {len(transplant_patient_ids)} unique transplant patients")
    print(f"Number of unique transplant patient IDs: {len(transplant_patient_ids)}")
    print(f"Sample patient IDs: {transplant_patient_ids[:5]}")

    # Convert patient IDs to string for consistent matc`hing
    transplant_patient_ids_str = [str(pid) for pid in transplant_patient_ids]

    #rename service_date to transplant_date
    transplant_df = transplant_df.rename(columns={'service_date': 'transplant_date'})
    logger.info("✓ Renamed service_date to transplant_date")

    #arrange columns in order of patient_id, transplant_date, transplant_type
    transplant_df = transplant_df[['patient_id', 'transplant_date', 'organ']]

    #create a new column called transplant number that is a running number of the transplant for each patient
    transplant_df['transplant_number'] = transplant_df.groupby('patient_id').cumcount() + 1

    #create a new column that is total number of transplants for each patient
    transplant_df['total_transplants'] = transplant_df.groupby('patient_id')['patient_id'].transform('count')
    logger.info("✓ Added transplant numbering columns")

    # Count transplants by organ type
    organ_counts = transplant_df['organ'].value_counts()
    logger.info("✓ Transplant counts by organ:")
    for organ, count in organ_counts.items():
        logger.info(f"  - {organ}: {count} transplants")

    # heart transplant recipient ids
    heart_transplant_ids = transplant_df[transplant_df['organ'] == 'heart']['patient_id'].unique()
    print(f"Number of unique heart transplant patient IDs: {len(heart_transplant_ids)}")
    print(f"Sample heart transplant patient IDs: {heart_transplant_ids[:5]}")

    # lung transplant recipient ids
    lung_transplant_ids = transplant_df[transplant_df['organ'] == 'lung']['patient_id'].unique()
    print(f"Number of unique lung transplant patient IDs: {len(lung_transplant_ids)}")
    print(f"Sample lung transplant patient IDs: {lung_transplant_ids[:5]}")

    # liver transplant recipient ids
    liver_transplant_ids = transplant_df[transplant_df['organ'] == 'liver']['patient_id'].unique()
    print(f"Number of unique liver transplant patient IDs: {len(liver_transplant_ids)}")
    print(f"Sample liver transplant patient IDs: {liver_transplant_ids[:5]}")

    # kidney transplant recipient ids
    kidney_transplant_ids = transplant_df[transplant_df['organ'] == 'kidney']['patient_id'].unique()
    print(f"Number of unique kidney transplant patient IDs: {len(kidney_transplant_ids)}")
    print(f"Sample kidney transplant patient IDs: {kidney_transplant_ids[:5]}")
    
    logger.info(f"✓ Step 2 completed - Transplant patient data processed")
    return heart_transplant_ids, transplant_df, transplant_patient_ids_str


@app.cell
def _(transplant_df):
    # tabulate transplant patient types 
    transplant_df['organ'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 3: Filter Patient Table to Transplant Recipients""")
    return


@app.cell
def _(Path, config, logger, pd, transplant_patient_ids_str):
    def load_clif_table(table_name, config):
        """Load a CLIF table based on configuration"""
        tables_path = Path(config['tables_path'])
        file_type = config['file_type']
        file_path = tables_path / f"{table_name}.{file_type}"

        if not file_path.exists():
            logger.warning(f"Table not found: {file_path}")
            print(f"Warning: {file_path} not found")
            return None

        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"✓ Loaded {table_name}: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            print(f"Error loading {file_path}: {str(e)}")
            return None

    # Load the patient table
    logger.info("Step 3: Loading and filtering CLIF patient table...")
    print("Loading CLIF patient table...")
    patient_df = load_clif_table('clif_patient', config)

    if patient_df is not None:
        original_patients = len(patient_df)
        #filter the patient_df to just transplant_patient_ids_str
        patient_df = patient_df[patient_df['patient_id'].astype(str).isin(transplant_patient_ids_str)]
        logger.info(f"✓ Filtered patient table: {original_patients} → {len(patient_df)} patients")
        logger.info(f"✓ Step 3 completed - Patient table filtered to transplant recipients")
    else:
        logger.error("Failed to load patient table")

    patient_df.shape
    return load_clif_table, patient_df


@app.cell
def _(config, load_clif_table, logger, transplant_patient_ids_str):
    # load in hospitalization table and filter to transplant recipients
    logger.info("Step 4: Loading and filtering hospitalization data...")
    hospitalization_df = load_clif_table('clif_hospitalization', config)

    if hospitalization_df is not None:
        original_hospitalizations = len(hospitalization_df)
        #filter to transplant recipients by patient_id
        hospitalization_df = hospitalization_df[hospitalization_df['patient_id'].isin(transplant_patient_ids_str)]

        logger.info(f"✓ Filtered hospitalizations: {original_hospitalizations} → {len(hospitalization_df)} records")
        print(f"number of hospitilizations for transplant recipients: {len(hospitalization_df)}")

        #count hospitalizations per patient
        hospitalization_df['hospitalization_count'] = hospitalization_df.groupby('patient_id')['patient_id'].transform('count')

        #check that all transplant recipients have at least one hospitalization
        unique_patients_with_hosp = len(hospitalization_df['patient_id'].unique())
        logger.info(f"✓ Patients with hospitalizations: {unique_patients_with_hosp}")
        logger.info(f"✓ Step 4 completed - Hospitalization data filtered")
    else:
        logger.error("Failed to load hospitalization table")
        
    len(hospitalization_df['patient_id'].unique())
    return (hospitalization_df,)


@app.cell
def _(transplant_patient_ids_str):
    len(transplant_patient_ids_str)
    return


@app.cell
def _(config, logger, os, pd):
    # Load continuous medication admin
    logger.info("Step 5: Loading continuous medication administration data...")
    med_cont_path = os.path.join(config['tables_path'], f"clif_medication_admin_continuous.{config['file_type']}")
    print(f"Loading continuous medication admin from: {med_cont_path}")

    try:
        if config['file_type'] == 'parquet':
            med_cont_df = pd.read_parquet(med_cont_path)
        else:
            med_cont_df = pd.read_csv(med_cont_path)

        logger.info(f"✓ Loaded continuous medication data: {len(med_cont_df)} records")
        logger.info(f"✓ Columns available: {len(med_cont_df.columns)}")
        print(f"Continuous medication admin loaded: {len(med_cont_df)} rows")
        print(f"Columns: {list(med_cont_df.columns)}")
        print("\nFirst few rows:")
        print(med_cont_df.head())
        logger.info(f"✓ Step 5 completed - Medication data loaded")
    except Exception as e:
        logger.error(f"Failed to load continuous medication data: {str(e)}")
        med_cont_df = None
        
    return (med_cont_df,)


@app.cell
def _(config, hospitalization_df, load_clif_table, logger):
    # Load in vitals table
    logger.info("Step 6: Loading and filtering vitals data...")
    vitals_df = load_clif_table('clif_vitals', config)

    if vitals_df is not None and hospitalization_df is not None:
        original_vitals = len(vitals_df)
        # filter to hospitalizations in hospitalization_df
        vitals_df = vitals_df[vitals_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        logger.info(f"✓ Filtered vitals data: {original_vitals} → {len(vitals_df)} records")
        logger.info(f"✓ Step 6 completed - Vitals data filtered to transplant hospitalizations")
    else:
        logger.error("Failed to filter vitals data")
        
    return (vitals_df,)


@app.cell
def _(hospitalization_df, pd, vitals_df):
    vital_start_stop = vitals_df.groupby('hospitalization_id').agg({'recorded_dttm': ['min', 'max']}).reset_index()
    vital_start_stop.columns = ['hospitalization_id', 'first_vital_dttm', 'last_vital_dttm']
    hospitalization_df_1 = pd.merge(hospitalization_df, vital_start_stop, on='hospitalization_id', how='left')
    hospitalization_df_1
    return (hospitalization_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Identify transplant hospitalization

    Using the transplant_date in the transplant table, identify the hospitalization which has the transplant_date between the start and end dates.b
    """
    )
    return


@app.cell
def _(hospitalization_df_1, pd, transplant_df):
    transplant_df['patient_id'] = transplant_df['patient_id'].astype(str)
    hospitalization_df_1['patient_id'] = hospitalization_df_1['patient_id'].astype(str)
    transplant_hospitalization = pd.merge(hospitalization_df_1, transplant_df, on='patient_id', how='left')
    transplant_hospitalization = transplant_hospitalization[transplant_hospitalization['admission_dttm'] < transplant_hospitalization['transplant_date']]
    transplant_hospitalization = transplant_hospitalization[transplant_hospitalization['discharge_dttm'] >= transplant_hospitalization['transplant_date']]
    transplant_hospitalization.shape
    missing_hosp = transplant_df[~transplant_df['patient_id'].isin(transplant_hospitalization['patient_id'])]
    transplant_df[transplant_df['patient_id'].isin(missing_hosp['patient_id'])]
    return (transplant_hospitalization,)


@app.cell
def _(hospitalization_df_1):
    hospitalization_df_1[hospitalization_df_1['patient_id'] == 43154]
    return


@app.cell
def _(pd):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    def plot_norepinephrine_dose_hourly(patient_id, transplant_norepinephrine_df, transplant_hospitalization_df, show_dates=False):
        """
        Plot norepinephrine dose per hour during transplant hospitalization for a single patient.

        Parameters:
        patient_id (str): Patient ID to plot
        transplant_norepinephrine_df (DataFrame): Filtered norepinephrine data for transplant patients
        transplant_hospitalization_df (DataFrame): Transplant hospitalization data
        show_dates (bool): Whether to show actual dates in legend (default: False)

        Returns:
        matplotlib.figure.Figure: The plot figure
        """
        patient_hosp = transplant_hospitalization_df[transplant_hospitalization_df['patient_id'] == str(patient_id)]
        if patient_hosp.empty:
            print(f'No transplant hospitalization found for patient {patient_id}')
            return None
        hosp_id = patient_hosp.iloc[0]['hospitalization_id']
        transplant_date = patient_hosp.iloc[0]['transplant_date']
        organ_type = patient_hosp.iloc[0]['organ']
        admission_date = patient_hosp.iloc[0]['admission_dttm']
        discharge_date = patient_hosp.iloc[0]['discharge_dttm']
        patient_norepi = transplant_norepinephrine_df[transplant_norepinephrine_df['hospitalization_id'] == hosp_id].copy()
        if patient_norepi.empty:
            print(f'No norepinephrine data found for patient {patient_id} during transplant hospitalization')
            return None
        patient_norepi['admin_dttm'] = pd.to_datetime(patient_norepi['admin_dttm'])
        patient_norepi = patient_norepi.sort_values('admin_dttm')
        transplant_date = pd.to_datetime(transplant_date)
        admission_date = pd.to_datetime(admission_date)
        discharge_date = pd.to_datetime(discharge_date)
        patient_norepi['days_since_admission'] = (patient_norepi['admin_dttm'] - admission_date).dt.total_seconds() / (24 * 3600)
        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)
        (_fig, ax) = plt.subplots(figsize=(12, 6))
        ax.plot(patient_norepi['days_since_admission'], patient_norepi['med_dose'], marker='o', markersize=3, linewidth=1, color='red', alpha=0.7)
        if show_dates:
            transplant_label = f"Transplant Date ({transplant_date.strftime('%Y-%m-%d %H:%M')})"
        else:
            transplant_label = f'Transplant (Day {transplant_days:.1f})'
        ax.axvline(x=transplant_days, color='green', linestyle='--', linewidth=2, label=transplant_label)
        if show_dates:
            admission_label = f"Admission ({admission_date.strftime('%Y-%m-%d %H:%M')})"
            discharge_label = f"Discharge ({discharge_date.strftime('%Y-%m-%d %H:%M')})"
        else:
            admission_label = 'Admission (Day 0)'
            discharge_label = f'Discharge (Day {discharge_days:.1f})'
        ax.axvline(x=0, color='blue', linestyle=':', alpha=0.7, label=admission_label)
        ax.axvline(x=discharge_days, color='orange', linestyle=':', alpha=0.7, label=discharge_label)
        ax.set_xlabel('Days Since Admission')
        ax.set_ylabel('Norepinephrine Dose (mcg/kg/min)')
        ax.set_title(f'Norepinephrine Dose During {organ_type.title()} Transplant Hospitalization\nPatient ID: {patient_id}')
        ax.set_xlim(left=-0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        print(f'Patient {patient_id} - {organ_type.title()} Transplant:')
        print(f'  Hospitalization ID: {hosp_id}')
        print(f'  Admission: {admission_date}')
        print(f'  Transplant: {transplant_date} (Day {transplant_days:.1f})')
        print(f'  Discharge: {discharge_date} (Day {discharge_days:.1f})')
        print(f'  Norepinephrine records: {len(patient_norepi)}')
        print(f"  Dose range: {patient_norepi['med_dose'].min():.2f} - {patient_norepi['med_dose'].max():.2f} mcg/kg/min")
        print(f"  Mean dose: {patient_norepi['med_dose'].mean():.2f} mcg/kg/min")
        return _fig
    return plot_norepinephrine_dose_hourly, plt


@app.cell
def _(
    med_cont_df,
    plot_norepinephrine_dose_hourly,
    plt,
    transplant_hospitalization,
):
    if 'transplant_norepinephrine' not in globals():
        print('Filtering norepinephrine data...')
        norepinephrine_df = med_cont_df[med_cont_df['med_category'] == 'norepinephrine'].copy()
        transplant_hosp_ids = transplant_hospitalization['hospitalization_id'].unique()
        transplant_norepinephrine = norepinephrine_df[norepinephrine_df['hospitalization_id'].isin(transplant_hosp_ids)].copy()
    if len(transplant_norepinephrine) > 0:
        _sample_hosp_id = transplant_norepinephrine['hospitalization_id'].iloc[0]
        _sample_patient_id = transplant_hospitalization[transplant_hospitalization['hospitalization_id'] == _sample_hosp_id]['patient_id'].iloc[0]
        print(f'Testing plot function with patient {_sample_patient_id}')
        _fig = plot_norepinephrine_dose_hourly(_sample_patient_id, transplant_norepinephrine, transplant_hospitalization)
        if _fig is not None:
            plt.show()
        else:
            print('Could not generate plot for sample patient')
    else:
        print('No norepinephrine data available for transplant patients')
    return transplant_hosp_ids, transplant_norepinephrine


@app.cell
def _(
    heart_transplant_ids,
    plot_norepinephrine_dose_hourly,
    plt,
    transplant_hospitalization,
    transplant_norepinephrine,
):
    _fig = plot_norepinephrine_dose_hourly(heart_transplant_ids[301], transplant_norepinephrine, transplant_hospitalization)
    if _fig is not None:
        plt.show()
    else:
        print('Could not generate plot for sample patient')
    return


@app.cell
def _(med_cont_df, pd, plt, transplant_hosp_ids, transplant_hospitalization):
    print('Filtering dobutamine data for transplant patients...')
    dobutamine_df = med_cont_df[med_cont_df['med_category'] == 'dobutamine'].copy()
    print(f'Total dobutamine records: {len(dobutamine_df)}')
    transplant_dobutamine = dobutamine_df[dobutamine_df['hospitalization_id'].isin(transplant_hosp_ids)].copy()
    print(f'Dobutamine records in transplant hospitalizations: {len(transplant_dobutamine)}')
    if len(transplant_dobutamine) > 0:
        sample_patient_hosp = transplant_dobutamine['hospitalization_id'].iloc[0]
        print(f'Sample hospitalization with dobutamine data: {sample_patient_hosp}')
        print('\nSample dobutamine data:')
        print(transplant_dobutamine.head())
    else:
        print('No dobutamine data found for transplant hospitalizations')

    def plot_dobutamine_dose_hourly(patient_id, transplant_dobutamine_df, transplant_hospitalization_df, show_dates=False):
        """
        Plot dobutamine dose per hour during transplant hospitalization for a single patient.

        Parameters:
        patient_id (str): Patient ID to plot
        transplant_dobutamine_df (DataFrame): Filtered dobutamine data for transplant patients
        transplant_hospitalization_df (DataFrame): Transplant hospitalization data
        show_dates (bool): Whether to show actual dates in legend (default: False)

        Returns:
        matplotlib.figure.Figure: The plot figure
        """
        patient_hosp = transplant_hospitalization_df[transplant_hospitalization_df['patient_id'] == str(patient_id)]
        if patient_hosp.empty:
            print(f'No transplant hospitalization found for patient {patient_id}')
            return None
        hosp_id = patient_hosp.iloc[0]['hospitalization_id']
        transplant_date = patient_hosp.iloc[0]['transplant_date']
        organ_type = patient_hosp.iloc[0]['organ']
        admission_date = patient_hosp.iloc[0]['admission_dttm']
        discharge_date = patient_hosp.iloc[0]['discharge_dttm']
        patient_dobutamine = transplant_dobutamine_df[transplant_dobutamine_df['hospitalization_id'] == hosp_id].copy()
        if patient_dobutamine.empty:
            print(f'No dobutamine data found for patient {patient_id} during transplant hospitalization')
            return None
        patient_dobutamine['admin_dttm'] = pd.to_datetime(patient_dobutamine['admin_dttm'])
        patient_dobutamine = patient_dobutamine.sort_values('admin_dttm')
        transplant_date = pd.to_datetime(transplant_date)
        admission_date = pd.to_datetime(admission_date)
        discharge_date = pd.to_datetime(discharge_date)
        patient_dobutamine['days_since_admission'] = (patient_dobutamine['admin_dttm'] - admission_date).dt.total_seconds() / (24 * 3600)
        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)
        (_fig, ax) = plt.subplots(figsize=(12, 6))
        ax.plot(patient_dobutamine['days_since_admission'], patient_dobutamine['med_dose'], marker='o', markersize=3, linewidth=1, color='blue', alpha=0.7)
        if show_dates:
            transplant_label = f"Transplant Date ({transplant_date.strftime('%Y-%m-%d %H:%M')})"
        else:
            transplant_label = f'Transplant (Day {transplant_days:.1f})'
        ax.axvline(x=transplant_days, color='green', linestyle='--', linewidth=2, label=transplant_label)
        if show_dates:
            admission_label = f"Admission ({admission_date.strftime('%Y-%m-%d %H:%M')})"
            discharge_label = f"Discharge ({discharge_date.strftime('%Y-%m-%d %H:%M')})"
        else:
            admission_label = 'Admission (Day 0)'
            discharge_label = f'Discharge (Day {discharge_days:.1f})'
        ax.axvline(x=0, color='blue', linestyle=':', alpha=0.7, label=admission_label)
        ax.axvline(x=discharge_days, color='orange', linestyle=':', alpha=0.7, label=discharge_label)
        ax.set_xlabel('Days Since Admission')
        ax.set_ylabel('Dobutamine Dose (mcg/kg/min)')
        ax.set_title(f'Dobutamine Dose During {organ_type.title()} Transplant Hospitalization\nPatient ID: {patient_id}')
        ax.set_xlim(left=-0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        print(f'Patient {patient_id} - {organ_type.title()} Transplant:')
        print(f'  Hospitalization ID: {hosp_id}')
        print(f'  Admission: {admission_date}')
        print(f'  Transplant: {transplant_date} (Day {transplant_days:.1f})')
        print(f'  Discharge: {discharge_date} (Day {discharge_days:.1f})')
        print(f'  Dobutamine records: {len(patient_dobutamine)}')
        print(f"  Dose range: {patient_dobutamine['med_dose'].min():.2f} - {patient_dobutamine['med_dose'].max():.2f} mcg/kg/min")
        print(f"  Mean dose: {patient_dobutamine['med_dose'].mean():.2f} mcg/kg/min")
        return _fig
    if len(transplant_dobutamine) > 0:
        _sample_hosp_id = transplant_dobutamine['hospitalization_id'].iloc[0]
        _sample_patient_id = transplant_hospitalization[transplant_hospitalization['hospitalization_id'] == _sample_hosp_id]['patient_id'].iloc[0]
        print(f'\nTesting dobutamine plot function with patient {_sample_patient_id}')
        _fig = plot_dobutamine_dose_hourly(_sample_patient_id, transplant_dobutamine, transplant_hospitalization)
        if _fig is not None:
            plt.show()
        else:
            print('Could not generate dobutamine plot for sample patient')
    else:
        print('No dobutamine data available for transplant patients - cannot test plotting function')
    return plot_dobutamine_dose_hourly, transplant_dobutamine


@app.cell
def _(
    heart_transplant_ids,
    plot_dobutamine_dose_hourly,
    transplant_dobutamine,
    transplant_hospitalization,
):
    _fig = plot_dobutamine_dose_hourly(heart_transplant_ids[302], transplant_dobutamine, transplant_hospitalization)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 4: Filter Remaining CLIF Tables""")
    return

@app.cell
def _(config, hospitalization_df, load_clif_table, logger):
    # Load and filter labs table
    logger.info("Step 7: Loading and filtering clinical data tables...")
    logger.info("  - Loading labs table...")
    print("Loading and filtering labs table...")
    labs_df = load_clif_table('clif_labs', config)
    if labs_df is not None:
        original_labs = len(labs_df)
        labs_df_filtered = labs_df[labs_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        logger.info(f"  ✓ Labs: {original_labs} → {len(labs_df_filtered)} records")
        print(f"Labs records: {len(labs_df)} -> {len(labs_df_filtered)} (filtered)")
    else:
        labs_df_filtered = None
        logger.warning("  - Labs table not found")
        print("Labs table not found")
    return (labs_df_filtered,)


@app.cell
def _(config, hospitalization_df, load_clif_table, logger):
    # Load and filter medication_admin_intermittent table
    logger.info("  - Loading intermittent medication table...")
    print("Loading and filtering medication_admin_intermittent table...")
    med_int_df = load_clif_table('clif_medication_admin_intermittent', config)
    if med_int_df is not None:
        original_med_int = len(med_int_df)
        med_int_df_filtered = med_int_df[med_int_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        logger.info(f"  ✓ Intermittent medications: {original_med_int} → {len(med_int_df_filtered)} records")
        print(f"Intermittent medication records: {len(med_int_df)} -> {len(med_int_df_filtered)} (filtered)")
    else:
        med_int_df_filtered = None
        logger.warning("  - Intermittent medication table not found")
        print("Medication admin intermittent table not found")
    return (med_int_df_filtered,)


@app.cell
def _(config, hospitalization_df, load_clif_table):
    # Load and filter respiratory_support table
    print("Loading and filtering respiratory_support table...")
    resp_df = load_clif_table('clif_respiratory_support', config)
    if resp_df is not None:
        resp_df_filtered = resp_df[resp_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        print(f"Respiratory support records: {len(resp_df)} -> {len(resp_df_filtered)} (filtered)")
    else:
        resp_df_filtered = None
        print("Respiratory support table not found")
    return (resp_df_filtered,)


@app.cell
def _(config, hospitalization_df, load_clif_table):
    # Load and filter ADT table
    print("Loading and filtering ADT table...")
    adt_df = load_clif_table('clif_adt', config)
    if adt_df is not None:
        adt_df_filtered = adt_df[adt_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        print(f"ADT records: {len(adt_df)} -> {len(adt_df_filtered)} (filtered)")
    else:
        adt_df_filtered = None
        print("ADT table not found")
    return (adt_df_filtered,)


@app.cell
def _(config, hospitalization_df, load_clif_table):
    # Load and filter CLIF 2.1 tables
    print("Loading and filtering CLIF 2.1 tables...")

    # CRRT therapy
    crrt_df = load_clif_table('clif_crrt_therapy', config)
    if crrt_df is not None:
        crrt_df_filtered = crrt_df[crrt_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        print(f"CRRT therapy records: {len(crrt_df)} -> {len(crrt_df_filtered)} (filtered)")
    else:
        crrt_df_filtered = None
        print("CRRT therapy table not found")

    # ECMO/MCS - will show as no data available
    ecmo_df = load_clif_table('clif_ecmo_mcs', config)
    if ecmo_df is not None:
        ecmo_df_filtered = ecmo_df[ecmo_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        print(f"ECMO/MCS records: {len(ecmo_df)} -> {len(ecmo_df_filtered)} (filtered)")
    else:
        ecmo_df_filtered = None
        print("ECMO/MCS table not found")

    # Patient procedures - will show as no data available
    proc_df = load_clif_table('clif_patient_procedure', config)
    if proc_df is not None:
        proc_df_filtered = proc_df[proc_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        print(f"Patient procedure records: {len(proc_df)} -> {len(proc_df_filtered)} (filtered)")
    else:
        proc_df_filtered = None
        print("Patient procedure table not found")

    # Microbiology culture
    micro_culture_df = load_clif_table('clif_microbiology_culture', config)
    if micro_culture_df is not None:
        micro_culture_df_filtered = micro_culture_df[micro_culture_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        print(f"Microbiology culture records: {len(micro_culture_df)} -> {len(micro_culture_df_filtered)} (filtered)")
    else:
        micro_culture_df_filtered = None
        print("Microbiology culture table not found")

    # Microbiology non-culture
    micro_non_culture_df = load_clif_table('clif_microbiology_non_culture', config)
    if micro_non_culture_df is not None:
        micro_non_culture_df_filtered = micro_non_culture_df[micro_non_culture_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        print(f"Microbiology non-culture records: {len(micro_non_culture_df)} -> {len(micro_non_culture_df_filtered)} (filtered)")
    else:
        micro_non_culture_df_filtered = None
        print("Microbiology non-culture table not found")

    # Microbiology susceptibility (filtered by organism_id from culture table)
    micro_susceptibility_df = load_clif_table('clif_microbiology_susceptibility', config)
    if micro_susceptibility_df is not None and micro_culture_df_filtered is not None and len(micro_culture_df_filtered) > 0:
        # Get organism IDs from filtered culture data
        transplant_organism_ids = micro_culture_df_filtered['organism_id'].unique()
        micro_susceptibility_df_filtered = micro_susceptibility_df[micro_susceptibility_df['organism_id'].isin(transplant_organism_ids)]
        print(f"Microbiology susceptibility records: {len(micro_susceptibility_df)} -> {len(micro_susceptibility_df_filtered)} (filtered)")
    else:
        micro_susceptibility_df_filtered = None
        if micro_susceptibility_df is None:
            print("Microbiology susceptibility table not found")
        else:
            print("Microbiology susceptibility table found but no culture data to filter against")

    return crrt_df_filtered, ecmo_df_filtered, proc_df_filtered, micro_culture_df_filtered, micro_non_culture_df_filtered, micro_susceptibility_df_filtered


@app.cell
def _(config, hospitalization_df, load_clif_table):
    # Load and filter additional tables
    print("Loading and filtering additional tables...")

    # Code status
    code_status_df = load_clif_table('clif_code_status', config)
    if code_status_df is not None:
        code_status_df_filtered = code_status_df[code_status_df['patient_id'].isin(hospitalization_df['patient_id'])]
        print(f"Code status records: {len(code_status_df)} -> {len(code_status_df_filtered)} (filtered)")
    else:
        code_status_df_filtered = None
        print("Code status table not found")

    # Hospital diagnosis
    hosp_diag_df = load_clif_table('clif_hospital_diagnosis', config)
    if hosp_diag_df is not None:
        hosp_diag_df_filtered = hosp_diag_df[hosp_diag_df['hospitalization_id'].isin(hospitalization_df['hospitalization_id'])]
        print(f"Hospital diagnosis records: {len(hosp_diag_df)} -> {len(hosp_diag_df_filtered)} (filtered)")
    else:
        hosp_diag_df_filtered = None
        print("Hospital diagnosis table not found")

    return code_status_df_filtered, hosp_diag_df_filtered


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 5: Create Valeos Output Directory and Export Tables""")
    return


@app.cell
def _(Path, config, logger):
    # Create Valeos output directory
    logger.info("Step 8: Setting up Valeos output directory...")
    tables_path = Path(config['tables_path'])
    valeos_dir = tables_path / 'Valeos'
    valeos_dir.mkdir(exist_ok=True)

    logger.info(f"✓ Created/verified Valeos directory: {valeos_dir}")
    print(f"Created Valeos directory: {valeos_dir}")

    # Get site name for file naming
    site_name = config['site_name']
    logger.info(f"✓ Site name for file naming: {site_name}")
    logger.info("✓ Step 8 completed - Output directory ready")

    return site_name, valeos_dir


@app.cell
def _(
    transplant_hospitalization,
    adt_df_filtered,
    code_status_df_filtered,
    crrt_df_filtered,
    ecmo_df_filtered,
    hosp_diag_df_filtered,
    hospitalization_df,
    # labs_df_filtered,
    logger,
    med_cont_df,
    # med_int_df_filtered,
    micro_culture_df_filtered,
    micro_non_culture_df_filtered,
    micro_susceptibility_df_filtered,
    patient_df,
    proc_df_filtered,
    resp_df_filtered,
    site_name,
    transplant_hosp_ids,
    valeos_dir,
    vitals_df,
):
    # Filter continuous medication data to transplant hospitalizations
    logger.info("Step 9: Finalizing data filtering and exporting tables...")
    logger.info("  - Filtering continuous medication data to transplant hospitalizations...")
    
    if med_cont_df is not None:
        original_med_cont = len(med_cont_df)
        transplant_hosp_ids = transplant_hospitalization['hospitalization_id'].unique()
        med_cont_df_filtered = med_cont_df[med_cont_df['hospitalization_id'].isin(transplant_hosp_ids)]
        logger.info(f"  ✓ Continuous medications: {original_med_cont} → {len(med_cont_df_filtered)} records")
        print(f"Continuous medication records: {len(med_cont_df)} -> {len(med_cont_df_filtered)} (filtered)")
    else:
        med_cont_df_filtered = None
        logger.warning("  - No continuous medication data available")

    # Dictionary of tables to export
    tables_to_export = {
        'patient': patient_df,
        'hospitalization': hospitalization_df,
        'vitals': vitals_df,
        # 'labs': labs_df_filtered,
        'medication_admin_continuous': med_cont_df_filtered,
        # 'medication_admin_intermittent': med_int_df_filtered,
        'respiratory_support': resp_df_filtered,
        'adt': adt_df_filtered,
        'crrt_therapy': crrt_df_filtered,
        'ecmo_mcs': ecmo_df_filtered,
        'patient_procedure': proc_df_filtered,
        'code_status': code_status_df_filtered,
        'hospital_diagnosis': hosp_diag_df_filtered,
        'microbiology_culture': micro_culture_df_filtered,
        'microbiology_non_culture': micro_non_culture_df_filtered,
        'microbiology_susceptibility': micro_susceptibility_df_filtered
    }

    # Export each table as parquet
    logger.info("  - Exporting tables to parquet format...")
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
                logger.info(f"  ✓ Exported {export_tbl_name}: {len(export_tbl_df)} records")
                print(f"Exported {export_tbl_name}: {len(export_tbl_df)} records -> {filepath}")
            except Exception as e:
                logger.error(f"  ✗ Failed to export {export_tbl_name}: {str(e)}")
        else:
            tables_without_data += 1
            logger.info(f"  - Skipped {export_tbl_name}: No data available")
            print(f"Skipped {export_tbl_name}: No data available")

    logger.info(f"✓ Export summary: {tables_with_data} tables exported, {tables_without_data} tables skipped")
    logger.info(f"✓ Step 9 completed - All clinical data tables exported")
    print(f"\nExported {len(exported_files)} tables to {valeos_dir}")
    return exported_files, tables_to_export


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 6: Create and Export Transplant Table""")
    return


@app.cell
def _(logger, site_name, transplant_df, valeos_dir):
    # Create the transplant table with unique transplant records
    logger.info("Step 10: Creating and exporting transplant table...")
    transplant_table = transplant_df.copy()

    # Rename 'organ' to 'transplant_type' for clarity
    transplant_table = transplant_table.rename(columns={'organ': 'transplant_type'})
    logger.info("  ✓ Renamed 'organ' to 'transplant_type'")

    # Ensure proper column order
    transplant_table = transplant_table[['patient_id', 'transplant_date', 'transplant_type', 'transplant_number', 'total_transplants']]
    logger.info("  ✓ Finalized column structure")

    # Export transplant table
    transplant_filename = f"{site_name}_valeos_inpatient_transplant.parquet"
    transplant_filepath = valeos_dir / transplant_filename
    
    try:
        transplant_table.to_parquet(transplant_filepath, index=False)
        logger.info(f"  ✓ Exported transplant table: {len(transplant_table)} records")
        logger.info(f"  ✓ File location: {transplant_filepath}")
        logger.info(f"✓ Step 10 completed - Transplant table exported")
        
        print(f"Exported transplant table: {len(transplant_table)} records -> {transplant_filepath}")
        print(f"Transplant table columns: {list(transplant_table.columns)}")
    except Exception as e:
        logger.error(f"  ✗ Failed to export transplant table: {str(e)}")

    return (transplant_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Step 7: Summary Statistics and Validation""")
    return


@app.cell
def _(
    dt_now,
    exported_files,
    logger,
    site_name,
    tables_to_export,
    transplant_table,
    valeos_dir,
):
    logger.info("Step 11: Generating final summary and statistics...")
    
    print("=== VALEOS DATABASE EXPORT SUMMARY ===")
    print(f"Site: {site_name}")
    print(f"Output directory: {valeos_dir}")
    print(f"Total files exported: {len(exported_files) + 1}")  # +1 for transplant table
    print()

    # Log detailed export summary
    logger.info("=== FINAL EXPORT SUMMARY ===")
    logger.info(f"Site: {site_name}")
    logger.info(f"Output directory: {valeos_dir}")
    logger.info(f"Total files exported: {len(exported_files) + 1}")

    print("Table Export Summary:")
    logger.info("Table Export Summary:")
    summary_tables_with_data = 0
    summary_tables_without_data = 0
    total_records_exported = 0
    
    for tbl_name, tbl_df in tables_to_export.items():
        if tbl_df is not None and len(tbl_df) > 0:
            summary_tables_with_data += 1
            total_records_exported += len(tbl_df)
            logger.info(f"  ✓ {tbl_name}: {len(tbl_df):,} records")
            print(f"  ✓ {tbl_name}: {len(tbl_df):,} records")
        else:
            summary_tables_without_data += 1
            logger.info(f"  ✗ {tbl_name}: No data")
            print(f"  ✗ {tbl_name}: No data")

    # Add transplant table to summary
    total_records_exported += len(transplant_table)
    logger.info(f"  ✓ transplant: {len(transplant_table):,} records")
    print(f"  ✓ transplant: {len(transplant_table):,} records")
    print()

    print("Transplant Summary:")
    logger.info("Transplant Summary by Organ:")
    transplant_counts = transplant_table['transplant_type'].value_counts()
    for organ_type, transplant_count in transplant_counts.items():
        logger.info(f"  - {organ_type}: {transplant_count:,} transplants")
        print(f"  {organ_type}: {transplant_count:,} transplants")

    unique_patients = transplant_table['patient_id'].nunique()
    total_procedures = len(transplant_table)
    
    logger.info(f"Total unique transplant patients: {unique_patients:,}")
    logger.info(f"Total transplant procedures: {total_procedures:,}")
    logger.info(f"Total clinical records exported: {total_records_exported:,}")
    
    print(f"\nTotal unique transplant patients: {unique_patients:,}")
    print(f"Total transplant procedures: {total_procedures:,}")

    # Multi-transplant patients
    multi_transplant = transplant_table[transplant_table['total_transplants'] > 1]
    if len(multi_transplant) > 0:
        multi_patients = multi_transplant['patient_id'].nunique()
        logger.info(f"Patients with multiple transplants: {multi_patients:,}")
        print(f"Patients with multiple transplants: {multi_patients:,}")

    # Final completion log
    logger.info(f"Processing completed at: {dt_now.now()}")
    logger.info(f"✓ SUCCESS: All {summary_tables_with_data + 1} tables exported successfully")
    logger.info(f"Tables with data: {summary_tables_with_data + 1}, Tables skipped: {summary_tables_without_data}")
    logger.info("=== CLIF TO VALEOS PROCESSING COMPLETED SUCCESSFULLY ===")

    return


if __name__ == "__main__":
    app.run()
