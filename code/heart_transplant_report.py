import marimo

__generated_with = "0.14.17"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(mo):
    mo.md("""# Setup & Configuration""")
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
        Patient, Hospitalization,
        Vitals, Labs, RespiratorySupport,
        MedicationAdminContinuous,
        CrrtTherapy, EcmoMcs, PatientProcedures
    )

    # Heart transplant CPT codes
    HEART_TRANSPLANT_CPTS = ['33945', '33935']

    # Suppression threshold for small cell protection
    SUPPRESSION_THRESHOLD = 10
    return (
        CrrtTherapy,
        EcmoMcs,
        HEART_TRANSPLANT_CPTS,
        Hospitalization,
        Labs,
        MedicationAdminContinuous,
        Path,
        Patient,
        PatientProcedures,
        RespiratorySupport,
        SUPPRESSION_THRESHOLD,
        Vitals,
        alt,
        datetime,
        json,
        load_config,
        mo,
        pd,
        plt,
    )


@app.cell
def _(load_config):
    # Load configuration
    config = load_config()
    print(f"Site: {config['site_name']}")
    print(f"Tables path: {config['tables_path']}")
    return (config,)


@app.cell
def _(SUPPRESSION_THRESHOLD):
    # Suppression helper functions for small cell protection

    def suppress_count(count, threshold=SUPPRESSION_THRESHOLD):
        """Return suppressed string if count < threshold"""
        if count < threshold:
            return "< 10"
        return str(count)

    def suppress_value(count, value, threshold=SUPPRESSION_THRESHOLD):
        """Return value or suppressed marker based on count"""
        if count < threshold:
            return "--"
        return value

    def format_count_pct(count, total, threshold=SUPPRESSION_THRESHOLD):
        """Format as 'N (%)' with suppression"""
        if count < threshold:
            return "< 10 (--)"
        if total == 0:
            return f"{count} (--)"
        pct = (count / total * 100)
        return f"{count} ({pct:.1f}%)"

    def is_suppressed(count, threshold=SUPPRESSION_THRESHOLD):
        """Check if count should be suppressed"""
        return count < threshold

    def format_median_iqr(series, count, threshold=SUPPRESSION_THRESHOLD):
        """Format median [IQR] with suppression"""
        if count < threshold or len(series) == 0:
            return "--"
        median = series.median()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        return f"{median:.1f} [{q25:.1f}-{q75:.1f}]"

    def format_mean_sd(series, count, threshold=SUPPRESSION_THRESHOLD):
        """Format mean ± SD with suppression"""
        if count < threshold or len(series) == 0:
            return "--"
        mean = series.mean()
        std = series.std()
        return f"{mean:.1f} ± {std:.1f}"
    return (
        format_count_pct,
        format_mean_sd,
        format_median_iqr,
        is_suppressed,
        suppress_count,
    )


@app.cell
def _(Hospitalization, Path, Patient, config):
    # Setup paths and load core tables using clifpy
    tables_path = Path(config['tables_path'])
    output_dir = tables_path.parent / 'output' / 'final'
    site_name = config['site_name']
    file_type = config.get('file_type', 'parquet')

    # Load Patient table using clifpy
    try:
        patient_table = Patient.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone='UTC'
        )
        heart_patient_df = patient_table.df
        print(f"Loaded patient via clifpy: {len(heart_patient_df):,} records")
    except Exception as e:
        print(f"Error loading patient table: {e}")
        heart_patient_df = None

    # Load Hospitalization table using clifpy
    try:
        hosp_table = Hospitalization.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone='UTC'
        )
        heart_hospitalization_df = hosp_table.df
        print(f"Loaded hospitalization via clifpy: {len(heart_hospitalization_df):,} records")
    except Exception as e:
        print(f"Error loading hospitalization table: {e}")
        heart_hospitalization_df = None
    return (
        file_type,
        heart_hospitalization_df,
        heart_patient_df,
        output_dir,
        site_name,
        tables_path,
    )


@app.cell(column=1)
def _(mo):
    mo.md("""# Data Loading & Cohort Identification""")
    return


@app.cell
def _(heart_transplant_df, pd):
    # Filter to heart transplants only
    if heart_transplant_df is not None:
        heart_only_df = heart_transplant_df[
            heart_transplant_df['transplant_type'] == 'heart'
        ].copy()
        heart_only_df['transplant_date'] = pd.to_datetime(
            heart_only_df['transplant_date'], utc=True
        )
        heart_patient_ids = heart_only_df['patient_id'].unique()
        total_heart_n = len(heart_patient_ids)
        print(f"Heart transplant patients: {total_heart_n}")
    else:
        heart_only_df = None
        heart_patient_ids = []
        total_heart_n = 0
        print("No transplant data available")
    return heart_only_df, heart_patient_ids, total_heart_n


@app.cell
def _(heart_hospitalization_df, heart_only_df, heart_patient_ids, pd):
    # Get hospitalization IDs for heart transplant patients
    if heart_hospitalization_df is not None and len(heart_patient_ids) > 0:
        # Filter hospitalizations to heart transplant patients
        heart_hosp_df = heart_hospitalization_df[
            heart_hospitalization_df['patient_id'].isin(heart_patient_ids)
        ].copy()

        # Convert datetime columns
        heart_hosp_df['admission_dttm'] = pd.to_datetime(
            heart_hosp_df['admission_dttm'], utc=True
        )
        heart_hosp_df['discharge_dttm'] = pd.to_datetime(
            heart_hosp_df['discharge_dttm'], utc=True
        )

        # Merge with transplant data to get transplant hospitalization
        heart_hosp_merged = pd.merge(
            heart_hosp_df,
            heart_only_df[['patient_id', 'transplant_date']],
            on='patient_id',
            how='inner'
        )

        # Find the hospitalization containing the transplant
        heart_transplant_hosp = heart_hosp_merged[
            (heart_hosp_merged['admission_dttm'] <= heart_hosp_merged['transplant_date']) &
            (heart_hosp_merged['discharge_dttm'] >= heart_hosp_merged['transplant_date'])
        ].copy()

        heart_hosp_ids = heart_transplant_hosp['hospitalization_id'].unique()
        print(f"Heart transplant hospitalizations: {len(heart_hosp_ids)}")
    else:
        heart_hosp_df = None
        heart_hosp_merged = None
        heart_transplant_hosp = None
        heart_hosp_ids = []
        print("No hospitalization data available")
    return heart_hosp_ids, heart_transplant_hosp


@app.cell
def _(
    HEART_TRANSPLANT_CPTS,
    PatientProcedures,
    file_type,
    heart_hospitalization_df,
    pd,
    tables_path,
):
    # Load Procedure table and identify heart transplants via CPT codes
    try:
        proc_table = PatientProcedures.from_file(
            data_directory=str(tables_path),
            filetype=file_type,
            timezone='UTC'
        )
        procedures_df = proc_table.df
        print(f"Loaded procedures via clifpy: {len(procedures_df):,} records")

        # Filter to heart transplant CPT codes
        heart_procedures = procedures_df[
            procedures_df['procedure_code'].astype(str).isin(HEART_TRANSPLANT_CPTS)
        ].copy()

        if len(heart_procedures) > 0:
            # Create transplant records with patient_id, hospitalization_id, and transplant_date
            heart_transplant_df = heart_procedures[['hospitalization_id', 'procedure_billed_dttm']].copy()
            heart_transplant_df = heart_transplant_df.rename(columns={'procedure_billed_dttm': 'transplant_date'})

            # Merge with hospitalization to get patient_id
            if heart_hospitalization_df is not None:
                heart_transplant_df = pd.merge(
                    heart_transplant_df,
                    heart_hospitalization_df[['hospitalization_id', 'patient_id']],
                    on='hospitalization_id',
                    how='inner'
                )
            heart_transplant_df['transplant_type'] = 'heart'
            print(f"Found heart transplants via CPT codes: {len(heart_transplant_df):,} procedures")
        else:
            heart_transplant_df = None
            print("No heart transplant CPT codes found in procedures")
    except Exception as e:
        print(f"Error loading procedures: {e}")
        heart_transplant_df = None
    return (heart_transplant_df,)


@app.cell(column=2)
def _(mo):
    mo.md("""# Clinical Data & Perioperative Filtering""")
    return


@app.cell
def _(
    CrrtTherapy,
    EcmoMcs,
    Labs,
    MedicationAdminContinuous,
    RespiratorySupport,
    Vitals,
    file_type,
    heart_hosp_ids,
    tables_path,
):
    # Load clinical tables using clifpy with hospitalization_id filters
    heart_hosp_ids_list = list(heart_hosp_ids) if len(heart_hosp_ids) > 0 else []

    # Load Vitals using clifpy
    if len(heart_hosp_ids_list) > 0:
        try:
            vitals_table = Vitals.from_file(
                data_directory=str(tables_path),
                filetype=file_type,
                timezone='UTC',
                filters={'hospitalization_id': heart_hosp_ids_list}
            )
            heart_vitals_df = vitals_table.df
            print(f"Loaded vitals via clifpy: {len(heart_vitals_df):,} records")
        except Exception as e:
            print(f"Error loading vitals: {e}")
            heart_vitals_df = None

        # Load Labs using clifpy
        try:
            labs_table = Labs.from_file(
                data_directory=str(tables_path),
                filetype=file_type,
                timezone='UTC',
                filters={'hospitalization_id': heart_hosp_ids_list}
            )
            heart_labs_df = labs_table.df
            print(f"Loaded labs via clifpy: {len(heart_labs_df):,} records")
        except Exception as e:
            print(f"Error loading labs: {e}")
            heart_labs_df = None

        # Load Medications using clifpy
        try:
            med_table = MedicationAdminContinuous.from_file(
                data_directory=str(tables_path),
                filetype=file_type,
                timezone='UTC',
                filters={'hospitalization_id': heart_hosp_ids_list}
            )
            heart_med_continuous_df = med_table.df
            print(f"Loaded medications via clifpy: {len(heart_med_continuous_df):,} records")
        except Exception as e:
            print(f"Error loading medications: {e}")
            heart_med_continuous_df = None

        # Load Respiratory Support using clifpy
        try:
            resp_table = RespiratorySupport.from_file(
                data_directory=str(tables_path),
                filetype=file_type,
                timezone='UTC',
                filters={'hospitalization_id': heart_hosp_ids_list}
            )
            heart_respiratory_df = resp_table.df
            print(f"Loaded respiratory via clifpy: {len(heart_respiratory_df):,} records")
        except Exception as e:
            print(f"Error loading respiratory: {e}")
            heart_respiratory_df = None

        # Load ECMO/MCS using clifpy
        try:
            ecmo_table = EcmoMcs.from_file(
                data_directory=str(tables_path),
                filetype=file_type,
                timezone='UTC',
                filters={'hospitalization_id': heart_hosp_ids_list}
            )
            heart_ecmo_df = ecmo_table.df
            print(f"Loaded ECMO via clifpy: {len(heart_ecmo_df):,} records")
        except Exception as e:
            print(f"ECMO table not available: {e}")
            heart_ecmo_df = None

        # Load CRRT using clifpy
        try:
            crrt_table = CrrtTherapy.from_file(
                data_directory=str(tables_path),
                filetype=file_type,
                timezone='UTC',
                filters={'hospitalization_id': heart_hosp_ids_list}
            )
            heart_crrt_df = crrt_table.df
            print(f"Loaded CRRT via clifpy: {len(heart_crrt_df):,} records")
        except Exception as e:
            print(f"CRRT table not available: {e}")
            heart_crrt_df = None
    else:
        heart_vitals_df = None
        heart_labs_df = None
        heart_med_continuous_df = None
        heart_respiratory_df = None
        heart_ecmo_df = None
        heart_crrt_df = None
        print("No hospitalization IDs to filter clinical data")
    return (
        heart_crrt_df,
        heart_ecmo_df,
        heart_labs_df,
        heart_med_continuous_df,
        heart_respiratory_df,
        heart_vitals_df,
    )


@app.cell
def _(heart_hosp_ids, heart_transplant_hosp, heart_vitals_df, pd):
    # Filter vitals to perioperative window (±14 days)
    PERIOP_DAYS = 14

    if heart_vitals_df is not None and len(heart_hosp_ids) > 0:
        # Filter to heart transplant hospitalizations
        periop_vitals_df = heart_vitals_df[
            heart_vitals_df['hospitalization_id'].isin(heart_hosp_ids)
        ].copy()

        # Convert datetime
        periop_vitals_df['recorded_dttm'] = pd.to_datetime(
            periop_vitals_df['recorded_dttm'], utc=True
        )

        # Merge with transplant dates
        periop_vitals_df = pd.merge(
            periop_vitals_df,
            heart_transplant_hosp[['hospitalization_id', 'transplant_date']],
            on='hospitalization_id',
            how='inner'
        )

        # Calculate days from transplant
        periop_vitals_df['days_from_transplant'] = (
            periop_vitals_df['recorded_dttm'] - periop_vitals_df['transplant_date']
        ).dt.total_seconds() / (24 * 3600)

        # Filter to ±14 days
        periop_vitals_df = periop_vitals_df[
            (periop_vitals_df['days_from_transplant'] >= -PERIOP_DAYS) &
            (periop_vitals_df['days_from_transplant'] <= PERIOP_DAYS)
        ]

        print(f"Perioperative vitals records: {len(periop_vitals_df):,}")
    else:
        periop_vitals_df = None
        print("No vitals data available")
    return PERIOP_DAYS, periop_vitals_df


@app.cell
def _(PERIOP_DAYS, heart_hosp_ids, heart_labs_df, heart_transplant_hosp, pd):
    # Filter labs to perioperative window
    if heart_labs_df is not None and len(heart_hosp_ids) > 0:
        periop_labs_df = heart_labs_df[
            heart_labs_df['hospitalization_id'].isin(heart_hosp_ids)
        ].copy()

        periop_labs_df['lab_result_dttm'] = pd.to_datetime(
            periop_labs_df['lab_result_dttm'], utc=True
        )

        periop_labs_df = pd.merge(
            periop_labs_df,
            heart_transplant_hosp[['hospitalization_id', 'transplant_date']],
            on='hospitalization_id',
            how='inner'
        )

        periop_labs_df['days_from_transplant'] = (
            periop_labs_df['lab_result_dttm'] - periop_labs_df['transplant_date']
        ).dt.total_seconds() / (24 * 3600)

        periop_labs_df = periop_labs_df[
            (periop_labs_df['days_from_transplant'] >= -PERIOP_DAYS) &
            (periop_labs_df['days_from_transplant'] <= PERIOP_DAYS)
        ]

        print(f"Perioperative labs records: {len(periop_labs_df):,}")
    else:
        periop_labs_df = None
        print("No labs data available")
    return (periop_labs_df,)


@app.cell
def _(
    PERIOP_DAYS,
    heart_hosp_ids,
    heart_med_continuous_df,
    heart_transplant_hosp,
    pd,
):
    # Filter medications to vasoactives in perioperative window
    VASOACTIVE_MEDS = [
        'norepinephrine', 'epinephrine', 'dopamine', 'dobutamine',
        'milrinone', 'vasopressin', 'phenylephrine', 'angiotensin'
    ]

    if heart_med_continuous_df is not None and len(heart_hosp_ids) > 0:
        # Filter to vasoactives
        periop_meds_df = heart_med_continuous_df[
            heart_med_continuous_df['med_category'].isin(VASOACTIVE_MEDS)
        ].copy()

        periop_meds_df['admin_dttm'] = pd.to_datetime(
            periop_meds_df['admin_dttm'], utc=True
        )

        periop_meds_df = pd.merge(
            periop_meds_df,
            heart_transplant_hosp[['hospitalization_id', 'transplant_date']],
            on='hospitalization_id',
            how='inner'
        )

        periop_meds_df['days_from_transplant'] = (
            periop_meds_df['admin_dttm'] - periop_meds_df['transplant_date']
        ).dt.total_seconds() / (24 * 3600)

        periop_meds_df = periop_meds_df[
            (periop_meds_df['days_from_transplant'] >= -PERIOP_DAYS) &
            (periop_meds_df['days_from_transplant'] <= PERIOP_DAYS)
        ]

        print(f"Perioperative vasoactive records: {len(periop_meds_df):,}")
    else:
        periop_meds_df = None
        print("No medication data available")
    return VASOACTIVE_MEDS, periop_meds_df


@app.cell
def _(
    PERIOP_DAYS,
    heart_hosp_ids,
    heart_respiratory_df,
    heart_transplant_hosp,
    pd,
):
    # Filter respiratory data to perioperative window
    if heart_respiratory_df is not None and len(heart_hosp_ids) > 0:
        periop_resp_df = heart_respiratory_df.copy()

        periop_resp_df['recorded_dttm'] = pd.to_datetime(
            periop_resp_df['recorded_dttm'], utc=True
        )

        periop_resp_df = pd.merge(
            periop_resp_df,
            heart_transplant_hosp[['hospitalization_id', 'transplant_date']],
            on='hospitalization_id',
            how='inner'
        )

        periop_resp_df['days_from_transplant'] = (
            periop_resp_df['recorded_dttm'] - periop_resp_df['transplant_date']
        ).dt.total_seconds() / (24 * 3600)

        periop_resp_df = periop_resp_df[
            (periop_resp_df['days_from_transplant'] >= -PERIOP_DAYS) &
            (periop_resp_df['days_from_transplant'] <= PERIOP_DAYS)
        ]

        print(f"Perioperative respiratory records: {len(periop_resp_df):,}")
    else:
        periop_resp_df = None
        print("No respiratory data available")
    return (periop_resp_df,)


@app.cell
def _():
    return


@app.cell(column=3)
def _(mo):
    mo.md("""# Report & Visualizations""")
    return


@app.cell
def _(datetime, heart_only_df, mo, site_name, suppress_count, total_heart_n):
    # Report Header
    _generation_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    if heart_only_df is not None and len(heart_only_df) > 0:
        _earliest = heart_only_df['transplant_date'].min().strftime("%Y-%m-%d")
        _latest = heart_only_df['transplant_date'].max().strftime("%Y-%m-%d")
        _date_range = f"{_earliest} to {_latest}"
    else:
        _date_range = "N/A"

    report_header_md = mo.md(f"""
    # Heart Transplant Federated Report

    **Site:** {site_name}
    **Generated:** {_generation_date}
    **Total Heart Transplant Recipients:** {suppress_count(total_heart_n)}
    **Date Range:** {_date_range}

    ---

    **Privacy Notice:** This report contains aggregate data only. All counts < 10 are suppressed
    to protect patient privacy and prevent potential re-identification.
    """)
    report_header_md
    return


@app.cell
def _(mo):
    mo.md("""## Patient Demographics""")
    return


@app.cell
def _(mo):
    mo.md("""## Transplant Volume""")
    return


@app.cell
def _(mo):
    mo.md("""## Perioperative Vitals (±14 days from transplant)""")
    return


@app.cell
def _(mo):
    mo.md("""## Perioperative Labs (±14 days from transplant)""")
    return


@app.cell
def _(mo):
    mo.md("""## Vasoactive Medication Use (±14 days from transplant)""")
    return


@app.cell
def _(mo):
    mo.md("""## Respiratory Support (±14 days from transplant)""")
    return


@app.cell
def _(mo):
    mo.md("""## Advanced Therapies""")
    return


@app.cell
def _(mo):
    mo.md("""## Outcomes""")
    return


@app.cell
def _(mo):
    mo.md("""## Export Federated Results""")
    return


@app.cell
def _(
    format_count_pct,
    format_mean_sd,
    format_median_iqr,
    heart_only_df,
    heart_patient_df,
    is_suppressed,
    mo,
    pd,
    total_heart_n,
):
    # Demographics Table with suppression
    def create_demographics_table():
        if heart_patient_df is None or heart_only_df is None:
            return "No demographic data available"

        if total_heart_n == 0:
            return "No heart transplant patients found"

        # Merge patient demographics with transplant data
        _merged = pd.merge(
            heart_patient_df,
            heart_only_df[['patient_id', 'transplant_date']],
            on='patient_id',
            how='inner'
        )

        # Get unique patients (in case of multiple transplants)
        _unique_patients = _merged.drop_duplicates('patient_id')
        _n = len(_unique_patients)

        # Calculate age at transplant
        if 'birth_date' in _unique_patients.columns:
            _unique_patients = _unique_patients.copy()
            _unique_patients['birth_date'] = pd.to_datetime(
                _unique_patients['birth_date']
            ).dt.tz_localize('UTC')
            _unique_patients['age_at_transplant'] = (
                _unique_patients['transplant_date'] - _unique_patients['birth_date']
            ).dt.days / 365.25

        # Build demographics table rows
        _rows = []
        _rows.append(f"| **Characteristic** | **Value** |")
        _rows.append(f"|:---|:---|")
        _rows.append(f"| **N** | {_n if not is_suppressed(_n) else '< 10'} |")

        # Age statistics
        if 'age_at_transplant' in _unique_patients.columns:
            _ages = _unique_patients['age_at_transplant'].dropna()
            _rows.append(f"| Age at transplant, mean ± SD | {format_mean_sd(_ages, _n)} |")
            _rows.append(f"| Age at transplant, median [IQR] | {format_median_iqr(_ages, _n)} |")

        # Sex distribution
        if 'sex_category' in _unique_patients.columns:
            _rows.append(f"| **Sex** | |")
            for _sex in ['Male', 'Female']:
                _count = len(_unique_patients[_unique_patients['sex_category'] == _sex])
                _rows.append(f"| - {_sex} | {format_count_pct(_count, _n)} |")

        # Race distribution
        if 'race_category' in _unique_patients.columns:
            _rows.append(f"| **Race** | |")
            _race_counts = _unique_patients['race_category'].value_counts()
            for _race in _race_counts.index[:5]:  # Top 5 categories
                _count = _race_counts[_race]
                _rows.append(f"| - {_race} | {format_count_pct(_count, _n)} |")

        # Ethnicity distribution
        if 'ethnicity_category' in _unique_patients.columns:
            _rows.append(f"| **Ethnicity** | |")
            _eth_counts = _unique_patients['ethnicity_category'].value_counts()
            for _eth in _eth_counts.index:
                _count = _eth_counts[_eth]
                _rows.append(f"| - {_eth} | {format_count_pct(_count, _n)} |")

        return "\n".join(_rows)

    demographics_table_md = mo.md(create_demographics_table())
    demographics_table_md
    return


@app.cell
def _(SUPPRESSION_THRESHOLD, alt, heart_only_df, mo, site_name):
    # Annual volume chart with suppression
    def create_annual_volume_chart():
        if heart_only_df is None or len(heart_only_df) == 0:
            return None, "No transplant data available"

        _work_df = heart_only_df.copy()
        _work_df['year'] = _work_df['transplant_date'].dt.year

        _yearly_counts = _work_df.groupby('year').size().reset_index(name='count')

        # Apply suppression - remove years with < 10
        _display_df = _yearly_counts[
            _yearly_counts['count'] >= SUPPRESSION_THRESHOLD
        ].copy()

        if len(_display_df) == 0:
            return None, "All years suppressed (< 10 transplants per year)"

        _chart = alt.Chart(_display_df).mark_bar(
            color='#2ca02c'
        ).encode(
            x=alt.X('year:O', axis=alt.Axis(title='Year', labelAngle=0)),
            y=alt.Y('count:Q', axis=alt.Axis(title='Number of Transplants')),
            tooltip=[
                alt.Tooltip('year:O', title='Year'),
                alt.Tooltip('count:Q', title='Count')
            ]
        ).properties(
            width=500,
            height=300,
            title=f'Annual Heart Transplant Volume - {site_name}'
        )

        return _chart, None

    annual_chart, annual_chart_error = create_annual_volume_chart()

    if annual_chart is not None:
        annual_chart
    else:
        mo.md(f"*{annual_chart_error}*")
    return


@app.cell
def _(SUPPRESSION_THRESHOLD, mo, periop_vitals_df, plt):
    # Vitals trajectory plot
    def create_vitals_trajectory():
        if periop_vitals_df is None or len(periop_vitals_df) == 0:
            return None, "No vitals data available"

        # Vitals to plot
        _vitals_config = {
            'heart_rate': {'label': 'Heart Rate (bpm)', 'color': '#e41a1c'},
            'sbp': {'label': 'Systolic BP (mmHg)', 'color': '#377eb8'},
            'dbp': {'label': 'Diastolic BP (mmHg)', 'color': '#4daf4a'},
            'spo2': {'label': 'SpO2 (%)', 'color': '#984ea3'},
            'temp_c': {'label': 'Temperature (°C)', 'color': '#ff7f00'},
            'respiratory_rate': {'label': 'Resp Rate (breaths/min)', 'color': '#a65628'},
        }

        # Get available vitals
        _available_vitals = [
            v for v in _vitals_config.keys()
            if v in periop_vitals_df['vital_category'].unique()
        ]

        if len(_available_vitals) == 0:
            return None, "No recognized vital categories in data"

        _n_vitals = len(_available_vitals)
        _n_cols = 2
        _n_rows = (_n_vitals + 1) // 2

        _fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(14, 4 * _n_rows))
        _axes = _axes.flatten() if _n_vitals > 1 else [_axes]

        for _idx, _vital in enumerate(_available_vitals):
            _ax = _axes[_idx]
            _config = _vitals_config[_vital]

            # Filter to this vital
            _vital_data = periop_vitals_df[
                periop_vitals_df['vital_category'] == _vital
            ].copy()

            if len(_vital_data) == 0:
                _ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=_ax.transAxes)
                _ax.set_title(_config['label'])
                continue

            # Round days and calculate statistics
            _vital_data['day_rounded'] = _vital_data['days_from_transplant'].round()

            _daily_stats = _vital_data.groupby('day_rounded').agg(
                median=('vital_value', 'median'),
                q25=('vital_value', lambda x: x.quantile(0.25)),
                q75=('vital_value', lambda x: x.quantile(0.75)),
                n=('vital_value', 'count')
            ).reset_index()

            # Suppress days with < 10 observations
            _daily_stats = _daily_stats[_daily_stats['n'] >= SUPPRESSION_THRESHOLD]

            if len(_daily_stats) == 0:
                _ax.text(0.5, 0.5, 'Data suppressed\n(< 10 per day)', ha='center', va='center', transform=_ax.transAxes)
                _ax.set_title(_config['label'])
                continue

            # Plot IQR ribbon and median line
            _ax.fill_between(
                _daily_stats['day_rounded'],
                _daily_stats['q25'],
                _daily_stats['q75'],
                alpha=0.3,
                color=_config['color'],
                label='IQR'
            )
            _ax.plot(
                _daily_stats['day_rounded'],
                _daily_stats['median'],
                color=_config['color'],
                linewidth=2,
                marker='o',
                markersize=4,
                label='Median'
            )

            # Add transplant day reference line
            _ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Transplant')

            _ax.set_xlabel('Days from Transplant')
            _ax.set_ylabel(_config['label'])
            _ax.set_title(_config['label'])
            _ax.grid(True, alpha=0.3)
            _ax.set_xlim(-14, 14)

            if _idx == 0:
                _ax.legend(loc='upper right', fontsize=8)

        # Hide unused subplots
        for _idx in range(_n_vitals, len(_axes)):
            _axes[_idx].set_visible(False)

        plt.tight_layout()
        return _fig, None

    vitals_fig, vitals_error = create_vitals_trajectory()

    if vitals_fig is not None:
        mo.as_html(vitals_fig)
    else:
        mo.md(f"*{vitals_error}*")
    return


@app.cell
def _(SUPPRESSION_THRESHOLD, mo, periop_labs_df, plt):
    # Labs trajectory plot
    def create_labs_trajectory():
        if periop_labs_df is None or len(periop_labs_df) == 0:
            return None, "No labs data available"

        # Labs to plot (cardiac, renal, hepatic, hematologic, metabolic)
        _labs_config = {
            'troponin_i': {'label': 'Troponin I (ng/L)', 'color': '#e41a1c'},
            'troponin_t': {'label': 'Troponin T (ng/L)', 'color': '#d73027'},
            'creatinine': {'label': 'Creatinine (mg/dL)', 'color': '#377eb8'},
            'bun': {'label': 'BUN (mg/dL)', 'color': '#4575b4'},
            'bilirubin_total': {'label': 'Total Bilirubin (mg/dL)', 'color': '#ff7f00'},
            'alt': {'label': 'ALT (U/L)', 'color': '#fdae61'},
            'ast': {'label': 'AST (U/L)', 'color': '#fee090'},
            'inr': {'label': 'INR', 'color': '#984ea3'},
            'hemoglobin': {'label': 'Hemoglobin (g/dL)', 'color': '#4daf4a'},
            'platelet_count': {'label': 'Platelets (10³/μL)', 'color': '#66c2a5'},
            'lactate': {'label': 'Lactate (mmol/L)', 'color': '#a65628'},
            'sodium': {'label': 'Sodium (mmol/L)', 'color': '#f781bf'},
            'potassium': {'label': 'Potassium (mmol/L)', 'color': '#999999'},
        }

        _available_labs = [
            lab for lab in _labs_config.keys()
            if lab in periop_labs_df['lab_category'].unique()
        ]

        if len(_available_labs) == 0:
            return None, "No recognized lab categories in data"

        _n_labs = min(len(_available_labs), 12)  # Limit to 12 most relevant
        _available_labs = _available_labs[:_n_labs]
        _n_cols = 3
        _n_rows = (_n_labs + 2) // 3

        _fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(15, 4 * _n_rows))
        _axes = _axes.flatten() if _n_labs > 1 else [_axes]

        for _idx, _lab in enumerate(_available_labs):
            _ax = _axes[_idx]
            _config = _labs_config[_lab]

            _lab_data = periop_labs_df[
                periop_labs_df['lab_category'] == _lab
            ].copy()

            if len(_lab_data) == 0:
                _ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=_ax.transAxes)
                _ax.set_title(_config['label'])
                continue

            _lab_data['day_rounded'] = _lab_data['days_from_transplant'].round()

            _daily_stats = _lab_data.groupby('day_rounded').agg(
                median=('lab_value_numeric', 'median'),
                q25=('lab_value_numeric', lambda x: x.quantile(0.25)),
                q75=('lab_value_numeric', lambda x: x.quantile(0.75)),
                n=('lab_value_numeric', 'count')
            ).reset_index()

            _daily_stats = _daily_stats[_daily_stats['n'] >= SUPPRESSION_THRESHOLD]

            if len(_daily_stats) == 0:
                _ax.text(0.5, 0.5, 'Data suppressed\n(< 10 per day)', ha='center', va='center', transform=_ax.transAxes)
                _ax.set_title(_config['label'])
                continue

            _ax.fill_between(
                _daily_stats['day_rounded'],
                _daily_stats['q25'],
                _daily_stats['q75'],
                alpha=0.3,
                color=_config['color']
            )
            _ax.plot(
                _daily_stats['day_rounded'],
                _daily_stats['median'],
                color=_config['color'],
                linewidth=2,
                marker='o',
                markersize=3
            )

            _ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
            _ax.set_xlabel('Days from Transplant')
            _ax.set_title(_config['label'])
            _ax.grid(True, alpha=0.3)
            _ax.set_xlim(-14, 14)

        for _idx in range(_n_labs, len(_axes)):
            _axes[_idx].set_visible(False)

        plt.tight_layout()
        return _fig, None

    labs_fig, labs_error = create_labs_trajectory()

    if labs_fig is not None:
        mo.as_html(labs_fig)
    else:
        mo.md(f"*{labs_error}*")
    return


@app.cell
def _(VASOACTIVE_MEDS, format_count_pct, heart_hosp_ids, mo, periop_meds_df):
    # Vasoactive use rates table
    def create_vasoactive_table():
        if periop_meds_df is None or len(periop_meds_df) == 0:
            return "No vasoactive medication data available"

        _total_patients = len(heart_hosp_ids)

        _rows = []
        _rows.append("| **Medication** | **Patients Receiving** |")
        _rows.append("|:---|:---|")

        for _med in VASOACTIVE_MEDS:
            _med_patients = periop_meds_df[
                periop_meds_df['med_category'] == _med
            ]['hospitalization_id'].nunique()
            _rows.append(f"| {_med.title()} | {format_count_pct(_med_patients, _total_patients)} |")

        return "\n".join(_rows)

    vasoactive_table_md = mo.md(create_vasoactive_table())
    vasoactive_table_md
    return


@app.cell
def _(format_count_pct, heart_hosp_ids, mo, periop_resp_df):
    # Respiratory support rates table
    def create_respiratory_table():
        if periop_resp_df is None or len(periop_resp_df) == 0:
            return "No respiratory support data available"

        _total_patients = len(heart_hosp_ids)

        # Calculate rates for each device category
        _device_counts = periop_resp_df.groupby('device_category')['hospitalization_id'].nunique()

        _rows = []
        _rows.append("| **Device Category** | **Patients** |")
        _rows.append("|:---|:---|")

        for _device in _device_counts.index:
            _count = _device_counts[_device]
            _rows.append(f"| {_device} | {format_count_pct(_count, _total_patients)} |")

        # Add IMV-specific stats
        _imv_patients = periop_resp_df[
            periop_resp_df['device_category'] == 'IMV'
        ]['hospitalization_id'].nunique()
        _rows.append(f"| **Mechanical Ventilation (IMV)** | {format_count_pct(_imv_patients, _total_patients)} |")

        return "\n".join(_rows)

    respiratory_table_md = mo.md(create_respiratory_table())
    respiratory_table_md
    return


@app.cell
def _(
    PERIOP_DAYS,
    format_count_pct,
    heart_crrt_df,
    heart_ecmo_df,
    heart_hosp_ids,
    heart_transplant_hosp,
    mo,
    pd,
):
    # Advanced therapies: ECMO/MCS and CRRT rates
    def create_advanced_therapies_table():
        _total_patients = len(heart_hosp_ids)
        _rows = []
        _rows.append("| **Therapy** | **Patients Receiving** |")
        _rows.append("|:---|:---|")

        # ECMO/MCS
        if heart_ecmo_df is not None and len(heart_ecmo_df) > 0:
            _ecmo_df = heart_ecmo_df.copy()
            _ecmo_df['device_start_dttm'] = pd.to_datetime(
                _ecmo_df['device_start_dttm'], utc=True
            )
            _ecmo_merged = pd.merge(
                _ecmo_df,
                heart_transplant_hosp[['hospitalization_id', 'transplant_date']],
                on='hospitalization_id',
                how='inner'
            )
            _ecmo_merged['days_from_transplant'] = (
                _ecmo_merged['device_start_dttm'] - _ecmo_merged['transplant_date']
            ).dt.total_seconds() / (24 * 3600)

            # Pre-transplant ECMO
            _pre_ecmo = _ecmo_merged[
                (_ecmo_merged['days_from_transplant'] >= -PERIOP_DAYS) &
                (_ecmo_merged['days_from_transplant'] < 0)
            ]['hospitalization_id'].nunique()
            _rows.append(f"| ECMO/MCS Pre-transplant | {format_count_pct(_pre_ecmo, _total_patients)} |")

            # Post-transplant ECMO
            _post_ecmo = _ecmo_merged[
                (_ecmo_merged['days_from_transplant'] >= 0) &
                (_ecmo_merged['days_from_transplant'] <= PERIOP_DAYS)
            ]['hospitalization_id'].nunique()
            _rows.append(f"| ECMO/MCS Post-transplant | {format_count_pct(_post_ecmo, _total_patients)} |")
        else:
            _rows.append("| ECMO/MCS | Data not available |")

        # CRRT
        if heart_crrt_df is not None and len(heart_crrt_df) > 0:
            _crrt_df = heart_crrt_df.copy()
            _crrt_df['crrt_start_dttm'] = pd.to_datetime(
                _crrt_df['crrt_start_dttm'], utc=True
            )
            _crrt_merged = pd.merge(
                _crrt_df,
                heart_transplant_hosp[['hospitalization_id', 'transplant_date']],
                on='hospitalization_id',
                how='inner'
            )
            _crrt_merged['days_from_transplant'] = (
                _crrt_merged['crrt_start_dttm'] - _crrt_merged['transplant_date']
            ).dt.total_seconds() / (24 * 3600)

            _crrt_patients = _crrt_merged[
                (_crrt_merged['days_from_transplant'] >= -PERIOP_DAYS) &
                (_crrt_merged['days_from_transplant'] <= PERIOP_DAYS)
            ]['hospitalization_id'].nunique()
            _rows.append(f"| CRRT | {format_count_pct(_crrt_patients, _total_patients)} |")
        else:
            _rows.append("| CRRT | Data not available |")

        return "\n".join(_rows)

    advanced_therapies_md = mo.md(create_advanced_therapies_table())
    advanced_therapies_md
    return


@app.cell
def _(
    format_count_pct,
    format_median_iqr,
    heart_transplant_hosp,
    is_suppressed,
    mo,
):
    # Outcomes: LOS, discharge disposition, mortality
    def create_outcomes_table():
        if heart_transplant_hosp is None or len(heart_transplant_hosp) == 0:
            return "No outcome data available"

        _df = heart_transplant_hosp.copy()
        _n = len(_df)

        # Calculate length of stay
        _df['los_days'] = (
            _df['discharge_dttm'] - _df['admission_dttm']
        ).dt.total_seconds() / (24 * 3600)

        _rows = []
        _rows.append("| **Outcome** | **Value** |")
        _rows.append("|:---|:---|")
        _rows.append(f"| **N** | {_n if not is_suppressed(_n) else '< 10'} |")

        # Length of stay
        _rows.append(f"| Hospital LOS, median [IQR] days | {format_median_iqr(_df['los_days'], _n)} |")

        # Discharge disposition
        if 'discharge_category' in _df.columns:
            _rows.append("| **Discharge Disposition** | |")
            _discharge_counts = _df['discharge_category'].value_counts()
            for _disp in _discharge_counts.index:
                _count = _discharge_counts[_disp]
                _rows.append(f"| - {_disp} | {format_count_pct(_count, _n)} |")

            # Mortality (Expired)
            _mortality = _discharge_counts.get('Expired', 0)
            _rows.append(f"| **In-hospital Mortality** | {format_count_pct(_mortality, _n)} |")

        return "\n".join(_rows)

    outcomes_md = mo.md(create_outcomes_table())
    outcomes_md
    return


@app.cell
def _(
    datetime,
    heart_only_df,
    heart_patient_df,
    heart_transplant_hosp,
    json,
    mo,
    output_dir,
    pd,
    site_name,
    total_heart_n,
):
    # Generate federated report JSON and CSV exports
    def generate_federated_export():
        output_dir.mkdir(parents=True, exist_ok=True)

        _generation_date = datetime.now().isoformat()

        # Build report data structure
        _report = {
            "metadata": {
                "site_name": site_name,
                "generation_date": _generation_date,
                "clif_version": "2.1",
                "report_version": "1.0",
                "suppression_threshold": 10
            },
            "cohort": {
                "total_n": int(total_heart_n),
                "suppressed": total_heart_n < 10
            }
        }

        # Add demographics if available
        if heart_patient_df is not None and heart_only_df is not None:
            _merged = pd.merge(
                heart_patient_df,
                heart_only_df[['patient_id', 'transplant_date']],
                on='patient_id',
                how='inner'
            ).drop_duplicates('patient_id')

            if 'birth_date' in _merged.columns and len(_merged) >= 10:
                _merged['birth_date'] = pd.to_datetime(_merged['birth_date']).dt.tz_localize('UTC')
                _merged['age'] = (_merged['transplant_date'] - _merged['birth_date']).dt.days / 365.25
                _report["demographics"] = {
                    "age_mean": round(_merged['age'].mean(), 1),
                    "age_sd": round(_merged['age'].std(), 1),
                    "age_median": round(_merged['age'].median(), 1)
                }

        # Add outcomes if available
        if heart_transplant_hosp is not None and len(heart_transplant_hosp) >= 10:
            _hosp = heart_transplant_hosp.copy()
            _hosp['los'] = (_hosp['discharge_dttm'] - _hosp['admission_dttm']).dt.total_seconds() / (24*3600)
            _report["outcomes"] = {
                "los_median": round(_hosp['los'].median(), 1),
                "los_q25": round(_hosp['los'].quantile(0.25), 1),
                "los_q75": round(_hosp['los'].quantile(0.75), 1)
            }
            if 'discharge_category' in _hosp.columns:
                _mortality = len(_hosp[_hosp['discharge_category'] == 'Expired'])
                if _mortality >= 10:
                    _report["outcomes"]["mortality_n"] = _mortality
                    _report["outcomes"]["mortality_pct"] = round(_mortality / len(_hosp) * 100, 1)

        # Save JSON
        _json_path = output_dir / f"{site_name}_heart_transplant_report.json"
        with open(_json_path, 'w') as f:
            json.dump(_report, f, indent=2)

        return f"Report exported to: {_json_path}"

    export_result = generate_federated_export()
    mo.md(f"**{export_result}**")
    return


@app.cell
def _(HEART_TRANSPLANT_CPTS, Path, PatientProcedures, config):
    # Debug: Load and display heart transplant procedures
    _tables_path = Path(config['tables_path'])
    _file_type = config.get('file_type', 'parquet')

    _proc_table = PatientProcedures.from_file(
        data_directory=str(_tables_path),
        filetype=_file_type,
        timezone='UTC'
    )

    heart_procedures_df = _proc_table.df[
        _proc_table.df['procedure_code'].astype(str).isin(HEART_TRANSPLANT_CPTS)
    ].copy()

    # Deduplicate by code, hospitalization_id, and procedure_billed_dttm
    _before = len(heart_procedures_df)
    heart_procedures_df = heart_procedures_df.drop_duplicates(
        subset=['procedure_code', 'hospitalization_id', 'procedure_billed_dttm']
    )
    print(f"Heart transplant procedures found: {len(heart_procedures_df):,} (removed {_before - len(heart_procedures_df):,} duplicates)")
    heart_procedures_df
    return (heart_procedures_df,)


@app.cell
def _(Hospitalization, Path, config, heart_procedures_df):
    # Filter hospitalization table to heart transplant hospitalization_ids
    _tables_path = Path(config['tables_path'])
    _file_type = config.get('file_type', 'parquet')

    _hosp_table = Hospitalization.from_file(
        data_directory=str(_tables_path),
        filetype=_file_type,
        timezone='UTC'
    )

    _heart_hosp_ids = heart_procedures_df['hospitalization_id'].unique()
    debug_heart_hosp_df = _hosp_table.df[
        _hosp_table.df['hospitalization_id'].isin(_heart_hosp_ids)
    ].copy()

    print(f"Heart transplant hospitalizations in CLIF: {len(debug_heart_hosp_df):,} of {len(_heart_hosp_ids):,} procedure hospitalization_ids")
    debug_heart_hosp_df
    return


@app.cell
def _(Path, config, pd):
    # Load national registry data filtered by site_name
    _registry_path = Path(__file__).resolve().parent.parent / 'public' / 'data' / 'clif_hr_tx_counts.csv'
    _registry_df = pd.read_csv(_registry_path)

    _site_name = config['site_name']
    registry_ucmc_df = _registry_df[
        (_registry_df['clif_site'] == _site_name) &
        (_registry_df['ORG_TY'] == 'HR')
    ].copy()

    print(f"National Registry Heart Transplants for {_site_name}:")
    registry_ucmc_df
    return


@app.cell
def _(heart_procedures_df, pd):
    # Group heart procedures by year for comparison with registry
    _df = heart_procedures_df.copy()
    _df['year'] = pd.to_datetime(_df['procedure_billed_dttm']).dt.year

    clif_by_year = _df.groupby('year').size().reset_index(name='clif_count')
    print("CLIF Heart Transplant Procedures by Year:")
    clif_by_year
    return


@app.cell
def _():
    return


@app.cell(column=4)
def _(mo):
    mo.md("""# Debug & Registry Comparison""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
