import marimo

__generated_with = "0.14.13"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import altair as alt
    import seaborn as sns

    # Add utils directory relative to current script location
    CURRENT_DIR = Path(__file__).resolve().parent
    UTILS_DIR = CURRENT_DIR.parent / 'utils'
    sys.path.append(str(UTILS_DIR))

    from load_config import load_config
    return Path, alt, load_config, mo, pd, plt


@app.cell
def _(load_config):
    # Load configuration
    config = load_config()
    print(f"Site: {config['site_name']}")
    print(f"Tables path: {config['tables_path']}")
    return (config,)


@app.cell
def _(Path, config, pd):
    # Load Valeos tables
    tables_path = Path(config['tables_path'])
    valeos_dir = tables_path / 'Valeos'
    site_name = config['site_name']

    def load_valeos_table(table_name):
        """Load a Valeos table and return DataFrame or None"""
        file_path = valeos_dir / f"{site_name}_valeos_inpatient_{table_name}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            print(f"Loaded {table_name} table: {len(df):,} records")
            return df
        else:
            print(f"{table_name} file not found: {file_path}")
            return None

    # Load core tables
    valeos_patient_df = load_valeos_table('patient')
    valeos_transplant_df = load_valeos_table('transplant')

    # Load additional clinical tables
    valeos_hospitalization_df = load_valeos_table('hospitalization')
    valeos_vitals_df = load_valeos_table('vitals')
    valeos_labs_df = load_valeos_table('labs')
    valeos_respiratory_df = load_valeos_table('respiratory_support')
    valeos_med_continuous_df = load_valeos_table('medication_admin_continuous')
    valeos_code_status_df = load_valeos_table('code_status')
    

    return (
        site_name,
        valeos_code_status_df,
        valeos_hospitalization_df,
        valeos_labs_df,
        valeos_med_continuous_df,
        valeos_patient_df,
        valeos_respiratory_df,
        valeos_transplant_df,
        valeos_vitals_df,
    )


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo, site_name):
    mo.md(f"""# Valeos inpatient transplant dashboard: {site_name}""")
    return


@app.cell
def _(
    valeos_code_status_df,
    valeos_labs_df,
    valeos_med_continuous_df,
    valeos_respiratory_df,
    valeos_vitals_df,
):
    # Display summary of loaded clinical tables
    clinical_tables = {
        'Vitals': valeos_vitals_df,
        'Labs': valeos_labs_df,
        'Respiratory Support': valeos_respiratory_df,
        'Medication Admin (Continuous)': valeos_med_continuous_df,
        'Code Status': valeos_code_status_df
    }

    print("=== CLINICAL DATA SUMMARY ===")
    for table_name, df in clinical_tables.items():
        if df is not None:
            unique_patients = df['hospitalization_id'].nunique() if 'hospitalization_id' in df.columns else 'N/A'
            print(f"{table_name}: {len(df):,} records across {unique_patients} hospitalizations")
        else:
            print(f"{table_name}: No data available")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Patient Demographics by Organ Type

    **Privacy Notice**: This dashboard shows all demographic breakdowns, including small cell counts. 
    When exporting or sharing results externally, cells with <10 patients should be suppressed 
    to protect patient privacy and prevent potential re-identification.
    """
    )
    return


@app.cell
def _(demographics_table):
    demographics_table
    return


@app.cell(hide_code=True)
def _(pd, valeos_patient_df, valeos_transplant_df):
    def create_demographics_table(patient_df, transplant_df):
        """Create demographics table with organ types as columns and demographic categories as rows"""
        if patient_df is None or transplant_df is None:
            return None

        # Merge patient and transplant data
        merged_df = pd.merge(patient_df, transplant_df, on='patient_id', how='inner')

        # Identify multi-organ recipients (>10 recipients for combinations)
        multi_organ_patients = merged_df.groupby('patient_id')['transplant_type'].apply(list).reset_index()
        multi_organ_patients['organ_combo'] = multi_organ_patients['transplant_type'].apply(
            lambda x: '+'.join(sorted(set(x))) if len(set(x)) > 1 else None
        )

        # Get all multi-organ combinations
        combo_counts = multi_organ_patients['organ_combo'].dropna().value_counts()
        all_combos = combo_counts.index.tolist()

        print(f"Multi-organ combinations found: {all_combos}")
        if len(combo_counts) > 0:
            print("Combination counts:", dict(combo_counts))

        # Create columns list: single organs + all multi-organ combos + overall
        single_organs = merged_df['transplant_type'].unique()
        all_columns = list(single_organs) + all_combos + ['Overall']

        # Initialize results dictionary
        demographics_data = {}

        # Calculate for single organ types
        for organ in single_organs:
            organ_data = merged_df[merged_df['transplant_type'] == organ]
            demographics_data[organ] = calculate_organ_demographics(organ_data)

        # Calculate for multi-organ combinations (if any)
        for combo in all_combos:
            combo_organs = combo.split('+')
            combo_patients = multi_organ_patients[multi_organ_patients['organ_combo'] == combo]['patient_id']
            combo_data = merged_df[merged_df['patient_id'].isin(combo_patients)]
            demographics_data[combo] = calculate_organ_demographics(combo_data)

        # Calculate overall
        demographics_data['Overall'] = calculate_organ_demographics(merged_df)

        # Convert to DataFrame with demographics as rows and organs as columns
        demographics_df = pd.DataFrame(demographics_data)

        return demographics_df

    def calculate_organ_demographics(group_df):
        """Calculate demographics for a group of patients"""
        stats = {}

        # Count
        stats['N'] = len(group_df.drop_duplicates('patient_id'))  # Unique patients

        # Get unique patients for demographic calculations
        unique_patients = group_df.drop_duplicates('patient_id')

        # Age at transplant - calculate from birth_date and transplant_date
        if 'birth_date' in unique_patients.columns and 'transplant_date' in unique_patients.columns:
            ages_at_transplant = []
            for _, patient in unique_patients.iterrows():
                if pd.notna(patient['birth_date']) and pd.notna(patient['transplant_date']):
                    birth = pd.to_datetime(patient['birth_date'], utc=True)
                    transplant = pd.to_datetime(patient['transplant_date'], utc=True)
                    age_at_transplant = (transplant - birth).days / 365.25
                    ages_at_transplant.append(age_at_transplant)

            if len(ages_at_transplant) > 0:
                ages_series = pd.Series(ages_at_transplant)
                stats['Age at transplant (mean ± SD)'] = f"{ages_series.mean():.1f} ± {ages_series.std():.1f}"
            else:
                stats['Age at transplant (mean ± SD)'] = "N/A"
        else:
            stats['Age at transplant (mean ± SD)'] = "N/A"

        # Sex
        if 'sex_category' in unique_patients.columns:
            sex_counts = unique_patients['sex_category'].value_counts()
            male_count = sex_counts.get('Male', 0)
            male_pct = (male_count / len(unique_patients)) * 100
            stats['Male (%)'] = f"{male_count} ({male_pct:.1f}%)"
        else:
            stats['Male (%)'] = "N/A"

        # Race categories - top 3 only
        if 'race_category' in unique_patients.columns:
            race_counts = unique_patients['race_category'].fillna('Unknown').value_counts()
            total_patients = len(unique_patients)
            # Get top 3 race categories
            top_3_races = race_counts.head(3)
            for race in top_3_races.index:
                count = top_3_races[race]
                pct = (count / total_patients) * 100
                stats[f'Race - {race}'] = f"{count} ({pct:.1f}%)"

        # Hispanic ethnicity as separate line
        if 'ethnicity_category' in unique_patients.columns:
            ethnicity_counts = unique_patients['ethnicity_category'].fillna('Unknown').value_counts()
            total_patients = len(unique_patients)
            # Only show Hispanic if it exists
            if 'Hispanic' in ethnicity_counts.index:
                hispanic_count = ethnicity_counts['Hispanic']
                hispanic_pct = (hispanic_count / total_patients) * 100
                stats['Hispanic/Latino (%)'] = f"{hispanic_count} ({hispanic_pct:.1f}%)"
            else:
                stats['Hispanic/Latino (%)'] = "0 (0.0%)"

        # Language categories - each as separate row
        if 'language_category' in unique_patients.columns:
            language_counts = unique_patients['language_category'].fillna('Unknown').value_counts()
            total_patients = len(unique_patients)
            for language in language_counts.index:
                count = language_counts[language]
                pct = (count / total_patients) * 100
                stats[f'Language - {language}'] = f"{count} ({pct:.1f}%)"

        return stats

    # Create the demographics table
    demographics_table = create_demographics_table(valeos_patient_df, valeos_transplant_df)

    return (demographics_table,)


@app.cell
def _(mo):
    mo.md(r"""## Transplant Volume Over Time""")
    return


@app.cell
def _(demographics_table, mo):
    organ_selected = mo.ui.dropdown(
        options=demographics_table.columns,
        label="Select organ type for volume analysis:",
        value="Overall"  # Set Overall as default
    )

    organ_selected
    return (organ_selected,)


@app.cell
def _(chart_display):
    chart_display
    return


@app.cell(hide_code=True)
def _(alt, config, mo, organ_selected, pd, valeos_transplant_df):
    def create_volume_chart_data(transplant_df, selected_organ):
        """Prepare data for Altair chart"""
        if transplant_df is None or selected_organ is None:
            return None, None

        # Convert transplant_date to datetime with UTC timezone
        work_df = transplant_df.copy()
        work_df['transplant_date'] = pd.to_datetime(work_df['transplant_date'], utc=True)
        work_df['month_year'] = work_df['transplant_date'].dt.to_period('M')

        # Get full date range to ensure zero months are included
        min_date = work_df['month_year'].min()
        max_date = work_df['month_year'].max()
        full_date_range = pd.period_range(start=min_date, end=max_date, freq='M')

        # Define consistent color scheme for all organ types
        organ_colors = {
            'kidney': '#1f77b4',    # Blue
            'liver': '#ff7f0e',     # Orange  
            'heart': '#2ca02c',     # Green
            'lung': '#d62728',      # Red
        }

        if selected_organ == 'Overall':
            # Create stacked bar chart data by organ type
            organ_monthly = work_df.groupby(['month_year', 'transplant_type']).size().reset_index(name='count')

            # Create complete date x organ grid to include zero months
            date_organ_grid = []
            for date in full_date_range:
                for organ in ['kidney', 'liver', 'heart', 'lung']:
                    if organ in work_df['transplant_type'].unique():
                        existing = organ_monthly[
                            (organ_monthly['month_year'] == date) & 
                            (organ_monthly['transplant_type'] == organ)
                        ]
                        if len(existing) > 0:
                            date_organ_grid.append({
                                'month_year': date.to_timestamp(),
                                'transplant_type': organ,
                                'count': existing['count'].iloc[0]
                            })
                        else:
                            date_organ_grid.append({
                                'month_year': date.to_timestamp(),
                                'transplant_type': organ,
                                'count': 0
                            })

            chart_data = pd.DataFrame(date_organ_grid)
            chart_data['month_year_str'] = chart_data['month_year'].dt.strftime('%m/%y')

            # Create Altair stacked bar chart
            chart = alt.Chart(chart_data).mark_bar().add_selection(
                alt.selection_interval()
            ).encode(
                x=alt.X('month_year:T', 
                       axis=alt.Axis(title='Month', 
                                   format='%m/%y',
                                   labelAngle=45)),
                y=alt.Y('count:Q', 
                       axis=alt.Axis(title='Number of Transplants'),
                       scale=alt.Scale(domain=[0, chart_data.groupby('month_year')['count'].sum().max() * 1.1])),
                color=alt.Color('transplant_type:N', 
                              scale=alt.Scale(domain=list(organ_colors.keys()),
                                            range=list(organ_colors.values())),
                              legend=alt.Legend(title="Organ Type")),
                order=alt.Order('transplant_type:N', sort=['kidney', 'liver', 'heart', 'lung']),
                tooltip=['month_year_str:N', 'transplant_type:N', 'count:Q']
            ).properties(
                width=600,
                height=400,
                title=f'Monthly Transplant Volume - {config["site_name"]}'
            )

        else:
            # Handle single organ or combinations
            if '+' in selected_organ:
                # For multi-organ combinations
                combo_organs = selected_organ.split('+')
                patient_organs = work_df.groupby('patient_id')['transplant_type'].apply(set).reset_index()
                combo_patients = patient_organs[
                    patient_organs['transplant_type'].apply(lambda x: set(combo_organs).issubset(x))
                ]['patient_id']
                filtered_df = work_df[work_df['patient_id'].isin(combo_patients)]
                bar_color = '#7f7f7f'  # Gray for combinations
            else:
                # Single organ type
                filtered_df = work_df[work_df['transplant_type'] == selected_organ]
                bar_color = organ_colors.get(selected_organ, '#1f77b4')

            if len(filtered_df) == 0:
                return None, f"No data available for {selected_organ}"

            # Aggregate by month and include zero months
            monthly_counts = filtered_df.groupby('month_year').size()
            monthly_counts = monthly_counts.reindex(full_date_range, fill_value=0)

            chart_data = pd.DataFrame({
                'month_year': monthly_counts.index.to_timestamp(),
                'count': monthly_counts.values
            })
            chart_data['month_year_str'] = chart_data['month_year'].dt.strftime('%m/%y')

            # Create Altair bar chart
            chart = alt.Chart(chart_data).mark_bar(
                color=bar_color,
                opacity=0.8
            ).add_selection(
                alt.selection_interval()
            ).encode(
                x=alt.X('month_year:T', 
                       axis=alt.Axis(title='Month', 
                                   format='%m/%y',
                                   labelAngle=45)),
                y=alt.Y('count:Q', 
                       axis=alt.Axis(title='Number of Transplants'),
                       scale=alt.Scale(domain=[0, chart_data['count'].max() * 1.1])),
                tooltip=['month_year_str:N', 'count:Q']
            ).properties(
                width=600,
                height=400,
                title=f'Monthly {selected_organ} Transplant Volume - {config["site_name"]}'
            )

        return chart, None

    # Get the selected organ value
    selected_organ = organ_selected.value if hasattr(organ_selected, 'value') else None

    # Create chart data
    volume_chart, error_msg = create_volume_chart_data(valeos_transplant_df, selected_organ)

    if error_msg:
        chart_display = mo.md(f"**{error_msg}**")
    elif volume_chart is not None:
        chart_display = mo.ui.altair_chart(volume_chart)
    else:
        chart_display = mo.md("**No data available for chart**")

    return (chart_display,)


@app.cell
def _(mo):
    mo.md(r"""## Annual Transplant Volume""")
    return


@app.cell
def _(yearly_chart_display):
    yearly_chart_display
    return


@app.cell(hide_code=True)
def _(alt, config, mo, pd, valeos_transplant_df):
    def create_yearly_volume_chart(transplant_df):
        """Create yearly transplant volume chart"""
        if transplant_df is None:
            return None, "No transplant data available"

        # Convert transplant_date to datetime with UTC timezone
        work_df = transplant_df.copy()
        work_df['transplant_date'] = pd.to_datetime(work_df['transplant_date'], utc=True)
        work_df['year'] = work_df['transplant_date'].dt.year

        # Define consistent color scheme for all organ types
        organ_colors = {
            'kidney': '#1f77b4',    # Blue
            'liver': '#ff7f0e',     # Orange  
            'heart': '#2ca02c',     # Green
            'lung': '#d62728',      # Red
        }

        # Group by year and organ type
        yearly_counts = work_df.groupby(['year', 'transplant_type']).size().reset_index(name='count')

        # Get full year range
        min_year = work_df['year'].min()
        max_year = work_df['year'].max()
        year_range = list(range(min_year, max_year + 1))

        # Create complete year x organ grid to include zero years
        year_organ_grid = []
        for year in year_range:
            for organ in ['kidney', 'liver', 'heart', 'lung']:
                if organ in work_df['transplant_type'].unique():
                    existing = yearly_counts[
                        (yearly_counts['year'] == year) & 
                        (yearly_counts['transplant_type'] == organ)
                    ]
                    if len(existing) > 0:
                        year_organ_grid.append({
                            'year': year,
                            'transplant_type': organ,
                            'count': existing['count'].iloc[0]
                        })
                    else:
                        year_organ_grid.append({
                            'year': year,
                            'transplant_type': organ,
                            'count': 0
                        })

        chart_data = pd.DataFrame(year_organ_grid)

        # Create Altair stacked bar chart
        yearly_chart = alt.Chart(chart_data).mark_bar().add_selection(
            alt.selection_interval()
        ).encode(
            x=alt.X('year:O', 
                   axis=alt.Axis(title='Year', labelAngle=0)),
            y=alt.Y('count:Q', 
                   axis=alt.Axis(title='Number of Transplants'),
                   scale=alt.Scale(domain=[0, chart_data.groupby('year')['count'].sum().max() * 1.1])),
            color=alt.Color('transplant_type:N', 
                          scale=alt.Scale(domain=list(organ_colors.keys()),
                                        range=list(organ_colors.values())),
                          legend=alt.Legend(title="Organ Type")),
            order=alt.Order('transplant_type:N', sort=['kidney', 'liver', 'heart', 'lung']),
            tooltip=['year:O', 'transplant_type:N', 'count:Q']
        ).properties(
            width=500,
            height=350,
            title=f'Annual Transplant Volume - {config["site_name"]}'
        )

        return yearly_chart, None

    # Create yearly chart
    yearly_volume_chart, yearly_error_msg = create_yearly_volume_chart(valeos_transplant_df)

    if yearly_error_msg:
        yearly_chart_display = mo.md(f"**{yearly_error_msg}**")
    elif yearly_volume_chart is not None:
        yearly_chart_display = mo.ui.altair_chart(yearly_volume_chart)
    else:
        yearly_chart_display = mo.md("**No yearly data available for chart**")

    return (yearly_chart_display,)


@app.cell(column=2)
def _(mo):
    mo.md(r"""# Individual patient tracker""")
    return


@app.cell(hide_code=True)
def _(mo, valeos_transplant_df):
    # create organ selector 
    mo.md(r"""## Patient Selector""")
    # Create organ
    organ_options = valeos_transplant_df['transplant_type'].unique().tolist()

    organ_selected_individual = mo.ui.dropdown(
        options=organ_options,
        label="Select organ type for patient selection:",
        value='heart'  # Set Overall as default
    )


    organ_selected_individual
    return (organ_selected_individual,)


@app.cell(hide_code=True)
def _(
    mo,
    organ_selected,
    organ_selected_individual,
    valeos_hospitalization_df,
    valeos_med_continuous_df,
    valeos_transplant_df,
):
    # Create the patient list and selector

    # Get the selected organ value
    selected_organ_individual = organ_selected_individual.value if hasattr(organ_selected, 'value') else None

    # Filter patients based on the selected organ
    if selected_organ_individual and valeos_transplant_df is not None:
        filtered_patient_options = valeos_transplant_df[valeos_transplant_df['transplant_type'] == selected_organ_individual]['patient_id'].unique().tolist()

        # Find a patient with any vasoactive medication data
        vasoactive_meds = [
            'norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin', 
            'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
        ]
        default_patient = None
        if valeos_med_continuous_df is not None and valeos_hospitalization_df is not None:
            for pt_id in filtered_patient_options:
                # Get patient's hospitalizations
                patient_hosp_ids = valeos_hospitalization_df[valeos_hospitalization_df['patient_id'] == pt_id]['hospitalization_id'].unique()

                # Check if patient has any vasoactive medication data
                has_vasoactives = valeos_med_continuous_df[
                    (valeos_med_continuous_df['hospitalization_id'].isin(patient_hosp_ids)) & 
                    (valeos_med_continuous_df['med_category'].isin(vasoactive_meds))
                ].shape[0] > 0

                if has_vasoactives:
                    default_patient = pt_id
                    break

        # Create or update the patient selector with the filtered options
        patient_selected = mo.ui.dropdown(
            options=filtered_patient_options,
            label="Select patient ID:",
            value=default_patient if default_patient else (filtered_patient_options[0] if filtered_patient_options else None)
        )

    else:
        mo.md("**No transplant data available for patient selection**")
        patient_selected = None


    patient_selected
    return (patient_selected,)


@app.cell
def _(patient_selected):
    # get patient ID from patient_selected
    patient_id = patient_selected.value if hasattr(patient_selected, 'value') else None
    return (patient_id,)


@app.cell
def _(mo):
    mo.md(r"""## Individual Patient Vasoactive Course""")
    return


@app.cell(hide_code=True)
def _(
    create_patient_vasoactive_chart,
    mo,
    patient_id,
    valeos_hospitalization_df,
    valeos_med_continuous_df,
    valeos_transplant_df,
):
    # Create vasoactive dosing chart`
    patient_chart, vaso_error_msg = create_patient_vasoactive_chart(
        patient_id, valeos_med_continuous_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if vaso_error_msg:
        patient_vasoactive_display = mo.md(f"**{vaso_error_msg}**")
    elif patient_chart is not None:
        patient_vasoactive_display = mo.as_html(patient_chart)
    else:
        patient_vasoactive_display = mo.md("**No chart data available**")

    patient_vasoactive_display
    return


@app.cell(hide_code=True)
def _(pd, plt):
    def create_patient_vasoactive_chart(patient_id, med_df, transplant_df, hosp_df):
        """Create individual patient vasoactive course chart showing all vasoactives using proper CLIF data structure"""
        if patient_id is None or med_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Define all vasoactive medications from mCIDE schema
        vasoactive_meds = [
            'norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin', 
            'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
        ]

        # Load vasoactive dose ranges for relative scaling
        import os
        dose_ranges_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vasoactive_dose_ranges.csv')
        try:
            dose_ranges_df = pd.read_csv(dose_ranges_path)
            dose_ranges = dict(zip(dose_ranges_df['medication'], dose_ranges_df['typical_max_dose']))
        except FileNotFoundError:
            # Fallback ranges if CSV not found
            dose_ranges = {
                'norepinephrine': 1.0, 'epinephrine': 1.0, 'phenylephrine': 200.0,
                'angiotensin': 20.0, 'vasopressin': 0.04, 'dopamine': 20.0,
                'dobutamine': 20.0, 'milrinone': 0.75, 'isoproterenol': 0.2
            }

        # Get patient transplant info
        patient_transplant = transplant_df[transplant_df['patient_id'] == patient_id]
        if patient_transplant.empty:
            return None, f"No transplant data found for patient {patient_id}"

        transplant_date = pd.to_datetime(patient_transplant.iloc[0]['transplant_date'], utc=True)
        organ_type = patient_transplant.iloc[0]['transplant_type']

        # Get patient's hospitalizations to find hospitalization_ids (CLIF key structure)
        patient_hospitalizations = hosp_df[hosp_df['patient_id'] == patient_id].copy()
        if patient_hospitalizations.empty:
            return None, f"No hospitalization data found for patient {patient_id}"

        # Get hospitalization IDs for this patient
        patient_hosp_ids = patient_hospitalizations['hospitalization_id'].unique()

        # Filter medication data for all vasoactives using hospitalization_id (CLIF standard approach)
        patient_med_data = med_df[
            (med_df['hospitalization_id'].isin(patient_hosp_ids)) & 
            (med_df['med_category'].isin(vasoactive_meds))
        ].copy()

        if patient_med_data.empty:
            return None, f"No vasoactive medication data found for patient {patient_id}"

        # Find the transplant hospitalization (contains transplant_date)
        patient_hospitalizations['admission_dttm'] = pd.to_datetime(patient_hospitalizations['admission_dttm'], utc=True)
        patient_hospitalizations['discharge_dttm'] = pd.to_datetime(patient_hospitalizations['discharge_dttm'], utc=True)

        # Find hospitalization that contains the transplant date
        transplant_hosp = patient_hospitalizations[
            (patient_hospitalizations['admission_dttm'] <= transplant_date) &
            (patient_hospitalizations['discharge_dttm'] >= transplant_date)
        ]

        if transplant_hosp.empty:
            return None, f"Cannot find transplant hospitalization for patient {patient_id}"

        admission_date = transplant_hosp.iloc[0]['admission_dttm']
        discharge_date = transplant_hosp.iloc[0]['discharge_dttm']
        discharge_category = transplant_hosp.iloc[0]['discharge_category'] if 'discharge_category' in transplant_hosp.columns else 'Unknown'

        # Process medication dates and calculate days since admission
        patient_med_data['admin_dttm'] = pd.to_datetime(patient_med_data['admin_dttm'], utc=True)
        patient_med_data = patient_med_data.sort_values('admin_dttm')

        patient_med_data['days_since_admission'] = (
            patient_med_data['admin_dttm'] - admission_date
        ).dt.total_seconds() / (24 * 3600)

        # Calculate relative dose percentages
        patient_med_data['relative_dose_percent'] = patient_med_data.apply(
            lambda row: (row['med_dose'] / dose_ranges.get(row['med_category'], 1.0)) * 100 
            if pd.notna(row['med_dose']) and dose_ranges.get(row['med_category'], 1.0) > 0 else 0, 
            axis=1
        )

        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)

        # Create plot with relative dosing scale
        plt.style.use('default')  # Reset to default style
        fig, ax = plt.subplots(figsize=(12, 10))  # Increased height for legend below

        # Create consistent color mapping for all vasoactive medications
        vasoactive_color_map = {
            'norepinephrine': '#1f77b4',    # Blue
            'epinephrine': '#ff7f0e',       # Orange  
            'phenylephrine': '#2ca02c',     # Green
            'angiotensin': '#d62728',       # Red
            'vasopressin': '#9467bd',       # Purple
            'dopamine': '#8c564b',          # Brown
            'dobutamine': '#e377c2',        # Pink
            'milrinone': '#7f7f7f',         # Gray
            'isoproterenol': '#bcbd22'      # Olive
        }

        # Get unique vasoactives present in the data
        present_vasoactives = patient_med_data['med_category'].unique()

        # Plot each vasoactive medication using relative percentage (dots only, no lines)
        for med in present_vasoactives:
            med_data = patient_med_data[patient_med_data['med_category'] == med]
            if not med_data.empty:
                color = vasoactive_color_map.get(med, '#000000')  # Default to black if not found
                ax.scatter(med_data['days_since_admission'], med_data['relative_dose_percent'], 
                          color=color, s=50, alpha=0.8, label=med.title())

        # Add reference lines with discharge category in legend
        ax.axvline(x=0, color='blue', linestyle=':', linewidth=2, alpha=0.8, label='Admission (Day 0)')
        ax.axvline(x=transplant_days, color='green', linestyle='--', linewidth=3, alpha=0.8, 
                  label=f'Transplant (Day {transplant_days:.1f})')
        ax.axvline(x=discharge_days, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                  label=f'Discharge (Day {discharge_days:.1f}) to {discharge_category}')

        # Formatting
        ax.set_xlabel('Days Since Admission', fontsize=12)
        ax.set_ylabel('Relative Dose (% of Typical Maximum)', fontsize=12)
        ax.set_title(f'Vasoactive Medications - {organ_type.title()} Transplant - Patient {patient_id}', 
                    fontsize=14, fontweight='bold')

        # Set y-axis limits - cap at 100% unless patient data exceeds it
        max_dose_percent = patient_med_data['relative_dose_percent'].max()
        y_max = 100 if max_dose_percent <= 100 else max_dose_percent * 1.1
        ax.set_ylim(bottom=0, top=y_max)

        # Add main legend
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Grid for better readability
        ax.grid(True, alpha=0.3)

        # Set x-axis limits to admission and discharge
        ax.set_xlim(left=-0.5, right=discharge_days + 0.5)

        # Create dosing scale legend text
        dose_scale_text = "Dosing Scale (100% = max):\n"
        for med in present_vasoactives:
            if med in dose_ranges:
                dose_scale_text += f"• {med.title()}: {dose_ranges[med]} "
                # Add units from CSV if available
                try:
                    med_units = dose_ranges_df[dose_ranges_df['medication'] == med]['units'].iloc[0]
                    dose_scale_text += f"{med_units}\n"
                except:
                    dose_scale_text += "units\n"

        # Position the dosing scale annotation below the legend in upper right
        # Get legend position and place text box below it
        ax.text(0.98, 0.65, dose_scale_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        return fig, None


    return (create_patient_vasoactive_chart,)


@app.cell
def _(mo):
    mo.md(r"""## Individual Patient Respiratory Support""")
    return


@app.cell(hide_code=True)
def _(
    create_patient_respiratory_chart,
    mo,
    patient_id,
    valeos_hospitalization_df,
    valeos_respiratory_df,
    valeos_transplant_df,
):
    # Create respiratory chart
    patient_resp_chart, resp_error_msg = create_patient_respiratory_chart(
        patient_id, valeos_respiratory_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if resp_error_msg:
        patient_respiratory_display = mo.md(f"**{resp_error_msg}**")
    elif patient_resp_chart is not None:
        patient_respiratory_display = mo.as_html(patient_resp_chart)
    else:
        patient_respiratory_display = mo.md("**No respiratory chart data available**")


    patient_respiratory_display
    return


@app.cell(hide_code=True)
def _(pd, plt):
    def create_patient_respiratory_chart(patient_id, respiratory_df, transplant_df, hosp_df):
        """Create individual patient respiratory support chart using proper CLIF data structure"""
        if patient_id is None or respiratory_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Get patient transplant info
        patient_transplant = transplant_df[transplant_df['patient_id'] == patient_id]
        if patient_transplant.empty:
            return None, f"No transplant data found for patient {patient_id}"

        transplant_date = pd.to_datetime(patient_transplant.iloc[0]['transplant_date'], utc=True)
        organ_type = patient_transplant.iloc[0]['transplant_type']

        # Get patient's hospitalizations to find hospitalization_ids (CLIF key structure)
        patient_hospitalizations = hosp_df[hosp_df['patient_id'] == patient_id].copy()
        if patient_hospitalizations.empty:
            return None, f"No hospitalization data found for patient {patient_id}"

        # Get hospitalization IDs for this patient
        patient_hosp_ids = patient_hospitalizations['hospitalization_id'].unique()

        # Filter respiratory data using hospitalization_id (CLIF standard approach)
        patient_resp_data = respiratory_df[
            respiratory_df['hospitalization_id'].isin(patient_hosp_ids)
        ].copy()

        if patient_resp_data.empty:
            return None, f"No respiratory support data found for patient {patient_id}"

        # Find the transplant hospitalization (contains transplant_date)
        patient_hospitalizations['admission_dttm'] = pd.to_datetime(patient_hospitalizations['admission_dttm'], utc=True)
        patient_hospitalizations['discharge_dttm'] = pd.to_datetime(patient_hospitalizations['discharge_dttm'], utc=True)

        # Find hospitalization that contains the transplant date
        transplant_hosp = patient_hospitalizations[
            (patient_hospitalizations['admission_dttm'] <= transplant_date) &
            (patient_hospitalizations['discharge_dttm'] >= transplant_date)
        ]

        if transplant_hosp.empty:
            return None, f"Cannot find transplant hospitalization for patient {patient_id}"

        admission_date = transplant_hosp.iloc[0]['admission_dttm']
        discharge_date = transplant_hosp.iloc[0]['discharge_dttm']
        discharge_category = transplant_hosp.iloc[0]['discharge_category'] if 'discharge_category' in transplant_hosp.columns else 'Unknown'

        # Process respiratory dates and calculate days since admission
        patient_resp_data['recorded_dttm'] = pd.to_datetime(patient_resp_data['recorded_dttm'], utc=True)
        patient_resp_data = patient_resp_data.sort_values('recorded_dttm')

        patient_resp_data['days_since_admission'] = (
            patient_resp_data['recorded_dttm'] - admission_date
        ).dt.total_seconds() / (24 * 3600)

        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)

        # Create seaborn plot
        plt.style.use('default')  # Reset to default style
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get unique device categories and assign colors
        device_categories = patient_resp_data['device_category'].unique()
        colors = plt.cm.Set3(range(len(device_categories)))
        device_color_map = dict(zip(device_categories, colors))

        # Plot respiratory support devices as horizontal bars
        for i, device in enumerate(device_categories):
            device_data = patient_resp_data[patient_resp_data['device_category'] == device]
            y_position = i

            # Create horizontal bars for each time period
            for _, row in device_data.iterrows():
                ax.barh(y_position, 0.1, left=row['days_since_admission'], 
                       color=device_color_map[device], alpha=0.7, height=0.8)

        # Add reference lines with discharge category in legend
        ax.axvline(x=0, color='blue', linestyle=':', linewidth=2, alpha=0.8, label='Admission (Day 0)')
        ax.axvline(x=transplant_days, color='green', linestyle='--', linewidth=3, alpha=0.8, 
                  label=f'Transplant (Day {transplant_days:.1f})')
        ax.axvline(x=discharge_days, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                  label=f'Discharge (Day {discharge_days:.1f}) to {discharge_category}')

        # Formatting
        ax.set_xlabel('Days Since Admission', fontsize=12)
        ax.set_ylabel('Respiratory Device Category', fontsize=12)
        ax.set_title(f'Respiratory Support - {organ_type.title()} Transplant - Patient {patient_id}', 
                    fontsize=14, fontweight='bold')

        # Set y-axis labels
        ax.set_yticks(range(len(device_categories)))
        ax.set_yticklabels(device_categories)

        # Add legend in better position
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Grid for better readability
        ax.grid(True, alpha=0.3)

        # Set x-axis limits to admission and discharge
        ax.set_xlim(left=-0.5, right=discharge_days + 0.5)

        plt.tight_layout()

        return fig, None

    return (create_patient_respiratory_chart,)


@app.cell
def _(mo):
    mo.md(r"""## Individual Patient Liver Function Timeline""")
    return


@app.cell(hide_code=True)
def _(
    create_patient_liver_function_chart,
    mo,
    patient_id,
    valeos_hospitalization_df,
    valeos_labs_df,
    valeos_transplant_df,
):
    # Create liver function chart
    patient_liver_chart, liver_error_msg = create_patient_liver_function_chart(
        patient_id, valeos_labs_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if liver_error_msg:
        patient_liver_display = mo.md(f"**{liver_error_msg}**")
    elif patient_liver_chart is not None:
        patient_liver_display = mo.as_html(patient_liver_chart)
    else:
        patient_liver_display = mo.md("**No liver function chart data available**")


    patient_liver_display
    return


@app.cell
def _(mo):
    mo.md(r"""## Population Vasoactive Trajectories""")
    return


@app.cell
def _(mo, valeos_transplant_df):
    # Organ selector for population trajectories
    population_organ_options = valeos_transplant_df['transplant_type'].unique().tolist()

    population_organ_selected = mo.ui.dropdown(
        options=population_organ_options,
        label="Select organ type for population trajectory:",
        value='heart'
    )

    population_organ_selected
    return (population_organ_selected,)


@app.cell
def _(population_trajectory_display):
    population_trajectory_display
    return


@app.cell
def _(
    create_population_vasoactive_trajectory,
    mo,
    population_organ_selected,
    valeos_hospitalization_df,
    valeos_med_continuous_df,
    valeos_transplant_df,
):
    # Get selected organ type
    population_organ_type = population_organ_selected.value if hasattr(population_organ_selected, 'value') else None

    # Create population trajectory chart
    population_chart, population_error_msg = create_population_vasoactive_trajectory(
        population_organ_type, valeos_med_continuous_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if population_error_msg:
        population_trajectory_display = mo.md(f"**{population_error_msg}**")
    elif population_chart is not None:
        population_trajectory_display = mo.as_html(population_chart)
    else:
        population_trajectory_display = mo.md("**No population trajectory data available**")

    return (population_trajectory_display,)


@app.cell
def _(pd, plt):
    def create_population_vasoactive_trajectory(selected_organ_type, med_df, transplant_df, hosp_df):
        """Create population-level vasoactive medication trajectory around transplant"""
        if selected_organ_type is None or med_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Define all vasoactive medications from mCIDE schema (excluding isoproterenol)
        vasoactive_meds = [
            'norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin', 
            'vasopressin', 'dopamine', 'dobutamine', 'milrinone'
        ]

        # Load vasoactive dose ranges for relative scaling
        import os
        dose_ranges_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vasoactive_dose_ranges.csv')
        try:
            dose_ranges_df = pd.read_csv(dose_ranges_path)
            dose_ranges = dict(zip(dose_ranges_df['medication'], dose_ranges_df['typical_max_dose']))
        except FileNotFoundError:
            # Fallback ranges if CSV not found
            dose_ranges = {
                'norepinephrine': 1.0, 'epinephrine': 1.0, 'phenylephrine': 200.0,
                'angiotensin': 20.0, 'vasopressin': 0.04, 'dopamine': 20.0,
                'dobutamine': 20.0, 'milrinone': 0.75, 'isoproterenol': 0.2
            }

        # Filter transplant data for selected organ
        organ_transplants = transplant_df[transplant_df['transplant_type'] == selected_organ_type].copy()
        if organ_transplants.empty:
            return None, f"No {selected_organ_type} transplant data found"

        # Convert transplant dates
        organ_transplants['transplant_date'] = pd.to_datetime(organ_transplants['transplant_date'], utc=True)

        # Get all patients for this organ type
        organ_patient_ids = organ_transplants['patient_id'].unique()

        # Get hospitalization data for these patients
        organ_hospitalizations = hosp_df[hosp_df['patient_id'].isin(organ_patient_ids)].copy()
        organ_hospitalizations['admission_dttm'] = pd.to_datetime(organ_hospitalizations['admission_dttm'], utc=True)
        organ_hospitalizations['discharge_dttm'] = pd.to_datetime(organ_hospitalizations['discharge_dttm'], utc=True)

        # Get hospitalization IDs
        organ_hosp_ids = organ_hospitalizations['hospitalization_id'].unique()

        # Filter medication data for vasoactives
        organ_med_data = med_df[
            (med_df['hospitalization_id'].isin(organ_hosp_ids)) & 
            (med_df['med_category'].isin(vasoactive_meds))
        ].copy()

        if organ_med_data.empty:
            return None, f"No vasoactive medication data found for {selected_organ_type} patients"

        # Process medication dates
        organ_med_data['admin_dttm'] = pd.to_datetime(organ_med_data['admin_dttm'], utc=True)

        # Merge with transplant dates and hospitalization data
        organ_med_data = organ_med_data.merge(
            organ_hospitalizations[['hospitalization_id', 'patient_id']], 
            on='hospitalization_id', how='left'
        )
        organ_med_data = organ_med_data.merge(
            organ_transplants[['patient_id', 'transplant_date']], 
            on='patient_id', how='left'
        )

        # Calculate days relative to transplant
        organ_med_data['days_from_transplant'] = (
            organ_med_data['admin_dttm'] - organ_med_data['transplant_date']
        ).dt.total_seconds() / (24 * 3600)

        # Filter to -15 to +15 days around transplant
        trajectory_data = organ_med_data[
            (organ_med_data['days_from_transplant'] >= -15) & 
            (organ_med_data['days_from_transplant'] <= 15)
        ].copy()

        if trajectory_data.empty:
            return None, f"No vasoactive data found in ±15 days around {selected_organ_type} transplant"

        # Calculate relative dose percentages
        trajectory_data['relative_dose_percent'] = trajectory_data.apply(
            lambda row: (row['med_dose'] / dose_ranges.get(row['med_category'], 1.0)) * 100 
            if pd.notna(row['med_dose']) and dose_ranges.get(row['med_category'], 1.0) > 0 else 0, 
            axis=1
        )

        # Round days to nearest integer for grouping
        trajectory_data['day_rounded'] = trajectory_data['days_from_transplant'].round().astype(int)

        # Calculate median dose for each medication by day
        daily_medians = trajectory_data.groupby(['day_rounded', 'med_category'])['relative_dose_percent'].median().reset_index()

        # Create plot
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(14, 8))

        # Use consistent color mapping
        vasoactive_color_map = {
            'norepinephrine': '#1f77b4',    # Blue
            'epinephrine': '#ff7f0e',       # Orange  
            'phenylephrine': '#2ca02c',     # Green
            'angiotensin': '#d62728',       # Red
            'vasopressin': '#9467bd',       # Purple
            'dopamine': '#8c564b',          # Brown
            'dobutamine': '#e377c2',        # Pink
            'milrinone': '#7f7f7f',         # Gray
            'isoproterenol': '#bcbd22'      # Olive
        }

        # Get unique medications present in data
        present_medications = daily_medians['med_category'].unique()

        # Plot each medication trajectory
        for med in present_medications:
            med_data = daily_medians[daily_medians['med_category'] == med]

            if not med_data.empty:
                color = vasoactive_color_map.get(med, '#000000')
                ax.plot(med_data['day_rounded'], med_data['relative_dose_percent'], 
                       color=color, linewidth=2, marker='o', markersize=4, 
                       label=med.title(), alpha=0.8)

        # Add transplant reference line
        ax.axvline(x=0, color='green', linestyle='--', linewidth=3, alpha=0.8, 
                  label='Transplant Day')

        # Formatting
        ax.set_xlabel('Days Relative to Transplant', fontsize=12)
        ax.set_ylabel('Median Relative Dose (% of Typical Maximum)', fontsize=12)
        ax.set_title(f'Population Vasoactive Trajectories - {selected_organ_type.title()} Transplant Recipients', 
                    fontsize=14, fontweight='bold')

        # Set x-axis limits and ticks
        ax.set_xlim(-15, 15)
        ax.set_xticks(range(-15, 16, 5))

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        # Add legend
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

        # Grid for better readability
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig, None

    return (create_population_vasoactive_trajectory,)


@app.cell
def _(mo):
    mo.md(r"""## Population Liver Function Tests

    This visualization shows liver function test trends around transplant for different organ types. 
    The chart displays median values with interquartile range (IQR) bars and normal reference ranges.

    **Tests included:**
    - Total Bilirubin (mg/dL)
    - ALT - Alanine Aminotransferase (U/L) 
    - AST - Aspartate Aminotransferase (U/L)
    - INR - International Normalized Ratio (no units)
    """)
    return


@app.cell
def _(mo, valeos_transplant_df):
    # Organ selector for population liver function
    liver_function_organ_options = valeos_transplant_df['transplant_type'].unique().tolist()

    liver_function_organ_selected = mo.ui.dropdown(
        options=liver_function_organ_options,
        label="Select organ type for population liver function analysis:",
        value='liver'
    )

    liver_function_organ_selected
    return (liver_function_organ_selected,)


@app.cell
def _(population_liver_function_display):
    population_liver_function_display
    return


@app.cell
def _(
    create_population_liver_function_chart,
    liver_function_organ_selected,
    mo,
    valeos_hospitalization_df,
    valeos_labs_df,
    valeos_transplant_df,
):
    # Get selected organ type for liver function analysis
    liver_function_organ_type = liver_function_organ_selected.value if hasattr(liver_function_organ_selected, 'value') else None

    # Create population liver function chart
    population_lft_chart, population_lft_error_msg = create_population_liver_function_chart(
        liver_function_organ_type, valeos_labs_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if population_lft_error_msg:
        population_liver_function_display = mo.md(f"**{population_lft_error_msg}**")
    elif population_lft_chart is not None:
        population_liver_function_display = mo.as_html(population_lft_chart)
    else:
        population_liver_function_display = mo.md("**No population liver function data available**")

    return (population_liver_function_display,)


@app.cell
def _(pd, plt):
    def create_population_liver_function_chart(selected_organ_type, labs_df, transplant_df, hosp_df):
        """Create population-level liver function test summary around transplant"""
        if selected_organ_type is None or labs_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Define liver function test lab categories from CLIF schema
        liver_function_labs = {
            'Total Bilirubin': 'bilirubin_total',
            'ALT': 'alt', 
            'AST': 'ast',
            'INR': 'inr'
        }

        # Filter transplant data for selected organ
        organ_transplants = transplant_df[transplant_df['transplant_type'] == selected_organ_type].copy()
        if organ_transplants.empty:
            return None, f"No {selected_organ_type} transplant data found"

        # Convert transplant dates
        organ_transplants['transplant_date'] = pd.to_datetime(organ_transplants['transplant_date'], utc=True)

        # Get all patients for this organ type
        organ_patient_ids = organ_transplants['patient_id'].unique()

        # Get hospitalization data for these patients
        organ_hospitalizations = hosp_df[hosp_df['patient_id'].isin(organ_patient_ids)].copy()
        organ_hospitalizations['admission_dttm'] = pd.to_datetime(organ_hospitalizations['admission_dttm'], utc=True)
        organ_hospitalizations['discharge_dttm'] = pd.to_datetime(organ_hospitalizations['discharge_dttm'], utc=True)

        # Get hospitalization IDs
        organ_hosp_ids = organ_hospitalizations['hospitalization_id'].unique()

        # Filter lab data for liver function tests
        organ_lab_data = labs_df[
            (labs_df['hospitalization_id'].isin(organ_hosp_ids)) & 
            (labs_df['lab_category'].isin(liver_function_labs.values()))
        ].copy()

        if organ_lab_data.empty:
            return None, f"No liver function test data found for {selected_organ_type} patients"

        # Process lab dates
        organ_lab_data['lab_result_dttm'] = pd.to_datetime(organ_lab_data['lab_result_dttm'], utc=True)

        # Merge with transplant dates and hospitalization data
        organ_lab_data = organ_lab_data.merge(
            organ_hospitalizations[['hospitalization_id', 'patient_id']], 
            on='hospitalization_id', how='left'
        )
        organ_lab_data = organ_lab_data.merge(
            organ_transplants[['patient_id', 'transplant_date']], 
            on='patient_id', how='left'
        )

        # Calculate days relative to transplant
        organ_lab_data['days_from_transplant'] = (
            organ_lab_data['lab_result_dttm'] - organ_lab_data['transplant_date']
        ).dt.total_seconds() / (24 * 3600)

        # Filter to -30 to +30 days around transplant
        lft_data = organ_lab_data[
            (organ_lab_data['days_from_transplant'] >= -30) & 
            (organ_lab_data['days_from_transplant'] <= 30)
        ].copy()

        if lft_data.empty:
            return None, f"No liver function data found in ±30 days around {selected_organ_type} transplant"

        # Filter out non-numeric values
        lft_data = lft_data.dropna(subset=['lab_value_numeric'])
        lft_data = lft_data[lft_data['lab_value_numeric'] > 0]  # Remove zero or negative values

        # Round days to nearest 3-day period for grouping
        lft_data['day_rounded'] = (lft_data['days_from_transplant'] / 3).round() * 3

        # Calculate median, quartiles, and count for each lab by day period
        daily_stats = lft_data.groupby(['day_rounded', 'lab_category'])['lab_value_numeric'].agg([
            'median', 'count', 
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75)   # Q3
        ]).reset_index()
        daily_stats.columns = ['day_rounded', 'lab_category', 'median_value', 'sample_count', 'q1_value', 'q3_value']

        # Filter out periods with very few samples
        daily_stats = daily_stats[daily_stats['sample_count'] >= 3]

        if daily_stats.empty:
            return None, f"Insufficient liver function data for {selected_organ_type} analysis"

        # Create subplot for each liver function test
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()

        # Define colors and normal ranges for each lab
        lab_colors = {
            'bilirubin_total': '#ff7f0e',   # Orange
            'alt': '#2ca02c',               # Green
            'ast': '#d62728',               # Red
            'inr': '#9467bd'                # Purple
        }

        normal_ranges = {
            'bilirubin_total': (0.2, 1.2),     # mg/dL
            'alt': (7, 35),                     # U/L (may vary by lab)
            'ast': (8, 40),                     # U/L (may vary by lab)
            'inr': (0.8, 1.2)                   # (no units)
        }

        lab_units = {
            'bilirubin_total': 'mg/dL',
            'alt': 'U/L',
            'ast': 'U/L', 
            'inr': '(no units)'
        }

        # Plot each liver function test
        for i, (display_name, lab_cat) in enumerate(liver_function_labs.items()):
            ax = axes[i]
            lab_data = daily_stats[daily_stats['lab_category'] == lab_cat]

            # Add normal range shading first
            if lab_cat in normal_ranges:
                normal_min, normal_max = normal_ranges[lab_cat]
                ax.axhspan(normal_min, normal_max, alpha=0.2, color=lab_colors[lab_cat], label='Normal Range')

            if not lab_data.empty:
                color = lab_colors.get(lab_cat, '#1f77b4')

                # Plot median line with markers
                ax.plot(lab_data['day_rounded'], lab_data['median_value'], 
                       color=color, linewidth=2, marker='o', markersize=6, 
                       alpha=0.8, label=f'Median {display_name}')

                # Add IQR vertical bars
                for _, row in lab_data.iterrows():
                    x_pos = row['day_rounded']
                    median_val = row['median_value']
                    q1_val = row['q1_value']
                    q3_val = row['q3_value']

                    # Draw vertical line from Q1 to Q3
                    ax.plot([x_pos, x_pos], [q1_val, q3_val], 
                           color=color, linewidth=3, alpha=0.6)

                    # Draw horizontal lines at Q1 and Q3
                    bar_width = 1.0  # Width of horizontal bars
                    ax.plot([x_pos - bar_width/2, x_pos + bar_width/2], [q1_val, q1_val], 
                           color=color, linewidth=2, alpha=0.6)
                    ax.plot([x_pos - bar_width/2, x_pos + bar_width/2], [q3_val, q3_val], 
                           color=color, linewidth=2, alpha=0.6)

                # Set appropriate y-axis limits including IQR data
                data_min = min(lab_data['q1_value'].min(), lab_data['median_value'].min())
                data_max = max(lab_data['q3_value'].max(), lab_data['median_value'].max())

                if lab_cat in normal_ranges:
                    norm_min, norm_max = normal_ranges[lab_cat]
                    y_min = min(data_min * 0.9, norm_min * 0.5)
                    y_max = max(data_max * 1.1, norm_max * 1.5)
                else:
                    y_min = data_min * 0.9
                    y_max = data_max * 1.1

                ax.set_ylim(y_min, y_max)

            # Add transplant reference line
            ax.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Transplant Day')

            # Formatting for each subplot
            unit = lab_units.get(lab_cat, '')
            ax.set_ylabel(f'{display_name} ({unit})', fontsize=11)
            ax.set_title(f'{display_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-30, 30)

            # Add legend for first subplot with IQR explanation
            if i == 0:
                # Create custom legend elements
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Transplant Day'),
                    Line2D([0], [0], color=lab_colors['bilirubin_total'], alpha=0.2, linewidth=10, label='Normal Range'),
                    Line2D([0], [0], color='black', marker='o', linewidth=2, markersize=6, label='Median'),
                    Line2D([0], [0], color='black', linewidth=3, alpha=0.6, label='Interquartile Range (IQR)')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Set shared x-axis labels
        for ax in axes[2:]:  # Bottom row
            ax.set_xlabel('Days Relative to Transplant', fontsize=12)

        # Add overall title
        fig.suptitle(f'Population Liver Function Tests - {selected_organ_type.title()} Transplant Recipients', 
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        return fig, None

    return (create_population_liver_function_chart,)


@app.cell(hide_code=True)
def _(pd, plt):
    def create_patient_liver_function_chart(patient_id, labs_df, transplant_df, hosp_df):
        """Create individual patient liver function timeline showing MELD components"""
        if patient_id is None or labs_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Get patient transplant info
        patient_transplant = transplant_df[transplant_df['patient_id'] == patient_id]
        if patient_transplant.empty:
            return None, f"No transplant data found for patient {patient_id}"

        transplant_date = pd.to_datetime(patient_transplant.iloc[0]['transplant_date'], utc=True)
        organ_type = patient_transplant.iloc[0]['transplant_type']

        # Get patient's hospitalizations to find hospitalization_ids (CLIF key structure)
        patient_hospitalizations = hosp_df[hosp_df['patient_id'] == patient_id].copy()
        if patient_hospitalizations.empty:
            return None, f"No hospitalization data found for patient {patient_id}"

        # Get hospitalization IDs for this patient
        patient_hosp_ids = patient_hospitalizations['hospitalization_id'].unique()

        # Define MELD component lab categories
        meld_labs = {
            'INR': 'inr',
            'Total Bilirubin': 'bilirubin_total', 
            'Creatinine': 'creatinine',
            'Sodium': 'sodium'
        }

        # Filter lab data for MELD components using hospitalization_id (CLIF standard approach)
        patient_lab_data = labs_df[
            (labs_df['hospitalization_id'].isin(patient_hosp_ids)) & 
            (labs_df['lab_category'].isin(meld_labs.values()))
        ].copy()

        if patient_lab_data.empty:
            return None, f"No MELD component lab data found for patient {patient_id}"

        # Find the transplant hospitalization (contains transplant_date)
        patient_hospitalizations['admission_dttm'] = pd.to_datetime(patient_hospitalizations['admission_dttm'], utc=True)
        patient_hospitalizations['discharge_dttm'] = pd.to_datetime(patient_hospitalizations['discharge_dttm'], utc=True)

        # Find hospitalization that contains the transplant date
        transplant_hosp = patient_hospitalizations[
            (patient_hospitalizations['admission_dttm'] <= transplant_date) &
            (patient_hospitalizations['discharge_dttm'] >= transplant_date)
        ]

        if transplant_hosp.empty:
            return None, f"Cannot find transplant hospitalization for patient {patient_id}"

        admission_date = transplant_hosp.iloc[0]['admission_dttm']
        discharge_date = transplant_hosp.iloc[0]['discharge_dttm']
        discharge_category = transplant_hosp.iloc[0]['discharge_category'] if 'discharge_category' in transplant_hosp.columns else 'Unknown'

        # Process lab dates and calculate days since admission
        patient_lab_data['lab_result_dttm'] = pd.to_datetime(patient_lab_data['lab_result_dttm'], utc=True)
        patient_lab_data = patient_lab_data.sort_values('lab_result_dttm')

        patient_lab_data['days_since_admission'] = (
            patient_lab_data['lab_result_dttm'] - admission_date
        ).dt.total_seconds() / (24 * 3600)

        transplant_days = (transplant_date - admission_date).total_seconds() / (24 * 3600)
        discharge_days = (discharge_date - admission_date).total_seconds() / (24 * 3600)

        # Create subplots (4 rows, 1 column)
        plt.style.use('default')  # Reset to default style
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        # Define colors and normal ranges for each lab
        lab_colors = {'inr': '#d62728', 'bilirubin_total': '#ff7f0e', 'creatinine': '#2ca02c', 'sodium': '#1f77b4'}
        normal_ranges = {
            'inr': (0.8, 1.2),
            'bilirubin_total': (0.2, 1.2),
            'creatinine': (0.7, 1.3), 
            'sodium': (135, 145)  # Note: CLIF uses mmol/L which is same numeric range as mEq/L
        }

        lab_titles = ['INR', 'Total Bilirubin (mg/dL)', 'Creatinine (mg/dL)', 'Sodium (mmol/L)']

        # Plot each MELD component
        for i, (display_name, lab_cat) in enumerate(meld_labs.items()):
            ax = axes[i]
            lab_data = patient_lab_data[patient_lab_data['lab_category'] == lab_cat]

            # Add normal range shading first
            if lab_cat in normal_ranges:
                normal_min, normal_max = normal_ranges[lab_cat]
                ax.axhspan(normal_min, normal_max, alpha=0.2, color=lab_colors[lab_cat])

            if not lab_data.empty:
                # Filter out non-numeric values and use lab_value_numeric
                numeric_data = lab_data.dropna(subset=['lab_value_numeric'])

                if not numeric_data.empty:
                    # Plot lab values as scatter points
                    ax.scatter(numeric_data['days_since_admission'], numeric_data['lab_value_numeric'], 
                              color=lab_colors[lab_cat], s=40, alpha=0.8)

                    # Set appropriate y-axis limits based on data and normal ranges
                    data_min = numeric_data['lab_value_numeric'].min()
                    data_max = numeric_data['lab_value_numeric'].max()

                    if lab_cat in normal_ranges:
                        norm_min, norm_max = normal_ranges[lab_cat]
                        y_min = min(data_min * 0.9, norm_min * 0.8)
                        y_max = max(data_max * 1.1, norm_max * 1.2)
                    else:
                        y_min = data_min * 0.9
                        y_max = data_max * 1.1

                    ax.set_ylim(y_min, y_max)

            # Add reference lines
            ax.axvline(x=0, color='blue', linestyle=':', linewidth=2, alpha=0.8)
            ax.axvline(x=transplant_days, color='green', linestyle='--', linewidth=3, alpha=0.8)
            ax.axvline(x=discharge_days, color='orange', linestyle=':', linewidth=2, alpha=0.8)

            # Formatting for each subplot
            ax.set_ylabel(lab_titles[i], fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=-0.5, right=discharge_days + 0.5)

            # Add legend only for first subplot
            if i == 0:
                # Create legend handles manually for proper labels
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='blue', linestyle=':', linewidth=2, label='Admission (Day 0)'),
                    Line2D([0], [0], color='green', linestyle='--', linewidth=3, label=f'Transplant (Day {transplant_days:.1f})'),
                    Line2D([0], [0], color='orange', linestyle=':', linewidth=2, label=f'Discharge (Day {discharge_days:.1f}) to {discharge_category}')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Set shared x-axis label
        axes[-1].set_xlabel('Days Since Admission', fontsize=12)

        # Add overall title
        fig.suptitle(f'Liver Function (MELD Components) - {organ_type.title()} Transplant - Patient {patient_id}', 
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        return fig, None

    return (create_patient_liver_function_chart,)


@app.cell
def _(mo):
    mo.md(r"""## Individual Patient Code Status""")
    return


@app.cell(hide_code=True)
def _(
    create_patient_code_status_chart,
    mo,
    patient_id,
    valeos_code_status_df,
    valeos_hospitalization_df,
    valeos_transplant_df,
):
    # Create code status chart
    patient_code_status_chart, code_status_error_msg = create_patient_code_status_chart(
        patient_id, valeos_code_status_df, valeos_transplant_df, valeos_hospitalization_df
    )

    if code_status_error_msg:
        patient_code_status_display = mo.md(f"**{code_status_error_msg}**")
    elif patient_code_status_chart is not None:
        patient_code_status_display = mo.as_html(patient_code_status_chart)
    else:
        patient_code_status_display = mo.md("**No code status data available**")

    patient_code_status_display
    return


@app.cell(hide_code=True)
def _(pd, plt):
    def create_patient_code_status_chart(patient_id, code_status_df, transplant_df, hosp_df):
        """Create individual patient code status timeline during transplant hospitalization"""
        if patient_id is None or code_status_df is None or transplant_df is None or hosp_df is None:
            return None, "Missing required data"

        # Get patient transplant info
        patient_transplant = transplant_df[transplant_df['patient_id'] == patient_id]
        if patient_transplant.empty:
            return None, f"No transplant data found for patient {patient_id}"

        transplant_date = pd.to_datetime(patient_transplant.iloc[0]['transplant_date'], utc=True)
        organ_type = patient_transplant.iloc[0]['transplant_type']

        # Get patient's hospitalizations to find hospitalization_ids (CLIF key structure)
        patient_hospitalizations = hosp_df[hosp_df['patient_id'] == patient_id].copy()
        if patient_hospitalizations.empty:
            return None, f"No hospitalization data found for patient {patient_id}"

        # Get hospitalization IDs for this patient
        patient_hosp_ids = patient_hospitalizations['hospitalization_id'].unique()

        # Filter code status data 
        patient_code_status_data = code_status_df[
            code_status_df['patient_id'].isin(patient_hosp_ids)
        ].copy()

        if patient_code_status_data.empty:
            return None, f"No code status data found for patient {patient_id}"

        # Find the transplant hospitalization (contains transplant_date)
        patient_hospitalizations['admission_dttm'] = pd.to_datetime(patient_hospitalizations['admission_dttm'], utc=True)
        patient_hospitalizations['discharge_dttm'] = pd.to_datetime(patient_hospitalizations['discharge_dttm'], utc=True)

        # Find hospitalization that contains the transplant date
        transplant_hosp = patient_hospitalizations[
            (patient_hospitalizations['admission_dttm'] <= transplant_date) &
            (patient_hospitalizations['discharge_dttm'] >= transplant_date)
        ]

        if transplant_hosp.empty:
            return None, f"Cannot find transplant hospitalization for patient {patient_id}"

        admission_date = transplant_hosp.iloc[0]['admission_dttm']
        discharge_date = transplant_hosp.iloc[0]['discharge_dttm']
        transplant_hosp_id = transplant_hosp.iloc[0]['hospitalization_id']

        # Filter code status data to transplant hospitalization only
        transplant_code_status = patient_code_status_data[
            patient_code_status_data['hospitalization_id'] == transplant_hosp_id
        ].copy()

        if transplant_code_status.empty:
            return None, f"No code status data found during transplant hospitalization for patient {patient_id}"

        # Convert datetime and sort by start time
        transplant_code_status['start_dttm'] = pd.to_datetime(transplant_code_status['start_dttm'], utc=True)
        transplant_code_status = transplant_code_status.sort_values('start_dttm')

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Define color mapping for code status categories
        status_colors = {
            'Full': '#2E8B57',           # Sea Green - most active
            'Presume Full': '#90EE90',   # Light Green - presumed active
            'DNR': '#FF6347',            # Tomato Red - do not resuscitate
            'DNR/DNI': '#B22222',        # Fire Brick - do not resuscitate/intubate
            'UDNR': '#FF69B4',           # Hot Pink - uncertain DNR
            'Other': '#808080'           # Gray - other/unknown
        }

        # Create timeline visualization
        y_pos = 0
        prev_end_time = admission_date
        
        for idx, row in transplant_code_status.iterrows():
            start_time = row['start_dttm']
            status = row['code_status_category']
            status_name = row['code_status_name']
            
            # Calculate hours from admission for positioning
            start_hours = (start_time - admission_date).total_seconds() / 3600
            
            # For visualization, assume status continues until next change or discharge
            if idx < len(transplant_code_status) - 1:
                next_start = transplant_code_status.iloc[idx + 1]['start_dttm']
                end_time = min(next_start, discharge_date)
            else:
                end_time = discharge_date
                
            end_hours = (end_time - admission_date).total_seconds() / 3600
            duration_hours = end_hours - start_hours
            
            # Create rectangle for this status period
            color = status_colors.get(status, '#808080')
            rect = plt.Rectangle((start_hours, y_pos - 0.3), duration_hours, 0.6,
                               facecolor=color, alpha=0.7, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add status label if duration is long enough
            if duration_hours > 6:  # Only label if at least 6 hours
                ax.text(start_hours + duration_hours/2, y_pos, status, 
                       ha='center', va='center', fontweight='bold', fontsize=10)

        # Add reference lines
        transplant_hours = (transplant_date - admission_date).total_seconds() / 3600
        discharge_hours = (discharge_date - admission_date).total_seconds() / 3600
        
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Admission')
        ax.axvline(x=transplant_hours, color='red', linestyle='-', alpha=0.9, linewidth=3, label='Transplant')
        ax.axvline(x=discharge_hours, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='Discharge')

        # Format axes
        ax.set_xlim(-2, discharge_hours + 2)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel('Hours from Admission', fontsize=12)
        ax.set_ylabel('')
        ax.set_yticks([])
        
        # Set title
        ax.set_title(f'Code Status Timeline - Patient {patient_id} ({organ_type.title()} Transplant)', 
                     fontsize=14, fontweight='bold', pad=20)

        # Add legend for reference lines
        ax.legend(loc='upper right')
        
        # Add legend for code status colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=status) 
                          for status, color in status_colors.items() 
                          if status in transplant_code_status['code_status_category'].values]
        
        ax2 = ax.twinx()
        ax2.legend(handles=legend_elements, loc='upper left', title='Code Status Categories')
        ax2.set_yticks([])
        
        plt.tight_layout()
        
        return fig, None

    return (create_patient_code_status_chart,)

if __name__ == "__main__":
    app.run()
