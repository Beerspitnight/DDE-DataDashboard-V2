# 1) Imports and Configuration
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Middle School Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2) Title and Header
st.title("Middle School Data Dashboard")
st.header("Analyze and Visualize Student Performance")

# 3) Essential Columns List
essential_columns = [
    'Student Grade', 'Class Teacher(s)', 'School'
]

# 4) Helper Function Definitions
@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded CSV file."""
    data = pd.read_csv(uploaded_file)
    return data

# --- MODIFIED: This function now works for both DataFrames and Series ---
def check_cols(df_or_series, cols):
    """A helper function to check if a list of columns/indices exists in a dataframe or series."""
    if isinstance(df_or_series, pd.DataFrame):
        return all(col in df_or_series.columns for col in cols)
    elif isinstance(df_or_series, pd.Series):
        return all(col in df_or_series.index for col in cols)
    return False

# --- Sidebar ---
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file with student data", type=["csv"]
)

# 5) Main Application Logic
if uploaded_file:
    data = load_data(uploaded_file)
    
    # --- MODIFIED: Safely create the 'Student Name' column ---
    if check_cols(data, ['First Name', 'Last Name']):
        data['Student Name'] = data['First Name'] + ' ' + data['Last Name']

    # (A) Check for ESSENTIAL columns only
    # ... (this section is unchanged) ...
    missing_essential_columns = [col for col in essential_columns if col not in data.columns]
    if missing_essential_columns:
        st.error(
            "The following essential columns are missing and required for the app to run: "
            f"`{', '.join(missing_essential_columns)}`"
        )
        st.stop()

    # (B) Data Preview
    with st.expander("Preview of Uploaded Data", expanded=False):
        st.dataframe(data.head())

    # (C) Sidebar Filters (now more robust)
    # ... (this section is unchanged) ...
    st.sidebar.header("Filters")
    
    filtered_data = data.copy()

    student_grade_filter = st.sidebar.multiselect("Grade", options=data['Student Grade'].unique(), default=data['Student Grade'].unique())
    filtered_data = filtered_data[filtered_data['Student Grade'].isin(student_grade_filter)]

    school_filter = st.sidebar.multiselect("School", options=data['School'].unique(), default=data['School'].unique())
    filtered_data = filtered_data[filtered_data['School'].isin(school_filter)]
    
    if 'Subject' in data.columns:
        subject_filter = st.sidebar.multiselect("Subject", options=data['Subject'].unique(), default=data['Subject'].unique())
        filtered_data = filtered_data[filtered_data['Subject'].isin(subject_filter)]

    grade_teacher_map = filtered_data.groupby('Student Grade')['Class Teacher(s)'].unique().to_dict()
    teacher_options_labeled = [f"{g}: {t}" for g, teachers in grade_teacher_map.items() for t in teachers]
    
    if teacher_options_labeled:
        enhanced_teacher_filter = st.sidebar.multiselect(
            "Select Class Teachers by Grade",
            options=teacher_options_labeled,
            default=teacher_options_labeled
        )
        selected_teachers = {label.split(': ')[1] for label in enhanced_teacher_filter}
        filtered_data = filtered_data[filtered_data['Class Teacher(s)'].isin(selected_teachers)]

    if filtered_data.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
        st.stop()

     # (D) Key Metrics Analysis: i-Ready (Optional Section)
    iready_columns = [
        'Total Lesson Time-on-Task (min)', 'i-Ready Overall: Lessons Passed',
        'i-Ready Overall: Lessons Completed', 'i-Ready Overall: % Lessons Passed'
    ]
    if check_cols(data, iready_columns):
        st.header("Key Metrics Analysis: i-Ready")
        
        # This small table is used for the Box Plot and Heatmap
        iready_data = filtered_data[iready_columns]

        st.sidebar.header("i-Ready Visualization Options")
        iready_vis_type = st.sidebar.selectbox(
            "Visualization Type for i-Ready",
            ["Bar Chart", "Box Plot", "Scatter Plot", "Correlation Heatmap"],
            key="iready_vis"
        )
        if iready_vis_type == "Bar Chart":
            avg_series = iready_data.mean()
            avg_values = avg_series.reset_index()
            avg_values.columns = ['Metric', 'Average Value']
            fig = px.bar(avg_values, x='Metric', y='Average Value', title="Average i-Ready Metrics")
            st.plotly_chart(fig, use_container_width=True)

        elif iready_vis_type == "Box Plot":
            fig = px.box(iready_data, title="Distribution of i-Ready Metrics")
            st.plotly_chart(fig, use_container_width=True)

        elif iready_vis_type == "Scatter Plot":
            col_x = st.sidebar.selectbox("X-axis", options=iready_columns, index=1, key="iready_x")
            col_y = st.sidebar.selectbox("Y-axis", options=iready_columns, index=3, key="iready_y")
            
            # --- CORRECTED LOGIC ---
            # We use the main 'filtered_data' table here to access the Student Name column.
            scatter_args = {
                'x': col_x, 
                'y': col_y, 
                'title': f"Scatter Plot: {col_x} vs {col_y}"
            }
            
            # Only add the hover_name argument if the "Student Name" column exists
            if 'Student Name' in filtered_data.columns:
                scatter_args['hover_name'] = "Student Name"

            fig = px.scatter(filtered_data, **scatter_args)
            
            # Use st.plotly_chart for Plotly figures, not st.pyplot
            st.plotly_chart(fig, use_container_width=True)

        elif iready_vis_type == "Correlation Heatmap":
            corr = iready_data.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='coolwarm',
                            title="Correlation Heatmap of i-Ready Metrics")
            st.plotly_chart(fig, use_container_width=True)

    
    # (E) Domain Analysis (Optional Section)
    # ... (this section is unchanged) ...
    st.header("Domain Analysis")
    all_domains = {
        "Algebra and Algebraic Thinking": ['i-Ready Algebra and Algebraic Thinking: Lessons Passed', 'i-Ready Algebra and Algebraic Thinking: Lessons Completed', 'i-Ready Algebra and Algebraic Thinking: % Lessons Passed'],
        "Number and Operations": ['i-Ready Number and Operations: Lessons Passed', 'i-Ready Number and Operations: Lessons Completed', 'i-Ready Number and Operations: % Lessons Passed'],
        "Measurement and Data": ['i-Ready Measurement and Data: Lessons Passed', 'i-Ready Measurement and Data: Lessons Completed', 'i-Ready Measurement and Data: % Lessons Passed'],
        "Geometry": ['i-Ready Geometry: Lessons Passed', 'i-Ready Geometry: Lessons Completed', 'i-Ready Geometry: % Lessons Passed'],
        "Pro Whole Numbers and Operations": ["i-Ready Pro Whole Numbers and Operations: Skills Successful", "i-Ready Pro Whole Numbers and Operations: Skills Completed", "i-Ready Pro Whole Numbers and Operations: % Skills Successful"],
    }
    available_domains = {name: cols for name, cols in all_domains.items() if check_cols(data, cols)}

    if not available_domains:
        st.warning("No complete domain data found in the uploaded file to analyze.")
    else:
        selected_domain = st.selectbox("Select Domain for Analysis", options=list(available_domains.keys()))

        # Ensure a domain was selected before proceeding
        if selected_domain is not None:
            domain_columns = available_domains[selected_domain]

            filtered_comparison_data = filtered_data.copy()

            student_info_cols = ['Last Name', 'First Name', 'Student ID']

            if not filtered_comparison_data.empty:
                tab1, tab2, tab3 = st.tabs(["Failing Students", "Class Comparison", "Progress Distribution"])

                with tab1:
                    if not check_cols(data, student_info_cols):
                        st.warning("Cannot display student list because 'Last Name', 'First Name', or 'Student ID' columns are missing.")
                    else:
                        st.sidebar.header(f"{selected_domain} Analysis Options")
                        pass_threshold = st.sidebar.slider("Select % Passed Threshold", 0, 100, 70)
                        failing_students = filtered_comparison_data[filtered_comparison_data[domain_columns[2]] < pass_threshold]
                        st.write(f"#### Students with {selected_domain} % Passed Below {pass_threshold}%")
                        st.dataframe(failing_students[student_info_cols + domain_columns])

                with tab2:
                    st.write(f"#### Class Progress vs. Other Selected Classes in {selected_domain}")
                    class_avg = filtered_comparison_data.groupby("Class Teacher(s)")[domain_columns].mean().reset_index()
                    class_avg = class_avg.sort_values(domain_columns[0], ascending=False)

                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(x=class_avg["Class Teacher(s)"], y=class_avg[domain_columns[0]], name=domain_columns[0]), secondary_y=False)
                    fig.add_trace(go.Bar(x=class_avg["Class Teacher(s)"], y=class_avg[domain_columns[1]], name=domain_columns[1]), secondary_y=False)
                    fig.add_trace(go.Scatter(x=class_avg["Class Teacher(s)"], y=class_avg[domain_columns[2]], name=domain_columns[2], mode='lines+markers'), secondary_y=True)

                    fig.update_layout(title_text=f"{selected_domain} - Class Averages", barmode="group")
                    fig.update_yaxes(title_text="Count Metrics (Passed/Completed)", secondary_y=False)
                    fig.update_yaxes(title_text="Percentage (%)", range=[0, 101], secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.write(f"#### Student Progress Distribution in {selected_domain}")
                    fig = px.histogram(filtered_comparison_data, x=domain_columns[2], nbins=20,
                                       title=f"Distribution of % Lessons Passed in {selected_domain}",
                                       labels={domain_columns[2]: "Percentage of Lessons Passed"})
                    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    #              NEW FEATURES
    # ==========================================
    st.markdown("---")

    # --- FEATURE 1: Student Deep Dive ---
    st.header("🧑‍🎓 Student Deep Dive")
    if 'Student Name' in filtered_data.columns:
        student_list = sorted(filtered_data['Student Name'].unique())
        selected_student = st.selectbox("Select a Student to Analyze", options=student_list)

        student_data = filtered_data[filtered_data['Student Name'] == selected_student].iloc[0]
        
        st.subheader(f"Report Card for: {selected_student}")

        # Overall Metrics
        # --- MODIFIED: The check_cols function now correctly handles the student_data Series ---
        if check_cols(student_data, iready_columns):
            class_avg_pass_rate = filtered_data['i-Ready Overall: % Lessons Passed'].mean()
            class_avg_time = filtered_data['Total Lesson Time-on-Task (min)'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric(
                label="Overall Pass Rate",
                value=f"{student_data['i-Ready Overall: % Lessons Passed']:.0f}%",
                delta=f"{student_data['i-Ready Overall: % Lessons Passed'] - class_avg_pass_rate:.0f} vs class avg"
            )
            col2.metric(
                label="Time on Task (min)",
                value=f"{student_data['Total Lesson Time-on-Task (min)']:.0f}",
                delta=f"{student_data['Total Lesson Time-on-Task (min)'] - class_avg_time:.0f} vs class avg"
            )
            col3.metric(
                label="Lessons Passed",
                value=student_data['i-Ready Overall: Lessons Passed']
            )

        # Domain performance chart
        student_domain_data = []
        for domain_name, domain_cols in available_domains.items():
            pass_rate_col = domain_cols[2]
            student_pass_rate = student_data.get(pass_rate_col, 0)
            class_avg_pass_rate = filtered_data[pass_rate_col].mean()
            student_domain_data.append({
                "Domain": domain_name,
                "Who": "Student",
                "Pass Rate": student_pass_rate
            })
            student_domain_data.append({
                "Domain": domain_name,
                "Who": "Class Average",
                "Pass Rate": class_avg_pass_rate
            })
        
        domain_df = pd.DataFrame(student_domain_data)
        fig = px.bar(
            domain_df, 
            x="Domain", y="Pass Rate", color="Who", 
            barmode="group", title="Student Domain Performance vs. Class Average",
            labels={"Pass Rate": "% Lessons Passed"}
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Add 'First Name' and 'Last Name' columns to the data to enable the Student Deep Dive.")


     # --- FEATURE 2: Performance vs. Time-on-Task ---
    st.markdown("---")
    st.header("⏱️ Performance vs. Time-on-Task")
    time_perf_cols = ['Total Lesson Time-on-Task (min)', 'i-Ready Overall: % Lessons Passed']
    if check_cols(filtered_data, time_perf_cols):
        color_by = st.radio("Color points by:", ('Grade', 'Teacher'), horizontal=True, key='time_color')
        
        color_map = {'Grade': 'Student Grade', 'Teacher': 'Class Teacher(s)'}
        
        # Determine which column to use for hover text, if available
        hover_name_col = "Student Name" if 'Student Name' in filtered_data.columns else None

        # Call the plotting function directly with the correct parameters
        fig = px.scatter(
            data_frame=filtered_data,
            x='Total Lesson Time-on-Task (min)',
            y='i-Ready Overall: % Lessons Passed',
            color=color_map[color_by or 'Grade'], # Use 'Grade' as default if color_by is None
            title="Time on Task vs. Overall Pass Rate",
            trendline="ols",
            hover_name=hover_name_col
        )
        st.plotly_chart(fig, use_container_width=True)

     # --- FEATURE 3: Grade-Level Comparison ---
    st.markdown("---")
    st.header("📊 Grade-Level Comparison")
    if check_cols(filtered_data, iready_columns):
        # Calculate the average of all i-Ready metrics for each grade
        grade_avg_data = filtered_data.groupby('Student Grade')[iready_columns].mean().reset_index()
        
        # "Melt" the data to turn the metric columns into rows, which is needed for faceting
        melted_data = grade_avg_data.melt(
            id_vars='Student Grade', 
            value_vars=iready_columns, 
            var_name='Metric', 
            value_name='Average Value'
        )
        
        # Create a faceted bar chart
        fig = px.bar(
            melted_data,
            x='Student Grade',
            y='Average Value',
            color='Student Grade',
            facet_row='Metric',  # This creates a separate chart row for each metric
            title='Grade-Level Performance Comparison',
            labels={'Average Value': 'Average'}
        )
        
        # Update axes to have independent scales and remove redundant x-axis labels
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_layout(height=600) # Give the chart more vertical space

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file using the sidebar to begin analyzing student data.")