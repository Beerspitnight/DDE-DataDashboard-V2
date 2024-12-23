# 1) Imports and Configuration
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Middle School Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2) Title and Header
st.title("Middle School Data Dashboard")
st.header("Analyze and Visualize Student Performance")

# 3) Upload Section (Outside if statement, at top level)
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file with student data", type=["csv"]
)

# 4) Required Columns List
required_columns = [
    # ... your list of required columns ...
]

# 5) Helper Function Definitions
def fetch_ai_insights(metric, mean_value=None, max_value=None):
    """
    Returns a dictionary of dynamic explanations/suggestions 
    based on the passed-in metric and optional numeric stats.
    """
    explanation = f"Analyzing '{metric}' for the current filtered dataset."
    if mean_value is not None:
        explanation += f" The average value is {mean_value:.2f}."
    if max_value is not None:
        explanation += f" The maximum recorded value is {max_value:.2f}."

    suggestions = [
        f"Investigate potential reasons behind '{metric}' peaking at {max_value:.2f}." if max_value else "",
        f"Compare '{metric}' across different teacher groups or grade levels."
    ]
    return {"explanation": explanation, "suggestions": suggestions}


def filter_non_attempted_students(data, domain_columns):
    """Filter out students who have not attempted any lessons in a domain."""
    return data[
        (data[domain_columns[0]] > 0)
        | (data[domain_columns[1]] > 0)
        | (data[domain_columns[2]].notnull())
    ]

def overview_component(df):
    """
    Display an overview of the dataset, including column stats and unique counts.
    """
    st.write("## Data Overview")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write("Columns and unique values:")
    unique_counts = {col: df[col].nunique() for col in df.columns}
    st.write(unique_counts)

# 6) Main Logic: Only run if a file is uploaded
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with student data", type=["csv"])

# In the sidebar:
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file with student data", 
    type=["csv"]
)

if uploaded_file:
    # Read the CSV
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

else:
    st.write("Please upload a CSV file to start analyzing data.")

    # (B) Check required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(
            "The following required columns are missing from the uploaded file: "
            + ", ".join(missing_columns)
        )
        st.stop()

    # (C) Display the first few rows
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    # (D) Sidebar Filters
    st.sidebar.header("Filters")

    # Organize filters into 3 columns
    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        student_grade_filter = st.multiselect(
            "Grade", 
            options=data['Student Grade'].unique(), 
            default=data['Student Grade'].unique()
        )
    with col2:
        subject_filter = st.multiselect(
            "Subject", 
            options=data['Subject'].unique(), 
            default=data['Subject'].unique()
        )
    with col3:
        school_filter = st.multiselect(
            "School", 
            options=data['School'].unique(), 
            default=data['School'].unique()
        )
# Build a mapping of grade -> set of teachers
grade_teacher_map = {}
for idx, row in data.iterrows():
    grade = row['Student Grade']
    teacher = row['Class Teacher(s)']
    grade_teacher_map.setdefault(grade, set()).add(teacher)

# Convert mapping into labeled strings for the teacher filter
teacher_options_labeled = []
for g, teachers in grade_teacher_map.items():
    for t in teachers:
        teacher_options_labeled.append(f"{g}: {t}")

# Convert mapping into labeled strings for the teacher filter
teacher_options_labeled = []
for g, teachers in grade_teacher_map.items():
    for t in teachers:
        teacher_options_labeled.append(f"{g}: {t}")


    # Keep teacher filter separate since it requires additional logic
    enhanced_teacher_filter = st.sidebar.multiselect(
        "Select Class Teachers by Grade",
        options=teacher_options_labeled,
        default=teacher_options_labeled
    )

    # Parse teacher label
    def parse_teacher_label(label_string):
        parts = label_string.split(":")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        else:
            return None, label_string

    selected_teachers = set()
    for label in enhanced_teacher_filter:
        _, teacher_name = parse_teacher_label(label)
        selected_teachers.add(teacher_name)

    # Apply combined filter
    filtered_data = data[
        (data['Student Grade'].isin(student_grade_filter)) &
        (data['Subject'].isin(subject_filter)) &
        (data['School'].isin(school_filter)) &
        (data['Class Teacher(s)'].isin(selected_teachers))
    ]

    if filtered_data.empty:
        st.warning("No data available after filters! Please adjust your selections.")
        st.stop()  # Prevent further execution if no data is available

    st.subheader("Filtered Data")
    st.dataframe(filtered_data)

    # (E) Experimental Components
    st.subheader("Experimental Components")
    if st.checkbox("Show Data Overview"):
        overview_component(filtered_data)

    # (F) Key Metrics for i-Ready
    st.subheader("Key Metrics Analysis: i-Ready")
    iready_columns = [
        'Total Lesson Time-on-Task (min)',
        'i-Ready Overall: Lessons Passed',
        'i-Ready Overall: Lessons Completed',
        'i-Ready Overall: % Lessons Passed'
    ]
    iready_data = filtered_data[iready_columns]

    st.sidebar.header("i-Ready Visualization Options")
    iready_vis_type = st.sidebar.selectbox(
        "Visualization Type for i-Ready", 
        ["Scatter Plot", "Bar Chart", "Box Plot", "Correlation Heatmap"], 
        index=1,  # Default to "Bar Chart"
        key="iready_vis"
    )


    # ---------------------------
    # i-Ready Plot Logic
    # ---------------------------
    if iready_vis_type == "Scatter Plot":
        # Indent everything inside this IF block
        col_x = st.sidebar.selectbox(
            "X-axis", 
            options=iready_columns, 
            index=1,  # Default to 'Lessons Passed'
            key="iready_x"
        )
        col_y = st.sidebar.selectbox(
            "Y-axis", 
            options=iready_columns, 
            index=3,  # Default to '% Lessons Passed'
            key="iready_y"
        )

        fig, ax = plt.subplots()
        sns.scatterplot(data=iready_data, x=col_x, y=col_y, ax=ax)
        ax.set_title(f"Scatter Plot: {col_x} vs {col_y}")
        st.pyplot(fig)

        if st.button("Show AI Insights", key=f"{col_x}-{col_y}-insights"):
            mean_x = iready_data[col_x].mean()
            max_x = iready_data[col_x].max()
            insights = fetch_ai_insights(col_x, mean_value=mean_x, max_value=max_x)
            st.info(f"Explanation: {insights['explanation']}")
            st.write("Suggestions:")
            for suggestion in insights['suggestions']:
                if suggestion:  # only print if not empty
                    st.write(f"- {suggestion}")

    elif iready_vis_type == "Bar Chart":
        avg_values = iready_data.mean()
        st.bar_chart(avg_values)

        if st.button("Show AI Insights", key="iready-bar-chart-insights"):
            insights = fetch_ai_insights("i-Ready Metrics (Bar Chart)")
            st.info(f"Explanation: {insights['explanation']}")
            st.write("Suggestions:")
            for suggestion in insights['suggestions']:
                st.write(f"- {suggestion}")

    elif iready_vis_type == "Box Plot":
        fig, ax = plt.subplots()
        sns.boxplot(data=iready_data, ax=ax)
        ax.set_title("Distribution of i-Ready Metrics")
        st.pyplot(fig)

        if st.button("Show AI Insights", key="iready-box-plot-insights"):
            insights = fetch_ai_insights("i-Ready Metrics (Box Plot)")
            st.info(f"Explanation: {insights['explanation']}")
            st.write("Suggestions:")
            for suggestion in insights['suggestions']:
                st.write(f"- {suggestion}")

    elif iready_vis_type == "Correlation Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(iready_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap of i-Ready Metrics")
        st.pyplot(fig)

        if st.button("Show AI Insights", key="iready-heatmap-insights"):
            insights = fetch_ai_insights("i-Ready Metrics (Correlation Heatmap)")
            st.info(f"Explanation: {insights['explanation']}")
            st.write("Suggestions:")
            for suggestion in insights['suggestions']:
                st.write(f"- {suggestion}")

    # (G) Domain Analysis
    st.subheader("Domain Analysis")
    domains = {
        "Algebra and Algebraic Thinking": [
            'i-Ready Algebra and Algebraic Thinking: Lessons Passed',
            'i-Ready Algebra and Algebraic Thinking: Lessons Completed',
            'i-Ready Algebra and Algebraic Thinking: % Lessons Passed'
        ],
        "Number and Operations": [
            'i-Ready Number and Operations: Lessons Passed',
            'i-Ready Number and Operations: Lessons Completed',
            'i-Ready Number and Operations: % Lessons Passed'
        ],
        "Measurement and Data": [
            'i-Ready Measurement and Data: Lessons Passed',
            'i-Ready Measurement and Data: Lessons Completed',
            'i-Ready Measurement and Data: % Lessons Passed'
        ],
        "Geometry": [
            'i-Ready Geometry: Lessons Passed',
            'i-Ready Geometry: Lessons Completed',
            'i-Ready Geometry: % Lessons Passed'
        ],
        "Pro Whole Numbers and Operations": [
            "i-Ready Pro Whole Numbers and Operations: Skills Successful",
            "i-Ready Pro Whole Numbers and Operations: Skills Completed",
            "i-Ready Pro Whole Numbers and Operations: % Skills Successful"
        ],
    }

    selected_domain = st.selectbox("Select Domain for Analysis", options=list(domains.keys()))
    domain_columns = domains[selected_domain]

    filtered_comparison_data = filter_non_attempted_students(filtered_data, domain_columns)

    st.sidebar.header(f"{selected_domain} Analysis Options")
    pass_threshold = st.sidebar.slider("Select % Passed Threshold", min_value=0, max_value=100, value=50)
    failing_students = filtered_comparison_data[filtered_comparison_data[domain_columns[2]] < pass_threshold]

    st.write(f"### Students with {selected_domain} % Passed Below {pass_threshold}%")
    st.dataframe(failing_students[["Last Name", "First Name", "Student ID"] + domain_columns])

    # Class vs Other Selected Classes
    st.write("### Class Progress vs Other Selected Classes")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # 2) Identify the three columns: Passed, Completed, and Percentage
    passed_col = domain_columns[0]
    completed_col = domain_columns[1]
    pct_col = domain_columns[2]

    class_avg = filtered_comparison_data.groupby("Class Teacher(s)")[domain_columns].mean().reset_index()
    class_avg = class_avg.sort_values(passed_col, ascending=False)  # Now passed_col is known

    # 3) Create a subplot figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
        
    # 4) Add 'Lessons Passed' bar trace (primary y-axis)
    fig.add_trace(
        go.Bar(
            x=class_avg["Class Teacher(s)"],
            y=class_avg[passed_col],
            name=passed_col
        ),
        secondary_y=False
    )

    # 5) Add 'Lessons Completed' bar trace (primary y-axis)
    fig.add_trace(
        go.Bar(
            x=class_avg["Class Teacher(s)"],
            y=class_avg[completed_col],
            name=completed_col
        ),
        secondary_y=False
    )

    # 6) Add '% Lessons Passed' bar trace (secondary y-axis)
    fig.add_trace(
        go.Bar(
            x=class_avg["Class Teacher(s)"],
            y=class_avg[pct_col],
            name=pct_col
        ),
        secondary_y=True
    )

    # 7) Layout & axis formatting
    fig.update_layout(
        title_text=f"{selected_domain} - Class Progress vs Selected Classes",
        barmode="group"
    )
    fig.update_yaxes(title_text="Count Metrics", secondary_y=False)
    fig.update_yaxes(title_text="Percentage (%)", range=[0, 100], secondary_y=True)

    st.plotly_chart(fig)

    # Keep your AI Insights button if you want
    if st.button("Show AI Insights", key="domain-bar-chart-insights"):
        insights = fetch_ai_insights(f"{selected_domain} Domain Bar Chart")
        st.info(f"Explanation: {insights['explanation']}")
        st.write("Suggestions:")
        for suggestion in insights['suggestions']:
            st.write(f"- {suggestion}")



    # Student Progress Distribution
    st.write("### Student Progress Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_comparison_data[domain_columns[2]], bins=10, kde=True, ax=ax)
    ax.set_title(f"Distribution of % Lessons Passed in {selected_domain}")
    st.pyplot(fig)

    if st.button("Show AI Insights", key="domain-distribution-insights"):
        insights = fetch_ai_insights(f"{selected_domain} Progress Distribution")
        st.info(f"Explanation: {insights['explanation']}")
        st.write("Suggestions:")
        for suggestion in insights['suggestions']:
            st.write(f"- {suggestion}")

    # Correlation within Domain
    st.write("### Correlation Analysis for Domain Metrics")
    st.markdown("""
    **Heatmap Explanation:**
    - A correlation value near 1 means the two metrics increase together.
    - A correlation value near -1 means they move in opposite directions.
    - 0 means little to no linear relationship.
    """)

    if len(filtered_comparison_data) < 3:
        st.warning("Not enough data to display correlation meaningfully.")
    else:
        fig, ax = plt.subplots()
        sns.heatmap(filtered_comparison_data[domain_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if st.button("Show AI Insights", key="domain-correlation-heatmap-insights"):
        insights = fetch_ai_insights(f"{selected_domain} Correlation Heatmap")
        st.info(f"Explanation: {insights['explanation']}")
        st.write("Suggestions:")
        for suggestion in insights['suggestions']:
            st.write(f"- {suggestion}")

else:
    # If no file was uploaded:
    st.write("Please upload a CSV file to start analyzing data.")

# ==========================================
#              NEW FEATURES (optional)
# ==========================================
