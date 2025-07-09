# DDE Data Dashboard V2

Because staring at raw CSV files is a cry for help. This is a Streamlit dashboard designed to analyze and visualize i-Ready student performance data, saving you from the bleak, number-filled abyss.

![Dashboard Screenshot](https://imgur.com/a/zH4FVKZ)

---

## About The Project

This dashboard takes a CSV file of student i-Ready data and turns it into a series of charts and tables. The goal is to make sense of performance trends without needing a pot of coffee and three hours of uninterrupted spreadsheet time.

It's built with Python and Streamlit. You can filter data, view overall metrics, and drill down into specific domains, grades, or even individual students who may or may not be doing their work.

---

## Features

* **Dynamic Filtering**: Filter by grade, subject, school, and teacher. Because context is everything.
* **Key Metrics Overview**: See the 10,000-foot view with bar charts, box plots, and a heatmap that reveals correlations you can point to during meetings.
* **Domain Analysis**: A dedicated section to see how students are faring in topics like "Algebra" and "Geometry." Includes a handy list of students who are... underperforming.
* **Student Deep Dive**: A digital report card for any student, comparing them to the class average. Use this power responsibly.
* **Performance vs. Time-on-Task**: A scatter plot that attempts to answer the age-old question, "Are they actually learning or just staring at the screen?"
* **Grade-Level Comparison**: A multi-faceted chart comparing key metrics across grades, so you can see who's really winning.

---

## Getting Started

Follow these steps meticulously. Or don't. It's your computer.

### Prerequisites

You'll need Python 3 installed. That's probably it.

### Installation

1.  **Clone the repository** (if you haven't already).
    ```sh
    git clone [https://github.com/Beerspitnight/DDE-DataDashboard-V2.git](https://github.com/Beerspitnight/DDE-DataDashboard-V2.git)
    cd DDE-DataDashboard-V2
    ```
2.  **Create and activate a virtual environment**. It's good practice.
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required packages**.
    ```sh
    pip install -r requirements.txt
    ```
4.  **Run the app**.
    ```sh
    streamlit run app.py
    ```
    A tab should open in your browser. If it doesn't, the terminal will give you the URL.

---

## Usage

Once the app is running, the process is painfully simple:

1.  Use the sidebar to **upload a CSV file** with student data.
2.  Watch the dashboard appear.
3.  Click on things. The charts will update.

---

## Data Requirements

The script is surprisingly resilient. It was designed to not have a meltdown if a column is missing.

* **Essential Columns**: It really does need `Student Grade`, `Class Teacher(s)`, and `School` to function on a basic level.
* **Optional Data**: All other charts and features will simply not appear if their required columns are missing from your file. The dashboard won't complain; it will just silently judge your data for being incomplete. For full functionality, include all the i-Ready and student name columns.

