"""
Created on Wed Nov 27 10:28:29 2024
Author: Alejandro Sanchez Pellon
DATASET LINK: https://www.kaggle.com/datasets/yusufdelikkaya/datascience-salaries-2024
"""

"""
DATASET COLUMNS:

* EXPERIENCE LEVEL: The employee's experience level (e.g., Junior, Mid-level, Senior, Expert).

* EMPLOYMENT TYPE: The type of employment (e.g., Full-Time, Part-Time, Contract).

* JOB TITLE: The title or role of the employee in the data science field.

* SALARY IN USD: The employee's salary converted to USD for standardization.

* REMOTE RATIO: The percentage of remote work allowed for the position (e.g., 0, 50, 100).

* COMPANY SIZE: The size of the company based on employee count (e.g., Small, Medium, Large).

* COMPANY LOCATION: The location of the company where the employee works.
"""

import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.stats import anderson, binomtest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

class EDA:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.label_encoders = {}  # Store label encoders for potential inverse transformation

##---------------------------------------------------------------DATA CLEANING AND PRE-PROCESSING------------------------------------------------------------------------------------
    def load_data(self):
        """
        Loads data from a CSV file into a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            self.encode_categorical_data()
            print("Data loaded successfully!")
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")

    def encode_categorical_data(self):
        columns_to_encode = ['experience_level', 'employment_type', 'job_title',
                             'employee_residence', 'company_location', 'company_size']
        for column in columns_to_encode:
            if column in self.data.columns:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column])
                self.label_encoders[column] = le
        print("Categorical data encoded.")

    def decode_categorical_data(self, column):
        if column in self.label_encoders:
            return self.label_encoders[column].inverse_transform(self.data[column])
        else:
            print(f"No encoder found for {column}.")
            return None

    def get_encoded_values(self, column):
        """
        Returns a dictionary mapping of original categorical values to their
        corresponding encoded values for a specified column, with encoded values as plain integers.

        Args:
        column (str): The name of the column for which to retrieve the encoding.

        Returns:
        dict: A dictionary where keys are the original categories and values are the encoded numbers as plain integers.
        """
        if column in self.label_encoders:
            # Retrieve the LabelEncoder for the specified column
            encoder = self.label_encoders[column]
            # Create a mapping of original category names to encoded values
            category_mapping = {category: int(code) for category, code in
                                zip(encoder.classes_, encoder.transform(encoder.classes_))}
            return category_mapping
        else:
            print(f"No encoder found for the column '{column}'.")
            return None

    def check_and_fill_missing_values(self, method='drop'):
        """
        Checks for and handles missing values according to the specified method.

        Parameters:
        method (str): Method to handle missing values. Options are 'mean', 'median', 'mode', 'drop', or a specific value.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        if self.data.isnull().sum().sum() == 0:
            print("No missing values found.")
            return

        if method == 'mean':
            self.data.fillna(self.data.mean(), inplace=True)
        elif method == 'median':
            self.data.fillna(self.data.median(), inplace=True)
        elif method == 'mode':
            # Mode can return multiple values per column, take the first mode
            self.data.fillna(self.data.mode().iloc[0], inplace=True)
        elif method == 'drop':
            self.data.dropna(inplace=True)
        else:
            try:
                self.data.fillna(method, inplace=True)
            except:
                print(f"Invalid fill value: {method}")

        print(f"Missing values handled using method: {method}")

    def handle_outliers(self, method='cap', factor=1.5):
        """
        Handles outliers in numerical columns based on the Interquartile Range (IQR) method.
        Skips encoded categorical columns.

        Parameters:
        method (str): Method to handle outliers, either 'cap' (cap values using IQR) or 'remove' (remove outliers).
        factor (float): The multiplier for IQR to define bounds. Default is 1.5.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        # Identify numeric columns by checking those not in the label_encoders dictionary
        numeric_cols = [col for col in self.data.select_dtypes(include=[np.number]).columns if
                        col not in self.label_encoders]

        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            if method == 'cap':
                self.data[col] = np.clip(self.data[col], lower_bound, upper_bound)
            elif method == 'remove':
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            else:
                print(f"Invalid method: {method}")

        print(f"Outliers handled using {method} method.")

    def remove_columns(self, columns_to_remove):
        """
        Removes specified columns from the DataFrame.

        Parameters:
        columns_to_remove (list): A list of strings where each string is a column name to be removed.
        """
        if self.data is not None:
            try:
                self.data.drop(columns=columns_to_remove, inplace=True)
                print(f"Columns {columns_to_remove} removed successfully.")
            except Exception as e:
                print(f"An error occurred while removing columns: {e}")
        else:
            print("Data not loaded. Please load the data first.")

    def display_data(self, rows=100):
        """
        Displays the data. If rows is None, attempts to display the entire dataset.
        If rows is specified, displays the number of rows specified.

        Parameters:
        rows (int): Optional. Number of rows to display. If None, displays the entire DataFrame.
        """
        if self.data is not None:
            if rows:
                print(self.data.head(rows))
            else:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    print(self.data)
        else:
            print("Data not loaded. Please load the data first.")

##---------------------------------------------------------------DATA CLEANING AND PRE-PROCESSING------------------------------------------------------------------------------------

##---------------------------------------------------------------EXPLORATORY DATA ANALYSIS------------------------------------------------------------------------------------

    def summary_statistics(self):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        # Temporarily adjust the pandas display settings within this method
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format',
                               '{:.2f}'.format):
            # Print statistics for numeric columns
            print("Descriptive Statistics for Numeric Data:")
            print(self.data.describe())

        # Handle categorical data: print original category counts
        print("\nDescriptive Statistics for Categorical Data (decoded):")
        for col, encoder in self.label_encoders.items():
            if col in self.data.columns:
                decoded_col = encoder.inverse_transform(self.data[col])
                print(f"\nStatistics for {col}:")
                print(pd.Series(decoded_col).value_counts())
            else:
                print(f"\nColumn {col} not found in data.")

    def correlation_analysis(self):
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        # Calculate and display correlation matrix for numerical data only
        numeric_data = self.data.select_dtypes(include=[np.number])
        plt.figure(figsize=(17, 14))
        sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Matrix for Numeric Data")
        plt.show()

    def distribution_analysis(self, top_n=20):
        """
        Visualizes the distribution of all columns in the dataset. Special handling
        for high cardinality categorical data is applied selectively, except for 'employment_type',
        'experience_level', and 'company_size' which are shown in full due to their limited number of categories.
        'company_size' will be displayed using a pie chart with correctly scaled percentages.

        Args:
        top_n (int): The number of top categories to display for high cardinality categorical variables.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        temp_df = self.data.copy()  # Make a temporary copy of the data for plotting

        chart_height = 500
        chart_width = 800

        for col in temp_df.columns:
            # Check if the column is numeric and not encoded
            if temp_df[col].dtype in ['int64', 'float64'] and col not in self.label_encoders:
                if col == 'remote_ratio':
                    # Ensure remote_ratio is numeric
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                    # Drop NaN values for remote_ratio
                    plot_data = temp_df[col].dropna()
                    data_slice = plot_data.value_counts().sort_index().reset_index()
                    data_slice.columns = [col, 'counts']

                    fig = px.bar(
                        data_slice,
                        x=col,  # Always set x to the column name
                        y='counts',
                        title=f'Distribution for {col}',
                        text='counts'
                    )
                else:
                    fig = px.histogram(
                        temp_df,
                        x=col,
                        title=f'Numeric Distribution for {col}',
                        marginal='rug',
                        hover_data=temp_df.columns
                    )

                # Set x-axis title conditionally
                x_title = "Remote Ratio (%)" if col == 'remote_ratio' else col

                fig.update_layout(
                    xaxis_title=x_title,
                    yaxis_title='Count',
                    height=chart_height,
                    width=chart_width
                )
                fig.show()

            else:
                # Decode categorical data if encoded
                decoded_data = self.label_encoders[col].inverse_transform(
                    temp_df[col]
                ) if col in self.label_encoders else temp_df[col]
                data_slice = pd.Series(decoded_data).value_counts().reset_index()
                data_slice.columns = [col, 'counts']  # Ensure columns are named correctly

                if col in ['employment_type', 'experience_level']:
                    fig = px.bar(
                        data_slice,
                        x='counts',
                        y=col,
                        title=f'Category Distribution for {col}',
                        orientation='h'
                    )
                    fig.update_layout(
                        xaxis_title='Counts',
                        yaxis_title=col,
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis={'tickangle': -45},
                        bargap=0.2,
                        height=chart_height,
                        width=chart_width
                    )
                    fig.show()

                elif col == 'company_size':
                    fig = px.pie(
                        data_slice,
                        names=col,
                        values='counts',
                        title=f'Category Distribution for {col}'
                    )
                    fig.update_traces(textposition='outside', textinfo='percent+label')
                    fig.update_layout(height=chart_height, width=chart_width)
                    fig.show()

                else:
                    if data_slice.shape[0] > top_n:
                        top_categories = data_slice.head(top_n)
                        others = pd.DataFrame(data={col: ['Others'], 'counts': [data_slice['counts'][top_n:].sum()]})
                        top_categories = pd.concat([top_categories, others], ignore_index=True)
                    else:
                        top_categories = data_slice

                    fig = px.bar(
                        top_categories,
                        x='counts',
                        y=col,
                        title=f'Top {top_n} Categories in {col}',
                        orientation='h'
                    )
                    fig.update_layout(
                        xaxis_title='Counts',
                        yaxis_title=col,
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis={'tickangle': -45},
                        bargap=0.2,
                        height=chart_height,
                        width=chart_width
                    )
                    fig.show()
                    fig.show()

    def plot_salary_by_company_size(self):
        if self.data is not None:
            temp_df = self.data.copy()
            if 'company_size' in self.label_encoders:
                temp_df['company_size'] = self.label_encoders['company_size'].inverse_transform(temp_df['company_size'])
            fig = px.box(temp_df, x='company_size', y='salary_in_usd', title='Salary Distribution by Company Size')
            fig.update_layout(xaxis_title='Company Size', yaxis_title='Salary in USD')
            fig.show()
        else:
            print("Data not loaded. Please load the data first.")

    def plot_salary_by_top_employee_residences(self):
        if self.data is not None:
            temp_df = self.data.copy()
            if 'employee_residence' in self.label_encoders:
                temp_df['employee_residence'] = self.label_encoders['employee_residence'].inverse_transform(
                    temp_df['employee_residence'])
            top_residences = temp_df['employee_residence'].value_counts().nlargest(5).index
            filtered_data = temp_df[temp_df['employee_residence'].isin(top_residences)]
            fig = px.box(filtered_data, x='employee_residence', y='salary_in_usd',
                         title='Salary Distribution for Top Employee Residences')
            fig.update_layout(xaxis_title='Employee Residence', yaxis_title='Salary in USD')
            fig.show()
        else:
            print("Data not loaded. Please load the data first.")

    def plot_salary_by_top_job_titles(self, top_n = 20):
        if self.data is not None:
            temp_df = self.data.copy()
            if 'job_title' in self.label_encoders:
                temp_df['job_title'] = self.label_encoders['job_title'].inverse_transform(temp_df['job_title'])
            top_job_titles = temp_df['job_title'].value_counts().nlargest(top_n).index
            filtered_data = temp_df[temp_df['job_title'].isin(top_job_titles)]
            fig = px.box(filtered_data, x='job_title', y='salary_in_usd',
                         title=f'Salary Distribution for Top {top_n} Job Titles',
                         labels={'job_title': 'Job Title', 'salary_in_usd': 'Salary in USD'})
            fig.update_traces(quartilemethod="inclusive")
            fig.show()
        else:
            print("Data not loaded. Please load the data first.")

    def plot_salary_by_experience_level(self):
        if self.data is not None:
            temp_df = self.data.copy()
            if 'experience_level' in self.label_encoders:
                temp_df['experience_level'] = self.label_encoders['experience_level'].inverse_transform(
                    temp_df['experience_level'])
            if 'experience_level' in temp_df.columns:
                fig = px.box(temp_df, x='experience_level', y='salary_in_usd',
                             title='Salary Distribution by Experience Level',
                             labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Salary in USD'},
                             category_orders={"experience_level": ["Entry", "Junior", "Mid", "Senior", "Expert"]})
                fig.update_traces(quartilemethod="inclusive")
                fig.show()
            else:
                print("'experience_level' column not found in the dataset.")
        else:
            print("Data not loaded. Please load the data first.")

##---------------------------------------------------------------EXPLORATORY DATA ANALYSIS------------------------------------------------------------------------------------
##---------------------------------------------------------------EXPLORATORY DATA ANALYSIS------------------------------------------------------------------------------------

##---------------------------------------------------------------DATA ASSUMPTIONS VALIDATION FOR STATISTICAL TESTS------------------------------------------------------------------------------------
##---------------------------------------------------------------DATA ASSUMPTIONS VALIDATION FOR STATISTICAL TESTS------------------------------------------------------------------------------------
    ##---------------------------------------------------------------DATA LOADING AND PRE-PROCESSING------------------------------------------------------------------------------------

    ##---------------------------------------------------------------STATISTICAL TESTS------------------------------------------------------------------------------------
    def check_normality_anderson(self, data, column):
        """
        Checks if the specified column's data is normally distributed using the Anderson-Darling test.

        Args:
            data (pd.Series): The data to test.
            column (str): The name of the column being tested.

        Returns:
            bool: True if data is normally distributed at 5% significance level, False otherwise.
        """
        result = anderson(data)
        print(f"Anderson-Darling Test for {column}: Statistic={result.statistic:.4f}")
        significance_levels = result.significance_level
        critical_values = result.critical_values

        # Typically, check at 5% significance level
        for sl, cv in zip(significance_levels, critical_values):
            if sl == 5:
                if result.statistic < cv:
                    print(f"At {sl}% significance level, data appears normal (fail to reject H0).")
                    return True
                else:
                    print(f"At {sl}% significance level, data does not appear normal (reject H0).")
                    return False
        # If 5% not found, default to False
        print("5% significance level not found in Anderson-Darling test.")
        return False

    def plot_qq(self, data, column):
        """
        Plots a Q-Q plot to visually assess normality.

        Args:
            data (pd.Series): The data to plot.
            column (str): The name of the column being plotted.

        Returns:
            None: Displays the Q-Q plot.
        """
        plt.figure()
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {column}')
        plt.show()

    def check_normality_anderson(self, data, column):
        """
        Checks if the specified column's data is normally distributed using the Anderson-Darling test.

        Args:
            data (pd.Series): The data to test.
            column (str): The name of the column being tested.

        Returns:
            bool: True if data is normally distributed at 5% significance level, False otherwise.
        """
        result = anderson(data)
        print(f"Anderson-Darling Test for {column}: Statistic={result.statistic:.4f}")
        significance_levels = result.significance_level
        critical_values = result.critical_values

        # Typically, check at 5% significance level
        if 5 in significance_levels:
            index = list(significance_levels).index(5)
            cv = critical_values[index]
            if result.statistic < cv:
                print(f"At 5% significance level, data appears normal (fail to reject H0).")
                return True
            else:
                print(f"At 5% significance level, data does not appear normal (reject H0).")
                return False
        else:
            # If 5% not present, default to False
            print("5% significance level not found in Anderson-Darling test.")
            return False

    def plot_qq(self, data, column):
        """
        Plots a Q-Q plot to visually assess normality.

        Args:
            data (pd.Series): The data to plot.
            column (str): The name of the column being plotted.

        Returns:
            None: Displays the Q-Q plot.
        """
        plt.figure(figsize=(6,6))
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {column}')
        plt.show()

    def check_homogeneity(self, groups, column):
        """
        Checks if the groups have equal variances using Levene's Test.

        Args:
            groups (list of pd.Series): The data groups to test.
            column (str): The name of the column being tested.

        Returns:
            bool: True if variances are equal, False otherwise.
        """
        stat, p = stats.levene(*groups)
        print(f"Levene's Test for {column}: Statistics={stat:.4f}, p-value={p:.4f}")
        if p > 0.05:
            print(f"Variances are equal across groups for {column} (fail to reject H0).")
            return True
        else:
            print(f"Variances are not equal across groups for {column} (reject H0).")
            return False

    def calculate_effect_size_mannwhitney(self, u_stat, n1, n2):
        """
        Calculates the effect size for the Mann-Whitney U Test using the rank-biserial correlation.

        Args:
            u_stat (float): The U statistic from the Mann-Whitney U Test.
            n1 (int): Sample size of group 1.
            n2 (int): Sample size of group 2.

        Returns:
            float: Effect size (rank-biserial correlation).
        """
        r = 1 - (2 * u_stat) / (n1 * n2)
        return r

    def plot_violin_salary_by_job_title(self, *job_titles):
        """
        Plots violin plots for salaries of specified job titles.

        Args:
            *job_titles (str): Variable number of job titles to plot. For example:
                                'Data Scientist', 'Applied Scientist', 'Machine Learning Engineer'

        Returns:
            None: Displays the violin plots.
        """
        if self.data is not None:
            temp_df = self.data.copy()

            # Decode 'job_title' if it was label encoded
            if 'job_title' in self.label_encoders:
                temp_df['job_title'] = self.decode_categorical_data('job_title')

            # Filter data for the specified job titles
            filtered_data = temp_df[temp_df['job_title'].isin(job_titles)]

            if filtered_data.empty:
                print("No data available for the specified job titles. Please check the job titles and try again.")
                return

            # Set the aesthetic style of the plots
            sns.set(style="whitegrid")

            # Increase the figure size for better readability, especially with multiple job titles
            plt.figure(figsize=(10, 8))

            # Create the violin plot
            sns.violinplot(x='job_title', y='salary_in_usd', data=filtered_data, inner='quartile', palette="Set2")

            # Enhance the plot with titles and labels
            plt.title(f'Salary Distribution for {", ".join(job_titles)}', fontsize=16)
            plt.xlabel('Job Title', fontsize=14)
            plt.ylabel('Salary in USD', fontsize=14)

            # Rotate x-axis labels if there are many job titles to prevent overlap
            if len(job_titles) > 3:
                plt.xticks(rotation=45, ha='right')

            # Display the plot
            plt.tight_layout()
            plt.show()
        else:
            print("Data not loaded. Please load the data first.")

    def plot_violin_salary_by_experience_level(self):
        """
        Plots a violin plot to visualize the distribution of salaries across different experience levels,
        organized in the order: Entry, Junior, Senior, Expert.

        Returns:
            None: Displays the violin plot.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        temp_df = self.data.copy()

        # Decode 'experience_level' if it was label encoded
        if 'experience_level' in self.label_encoders:
            temp_df['experience_level'] = self.decode_categorical_data('experience_level')
        else:
            temp_df['experience_level'] = temp_df['experience_level']

        # Ensure 'salary_in_usd' is numeric
        temp_df['salary_in_usd'] = pd.to_numeric(temp_df['salary_in_usd'], errors='coerce')

        # Drop rows with missing values in 'experience_level' or 'salary_in_usd'
        temp_df = temp_df.dropna(subset=['experience_level', 'salary_in_usd'])

        # Define the mapping from codes to descriptive labels
        level_mapping = {'EN': 'Entry', 'MI': 'Junior', 'SE': 'Senior', 'EX': 'Expert'}
        temp_df['experience_level_label'] = temp_df['experience_level'].map(level_mapping)

        # Check for any unmapped values and handle them
        unmapped = temp_df['experience_level_label'].isnull().sum()
        if unmapped > 0:
            print(
                f"Warning: {unmapped} experience level entries could not be mapped and will be excluded from the plot.")
            temp_df = temp_df.dropna(subset=['experience_level_label'])

        # Define the desired order of experience levels
        desired_order = ['Entry', 'Junior', 'Senior', 'Expert']

        # Set the aesthetic style of the plots
        sns.set(style="whitegrid")

        # Increase the figure size for better readability
        plt.figure(figsize=(12, 8))

        # Create the violin plot with the specified order
        sns.violinplot(
            x='experience_level_label',
            y='salary_in_usd',
            data=temp_df,
            inner='quartile',  # Show quartiles inside the violin
            palette="Set2",  # Choose a color palette
            order=desired_order  # Set the correct order of categories
        )



        # Enhance the plot with titles and labels
        plt.title('Salary Distribution Across Experience Levels', fontsize=16)
        plt.xlabel('Experience Level', fontsize=14)
        plt.ylabel('Salary in USD', fontsize=14)

        # Rotate x-axis labels if they are long
        plt.xticks(rotation=45, ha='right')

        # Display the plot
        plt.tight_layout()
        plt.show()


    def compare_job_salaries_2tTest(self, job_title1, job_title2):
        """
        Compares the salary distributions between two job titles using appropriate statistical tests.

        Args:
            job_title1 (str): The first job title for comparison.
            job_title2 (str): The second job title for comparison.

        Null Hypothesis:
            The distributions of salaries for the two job titles are equal.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        # Decode job titles if they are encoded
        if 'job_title' in self.label_encoders:
            decoded_job_titles = self.decode_categorical_data('job_title')
            self.data['decoded_job_title'] = decoded_job_titles
        else:
            self.data['decoded_job_title'] = self.data['job_title']

        # Filter data for the two job titles
        data1 = self.data[self.data['decoded_job_title'] == job_title1]['salary_in_usd'].dropna()
        data2 = self.data[self.data['decoded_job_title'] == job_title2]['salary_in_usd'].dropna()

        if data1.empty or data2.empty:
            print(f"No sufficient data for jobs: '{job_title1}' or '{job_title2}'. Check the job titles or data availability.")
            return

        print(f"\nComparing Salaries between '{job_title1}' and '{job_title2}':")

        # Check Normality using Anderson-Darling Test
        print("\nChecking Normality:")
        normal1 = self.check_normality_anderson(data1, job_title1)
        normal2 = self.check_normality_anderson(data2, job_title2)

        # Check Homogeneity of Variances
        print("\nChecking Homogeneity of Variances:")
        homogeneity = self.check_homogeneity([data1, data2], 'salary_in_usd')

        # Decide on test based on assumptions
        if normal1 and normal2:
            if homogeneity:
                print("\nAssumptions met. Performing independent two-sample t-test with equal variances.")
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)
            else:
                print("\nAssumptions met. Performing independent two-sample t-test with unequal variances.")
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

            print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

            alpha = 0.05
            print(f"Null Hypothesis: The mean salaries of '{job_title1}' and '{job_title2}' are equal.")

            if p_value < alpha:
                print("Reject the null hypothesis - Significant difference in mean salaries.")
            else:
                print("Fail to reject the null hypothesis - No significant difference in mean salaries.")

        else:
            print("\nAssumptions not met. Performing Mann-Whitney U Test instead.")
            # Perform Mann-Whitney U Test
            u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            print(f"Mann-Whitney U Test: U-statistic={u_stat}, p-value={p_value:.4f}")

            # Calculate Effect Size
            effect_size = self.calculate_effect_size_mannwhitney(u_stat, len(data1), len(data2))
            print(f"Effect Size (Rank-Biserial Correlation): {effect_size:.4f}")

            alpha = 0.05
            print(f"Null Hypothesis: The distributions of salaries for '{job_title1}' and '{job_title2}' are equal.")

            if p_value < alpha:
                print("Reject the null hypothesis - Significant difference in salary distributions.")
            else:
                print("Fail to reject the null hypothesis - No significant difference in salary distributions.")

            # Plot Violin Plots for Visual Inspection
            self.plot_violin_salary_by_job_title(job_title1, job_title2)

    def salary_average_tTest(self, hypothesized_average, job_title):
        """
        Compares the median salary of a specific job title against a hypothesized average salary using appropriate statistical tests.

        Args:
            hypothesized_average (float): The hypothesized average salary to compare against.
            job_title (str): The job title to filter the salaries for comparison.

        Null Hypothesis:
            There is no significant difference between the median salary of the specified job title and the hypothesized average.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        if 'salary_in_usd' not in self.data.columns:
            print("Salary column does not exist in the data.")
            return

        # Decode job titles if they are encoded
        if 'job_title' in self.label_encoders:
            decoded_job_titles = self.decode_categorical_data('job_title')
            self.data['decoded_job_title'] = decoded_job_titles
        else:
            self.data['decoded_job_title'] = self.data['job_title']

        # Filter data for the specified job title
        salary_data = self.data[self.data['decoded_job_title'] == job_title]['salary_in_usd'].dropna()

        if salary_data.empty:
            print(f"No data available for job title: {job_title}")
            return

        print(f"\nComparing Median Salary for '{job_title}' against Hypothesized Average of ${hypothesized_average}:")

        # Check Normality using Anderson-Darling Test
        print("\nChecking Normality:")
        normal = self.check_normality_anderson(salary_data, 'salary_in_usd')

        if normal:
            print("\nAssumption met. Performing one-sample t-test.")
            # Perform one-sample t-test
            t_stat, p_value = stats.ttest_1samp(salary_data, hypothesized_average)
            print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

            alpha = 0.05
            print(f"Null Hypothesis: The mean salary for '{job_title}' is equal to ${hypothesized_average}.")

            if p_value < alpha:
                print("Reject the null hypothesis - There is a significant difference between the mean salary and the hypothesized average.")
            else:
                print("Fail to reject the null hypothesis - There is no significant difference between the mean salary and the hypothesized average.")
        else:
            print("\nAssumption not met. Performing Sign Test instead.")
            # Perform Sign Test using binomtest
            median = hypothesized_average
            greater = (salary_data > median).sum()
            less = (salary_data < median).sum()
            n = greater + less

            print(f"Number of salaries above ${median}: {greater}")
            print(f"Number of salaries below ${median}: {less}")

            if n == 0:
                print("All salaries are equal to the hypothesized average.")
                return

            # Perform binomial test (Sign Test)
            result = binomtest(min(greater, less), n, p=0.5, alternative='two-sided')
            p_value = result.pvalue
            print(f"Sign Test: p-value={p_value:.4f}")

            alpha = 0.05
            print(f"Null Hypothesis: The median salary for '{job_title}' is equal to ${hypothesized_average}.")

            if p_value < alpha:
                print("Reject the null hypothesis - The median salary is significantly different.")
                # Determine direction of difference
                if less > greater:
                    print(f"The median salary is lower than the hypothesized average of ${hypothesized_average}.")
                elif greater > less:
                    print(f"The median salary is higher than the hypothesized average of ${hypothesized_average}.")
                else:
                    print(f"The median salary is equal to the hypothesized average of ${hypothesized_average}.")
            else:
                print("Fail to reject the null hypothesis - The median salary is not significantly different.")

    def test_remote_ratio_impact_on_salary(self):
        """
        Performs a linear regression analysis to test the impact of remote work ratio on salary.
        - Null Hypothesis (H₀): Remote work ratio has no impact on salary.
        - Alternative Hypothesis (H₁): Remote work ratio has a significant impact on salary.

        Additionally, quantifies the impact of remote work ratio on salary.

        Returns:
            None: Prints the regression results and displays relevant plots.
        """
        import statsmodels.api as sm
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        # Ensure 'remote_ratio' and 'salary_in_usd' are present
        if 'remote_ratio' not in self.data.columns or 'salary_in_usd' not in self.data.columns:
            print("Required columns 'remote_ratio' and/or 'salary_in_usd' not found in the data.")
            return

        # Drop rows with missing values in 'remote_ratio' or 'salary_in_usd'
        regression_data = self.data[['remote_ratio', 'salary_in_usd']].dropna()

        # Define independent variable (with constant for intercept)
        X = regression_data['remote_ratio']
        X = sm.add_constant(X)  # Adds a constant term to the predictor

        # Define dependent variable
        y = regression_data['salary_in_usd']

        # Fit the linear regression model
        model = sm.OLS(y, X).fit()

        # Print the regression results
        print("\n=== Linear Regression Analysis: Remote Work Ratio vs. Salary ===")
        print(model.summary())

        # Plotting the regression line with scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='remote_ratio', y='salary_in_usd', data=regression_data, label='Data Points')
        sns.lineplot(x=regression_data['remote_ratio'], y=model.predict(X), color='red', label='Regression Line')
        plt.title('Salary vs. Remote Work Ratio')
        plt.xlabel('Remote Work Ratio (%)')
        plt.ylabel('Salary in USD')
        plt.legend()
        plt.show()

        # Residual Plot to assess assumptions
        residuals = model.resid
        fitted = model.fittedvalues

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=fitted, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.show()

    def summary_salary_statistics(self, top_n_job_titles=25, export=False):
        """
        Generates and displays summary statistics for 'salary_in_usd' across various categorical variables:
        - Experience Level
        - Company Size
        - Top N Job Titles

        The summary includes count, mean, median, standard deviation, min, max, and quartiles.

        Ensures that all categorical variables are decoded to their descriptive labels and that
        summary statistics tables are fully displayed without truncation.

        Parameters:
            top_n_job_titles (int): Number of top job titles to include based on frequency. Default is 25.
            export (bool): If True, exports the summary tables to CSV files. Default is False.

        Returns:
            summaries (dict): Dictionary containing summary DataFrames for each variable.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        # Define the categorical variables to analyze
        categorical_vars = ['experience_level', 'company_size', 'job_title']

        # Initialize a dictionary to store summary statistics
        summaries = {}

        # Adjust pandas display options to ensure full tables are printed
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', None)

        for var in categorical_vars:
            if var not in self.data.columns:
                print(f"Column '{var}' not found in the data. Skipping.")
                continue

            print(f"\n=== Summary Statistics for 'salary_in_usd' across '{var}' ===")

            # Decode the categorical variable if it's encoded
            if var in self.label_encoders:
                decoded_col = self.decode_categorical_data(var)
                temp_df = self.data.copy()
                temp_df[f'{var}_label'] = decoded_col
                group_var = f'{var}_label'
            else:
                # If not encoded, use the original column
                temp_df = self.data.copy()
                group_var = var

            if var == 'job_title':
                # Identify top N job titles by frequency
                top_jobs = temp_df[group_var].value_counts().nlargest(top_n_job_titles).index
                subset = temp_df[temp_df[group_var].isin(top_jobs)]
                summary = subset.groupby(group_var)['salary_in_usd'].describe().sort_values('mean', ascending=False)
                summary = summary.transpose()  # Transpose to have statistics as rows
                print(f"\nTop {top_n_job_titles} Job Titles by Count:")
                print(summary)
                summaries[var] = summary

                if export:
                    summary.to_csv(f'salary_summary_{var}_top_{top_n_job_titles}.csv')
                    print(
                        f"Summary statistics for '{var}' exported to 'salary_summary_{var}_top_{top_n_job_titles}.csv'.")
            else:
                # For other categorical variables
                summary = temp_df.groupby(group_var)['salary_in_usd'].describe()
                summary = summary.transpose()  # Transpose to have statistics as rows
                print(summary)
                summaries[var] = summary

                if export:
                    summary.to_csv(f'salary_summary_{var}.csv')
                    print(f"Summary statistics for '{var}' exported to 'salary_summary_{var}.csv'.")

        # Reset pandas display options to default to avoid affecting other parts of the code
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_rows')

        return summaries
    ##---------------------------------------------------------------STATISTICAL TESTS------------------------------------------------------------------------------------

##---------------------------------------------------------------STATISTICAL TESTS------------------------------------------------------------------------------------
##---------------------------------------------------------------STATISTICAL TESTS------------------------------------------------------------------------------------



eda = EDA( filepath = '/Users/alejandro/Documents/School/Florida International University/8- 2024 Fall Semester'
                      '/Fundamentals of Data Science/Final Project/data sets/datascienceSalaries.csv')
eda.load_data()

## HANDLE OUTLIERS
eda.handle_outliers(method='cap')
# Remove 'salary' and 'salary_currency' columns, if both exist
columns_to_remove = ['salary', 'salary_currency']
eda.remove_columns(columns_to_remove)


#eda.distribution_analysis()
#eda.summary_statistics()
eda.summary_salary_statistics()
eda.correlation_analysis()
eda.plot_salary_by_company_size()
eda.plot_salary_by_top_employee_residences()
eda.plot_salary_by_top_job_titles()
eda.plot_salary_by_experience_level()
# Test the impact of remote work ratio on salary
eda.test_remote_ratio_impact_on_salary()
# Plot the violin plot for salary distribution across experience levels
eda.plot_violin_salary_by_experience_level()
# 6e. Violin Plot for 'Data Science' vs 'Applied Scientist'
eda.plot_violin_salary_by_job_title('Data Science', 'Applied Scientist', 'Machine Learning Engineer')


# 6b. Compare salaries between 'Data Science' and 'Applied Scientist' roles
# Ensure that these job titles exist in your dataset; adjust if necessary
print("\n=== Test: Compare Salaries between 'Data Science' and 'Applied Scientist' ===")
eda.compare_job_salaries_2tTest('Data Science', 'Applied Scientist')

# 6c. Compare the average salary against a hypothesized average of $149,000
print("\n=== Test: Compare Average Salary against Hypothesized Average of $121,593 ===")
eda.salary_average_tTest(121593, 'Data Scientist')