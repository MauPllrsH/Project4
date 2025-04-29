import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import sys
from io import StringIO, BytesIO
from services.openai_service import call_openai_api
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend


def generate_visualization_code(df, analysis_suggestions, user_instructions=""):
    """Generate data visualization code using OpenAI"""
    system_prompt = '''
    You are a Python data visualization and analysis assistant. When given information about a dataset and analysis requirements,
    respond ONLY with executable Python code that creates data visualizations and provides analytical insights.

    Important guidelines:
    1. ONLY return valid Python code that can be executed directly
    2. Do not use any markdown formatting or backticks
    3. The original DataFrame is already available as 'df'
    4. Your code must create a list of figures called 'figures = []'
    5. Your code must create a string variable called 'insights' with key analytical findings in markdown format
    6. Include detailed comments explaining your analysis steps
    7. Use pandas, numpy, matplotlib, and seaborn for visualization
    8. Make figures clear with proper titles, labels, and legends
    9. Save each figure to the figures list using:
       - fig, ax = plt.subplots()
       - # plotting code here
       - figures.append(fig)
       - plt.close(fig)  # Important to prevent display issues
    10. For insightful analysis:
       - Look for correlations and patterns between variables
       - Identify outliers and unusual distributions
       - Highlight key trends with statistical context
       - Summarize findings with clear, specific statements
    11. VERY IMPORTANT: ONLY reference columns that actually exist in the dataset
    12. Limit to 3-5 of the most insightful visualizations
    13. Make sure figures are large enough to be readable
    14. Use plt.figure(figsize=(10, 6)) for good sized plots
    15. Use seaborn themes for professional looking visualizations: sns.set_theme()
    '''

    # Create a summary of the dataset
    column_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = str(df[col].iloc[0]) if not df[col].empty and not pd.isna(df[col].iloc[0]) else "N/A"
        column_info.append(f"{col} (type: {dtype}, sample: {sample})")

    # Prepare the prompt
    prompt = f"""
    Dataset Information:
    - Rows: {len(df)}
    - Columns: {len(df.columns)}

    Column names (EXACT spelling and capitalization):
    {", ".join([f"'{col}'" for col in df.columns])}

    Column information:
    {', '.join(column_info)}

    Sample data (first 5 rows):
    {df.head(5).to_string()}

    Data types summary:
    {df.dtypes.value_counts().to_string()}

    Numeric columns:
    {", ".join([f"'{col}'" for col in df.select_dtypes(include=['number']).columns])}

    Categorical/object columns:
    {", ".join([f"'{col}'" for col in df.select_dtypes(include=['object', 'category']).columns])}

    Datetime columns:
    {", ".join([f"'{col}'" for col in df.select_dtypes(include=['datetime']).columns])}

    Analysis suggestions:
    {analysis_suggestions}

    User specific instructions:
    {user_instructions}

    Generate Python code to create visualizations and analysis for this dataset according to the suggestions and user instructions.
    Start by initializing figures = [] and create 3-5 insightful visualizations that reveal important patterns or trends.
    Also generate an 'insights' string variable with your key findings formatted in markdown.

    REMEMBER: 
    - Save each figure to the figures list
    - Close figures after appending them to prevent display issues
    - Only use pandas, numpy, matplotlib, and seaborn 
    - Be careful with column names, use EXACT spelling and capitalization
    - Make figures clear with proper titles, labels, and legends
    - Provide thoughtful analysis in the insights variable
    - Don't just make generic plots - analyze the data and create visualizations that tell a meaningful story
    """

    # Call OpenAI API
    response, error = call_openai_api(prompt, system_prompt)

    if error:
        return "# Error generating visualization code\nfigures = []\ninsights = \"Error: Could not generate analysis\""

    if response:
        # Ensure required components are present
        if "figures = []" not in response:
            response = "# Initialize figures list\nfigures = []\n\n" + response

        if "insights = " not in response:
            response += "\n\n# Create insights summary\ninsights = \"No insights were generated.\"\n"

        return response

    return "# Error generating visualization code\nfigures = []\ninsights = \"Error: Could not generate analysis\""


def execute_visualization_code(df, code):
    """Validate and execute data visualization code"""
    # Create a namespace for execution
    exec_globals = {
        'df': df.copy(),
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'figures': [],
        'insights': "No insights were generated."
    }

    # Capture print output
    old_stdout = sys.stdout
    mystdout = StringIO()
    sys.stdout = mystdout

    try:
        # Execute the visualization code
        exec(code, exec_globals)

        # Get the figures and insights
        figures = exec_globals.get('figures', [])
        insights = exec_globals.get('insights', "No insights were generated.")

        # Restore stdout
        sys.stdout = old_stdout
        output = mystdout.getvalue()

        return figures, insights, output

    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = old_stdout
        return None, None, f"Error executing visualization code: {str(e)}\n{traceback.format_exc()}"


def generate_pdf_report(df, figures, insights):
    """Generate a PDF report with data analysis and visualizations"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import markdown
        from io import BytesIO
        import tempfile

        # Create a BytesIO object to store the PDF
        buffer = BytesIO()

        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        # Create custom styles WITHOUT reusing existing style names
        styles.add(ParagraphStyle(name='ReportTitle',
                                  fontName='Helvetica-Bold',
                                  fontSize=16,
                                  spaceAfter=12))
        styles.add(ParagraphStyle(name='ReportHeading2',
                                  fontName='Helvetica-Bold',
                                  fontSize=14,
                                  spaceAfter=10))
        styles.add(ParagraphStyle(name='ReportNormal',
                                  fontName='Helvetica',
                                  fontSize=10,
                                  spaceAfter=8))

        # Build the PDF content
        content = []

        # Add title
        content.append(Paragraph("Data Analysis Report", styles['ReportTitle']))
        content.append(Spacer(1, 0.2 * inch))

        # Add dataset information
        content.append(Paragraph("Dataset Overview", styles['ReportHeading2']))
        content.append(Paragraph(f"Rows: {len(df)}", styles['ReportNormal']))
        content.append(Paragraph(f"Columns: {len(df.columns)}", styles['ReportNormal']))
        content.append(Spacer(1, 0.2 * inch))

        # Add insights
        content.append(Paragraph("Analysis Insights", styles['ReportHeading2']))

        # Convert markdown to HTML for the insights
        from bs4 import BeautifulSoup
        import re

        # Handle bullet points and formatting in markdown
        html_insights = markdown.markdown(insights)
        soup = BeautifulSoup(html_insights, 'html.parser')

        # Process the HTML to create paragraphs
        for tag in soup.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4']):
            style = 'ReportNormal'
            if tag.name.startswith('h'):
                style = 'ReportHeading2'

            text = tag.get_text()
            if tag.name == 'li':
                text = f"• {text}"

            content.append(Paragraph(text, styles[style]))

        content.append(Spacer(1, 0.2 * inch))

        # Add visualizations
        content.append(Paragraph("Visualizations", styles['ReportHeading2']))

        # Save figures to temporary image files and add to the PDF
        for i, fig in enumerate(figures):
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
                img = Image(tmp.name, width=6 * inch, height=4 * inch)
                content.append(img)
                content.append(Spacer(1, 0.1 * inch))

        # Build the PDF
        doc.build(content)

        # Get the PDF data
        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    except Exception as e:
        import traceback
        error_msg = f"Error generating PDF report: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)

        # Create a simple error PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("Error Generating PDF Report", styles['Title']))
        content.append(Paragraph(str(e), styles['Normal']))

        doc.build(content)

        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data