import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io
import os
import uuid
import re

def clean_data(df):
    df = df.copy()
        # Data preprocessing 
    # 1. Clean column names: lowercase, replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 2. Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # 3. Extract new time-based features
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year

    # 4. Check for duplicates (optional step)
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")

    # 5. Remove duplicates if needed
    df.drop_duplicates(inplace=True)

        # 1. Rename 'sales_value_inr' to 'total_sales' for clarity (already lowercase)
    df.rename(columns={'sales_value_(inr)': 'total_sales'}, inplace=True)

    # 2. Create average unit price column
    df['average_unit_price'] = df['total_sales'] / df['units_sold']

    # 3. Add transaction count per region
    region_counts = df['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'transaction_count']

    # 4. Add transaction count per dealer
    dealer_counts = df['dealer_name'].value_counts().reset_index()
    dealer_counts.columns = ['dealer_name', 'transaction_count']

    # 5. Group sales by date (daily summary)
    daily_sales = df.groupby('date')['total_sales'].sum().reset_index()

    # 6. Group by month & year (monthly summary)
    monthly_sales = df.groupby(['year', 'month'])['total_sales'].sum().reset_index()

    # 7. Optional: Save enriched data to a new CSV
    df.to_csv("cleaned_vecv_sales_data.csv", index=False)
    return df

# Save cleaned CSV
def save_cleaned_csv(df):
    cleaned_file = f"cleaned_data_{uuid.uuid4().hex[:6]}.csv"
    df.to_csv(cleaned_file, index=False)
    return cleaned_file



def generate_detailed_sales_report(csv_path, save_as="vecv_detailed_report.txt"):
    global global_df
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['date'] = pd.to_datetime(df['date'])
    with open(save_as, "w", encoding="utf-8") as f:

        f.write("VECV SALES ANALYTICS REPORT\n")
        f.write("="*40 + "\n\n")

        # --- 1. Basic Overview ---
        f.write(" OVERALL SUMMARY\n")
        f.write(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write(f"Total Revenue: ₹{df['total_sales'].sum():,.2f}\n")
        f.write(f"Total Units Sold: {df['units_sold'].sum():,}\n")
        f.write(f"Average Unit Price: ₹{df['average_unit_price'].mean():,.2f}\n\n")

        # --- 2. Monthly Trend ---
        f.write(" MONTHLY SALES & REVENUE\n")
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('month')[['units_sold', 'total_sales']].sum()
        f.write(monthly.to_string())
        f.write("\n\n")

        # --- 3. REGION-WISE PERFORMANCE ---
        f.write(" REGION-WISE PERFORMANCE\n")
        region_perf = df.groupby('region')[['units_sold', 'total_sales', 'average_unit_price']].agg({
            'units_sold': 'sum',
            'total_sales': 'sum',
            'average_unit_price': 'mean'
        }).sort_values(by='total_sales', ascending=False)
        f.write(region_perf.to_string())
        f.write("\n\n")

        # --- 4. DEALER PERFORMANCE ---
        f.write(" TOP DEALERS BY REVENUE\n")
        top_dealers = df.groupby('dealer_name')['total_sales'].sum().sort_values(ascending=False).head(10)
        f.write(top_dealers.to_string())
        f.write("\n\n")

        # --- 5. SEGMENT-WISE ANALYSIS ---
        f.write(" CUSTOMER SEGMENT PERFORMANCE\n")
        segment_data = df.groupby('customer_segment')[['units_sold', 'total_sales']].sum()
        f.write(segment_data.to_string())
        f.write("\n\n")

        # --- 6. VEHICLE MODEL ANALYSIS ---
        f.write(" VEHICLE MODEL BREAKDOWN\n")
        model_perf = df.groupby('vehicle_model')[['units_sold', 'total_sales']].sum().sort_values(by='total_sales', ascending=False)
        f.write(model_perf.head(10).to_string())
        f.write("\n\n")

        # --- 7. CITY-WISE DEALER PERFORMANCE ---
        f.write(" CITY-WISE SALES\n")
        city_sales = df.groupby('dealer_name')['total_sales'].sum().sort_values(ascending=False).head(10)
        f.write(city_sales.to_string())
        f.write("\n\n")

        # --- 8. EXTREMES ---
        f.write(" BEST & WORST PERFORMERS\n")
        best_model = model_perf['total_sales'].idxmax()
        worst_model = model_perf['total_sales'].idxmin()
        f.write(f"Highest Selling Model: {best_model}\n")
        f.write(f"Lowest Selling Model: {worst_model}\n")
        best_region = region_perf['total_sales'].idxmax()
        worst_region = region_perf['total_sales'].idxmin()
        f.write(f"Top Region: {best_region}\n")
        f.write(f"Lowest Region: {worst_region}\n\n")

        # --- 9. DATA QUALITY CHECK ---
        f.write(" DATA SANITY CHECKS\n")
        f.write(f"Missing Values:\n{df.isnull().sum().to_string()}\n")
        f.write(f"\nDuplicate Records: {df.duplicated().sum()}\n")
        f.write(f"Unique Dealers: {df['dealer_name'].nunique()}\n")
        f.write(f"Unique Models: {df['vehicle_model'].nunique()}\n")
        f.write(f"Unique Regions: {df['region'].nunique()}\n")

        # --- 10. FORECAST FUTURE REVENUE & SALES ---
        f.write("\n FORECAST: Next 3 Months\n")

        # Prepare data for Prophet
        df_prophet = df.groupby('date')[['total_sales', 'units_sold']].sum().reset_index()

        # Forecast Revenue
        revenue_df = df_prophet[['date', 'total_sales']].rename(columns={'date': 'ds', 'total_sales': 'y'})
        revenue_model = Prophet()
        revenue_model.fit(revenue_df)
        future_revenue = revenue_model.make_future_dataframe(periods=90)
        forecast_revenue = revenue_model.predict(future_revenue)

        # Forecast Units Sold
        units_df = df_prophet[['date', 'units_sold']].rename(columns={'date': 'ds', 'units_sold': 'y'})
        units_model = Prophet()
        units_model.fit(units_df)
        future_units = units_model.make_future_dataframe(periods=90)
        forecast_units = units_model.predict(future_units)

        # Get last 3 forecasted months
        forecast_revenue['month'] = forecast_revenue['ds'].dt.to_period('M')
        forecast_units['month'] = forecast_units['ds'].dt.to_period('M')

        revenue_monthly = forecast_revenue.groupby('month')['yhat'].sum().tail(3)
        units_monthly = forecast_units.groupby('month')['yhat'].sum().tail(3)

        # Write to file
        f.write(" Predicted Revenue (₹):\n")
        f.write(revenue_monthly.apply(lambda x: f"₹{x:,.2f}").to_string())
        f.write("\n\n")
        f.write(" Predicted Units Sold:\n")
        f.write(units_monthly.apply(lambda x: f"{int(x):,} units").to_string())
        f.write("\n")
    

    return f"{save_as}"


chat_log = []


# --- Insight Functions ---
def top_regions(df, n=5):
    return df.groupby('region')['total_sales'].sum().sort_values(ascending=False).head(n)

def top_vehicles(df, n=5):
    return df.groupby('vehicle_model')['total_sales'].sum().sort_values(ascending=False).head(n)

def top_dealers(df, n=5):
    return df.groupby('dealer_name')['total_sales'].sum().sort_values(ascending=False).head(n)

def sales_between(df, start_date, end_date):
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    return df.loc[mask].groupby('region')['total_sales'].sum().sort_values(ascending=False)

def sales_by_segment(df):
    return df.groupby('customer_segment')['total_sales'].sum().sort_values(ascending=False)

def avg_unit_price(df, model_name):
    filtered = df[df['vehicle_model'].str.lower() == model_name.lower()]
    if filtered.empty:
        return f" No data for model: {model_name}"
    return f"₹{round(filtered['average_unit_price'].mean(), 2):,.2f}"

global_df= None

# Convert Matplotlib figure to image buffer
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# Generate Plots
def generate_plots(df):
    plots = []

    # Monthly Revenue Trend
    df['month'] = df['date'].dt.to_period("M").astype(str)
    monthly = df.groupby('month')['total_sales'].sum().reset_index()
    fig1, ax1 = plt.subplots()
    sns.lineplot(data=monthly, x='month', y='total_sales', marker='o', ax=ax1)
    ax1.set_title("Monthly Revenue Trend")
    ax1.tick_params(axis='x', rotation=45)
    plots.append(fig_to_image(fig1))

    # Region-wise Sales
    region_sales = df.groupby('region')['total_sales'].sum().reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(data=region_sales, x='region', y='total_sales', ax=ax2)
    ax2.set_title("Sales by Region")
    ax2.tick_params(axis='x', rotation=45)
    plots.append(fig_to_image(fig2))

    # Top 5 Vehicle Models
    top_models = df.groupby('vehicle_model')['total_sales'].sum().nlargest(5).reset_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(data=top_models, x='vehicle_model', y='total_sales', ax=ax3)
    ax3.set_title("Top 5 Vehicle Models")
    ax3.tick_params(axis='x', rotation=45)
    plots.append(fig_to_image(fig3))

    return plots

# Generate Visual PDF Report
def generate_visual_pdf_report(df):
    
    report_file = f"sales_report_{uuid.uuid4().hex[:6]}.pdf"
    c = canvas.Canvas(report_file, pagesize=A4)
    width, height = A4

    # Title and Summary
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, " VECV SALES REPORT")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Total Records: {len(df)}")
    c.drawString(50, height - 100, f"Total Revenue: ₹{df['total_sales'].sum():,.2f}")
    c.drawString(50, height - 120, f"Total Units Sold: {df['units_sold'].sum():,}")

    # Generate and Embed Plots
    plots = generate_plots(df)
    y = height - 170
    for img in plots:
        c.drawImage(ImageReader(img), 50, y - 200, width=500, height=200)
        y -= 220
        if y < 100:
            c.showPage()
            y = height - 50

    c.save()
    return report_file, " PDF report with charts created!"


# --- Query Handler ---
def handle_query(message, history):
    global chat_log
    global global_df
    df = global_df
    response = ""
    try:
        import re
        message = message.lower()
        print(" Query:", message)

        # 1. Top N regions
        if "top" in message and "region" in message:
            n = int(next((w for w in message.split() if w.isdigit()), 5))
            return df.groupby('region')['total_sales'].sum().sort_values(ascending=False).head(n).to_string()

        # 2. Top N vehicle models
        elif "top" in message and "model" in message:
            n = int(next((w for w in message.split() if w.isdigit()), 5))
            return df.groupby('vehicle_model')['total_sales'].sum().sort_values(ascending=False).head(n).to_string()

        # 3. Top dealers
        elif "top" in message and "dealer" in message:
            n = int(next((w for w in message.split() if w.isdigit()), 5))
            return df.groupby('dealer_name')['total_sales'].sum().sort_values(ascending=False).head(n).to_string()

        # 4. Sales between dates
        elif "sales between" in message:
            dates = re.findall(r'\d{4}-\d{2}-\d{2}', message)
            if len(dates) == 2:
                mask = (df['date'] >= dates[0]) & (df['date'] <= dates[1])
                return df.loc[mask].groupby('region')['total_sales'].sum().to_string()
            else:
                return " Use format: YYYY-MM-DD"

        # 5. Average unit price by model
        elif "average" in message or "unit price" in message:
            for model in df['vehicle_model'].dropna().unique():
                if model.lower() in message:
                    filtered = df[df['vehicle_model'].str.lower() == model.lower()]
                    avg = filtered['average_unit_price'].mean()
                    return f"Average unit price of {model}: ₹{avg:,.2f}"
            return " Model not found."

        # 6. Sales by segment
        elif "segment" in message:
            return df.groupby('customer_segment')['total_sales'].sum().to_string()

        # 7. Total revenue or total units sold (with optional month)
        elif "total revenue" in message or "total sales" in message:

            # Check if month is mentioned
            month_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)', message)
            
            if month_match:
                month_name = month_match.group(1)
                month_num = pd.to_datetime(month_name, format='%B').month
                filtered = df[df['date'].dt.month == month_num]

                if "revenue" in message:
                    total = filtered['total_sales'].sum()
                    return f" Total Revenue in {month_name.title()}: ₹{total:,.2f}"
                else:
                    total = filtered['units_sold'].sum()
                    return f" Total Units Sold in {month_name.title()}: {total:,}"

            else:
                if "revenue" in message:
                    total = df['total_sales'].sum()
                    return f" Total Revenue (All Time): ₹{total:,.2f}"
                else:
                    total = df['units_sold'].sum()
                    return f" Total Units Sold (All Time): {total:,}"


        # 8. Units sold in specific month
        elif "units sold" in message:
            month_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)', message)
            if month_match:
                month = month_match.group(1)
                month_num = pd.to_datetime(month, format='%B').month
                filtered = df[df['date'].dt.month == month_num]
                return f"Total units sold in {month.title()}: {filtered['units_sold'].sum():,.0f}"

        # 9. Compare two vehicle models
        elif "compare" in message and "and" in message:
            parts = message.split("compare")[1].strip().split("and")
            if len(parts) == 2:
                model1 = parts[0].strip()
                model2 = parts[1].strip()
                sales1 = df[df['vehicle_model'].str.lower() == model1.lower()]['total_sales'].sum()
                sales2 = df[df['vehicle_model'].str.lower() == model2.lower()]['total_sales'].sum()
                return f"{model1.title()}: ₹{sales1:,.0f}\n {model2.title()}: ₹{sales2:,.0f}"

        # 10. Sales or Revenue in a region
        elif ("revenue in" in message or "sales in" in message):
            for region in df['region'].dropna().unique():
                if region.lower() in message:
                    df_region = df[df['region'].str.lower() == region.lower()]
                    if "revenue" in message:
                        total = df_region['total_sales'].sum()
                        return f" Total **revenue** in {region}: ₹{total:,.2f}"
                    elif "sales" in message:
                        total = df_region['units_sold'].sum()
                        return f" Total **units sold** in {region}: {total:,}"


        # 11. Dealer performance in city
        elif "dealer" in message and "in" in message:
            for city in df['dealer_location'].dropna().unique():
                if city.lower() in message:
                    dealers = df[df['dealer_location'].str.lower() == city.lower()]
                    result = dealers.groupby('dealer_name')['total_sales'].sum().to_string()
                    return f"Dealer performance in {city.title()}:\n{result}"

        # 12. Sales trend per month
        elif "monthly sales trend" in message or "sales trend" in message:
            monthly = df.groupby(df['date'].dt.to_period("M"))['total_sales'].sum()
            return f" Monthly sales trend:\n{monthly.to_string()}"

        # 13. Yearly revenue comparison
        elif "compare" in message and "year" in message:
            yearly = df.groupby(df['date'].dt.year)['total_sales'].sum()
            return f" Year-wise revenue:\n{yearly.to_string()}"

        # 14. Best month for model
        elif "best month" in message:
            for model in df['vehicle_model'].unique():
                if model.lower() in message:
                    monthly = df[df['vehicle_model'].str.lower() == model.lower()]
                    result = monthly.groupby(df['date'].dt.to_period("M"))['total_sales'].sum()
                    best = result.idxmax()
                    return f" Best month for {model}: {best}"
            return " Model not found."

        # 15. List models sold in region
        elif "models sold in" in message:
            for region in df['region'].unique():
                if region.lower() in message:
                    models = df[df['region'].str.lower() == region.lower()]['vehicle_model'].unique()
                    return f" Models sold in {region}:\n" + ", ".join(models)

        # 16. Dealers who sold a model
        elif "dealers who sold" in message:
            for model in df['vehicle_model'].unique():
                if model.lower() in message:
                    dealers = df[df['vehicle_model'].str.lower() == model.lower()]['dealer_name'].unique()
                    return f" Dealers who sold {model}:\n" + ", ".join(dealers)       
                
        elif "save" in message or "export" in message:
            if not history or len(history) == 0:
                return " No conversation to export yet."


            # Generate unique file path
            file_name = f"chat_log_{uuid.uuid4().hex[:8]}.txt"
            file_path = os.path.join(os.getcwd(), file_name)

            try:
                with open(file_path, mode="w", encoding="utf-8") as f:
                    for user, bot in history:
                        f.write(f"User: {user}\n")
                        f.write(f"Bot: {bot}\n")
                        f.write("-" * 40 + "\n")
            except Exception as e:
                return f" Failed to save chat log: {e}"

            return (
                " Chat log saved. Download below:",
                gr.File(file_path, label="Download Chat Log")
            )

        # Fallback
        else:
            return (
                " Sorry, I didn’t understand.\n"
                "Try things like:\n"
                "- 'Top 3 regions'\n"
                "- 'Average unit price of Skyline 20.15'\n"
                "- 'Sales between 2024-01-01 and 2024-03-31'\n"
                "- 'Compare sales of Pro 2055 and Skyline 20.15'"
            )

    except Exception as e:

        print("ERROR:", e)
        return f"Internal Error: {str(e)}"

global_df=None    
# Handle CSV Upload
def process_upload(csv_file):
    global global_df
    df = pd.read_csv(csv_file.name, parse_dates=["Date"])
    df = clean_data(df)
    global_df = df 
    cleaned_file = save_cleaned_csv(df)
    pdf_file = generate_detailed_sales_report(cleaned_file)
    plots = generate_visual_pdf_report(df)
    return pdf_file, cleaned_file, plots,"✅ Data cleaned and report generated. Ask your questions below 👇"
    

    
# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("#  VECV AI Sales Analyst")
    gr.Markdown("**Step 1:** Upload CSV file to generate report\nand  Ask your questions in chat")

    
    file_input = gr.File(label="📁 Upload CSV", file_types=[".csv"])
    cleaned_csv_output = gr.File(label="🧼 Download Cleaned CSV")
    report_output = gr.File(label="📄 Download Report")
    status = gr.Textbox(label="Status", interactive=False)
    gallery = gr.Gallery(label="📈 Visual Insights", columns=2, rows=2)

    
    file_input.change(
        fn=process_upload,
        inputs=file_input,
        outputs=[cleaned_csv_output,report_output,status, gallery]
    )
    gr.Markdown("---")
    gr.Markdown("**Note**: Report includes charts like monthly trend, region-wise sales, and top vehicle models.")
    gr.Markdown("### 💬 Ask about your data:")
    chatbot = gr.ChatInterface(fn=handle_query, title="VECV Sales Chat", textbox=gr.Textbox(placeholder="Ask me about sales, models, regions..."))
    gr.Button("Exit")
demo.launch()
