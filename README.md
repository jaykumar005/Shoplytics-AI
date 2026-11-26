# Shoplytics AI & Prediction (Working MVP)

## What works in this version
- Upload an orders CSV/Excel file.
- Append daily order files into a local SQLite database.
- Deduplicate records by `order_id` (latest uploaded record wins).
- Validate order ID, date, quantity and unit price.
- Calculate sales as `quantity * unit_price`.
- View overview KPIs, daily/monthly trends and top products.
- Download a data-quality report.
- Train a Random Forest daily-sales forecasting model after at least 60 calendar days of history.
- Evaluate the model using a chronological holdout set (MAE, RMSE, R²).
- Forecast 1–14 days after a model has been trained.

## Important data note
The included original sample data contains only one month of order history. The application will load it and show analytics, but intentionally will not train a forecasting model yet. Add more real daily data over time. A model trained on too little history can look convincing while being unreliable.

## Requirements
- Windows 10/11
- Python 3.10 or newer
- VS Code recommended

## Run on Windows
1. Extract this ZIP.
2. Open the extracted `Smart_Ecommerce_Predictive_Analytics` folder in VS Code.
3. Open Terminal in VS Code.
4. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

5. Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Start the app:

```powershell
streamlit run app/app.py
```

The browser should open automatically. If not, open the local URL printed in the terminal.

## First use
1. Open **Upload & Update Data**.
2. Click **Load original project sample data** to load the included five CSV datasets (orders are used for current MVP analytics).
3. Visit **Overview**, **Data Quality**, and **Sales Analytics**.
4. For daily updates, upload the latest orders CSV in **Upload & Update Data**. It must contain:
   - `order_id`
   - `order_date`
   - `quantity`
   - `unit_price`
   - Optional: `customer_id`, `product_id`

Each daily upload is merged into SQLite. Do not upload only a tiny partial file if it contains updates to existing IDs; duplicate `order_id` values are replaced by the latest row.

## Sample order format

```csv
order_id,customer_id,product_id,order_date,quantity,unit_price
O10001,C0001,P0001,2026-11-25,2,499
O10002,C0002,P0002,2026-11-25,1,1299
```

## Model details
Target: daily total sales (`sum(quantity * unit_price)`).
Features: previous day's sales, sales 7 days ago, previous 7-day mean, day of week, day of year, month.
Model: RandomForestRegressor.
Validation: chronological train/test split. This is a starter model; for serious forecasting, collect a longer history and compare against seasonal-naive and other time-series models.

## Current limitations / next phase
- This MVP does not yet connect to Shopify/WooCommerce automatically.
- API integration requires a store you own/manage and its authorized API credentials.
- Payment/refund reconciliation and full customer/product joins need additional business rules.
- Current model forecasts sales totals, not demand by SKU or customer-level outcomes.


## Dashboard design
The Streamlit interface includes a custom gradient hero header, dark navigation sidebar,
clean KPI cards, rounded chart containers, and styled upload controls. The dashboard is
responsive within Streamlit's layout. Run it with `streamlit run app/app.py`.


## Product names
When a products reference file is imported (or the supplied sample data is loaded),
the Sales Analytics page displays product names alongside product IDs. The expected
products columns include `product_id` and `product_name`.

## Publish a shareable online demo (Streamlit Community Cloud)
1. Create a GitHub repository and upload the **contents of this project folder**
   (so `app/app.py`, `requirements.txt`, and `data/sample/` are at the repository root).
2. Keep only demo/sample data in the public repository. Do not commit real customer,
   payment, or other private business data.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the app file path
   to `app/app.py`.
5. Deploy. When the build finishes, Streamlit gives you a public URL you can share.

### Important hosting note
This MVP uses a local SQLite database and local model files. On Streamlit Community
Cloud, local files are not guaranteed to persist across app restarts or redeployments.
The public demo can load the bundled sample data, but durable multi-user storage needs
a hosted database (for example, managed PostgreSQL) and persistent model storage.
The current project ZIP is deployment-ready, but it has not been published to a live
URL because deployment requires access to your GitHub and Streamlit accounts.
