from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import io
import json
import traceback

app = Flask(__name__)

# ─── Core processing logic ───────────────────────────────────────────────────

def process_sales(sales_files, cust_file, material_file, mrp_file, config):
    exclude_types   = [x.strip().upper() for x in config.get("exclude_types", "").split(",") if x.strip()]
    flip_types      = [x.strip().upper() for x in config.get("flip_types", "").split(",") if x.strip()]
    fx_rate         = float(config.get("fx_rate", 1.0))

    logs = []

    # ── 1. Load & combine all sales files ────────────────────────────────────
    logs.append("Loading sales files...")
    dfs = []
    for f in sales_files:
        df = pd.read_excel(f, engine="openpyxl")
        logs.append(f"  ✓ {f.filename} — {len(df):,} rows")
        dfs.append(df)

    sales = pd.concat(dfs, ignore_index=True)
    logs.append(f"Combined: {len(sales):,} rows, {len(sales.columns)} columns")

    # ── 2. Filter: Cancelled = blank ─────────────────────────────────────────
    before = len(sales)
    sales = sales[sales["Cancelled"].isna() | (sales["Cancelled"].astype(str).str.strip() == "")]
    logs.append(f"After removing cancelled: {len(sales):,} rows (removed {before - len(sales):,})")

    # ── 3. Filter: Exclude billing types ─────────────────────────────────────
    before = len(sales)
    sales["Billing Type"] = sales["Billing Type"].astype(str).str.strip().str.upper()
    if exclude_types:
        sales = sales[~sales["Billing Type"].isin(exclude_types)]
    logs.append(f"After excluding billing types {exclude_types}: {len(sales):,} rows (removed {before - len(sales):,})")

    # ── 4. Net Value = column index 114 (DK, 0-based) ────────────────────────
    col_index = 114
    net_val_col = sales.columns[col_index]
    logs.append(f"Net Value column: '{net_val_col}' (column DK, index {col_index})")
    sales["_net_value"] = pd.to_numeric(sales[net_val_col], errors="coerce").fillna(0)
    sales["_cost"]      = pd.to_numeric(sales["Cost"], errors="coerce").fillna(0)
    sales["_qty"]       = pd.to_numeric(sales["Billed Quantity"], errors="coerce").fillna(0)

    # ── 5. Flip signs for return billing types ────────────────────────────────
    flip_mask = sales["Billing Type"].isin([x.upper() for x in flip_types])
    sales.loc[flip_mask, "_net_value"] = sales.loc[flip_mask, "_net_value"] * -1
    sales.loc[flip_mask, "_qty"]       = sales.loc[flip_mask, "_qty"] * -1
    sales.loc[flip_mask, "_cost"]      = sales.loc[flip_mask, "_cost"] * -1
    logs.append(f"Sign flipped for billing types {flip_types}: {flip_mask.sum():,} rows")

    # ── 6. Load master files ──────────────────────────────────────────────────
    cust_df     = pd.read_excel(cust_file, engine="openpyxl")
    material_df = pd.read_excel(material_file, engine="openpyxl")
    mrp_df      = pd.read_excel(mrp_file, engine="openpyxl")

    logs.append(f"Customer Grouping: {len(cust_df):,} rows")
    logs.append(f"Material Master: {len(material_df):,} rows")
    logs.append(f"MRP file: {len(mrp_df):,} rows")

    # ── 7. Join Customer Grouping ─────────────────────────────────────────────
    cust_df["Customer_Code"] = cust_df["Customer_Code"].astype(str).str.strip()
    sales["_payer"]          = sales["Payer"].astype(str).str.strip()
    sales = sales.merge(
        cust_df[["Customer_Code", "Customer_Group"]].drop_duplicates(),
        left_on="_payer", right_on="Customer_Code", how="left"
    )
    sales["Customer_Group"] = sales["Customer_Group"].fillna("Unknown")
    logs.append(f"Customer join: {sales['Customer_Group'].ne('Unknown').sum():,} matched, {sales['Customer_Group'].eq('Unknown').sum():,} unmatched")

    # ── 8. Join Material Master → tag R&K / Core / XOther ────────────────────
    material_df["Material"] = material_df["Material"].astype(str).str.strip()
    material_df["_status"]  = material_df["Plant-sp.matl status"].astype(str).str.strip().str.upper()
    material_df["_mat_tag"] = material_df["_status"].apply(
        lambda x: "R&K" if x in ["R", "K"] else "Core"
    )
    sales["_material"] = sales["Material"].astype(str).str.strip()
    sales = sales.merge(
        material_df[["Material", "_mat_tag"]].drop_duplicates(subset="Material"),
        left_on="_material", right_on="Material", how="left", suffixes=("", "_mat")
    )
    sales["_mat_tag"] = sales["_mat_tag"].fillna("XOther")
    logs.append(f"Material tag: R&K={sales['_mat_tag'].eq('R&K').sum():,}, Core={sales['_mat_tag'].eq('Core').sum():,}, XOther={sales['_mat_tag'].eq('XOther').sum():,}")

    # ── 9. Join MRP (primary: Material + Currency, fallback: EAN + Currency) ──
    mrp_df["Material"]      = mrp_df["Material"].astype(str).str.strip()
    mrp_df["Currency"]      = mrp_df["Currency"].astype(str).str.strip().str.upper()
    mrp_df["EAN/UPC"]       = mrp_df["EAN/UPC"].astype(str).str.strip()
    mrp_df["Retail Price"]  = pd.to_numeric(mrp_df["Retail Price"], errors="coerce").fillna(0)
    mrp_df["Wholesale Price"] = pd.to_numeric(mrp_df["Wholesale Price"], errors="coerce").fillna(0)

    sales["_currency"] = sales["Document Currency"].astype(str).str.strip().str.upper()
    sales["_ean"]      = sales["EAN/UPC"].astype(str).str.strip()

    # Primary join
    mrp_primary = mrp_df[["Material", "Currency", "Retail Price", "Wholesale Price"]].drop_duplicates(subset=["Material", "Currency"])
    sales = sales.merge(mrp_primary, left_on=["_material", "_currency"], right_on=["Material", "Currency"], how="left", suffixes=("", "_mrp"))

    # Fallback join for unmatched rows
    unmatched = sales["Retail Price"].isna()
    if unmatched.sum() > 0:
        mrp_fallback = mrp_df[["EAN/UPC", "Currency", "Retail Price", "Wholesale Price"]].drop_duplicates(subset=["EAN/UPC", "Currency"])
        fallback = sales[unmatched][["_ean", "_currency"]].merge(
            mrp_fallback, left_on=["_ean", "_currency"], right_on=["EAN/UPC", "Currency"], how="left"
        )
        sales.loc[unmatched, "Retail Price"]    = fallback["Retail Price"].values
        sales.loc[unmatched, "Wholesale Price"] = fallback["Wholesale Price"].values

    sales["Retail Price"]    = pd.to_numeric(sales["Retail Price"], errors="coerce").fillna(0)
    sales["Wholesale Price"] = pd.to_numeric(sales["Wholesale Price"], errors="coerce").fillna(0)

    logs.append(f"MRP join: {sales['Retail Price'].ne(0).sum():,} matched, {sales['Retail Price'].eq(0).sum():,} unmatched")

    # ── 10. Calculate values ──────────────────────────────────────────────────
    sales["_retail_value"]    = sales["Retail Price"]    * sales["_qty"]
    sales["_wholesale_value"] = sales["Wholesale Price"] * sales["_qty"]

    # ── 11. FX conversion (values only, not qty) ──────────────────────────────
    sales["_net_value_usd"]       = sales["_net_value"]       * fx_rate
    sales["_cost_usd"]            = sales["_cost"]            * fx_rate
    sales["_retail_value_usd"]    = sales["_retail_value"]    * fx_rate
    sales["_wholesale_value_usd"] = sales["_wholesale_value"] * fx_rate

    logs.append(f"FX rate applied: {fx_rate} → all values converted to USD")

    # ── 12. Pivot / Summary ───────────────────────────────────────────────────
    logs.append("Building summary...")

    grp = sales.groupby(["Customer_Group", "_mat_tag"]).agg(
        Billed_Qty        = ("_qty",               "sum"),
        Net_Sales_USD     = ("_net_value_usd",      "sum"),
        Cost_USD          = ("_cost_usd",           "sum"),
        Wholesale_Val_USD = ("_wholesale_value_usd","sum"),
        Retail_Val_USD    = ("_retail_value_usd",   "sum"),
    ).reset_index()

    # Customer totals
    totals = grp.groupby("Customer_Group").agg(
        Billed_Qty        = ("Billed_Qty",        "sum"),
        Net_Sales_USD     = ("Net_Sales_USD",      "sum"),
        Cost_USD          = ("Cost_USD",           "sum"),
        Wholesale_Val_USD = ("Wholesale_Val_USD",  "sum"),
        Retail_Val_USD    = ("Retail_Val_USD",     "sum"),
    ).reset_index()
    totals["_mat_tag"] = "Total"

    summary = pd.concat([grp, totals], ignore_index=True)

    # Ranking by total net sales descending
    rank_df = totals[["Customer_Group", "Net_Sales_USD"]].copy()
    rank_df["Ranking"] = rank_df["Net_Sales_USD"].rank(ascending=False, method="dense").astype(int)
    summary = summary.merge(rank_df[["Customer_Group", "Ranking"]], on="Customer_Group", how="left")

    # Sort
    mat_order = {"Core": 0, "R&K": 1, "XOther": 2, "Total": 3}
    summary["_mat_order"] = summary["_mat_tag"].map(mat_order).fillna(4)
    summary = summary.sort_values(["Ranking", "_mat_order"]).reset_index(drop=True)

    # Final columns
    summary = summary.rename(columns={
        "Customer_Group":    "Customer Name",
        "_mat_tag":          "Material Type",
        "Billed_Qty":        "Billed Qty",
        "Net_Sales_USD":     "Net Sales (USD)",
        "Cost_USD":          "Cost (USD)",
        "Wholesale_Val_USD": "Wholesale Value (USD)",
        "Retail_Val_USD":    "Retail Value (USD)",
    })

    summary = summary[[
        "Customer Name", "Ranking", "Material Type",
        "Billed Qty", "Net Sales (USD)", "Cost (USD)",
        "Wholesale Value (USD)", "Retail Value (USD)"
    ]]

    # Round values
    for col in ["Net Sales (USD)", "Cost (USD)", "Wholesale Value (USD)", "Retail Value (USD)"]:
        summary[col] = summary[col].round(2)
    summary["Billed Qty"] = summary["Billed Qty"].round(0).astype(int)

    logs.append(f"Summary: {len(summary):,} rows, {summary['Customer Name'].nunique():,} customers")

    return summary, logs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    try:
        sales_files   = request.files.getlist("sales_files")
        cust_file     = request.files.get("cust_file")
        material_file = request.files.get("material_file")
        mrp_file      = request.files.get("mrp_file")

        config = {
            "exclude_types": request.form.get("exclude_types", ""),
            "flip_types":    request.form.get("flip_types", ""),
            "fx_rate":       request.form.get("fx_rate", "1.0"),
        }

        if not sales_files or not cust_file or not material_file or not mrp_file:
            return jsonify({"success": False, "error": "Please upload all required files."})

        summary, logs = process_sales(sales_files, cust_file, material_file, mrp_file, config)

        # Return as JSON
        result = {
            "success": True,
            "logs":    logs,
            "columns": list(summary.columns),
            "rows":    summary.values.tolist(),
            "stats": {
                "customers": int(summary[summary["Material Type"] == "Total"]["Customer Name"].nunique()),
                "total_rows": len(summary),
                "net_sales":  round(float(summary[summary["Material Type"] == "Total"]["Net Sales (USD)"].sum()), 2),
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
