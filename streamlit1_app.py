# streamlit_app.py
# Streamlit wrapper for your FinRPG script (keeps your original code and logic intact)

import random
from datetime import datetime, timedelta
from collections import defaultdict
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # or any default font
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Arial, sans-serif", size=14, color="#333333"),
        title_font=dict(size=20, color="#222222"),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor="#DDDDDD")
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", zeroline=False)
    return fig


try:
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Keep your original rich imports (we won't use console printing in Streamlit UI,
# but we keep them so your code remains exactly present).
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.align import Align

# Original console (unused in Streamlit UI, kept for fidelity)
console = Console()

# -------------------------
# === ORIGINAL FUNCTIONS ===
# (kept as-is from your provided code)
# -------------------------
def simulate_transactions(seed=42, months=6, starting_balance=2000.0):
    """
    Simulate daily transactions for the past `months` months.
    Returns a pandas DataFrame of transactions and a starting balance.
    """
    random.seed(seed)
    np.random.seed(seed)

    today = datetime.today().date()
    start_date = today - timedelta(days=30 * months)
    dates = pd.date_range(start=start_date, end=today, freq='D').date

    categories = ["Food", "Transport", "Entertainment", "Bills", "Savings", "Income"]
    rows = []

    for d in dates:
        if d.day in (1, 15) and random.random() < 0.3:  # not always; some months single salary
            amount = round(random.gauss(2500, 300), 2)
            rows.append({"date": d, "category": "Income", "amount": amount})

        if d.day in (3, 5) and random.random() < 0.85:
            amount = -round(random.uniform(600, 1100), 2)
            rows.append({"date": d, "category": "Bills", "amount": amount})

        if d.day == 2 and random.random() < 0.9:
            amount = -round(random.uniform(150, 500), 2)
            rows.append({"date": d, "category": "Savings", "amount": amount})

        if random.random() < 0.6:
            amount = -round(abs(random.gauss(8, 12)) + random.choice([0, 5, 15]), 2)
            rows.append({"date": d, "category": "Food", "amount": amount})

        if random.random() < 0.35:
            amount = -round(abs(random.gauss(3, 4)) + random.choice([0, 2, 5]), 2)
            rows.append({"date": d, "category": "Transport", "amount": amount})

        if random.random() < 0.12:
            amount = -round(abs(random.gauss(20, 30)) + random.choice([0, 10, 25]), 2)
            rows.append({"date": d, "category": "Entertainment", "amount": amount})

    df = pd.DataFrame(rows)
    if df[df.category == "Income"].empty:
        df = df.append({"date": start_date, "category": "Income", "amount": 2500.0}, ignore_index=True)
    if df[df.category == "Savings"].empty:
        df = df.append({"date": start_date + timedelta(days=2), "category": "Savings", "amount": -200.0}, ignore_index=True)

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['running_balance'] = starting_balance + df['amount'].cumsum()
    return df, starting_balance

def category_summary(df):
    """Return a DataFrame summarizing spending by category (absolute amounts)."""
    summary = df.groupby('category')['amount'].sum().reset_index()
    summary['abs_amount'] = summary['amount'].abs()
    cat_order = ["Income", "Savings", "Bills", "Food", "Transport", "Entertainment"]
    summary['category'] = pd.Categorical(summary['category'], categories=cat_order, ordered=True)
    summary = summary.sort_values('category')
    return summary[['category', 'amount', 'abs_amount']]

def monthly_net_flows(df):
    """Aggregate to monthly net flows (income - expenses) for each calendar month in the data."""
    df2 = df.copy()
    df2['month'] = df2['date'].dt.to_period('M')
    monthly = df2.groupby('month')['amount'].sum().reset_index()
    monthly['month_start'] = monthly['month'].dt.to_timestamp()
    return monthly[['month_start', 'amount']]

def project_balance(monthly_net, last_balance, months_ahead=3):
    """
    Use linear regression on monthly net flows to predict next `months_ahead` net flows,
    then cumulatively compute projected balances.
    Returns arrays of projected dates and balances.
    """
    if len(monthly_net) < 3 or not SKLEARN_AVAILABLE:
        # moving average fallback: average of past nets
        avg_net = monthly_net['amount'].mean()
        last_month = monthly_net['month_start'].iloc[-1]
        proj_dates = [last_month + pd.DateOffset(months=i) for i in range(1, months_ahead+1)]
        proj_nets = [avg_net for _ in range(months_ahead)]
    else:
        X = np.arange(len(monthly_net)).reshape(-1, 1)
        y = monthly_net['amount'].values
        model = LinearRegression()
        model.fit(X, y)
        last_idx = len(monthly_net) - 1
        future_idx = np.arange(last_idx + 1, last_idx + 1 + months_ahead).reshape(-1, 1)
        proj_nets = model.predict(future_idx)
        last_month = monthly_net['month_start'].iloc[-1]
        proj_dates = [last_month + pd.DateOffset(months=i) for i in range(1, months_ahead+1)]

    proj_balances = []
    bal = last_balance
    for net in proj_nets:
        bal += net
        proj_balances.append(bal)

    hist_dates = list(monthly_net['month_start'])
    hist_balances = []
    bal_hist = last_balance - monthly_net['amount'].iloc[-1]  # rough starting point back one month
    cs = monthly_net['amount'].cumsum()
    starting_balance = last_balance - cs.iloc[-1]
    hist_balances = starting_balance + cs.values
    return hist_dates, hist_balances, proj_dates, proj_balances

def train_tiny_decision_tree():
    """
    Train a tiny DecisionTreeClassifier on synthetic labeled examples.
    Features:
      - savings_rate (0-1)
      - spend_income_ratio (spending / income) (0-2)
      - pct_food   (0-1)
      - pct_entertainment (0-1)

    Labels: 0=Needs Improvement, 1=Good, 2=Excellent
    Returns the trained classifier.
    """
    if not SKLEARN_AVAILABLE:
        return None  # We'll fallback to rule-based tree if sklearn not available

    X = []
    y = []

    excellent = [
        (0.25, 0.6, 0.12, 0.05),
        (0.20, 0.7, 0.10, 0.04),
        (0.30, 0.5, 0.08, 0.03),
    ]
    for ex in excellent:
        X.append(ex); y.append(2)

    good = [
        (0.12, 0.85, 0.18, 0.08),
        (0.10, 0.9, 0.16, 0.12),
        (0.15, 0.8, 0.14, 0.09),
    ]
    for ex in good:
        X.append(ex); y.append(1)

    needs = [
        (0.05, 1.3, 0.28, 0.15),
        (0.02, 1.6, 0.30, 0.22),
        (0.07, 1.2, 0.25, 0.18),
    ]
    for ex in needs:
        X.append(ex); y.append(0)

    X = np.array(X)
    y = np.array(y)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    return clf

def compute_financial_features(df):
    """
    Compute features used by the decision tree and advice engine:
      - savings_rate: ratio of abs(savings) / total income
      - spend_income_ratio: total spending (abs of negative non-income) / income
      - pct_food, pct_entertainment: spend in category / total spend (non-income)
    """
    totals = df.groupby('category')['amount'].sum().to_dict()
    income = totals.get('Income', 0.0)
    savings = -totals.get('Savings', 0.0)  # savings is negative amount in transactions
    spending_total = 0.0
    for k, v in totals.items():
        if k not in ['Income', 'Savings']:
            spending_total += -v  # categories are negative -> take abs

    # safety
    income = max(1.0, income)  # avoid division by zero
    savings_rate = savings / income
    spend_income_ratio = (spending_total / income) if income else 0.0

    pct_food = (-totals.get('Food', 0.0) / spending_total) if spending_total > 0 else 0.0
    pct_ent = (-totals.get('Entertainment', 0.0) / spending_total) if spending_total > 0 else 0.0

    features = {
        "income": income,
        "savings": savings,
        "spending_total": spending_total,
        "savings_rate": savings_rate,
        "spend_income_ratio": spend_income_ratio,
        "pct_food": pct_food,
        "pct_entertainment": pct_ent
    }
    return features

def decision_tree_score(features, clf=None):
    """
    If clf provided (sklearn), use it to predict label; otherwise use a simple manual decision tree.
    Returns label string and numeric code (0,1,2).
    """
    if clf is not None and SKLEARN_AVAILABLE:
        X = np.array([[features['savings_rate'], features['spend_income_ratio'],
                       features['pct_food'], features['pct_entertainment']]])
        label_code = int(clf.predict(X)[0])
    else:
        sr = features['savings_rate']
        sirl = features['spend_income_ratio']
        if sr >= 0.20 and sirl <= 0.8:
            label_code = 2
        elif sr >= 0.10 and sirl <= 1.0:
            label_code = 1
        else:
            label_code = 0
    mapping = {0: "Needs Improvement", 1: "Good", 2: "Excellent"}
    return mapping[label_code], label_code

def advice_engine(features):
    """
    Simple rule-based advice generator: return 2-3 personalized tips.
    """
    tips = []

    if features['pct_food'] > 0.22:  # >22% of spending on food
        tips.append("You're spending a high share on Food — try cooking 2–3 nights a week instead of eating out.")
    elif features['pct_food'] > 0.12:
        tips.append("Food is a noticeable portion of your spending — small grocery planning can save money.")

    if features['savings_rate'] < 0.10:
        tips.append("Savings are below 10% of income — consider automating at least 10% of your income to savings.")
    elif features['savings_rate'] >= 0.20:
        tips.append("Great job saving ~20%+ of your income. Keep building an emergency fund (3–6 months).")

    if features['pct_entertainment'] > 0.12:
        tips.append("Entertainment is comparatively high — try a 'no-spend' weekend challenge to reset habits.")

    if features['spend_income_ratio'] > 1.0:
        tips.append("You're spending more than or close to your income — review recurring subscriptions and bills.")
    elif features['spend_income_ratio'] > 0.85:
        tips.append("You're close to breaking even — track variable expenses and set week-by-week budgets.")

    if not tips:
        tips = ["Your finances look balanced — consider setting a medium-term saving goal to gamify progress.",
                "Try a weekly check-in: review spendings and reward progress with XP."]
    return tips[:3]

def compute_xp(features):
    """
    Award or penalize XP based on habits. The returned XP is the net change.
    Rules (example):
      +30 XP if savings_rate >= 0.15
      +20 XP if spend_income_ratio <= 0.8
      +15 XP if pct_food <= 0.12
      -20 XP if savings_rate < 0.05
      -25 XP if spend_income_ratio > 1.1
      -15 XP if pct_entertainment > 0.15
    Start at 0 XP (or could load persistent state).
    """
    xp = 0
    if features['savings_rate'] >= 0.15:
        xp += 30
    if features['spend_income_ratio'] <= 0.8:
        xp += 20
    if features['pct_food'] <= 0.12:
        xp += 15

    if features['savings_rate'] < 0.05:
        xp -= 20
    if features['spend_income_ratio'] > 1.1:
        xp -= 25
    if features['pct_entertainment'] > 0.15:
        xp -= 15

    xp = max(-100, min(200, xp))
    return int(xp)

def level_from_xp(total_xp):
    """Compute level (starting level 1) and xp progress to next level (every 100 XP)."""
    level = (total_xp // 100) + 1
    xp_into_level = total_xp % 100
    xp_to_next = 100 - xp_into_level
    return int(level), int(xp_into_level), int(xp_to_next)

def display_dashboard(df, summary, features, health_label, health_code, xp_change, total_xp, monthly_net,
                      hist_dates, hist_balances, proj_dates, proj_balances, tips):
    console.rule("[bold yellow]FinRPG — Your Financial Dashboard[/bold yellow]")

    t = Table(title="Spending Summary (last 6 months)", box=box.ROUNDED, show_lines=True)
    t.add_column("Category", style="cyan", justify="left")
    t.add_column("Net Amount", style="magenta", justify="right")
    t.add_column("Abs Total", style="green", justify="right")
    for _, row in summary.iterrows():
        amt = row['amount']
        abs_amt = row['abs_amount']
        t.add_row(str(row['category']), f"{amt:,.2f}", f"{abs_amt:,.2f}")
    console.print(t)

    health_color = {0: "red", 1: "yellow", 2: "green"}[health_code]
    health_text = Text(f"{health_label}", style=f"bold {health_color}")
    total_xp_after = max(0, total_xp + xp_change)  # ensure non-negative for display
    level, xp_into_level, xp_to_next = level_from_xp(total_xp_after)
    xp_panel = Panel.fit(
        f"[bold]Level {level}[/bold]\nXP: {xp_into_level} / 100\nNext: {xp_to_next} XP",
        title="RPG Progress",
        border_style="blue"
    )
    cols = Table.grid(expand=True)
    cols.add_column(ratio=2)
    cols.add_column(ratio=1)
    features_text = (
        f"Income (total): {features['income']:.2f}\n"
        f"Savings (total): {features['savings']:.2f}\n"
        f"Savings rate: {features['savings_rate']*100:.1f}%\n"
        f"Spending / Income: {features['spend_income_ratio']*100:.1f}%"
    )
    cols.add_row(Panel.fit(health_text, title="Financial Health", border_style=health_color),
                 Panel.fit(features_text, title="Key Metrics", border_style="magenta"))
    console.print(cols)
    console.print(xp_panel)

    with Progress() as progress:
        task = progress.add_task("[green]XP Progress", total=100)
        progress.update(task, completed=xp_into_level)

    console.rule("[bold purple]Balance Projection (next 3 months)[/bold purple]")

    plt.figure(figsize=(8, 4))
    plt.plot(hist_dates, hist_balances, marker='o', label='Historic Month Balance')
    plt.plot(proj_dates, list(proj_balances), marker='o', linestyle='--', label='Projected Balance')
    plt.xlabel('Month')
    plt.ylabel('Balance (EUR)')
    plt.title('Account Balance Projection')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        console.print("[bold red]Failed to display matplotlib chart in this environment.[/]")

    console.rule("[bold green]Personalized Tips[/bold green]")
    tip_table = Table(box=box.SIMPLE)
    tip_table.add_column("Tip", style="bright_white")
    for i, ttext in enumerate(tips, start=1):
        tip_table.add_row(f"{i}. {ttext}")
    console.print(tip_table)

    nf_table = Table(title="Monthly Net Flows", box=box.MINIMAL)
    nf_table.add_column("Month", style="cyan")
    nf_table.add_column("Net Flow", justify="right")
    for idx, row in monthly_net.iterrows():
        nf_table.add_row(str(row['month_start'].strftime("%Y-%m")), f"{row['amount']:,.2f}")
    console.print(nf_table)

    console.rule("[bold white]End of Report[/bold white]")

# -------------------------
# === STREAMLIT WRAPPER ===
# -------------------------
import streamlit as st

st.set_page_config(page_title="Your Personal Finance RPG", layout="wide")

st.set_page_config(page_title="Your Personal Finance RPG", layout="wide")

def run_app():
    st.title('\n' + "Your Personal Financial Advisor")
    
    # st.markdown(
    #     "This app runs your original FinRPG script in a browser. "
    #     "It uses your simulation, projection, decision tree, advice engine and XP logic unchanged."
    # )

    # Sidebar controls
    st.sidebar.header("Simulation Controls")
    seed = st.sidebar.number_input("Random seed", value=1234, step=1,
                                   help="Seed for random number generator to reproduce results.")
    months = st.sidebar.slider("Months to simulate", min_value=3, max_value=12, value=6,
                               help="How many months of transaction data to simulate.")
    starting_balance = st.sidebar.number_input("Starting balance (€)", value=2000.0, step=100.0,
                                               help="Your bank account balance at simulation start.")
    run_button = st.sidebar.button("Generate / Analyze")

    st.sidebar.markdown("---")
    st.sidebar.header("Upload your data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (columns: date, category, amount)", type=["csv"])

    # st.markdown("**Sections:** 1) Spending summary  2) Financial health & XP  3) Balance projection chart  4) Advice tips")
    # st.markdown("---")

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                st.warning("Uploaded CSV missing 'date' column — will use simulated data if you press Generate.")
                df = None
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            df = None

    if df is None and run_button:
        df, _ = simulate_transactions(seed=seed, months=months, starting_balance=starting_balance)

    if df is None:
        st.info("Upload a CSV or press **Generate / Analyze** to simulate data.")
        return

    # Process data
    summary = category_summary(df)
    summary = summary.apply(lambda col: col.fillna(0) if col.dtype != 'category' else col)

    monthly = monthly_net_flows(df)
    last_balance = df['running_balance'].iloc[-1]
    hist_dates, hist_balances, proj_dates, proj_balances = project_balance(monthly, last_balance, months_ahead=3)

    clf = train_tiny_decision_tree() if SKLEARN_AVAILABLE else None
    features = compute_financial_features(df)
    health_label, health_code = decision_tree_score(features, clf=clf)
    tips = advice_engine(features)
    xp_change = compute_xp(features)
    previous_total_xp = 40
    total_xp = max(0, previous_total_xp + xp_change)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Spending Summary (last {months} months)")
        display_summary = summary.copy()
        display_summary['category'] = display_summary['category'].astype(str)

        if "amount" in display_summary.columns:
            display_summary = display_summary.drop(columns=["amount"])

        # Sort by the column you want, e.g. 'rank'
        display_summary = display_summary.sort_index(ascending=True)

        styled_df = display_summary.style.format({"amount": "{:,.2f}", "abs_amount": "{:,.2f}"}).set_table_styles(
            [
                    {"selector": "tr:hover", "props": [("background-color", "#2a2a2a")]},  # dark hover
                    {"selector": "th", "props": [("background-color", "#4F46E5"), ("color", "white")]},
                    {"selector": "td", "props": [("background-color", "#1e1e1e"), ("color", "white")]}  # dark table cells
            ]
        )
        st.dataframe(styled_df, use_container_width=True)

        plot_df = display_summary.set_index('category')

        # Create custom dark background bar chart with Plotly
        fig = go.Figure(
            data=[go.Bar(
                x=plot_df.index,
                y=plot_df['abs_amount'],
                marker_color='#4F46E5'  # custom bar color (you can change this)
            )]
        )
        fig.update_layout(
            plot_bgcolor='white',  # dark plot area
            paper_bgcolor='white',  # dark background around plot
            font_color='black',                # white text
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickangle=45,
                tickfont=dict(color='black')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(100,100,100,0.3)',
                zeroline=False,
                tickfont=dict(color='black')
            ),
            margin=dict(l=40, r=20, t=40, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Financial Health & XP")
        health_color = {0: "Needs Improvement", 1: "Good", 2: "Excellent"}[health_code]
        st.markdown(f"**Health:** {health_color}")

        st.markdown("**Key metrics:**")
        st.write(f"- Total income: €{features['income']:.2f}")
        st.write(f"- Total savings: €{features['savings']:.2f}")
        st.write(f"- Savings rate: {features['savings_rate']*100:.1f}%")
        st.write(f"- Spending / Income: {features['spend_income_ratio']*100:.1f}%")

        level, xp_into_level, xp_to_next = level_from_xp(total_xp)
        st.markdown(f"**Level:** {level}")
        st.markdown(f"**XP:** {xp_into_level} / 100  (Next level in {xp_to_next} XP)")

        fig = go.Figure()

        # Background bar - full length, lighter color
        fig.add_trace(go.Bar(
            x=[100],
            y=['XP Progress'],
            orientation='h',
            marker=dict(color='#d3d3d3'),  # light gray for background track
            width=0.5,
            showlegend=False,
            hoverinfo='none',
        ))

        # Foreground bar - actual XP progress, colored
        fig.add_trace(go.Bar(
            x=[xp_into_level],
            y=['XP Progress'],
            orientation='h',
            marker=dict(color='#4F46E5'),  # your chosen color
            width=0.5,
            showlegend=False,
            hoverinfo='none',
        ))

        fig.update_layout(
            barmode='overlay',  # overlay bars on top of each other
            xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False),
            height=40,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',  # transparent
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**XP change this run:**")
        if xp_change >= 0:
            st.success(f"+{xp_change} XP")
        else:
            st.error(f"{xp_change} XP")

    st.markdown("---")

    st.subheader("Balance Projection (historic + next 3 months)")
    fig = apply_plotly_theme(go.Figure())
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_balances, mode='lines+markers', name='Historic'))
    fig.add_trace(go.Scatter(x=proj_dates, y=proj_balances, mode='lines+markers', name='Projected',
                             line=dict(dash='dash')))
    fig.update_layout(
        title='Account Balance Projection',
        xaxis_title='Month',
        yaxis_title='Balance (EUR)',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("💡 Personalized Tips")
    for i, ttext in enumerate(tips, start=1):
        st.markdown(f"**{i}.** {ttext}")

    st.markdown("---")

    st.subheader("Monthly Net Flows")
    monthly_display = monthly.copy()
    monthly_display['month'] = monthly_display['month_start'].dt.strftime("%Y-%m")
    monthly_display = monthly_display[['month', 'amount']]
    styled_monthly = monthly_display.style.format({"amount": "{:,.2f}"}).set_table_styles(
        [
            {"selector": "tr:hover", "props": [("background-color", "#f0f0f0")]},
            {"selector": "th", "props": [("background-color", "#4F46E5"), ("color", "white")]},
        ]
    )
    st.dataframe(styled_monthly, use_container_width=True)

    if SKLEARN_AVAILABLE and clf is not None:
        with st.expander("🔍 Show Decision Tree Rules"):
            try:
                tree_text = export_text(clf, feature_names=["savings_rate", "spend_income_ratio", "pct_food", "pct_entertainment"])
                st.code(tree_text)
            except Exception:
                st.info("Couldn't show tree text representation.")

    st.success("Thanks for trying FinRPG — improve habits to earn XP and level up!")

if __name__ == "__main__":
    run_app()