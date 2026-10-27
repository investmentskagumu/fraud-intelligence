# ============================================================
# FRAUD INTELLIGENCE 001 — STAGE 27 INTERACTIVE DASHBOARD
# Run from the project root: python scripts/27_dashboard.py
# ============================================================
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
METRICS = ROOT / 'outputs' / 'metrics'
OUTDIR = ROOT / 'outputs' / 'dashboard'
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / 'fraud_intelligence_dashboard.html'


def load_csv(path):
    try:
        return pd.read_csv(path) if path and path.exists() else pd.DataFrame()
    except Exception as exc:
        print(f'WARNING: Could not read {path}: {exc}')
        return pd.DataFrame()


def first_col(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def num(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def table(df, rows=30):
    if df.empty:
        return '<p class="muted">No saved output was found for this section.</p>'
    return df.head(rows).to_html(index=False, classes='data-table', border=0, na_rep='—', float_format=lambda x: f'{x:.4f}')


def find_dataset():
    candidates = [
        ROOT / 'data/raw/Synthetic_Financial_datasets_log.csv',
        ROOT / 'data/processed/Synthetic_Financial_datasets_log.csv',
        ROOT / 'Synthetic_Financial_datasets_log.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    for base in [ROOT / 'data', ROOT]:
        if base.exists():
            for p in base.glob('**/*Synthetic*Financial*datasets*log*.csv'):
                if '.venv' not in p.parts and 'site-packages' not in p.parts:
                    return p
    return None


model_df = load_csv(METRICS / 'stage_26_final_model_summary.csv')
ranking_df = load_csv(METRICS / 'stage_26_final_model_selection.csv')
risk_df = load_csv(METRICS / 'stage_22_risk_summary.csv')
queue_df = load_csv(METRICS / 'stage_23_investigation_summary.csv')
temporal_df = load_csv(METRICS / 'stage_24_temporal_validation.csv')
stress_df = load_csv(METRICS / 'stage_25_stress_testing.csv')
dataset_path = find_dataset()
df = load_csv(dataset_path)

# Dataset KPIs and fraud-pattern charts
has_data = not df.empty
total = len(df) if has_data else 0
fraud = int((df['isFraud'] == 1).sum()) if has_data and 'isFraud' in df else 0
legitimate = int((df['isFraud'] == 0).sum()) if has_data and 'isFraud' in df else 0
fraud_rate = fraud / total if total else 0
flagged = int((df['isFlaggedFraud'] == 1).sum()) if has_data and 'isFlaggedFraud' in df else 0
fraud_type, rate_type, volume_type, fraud_steps, amount_bands = [], [], [], [], []
if has_data and 'type' in df:
    by_type = df.groupby('type', dropna=False)
    volume_type = [{'type': str(k), 'count': int(v)} for k, v in by_type.size().items()]
    if 'isFraud' in df:
        fraud_type = [{'type': str(k), 'count': int(v)} for k, v in by_type['isFraud'].sum().items()]
        rate_type = [{'type': str(k), 'rate': float(v * 100)} for k, v in by_type['isFraud'].mean().items()]
if has_data and {'step', 'isFraud'}.issubset(df.columns):
    tmp = df.groupby('step')['isFraud'].agg(['sum', 'size']).reset_index()
    if len(tmp) > 200:
        tmp = tmp.iloc[np.linspace(0, len(tmp)-1, 200).astype(int)]
    fraud_steps = [{'step': int(r.step), 'fraud': int(r['sum'])} for _, r in tmp.iterrows()]
if has_data and {'amount', 'isFraud'}.issubset(df.columns):
    amounts = pd.to_numeric(df['amount'], errors='coerce')
    positive = amounts[amounts > 0]
    if not positive.empty:
        bins = np.unique(np.quantile(positive, np.linspace(0, 1, 11)))
        if len(bins) > 1:
            tmp = pd.DataFrame({'amount': amounts.loc[positive.index], 'isFraud': df.loc[positive.index, 'isFraud']})
            tmp['band'] = pd.cut(tmp['amount'], bins=bins, include_lowest=True, duplicates='drop')
            g = tmp.groupby('band', observed=False)['isFraud'].agg(['sum', 'size']).reset_index()
            amount_bands = [{'band': str(r.band), 'fraud': int(r['sum']), 'legitimate': int(r['size']-r['sum'])} for _, r in g.iterrows()]

# Selected-model summary
final_model = 'N/A'; precision = recall = f1 = pr_auc = roc_auc = 0.0; threshold = None
if not model_df.empty:
    r = model_df.iloc[0]
    final_model = str(r.get('Final_Model', final_model))
    precision = num(r.get('Precision', 0)); recall = num(r.get('Recall', 0))
    f1 = num(r.get('F1', r.get('F1_Score', 0)))
    pr_auc = num(r.get('PR_AUC', 0)); roc_auc = num(r.get('ROC_AUC', 0))
    t = num(r.get('Decision_Threshold', np.nan), np.nan)
    threshold = t if np.isfinite(t) else None

# Model comparison chart data, supporting known column variants
model_chart = []
if not ranking_df.empty:
    cols = {
        'model': first_col(ranking_df, ['model', 'Model', 'model_name']),
        'precision': first_col(ranking_df, ['precision', 'Precision']),
        'recall': first_col(ranking_df, ['recall', 'Recall']),
        'f1': first_col(ranking_df, ['f1_score', 'f1', 'F1', 'F1_Score']),
        'pr_auc': first_col(ranking_df, ['pr_auc', 'PR_AUC', 'PR-AUC']),
        'roc_auc': first_col(ranking_df, ['roc_auc', 'ROC_AUC', 'ROC-AUC']),
    }
    if cols['model']:
        for _, r in ranking_df.iterrows():
            model_chart.append({k: (str(r[c]) if k == 'model' else num(r[c]) if c else 0) for k, c in cols.items()})

# Optional Stage 21 importance output
importance = []
for p in sorted((ROOT / 'outputs').glob('**/*importance*.csv')) if (ROOT / 'outputs').exists() else []:
    if '21' in p.name or 'stage_21' in str(p).lower():
        imp = load_csv(p)
        fc = first_col(imp, ['feature', 'Feature', 'feature_name', 'Feature_Name'])
        ic = first_col(imp, ['importance', 'Importance', 'permutation_importance', 'importance_mean', 'Importance_Mean'])
        if fc and ic:
            imp[ic] = pd.to_numeric(imp[ic], errors='coerce')
            imp = imp.dropna(subset=[ic]).sort_values(ic, ascending=False).head(12)
            importance = [{'feature': str(r[fc]), 'value': num(r[ic])} for _, r in imp.iterrows()]
            break

# Convert pandas records into safe JSON (NaN values become null)
def records(df):
    if df.empty: return []
    return json.loads(df.replace({np.nan: None}).to_json(orient='records'))

chart_data = json.dumps({
    'fraudType': fraud_type, 'rateType': rate_type, 'volumeType': volume_type,
    'fraudSteps': fraud_steps, 'amountBands': amount_bands, 'models': model_chart,
    'risk': records(risk_df), 'temporal': records(temporal_df),
    'stress': records(stress_df), 'importance': importance
}).replace('</', '<\\/')

ranking_html = table(ranking_df, 25)
risk_html = table(risk_df)
queue_html = table(queue_df)
temporal_html = table(temporal_df)
stress_html = table(stress_df)
dataset_label = html.escape(str(dataset_path.relative_to(ROOT)) if dataset_path else 'Dataset not found')
threshold_label = f'{threshold:.2f}' if threshold is not None else 'Not validated'

page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FRAUD INTELLIGENCE 001 — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{{--bg:#0b1220;--panel:#111c2e;--panel2:#17243a;--text:#e5edf7;--muted:#9caec4;--line:#26364e;--blue:#38bdf8;--red:#fb7185;--green:#34d399;--gold:#fbbf24;--purple:#a78bfa}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Segoe UI,Arial,sans-serif}}header{{padding:28px 4vw;background:linear-gradient(120deg,#101a2d,#172d49);border-bottom:1px solid var(--line)}}header h1{{margin:0;font-size:clamp(24px,3vw,36px)}}header p,.muted{{color:var(--muted)}}main{{width:94%;max-width:1600px;margin:24px auto 40px}}.section{{margin:30px 0}}h2{{font-size:21px;border-left:4px solid var(--blue);padding-left:12px}}h3{{margin-top:2px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px}}.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:0 8px 24px #0002}}.card{{padding:18px}}.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}}.value{{font-size:clamp(22px,2.4vw,30px);font-weight:700;margin-top:10px;overflow-wrap:anywhere}}.panel{{padding:18px;min-width:0}}.grid2{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.chart{{height:350px;width:100%}}.tall{{height:420px}}.table-wrap{{overflow:auto;max-height:500px}}.data-table{{width:100%;border-collapse:collapse;font-size:13px}}.data-table th,.data-table td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}.data-table th{{background:var(--panel2);position:sticky;top:0}}.notice{{background:#332a16;border:1px solid #6c5523;color:#f6dda0;padding:13px;border-radius:10px;line-height:1.5}}.note-good{{background:#102b27;border:1px solid #1e6254;color:#b7f7df;padding:13px;border-radius:10px;line-height:1.6}}footer{{text-align:center;color:var(--muted);padding:24px;border-top:1px solid var(--line);font-size:12px}}@media(max-width:850px){{.grid2{{grid-template-columns:1fr}}.chart{{height:320px}}}}
</style></head><body>
<header><h1>FRAUD INTELLIGENCE 001</h1><p>Fraud Detection · Risk Intelligence · Model Evaluation</p></header><main>
<section class="section"><h2>Executive Summary</h2><div class="cards">
<div class="card"><div class="label">Transactions analysed</div><div class="value">{total:,}</div><p class="muted">{dataset_label}</p></div>
<div class="card"><div class="label">Fraud transactions</div><div class="value">{fraud:,}</div></div>
<div class="card"><div class="label">Fraud prevalence</div><div class="value">{fraud_rate:.3%}</div></div>
<div class="card"><div class="label">Final selected model</div><div class="value">{html.escape(final_model)}</div></div>
<div class="card"><div class="label">PR-AUC</div><div class="value">{pr_auc:.4f}</div></div>
<div class="card"><div class="label">Fraud recall</div><div class="value">{recall:.2%}</div></div>
</div></section>
<section class="section"><h2>Fraud Analysis</h2><p class="muted">These charts are calculated from the dataset found by this script.</p><div class="grid2">
<div class="panel"><h3>Fraud Transactions by Type</h3><div id="fraudType" class="chart"></div></div>
<div class="panel"><h3>Fraud Rate by Transaction Type</h3><div id="rateType" class="chart"></div></div>
<div class="panel"><h3>Transaction Volume by Type</h3><div id="volumeType" class="chart"></div></div>
<div class="panel"><h3>Fraud Across Simulation Steps</h3><div id="fraudSteps" class="chart"></div></div>
<div class="panel"><h3>Fraud vs Legitimate by Amount Band</h3><div id="amountBands" class="chart tall"></div></div>
</div></section>
<section class="section"><h2>Model Performance</h2><div class="cards">
<div class="card"><div class="label">Precision</div><div class="value">{precision:.2%}</div></div><div class="card"><div class="label">Recall</div><div class="value">{recall:.2%}</div></div><div class="card"><div class="label">F1-score</div><div class="value">{f1:.4f}</div></div><div class="card"><div class="label">ROC-AUC</div><div class="value">{roc_auc:.4f}</div></div><div class="card"><div class="label">Decision threshold</div><div class="value">{threshold_label}</div><p class="muted">Validate for the selected model.</p></div>
</div><div class="grid2" style="margin-top:16px"><div class="panel"><h3>Model Comparison — PR-AUC</h3><div id="modelPRAUC" class="chart tall"></div></div><div class="panel"><h3>Model Comparison — Precision and Recall</h3><div id="modelPR" class="chart tall"></div></div></div><div class="panel" style="margin-top:16px"><h3>Model Ranking Table</h3><div class="table-wrap">{ranking_html}</div></div>
<div class="notice" style="margin-top:16px">Important consistency note: the Stage 26 selected model is XGBoost, but earlier risk scores and the threshold analysis were generated using Logistic Regression. Those risk outputs must not be described as XGBoost results until regenerated from XGBoost predictions.</div></section>
<section class="section"><h2>Risk Intelligence</h2><div class="grid2"><div class="panel"><h3>Risk Summary Chart</h3><div id="riskChart" class="chart"></div><div class="table-wrap">{risk_html}</div></div><div class="panel"><h3>Stage 21 Feature Importance</h3><div id="importanceChart" class="chart tall"></div><p class="muted">If no feature-importance CSV was found, check the filename and column names from Stage 21.</p></div></div><div class="panel" style="margin-top:16px"><h3>Investigation Queue Summary</h3><div class="table-wrap">{queue_html}</div></div></section>
<section class="section"><h2>Temporal Validation & Stress Testing</h2><div class="grid2"><div class="panel"><h3>Temporal Validation Chart</h3><div id="temporalChart" class="chart"></div><div class="table-wrap">{temporal_html}</div></div><div class="panel"><h3>Stress Test Chart</h3><div id="stressChart" class="chart"></div><div class="table-wrap">{stress_html}</div></div></div><p class="notice" style="margin-top:16px">Stress tests are scenario experiments, not external validation. Check the decision threshold used in each stage before comparing model performance values.</p></section>
<section class="section"><h2>Data and Method Notes</h2><div class="note-good"><b>Transactions:</b> {total:,}<br><b>Legitimate transactions:</b> {legitimate:,}<br><b>Fraud transactions:</b> {fraud:,}<br><b>isFlaggedFraud = 1:</b> {flagged:,}<br><b>Leakage controls:</b> Never use <code>isFraud</code> as a predictor. Balance columns are excluded from the planned final predictive feature set. Account-behaviour features should be checked for prediction-time leakage.</div></section>
</main><footer>FRAUD INTELLIGENCE 001 · Academic Research + Data Analytics + Machine Learning</footer>
<script>
const D={chart_data};const blue='#38bdf8',red='#fb7185',green='#34d399',gold='#fbbf24',purple='#a78bfa';
const base={{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:'#dbe7f5'}},margin:{{l:55,r:18,t:15,b:70}},xaxis:{{gridcolor:'#26364e'}},yaxis:{{gridcolor:'#26364e'}},legend:{{orientation:'h',y:-.25}}}};
function draw(id,traces,extra={{}}){{const el=document.getElementById(id);if(!traces.length||!traces[0].x||!traces[0].x.length){{el.innerHTML='<p class="muted">No chart data available.</p>';return}}Plotly.newPlot(el,traces,Object.assign({{}},base,extra),{{responsive:true,displaylogo:false}})}}
draw('fraudType',[{{x:D.fraudType.map(d=>d.type),y:D.fraudType.map(d=>d.count),type:'bar',marker:{{color:red}}}}],{{yaxis:{{title:'Fraud transactions'}}}});
draw('rateType',[{{x:D.rateType.map(d=>d.type),y:D.rateType.map(d=>d.rate),type:'bar',marker:{{color:gold}}}}],{{yaxis:{{title:'Fraud rate (%)'}}}});
draw('volumeType',[{{x:D.volumeType.map(d=>d.type),y:D.volumeType.map(d=>d.count),type:'bar',marker:{{color:blue}}}}],{{yaxis:{{title:'Transactions'}}}});
draw('fraudSteps',[{{x:D.fraudSteps.map(d=>d.step),y:D.fraudSteps.map(d=>d.fraud),type:'scatter',mode:'lines+markers',line:{{color:red}}}}],{{xaxis:{{title:'Simulation step'}},yaxis:{{title:'Fraud transactions'}}}});
draw('amountBands',[{{x:D.amountBands.map(d=>d.band),y:D.amountBands.map(d=>d.legitimate),name:'Legitimate',type:'bar',marker:{{color:blue}}}},{{x:D.amountBands.map(d=>d.band),y:D.amountBands.map(d=>d.fraud),name:'Fraud',type:'bar',marker:{{color:red}}}}],{{barmode:'group',xaxis:{{tickangle:-25}},yaxis:{{title:'Transactions'}}}});
draw('modelPRAUC',[{{x:D.models.map(d=>d.model),y:D.models.map(d=>d.pr_auc),type:'bar',marker:{{color:purple}}}}],{{xaxis:{{tickangle:-30}},yaxis:{{title:'PR-AUC',range:[0,1]}}}});
draw('modelPR',[{{x:D.models.map(d=>d.model),y:D.models.map(d=>d.precision),name:'Precision',type:'bar',marker:{{color:blue}}}},{{x:D.models.map(d=>d.model),y:D.models.map(d=>d.recall),name:'Recall',type:'bar',marker:{{color:green}}}}],{{barmode:'group',xaxis:{{tickangle:-30}},yaxis:{{title:'Score',range:[0,1]}}}});
if(D.risk.length){{const r=D.risk[0];const kc=['Risk_Level','risk_level','Risk Category','risk_category','Category','category','Risk_Band'].find(k=>r[k]!==undefined);const cc=['Count','count','Transactions','transactions','Transaction_Count','transaction_count','Total'].find(k=>r[k]!==undefined);if(kc&&cc)Plotly.newPlot('riskChart',[{{labels:D.risk.map(x=>String(x[kc])),values:D.risk.map(x=>Number(x[cc])||0),type:'pie',hole:.4}}],Object.assign({{}},base,{{margin:{{l:5,r:5,t:5,b:5}}}}),{{responsive:true,displaylogo:false}});else document.getElementById('riskChart').innerHTML='<p class="muted">Risk table is available; columns need mapping for this chart.</p>'}}else document.getElementById('riskChart').innerHTML='<p class="muted">No risk summary output found.</p>';
draw('importanceChart',[{{x:D.importance.map(d=>d.value),y:D.importance.map(d=>d.feature),type:'bar',orientation:'h',marker:{{color:purple}}}}],{{margin:{{l:160,r:20,t:15,b:40}},yaxis:{{autorange:'reversed'}},xaxis:{{title:'Importance'}}}});
const T=D.temporal;if(T.length){{const p=['period','Period','validation_period','Validation_Period'].find(k=>T[0][k]!==undefined);const rc=['recall','Recall'].find(k=>T[0][k]!==undefined);const fc=['f1','F1','f1_score','F1_Score'].find(k=>T[0][k]!==undefined);if(p&&rc&&fc)draw('temporalChart',[{{x:T.map(r=>String(r[p])),y:T.map(r=>Number(r[rc])),name:'Recall',type:'scatter',mode:'lines+markers',line:{{color:green}}}},{{x:T.map(r=>String(r[p])),y:T.map(r=>Number(r[fc])),name:'F1-score',type:'scatter',mode:'lines+markers',line:{{color:gold}}}}],{{yaxis:{{title:'Score',range:[0,1]}}}});else document.getElementById('temporalChart').innerHTML='<p class="muted">Temporal data is available in the table; columns need mapping for this chart.</p>'}}else document.getElementById('temporalChart').innerHTML='<p class="muted">No temporal output found.</p>';
const S=D.stress;if(S.length){{const sc=['scenario','Scenario','test','Test','stress_test','Stress_Test','condition','Condition'].find(k=>S[0][k]!==undefined);const pc=['pr_auc','PR_AUC','PR-AUC'].find(k=>S[0][k]!==undefined);if(sc&&pc)draw('stressChart',[{{x:S.map(r=>String(r[sc])),y:S.map(r=>Number(r[pc])),type:'bar',marker:{{color:gold}}}}],{{xaxis:{{tickangle:-25}},yaxis:{{title:'PR-AUC',range:[0,1]}}}});else document.getElementById('stressChart').innerHTML='<p class="muted">Stress data is available in the table; columns need mapping for this chart.</p>'}}else document.getElementById('stressChart').innerHTML='<p class="muted">No stress-test output found.</p>';
</script></body></html>'''
OUTFILE.write_text(page, encoding='utf-8')
print('\n==============================================')
print('STAGE 27 — INTERACTIVE GRAPHICAL DASHBOARD')
print('==============================================')
print(f'Dataset found: {dataset_path.relative_to(ROOT) if dataset_path else "NO"}')
print(f'Transactions loaded: {total:,}')
print(f'Fraud transactions: {fraud:,}')
print(f'Fraud rate: {fraud_rate:.3%}')
print(f'Final model in summary: {final_model}')
print(f'Dashboard created: {OUTFILE.relative_to(ROOT)}')
print('Note: Interactive Plotly charts need an internet connection when opening the HTML.')
print('STAGE 27 COMPLETE')
