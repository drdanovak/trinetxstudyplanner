# TriNetX Study Planner – Streamlit App (single-file)
# --------------------------------------------------
# This Streamlit app guides students through planning a retrospective, EHR-based
# study suitable for execution on platforms like TriNetX. It scaffolds decisions
# through a step-by-step wizard, captures structured inputs (PICO, cohorts,
# codes, windows, covariates, analysis choices), and exports a clean Markdown
# or JSON study plan. It also renders a simple cohort flow diagram.
#
# How to run:
#   1) Install:  pip install streamlit pandas numpy pydantic python-dateutil
#   2) Run:      streamlit run app.py
#   3) Your browser will open at http://localhost:8501
#
# Notes:
# - The app doesn’t connect to TriNetX; it prepares a plan you can implement there.
# - It avoids platform-specific details that may change over time.
# - Students can save/load plans via JSON and download a Markdown summary.

from __future__ import annotations

import json
import io
from datetime import date
from textwrap import dedent
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Constants & Helper Structures
# -----------------------------
APP_TITLE = "TriNetX Study Planner"
APP_TAGLINE = "A step-by-step wizard to design retrospective EHR studies"

COMPARISON_TYPES = {
    "Active comparator (Drug A vs Drug B)": {
        "desc": "Compare two active treatments to reduce confounding by indication.",
        "requires": ["exposureA", "exposureB"]
    },
    "Exposed vs Unexposed": {
        "desc": "Compare patients exposed to a drug/procedure/condition vs a matched control without exposure.",
        "requires": ["exposureA"]
    },
    "Condition vs General Population": {
        "desc": "Compare those with an index condition vs those without it.",
        "requires": ["indexCondition"]
    },
    "Dose/Intensity (High vs Low)": {
        "desc": "Compare different dose bands or intensity tiers of the same exposure.",
        "requires": ["exposureA", "doseBands"]
    },
    "Time-based self-controlled (Pre vs Post)": {
        "desc": "Within-person design comparing a pre-exposure window to a post-exposure window.",
        "requires": ["exposureA", "timeWindows"]
    }
}

CODE_SYSTEMS = [
    "ICD-10-CM (diagnoses)",
    "ICD-10-PCS (procedures)",
    "CPT (procedures)",
    "HCPCS (procedures)",
    "LOINC (labs)",
    "RxNorm (medications)",
    "ATC (medication class)",
]

COVARIATE_CATEGORIES = [
    "Demographics (age, sex)",
    "Race/Ethnicity",
    "Comorbidities (CCI, Elixhauser)",
    "Medication use (polypharmacy)",
    "Healthcare utilization (visits, admissions)",
    "Baseline labs (e.g., A1c, LDL)",
    "Lifestyle proxies (BMI, smoking codes)",
    "Socioeconomic proxies (insurance type)",
]

ANALYSIS_METHODS = [
    "Risk ratio / risk difference",
    "Odds ratio",
    "Time-to-event (KM / Cox)",
]

BALANCING_METHODS = [
    "Propensity score matching",
    "Propensity score weighting",
    "Stratification by propensity score",
    "Regression adjustment only",
]

FOLLOWUP_CHOICES = [
    "Fixed window (e.g., 365 days)",
    "Until outcome, death, or loss to follow-up",
    "Variable window by exposure duration",
]

# Minimal templates to help students get started quickly
TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Statins vs No Statin (AD outcomes)": {
        "objective": "Estimate whether statin exposure is associated with reduced risk of Alzheimer's disease (AD).",
        "comparison": "Exposed vs Unexposed",
        "exposures_df": [
            {"Domain": "Medication", "Code System": "RxNorm (medications)", "Code": "617314", "Description": "Atorvastatin", "Role": "Exposure", "Include?": True, "Window Start (days)": -90, "Window End (days)": 0},
            {"Domain": "Medication", "Code System": "RxNorm (medications)", "Code": "617318", "Description": "Rosuvastatin", "Role": "Exposure", "Include?": True, "Window Start (days)": -90, "Window End (days)": 0},
        ],
        "outcomes_df": [
            {"Outcome Type": "Primary", "Code System": "ICD-10-CM (diagnoses)", "Code": "G30%", "Description": "Alzheimer's disease (any)", "Time-at-risk Start (days)": 1, "Time-at-risk End (days)": 1825},
        ],
        "covariates": ["Demographics (age, sex)", "Race/Ethnicity", "Comorbidities (CCI, Elixhauser)", "Baseline labs (e.g., A1c, LDL)"]
    },
    "GLP-1 vs Sulfonylurea (CVD & cognitive outcomes)": {
        "objective": "Compare GLP-1 receptor agonists to sulfonylureas on cardiovascular and cognitive outcomes in T2D.",
        "comparison": "Active comparator (Drug A vs Drug B)",
        "exposures_df": [
            {"Domain": "Medication", "Code System": "RxNorm (medications)", "Code": "111"
             , "Description": "GLP-1 RA class (placeholder)", "Role": "Exposure A", "Include?": True, "Window Start (days)": -90, "Window End (days)": 0},
            {"Domain": "Medication", "Code System": "RxNorm (medications)", "Code": "222"
             , "Description": "Sulfonylurea class (placeholder)", "Role": "Exposure B", "Include?": True, "Window Start (days)": -90, "Window End (days)": 0},
        ],
        "outcomes_df": [
            {"Outcome Type": "Primary", "Code System": "ICD-10-CM (diagnoses)", "Code": "I63%", "Description": "Ischemic stroke", "Time-at-risk Start (days)": 1, "Time-at-risk End (days)": 1460},
            {"Outcome Type": "Secondary", "Code System": "ICD-10-CM (diagnoses)", "Code": "G30%", "Description": "Alzheimer's disease (any)", "Time-at-risk Start (days)": 1, "Time-at-risk End (days)": 1825},
        ],
        "covariates": ["Demographics (age, sex)", "Comorbidities (CCI, Elixhauser)", "Baseline labs (e.g., A1c, LDL)"]
    }
}

# ----------------------
# Session State Handling
# ----------------------
DEFAULT_COLUMNS_EXPOSURES = [
    "Domain", "Code System", "Code", "Description", "Role", "Include?", "Window Start (days)", "Window End (days)"
]
DEFAULT_COLUMNS_OUTCOMES = [
    "Outcome Type", "Code System", "Code", "Description", "Time-at-risk Start (days)", "Time-at-risk End (days)"
]
DEFAULT_COLUMNS_COVARS = [
    "Category", "Detail"
]

EMPTY_EXPOSURES_DF = pd.DataFrame([{c: (True if c == "Include?" else "") for c in DEFAULT_COLUMNS_EXPOSURES}]).astype({"Include?": bool})
EMPTY_OUTCOMES_DF = pd.DataFrame([{c: "" for c in DEFAULT_COLUMNS_OUTCOMES}])
EMPTY_COVARS_DF = pd.DataFrame([{c: "" for c in DEFAULT_COLUMNS_COVARS}])


def init_state():
    if "plan" not in st.session_state:
        st.session_state.plan = {
            "meta": {
                "title": "",
                "pi": "",
                "team": "",
                "date": str(date.today()),
                "domain": "",
            },
            "research_question": "",
            "pico": {
                "population": "",
                "intervention": "",
                "comparator": "",
                "outcome": "",
            },
            "comparison_type": "",
            "exposures_df": EMPTY_EXPOSURES_DF.copy(),
            "outcomes_df": EMPTY_OUTCOMES_DF.copy(),
            "cohort_criteria": {
                "age_min": 18,
                "age_max": 120,
                "sex": [],
                "index_date_start": "",
                "index_date_end": "",
                "washout_days": 365,
                "min_history_days": 365,
                "site_inclusion": "Any site",
                "notes": "",
            },
            "covariates": [],
            "covariates_df": EMPTY_COVARS_DF.copy(),
            "balancing": {
                "method": "Propensity score matching",
                "match_ratio": 1,
                "caliper": "(platform default)",
                "replacement": False,
                "post_match_checks": ["Standardized differences < 0.1", "Covariate balance plot"],
            },
            "analysis": {
                "methods": ["Risk ratio / risk difference"],
                "followup": "Fixed window (e.g., 365 days)",
                "followup_days": 365,
                "censoring": ["Outcome", "Death", "Loss to follow-up"],
                "multiple_testing": "Report per-outcome; consider FDR for many outcomes",
            },
            "subgroups": [],
            "sensitivity": ["Vary time-at-risk windows", "Alternate outcome definitions"],
            "feasibility": {
                "min_n_per_arm": 1000,
                "min_events": 50,
                "network_scope": "Global network",
                "lookback_required_days": 365,
            },
            "ethics": {
                "irb": "Consult local IRB; de-identified aggregate analysis",
                "data_governance": "Follow data use agreements and site policies",
            },
        }
    if "step" not in st.session_state:
        st.session_state.step = 1


# --------------
# UI Components
# --------------

def header():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)


def sidebar_nav():
    steps = [
        "Basics", "Comparison", "Exposure/Index", "Outcomes", "Cohort Criteria",
        "Covariates & Balancing", "Analysis Plan", "Subgroups & Sensitivity",
        "Feasibility & Ethics", "Preview & Export"
    ]
    total = len(steps)
    with st.sidebar:
        st.header("Plan Wizard")
        st.session_state.step = st.number_input(
            "Step",
            min_value=1,
            max_value=total,
            value=int(st.session_state.step),
            step=1,
            help="Use the arrows or type a number to jump between steps."
        )
        progress = st.session_state.step / total
        st.progress(progress, text=f"Step {st.session_state.step} of {total}")
        st.markdown("---")
        st.subheader("Quick Actions")
        template = st.selectbox("Load a template (optional)", ["—"] + list(TEMPLATES.keys()))
        if template != "—":
            apply_template(template)
            st.success(f"Loaded template: {template}")
        uploaded = st.file_uploader("Load saved plan (JSON)", type=["json"])
        if uploaded is not None:
            load_plan(uploaded)
            st.success("Plan loaded from JSON.")
        st.markdown("---")
        st.markdown("""
        **Tips**
        - Keep code lists broad initially (use wildcards like `G30%` responsibly).
        - Define a clear index date and washout period.
        - Choose an appropriate comparator to minimize bias.
        - Ensure adequate events and follow-up.
        """)


def apply_template(name: str):
    t = TEMPLATES[name]
    st.session_state.plan["research_question"] = t["objective"]
    st.session_state.plan["comparison_type"] = t["comparison"]
    st.session_state.plan["exposures_df"] = pd.DataFrame(t["exposures_df"]) if t.get("exposures_df") else EMPTY_EXPOSURES_DF.copy()
    st.session_state.plan["outcomes_df"] = pd.DataFrame(t["outcomes_df"]) if t.get("outcomes_df") else EMPTY_OUTCOMES_DF.copy()
    st.session_state.plan["covariates"] = t.get("covariates", [])


# ------------------
# Step Renderers
# ------------------

def step_basics():
    st.subheader("1) Project Basics")
    col1, col2, col3, col4 = st.columns([2,2,2,2])
    with col1:
        st.session_state.plan["meta"]["title"] = st.text_input("Project title", st.session_state.plan["meta"]["title"])    
    with col2:
        st.session_state.plan["meta"]["pi"] = st.text_input("PI / Lead", st.session_state.plan["meta"]["pi"])
    with col3:
        st.session_state.plan["meta"]["team"] = st.text_input("Team (comma-separated)", st.session_state.plan["meta"]["team"])
    with col4:
        st.session_state.plan["meta"]["domain"] = st.text_input("Clinical domain (e.g., neurology)", st.session_state.plan["meta"]["domain"])    

    st.text_area("Research question (one sentence)", key="rq", value=st.session_state.plan["research_question"],
                 on_change=lambda: st.session_state.plan.update({"research_question": st.session_state.rq}))

    st.markdown("**PICO Elements**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.plan["pico"]["population"] = st.text_input("Population", st.session_state.plan["pico"]["population"])    
    with c2:
        st.session_state.plan["pico"]["intervention"] = st.text_input("Exposure/Intervention", st.session_state.plan["pico"]["intervention"])    
    with c3:
        st.session_state.plan["pico"]["comparator"] = st.text_input("Comparator", st.session_state.plan["pico"]["comparator"])    
    with c4:
        st.session_state.plan["pico"]["outcome"] = st.text_input("Outcome", st.session_state.plan["pico"]["outcome"])    



def step_comparison():
    st.subheader("2) Comparison Framework")
    cols = st.columns([2,3])
    with cols[0]:
        choice = st.selectbox("What kind of comparison are you trying to do?", list(COMPARISON_TYPES.keys()),
                              index=(list(COMPARISON_TYPES.keys()).index(st.session_state.plan["comparison_type"]) if st.session_state.plan["comparison_type"] in COMPARISON_TYPES else 0))
        st.session_state.plan["comparison_type"] = choice
    with cols[1]:
        st.info(COMPARISON_TYPES[choice]["desc"])    

    # Contextual prompts
    req = COMPARISON_TYPES[choice]["requires"]
    if "exposureA" in req:
        st.text_input("Define Exposure A (label)", key="label_exposure_a")
    if "exposureB" in req:
        st.text_input("Define Exposure B (label)", key="label_exposure_b")
    if "indexCondition" in req:
        st.text_input("Define the index condition (label)", key="label_index_condition")
    if "doseBands" in req:
        st.text_input("How will you define dose bands? (e.g., >40mg as 'high')", key="dose_bands_desc")
    if "timeWindows" in req:
        st.text_input("Specify pre/post windows (days)", value="-180 to -1 (pre), +1 to +180 (post)", key="time_windows_desc")



def step_exposure():
    st.subheader("3) Exposure / Index Definitions")
    st.caption("Add the structured codes that define your exposure(s) or index condition. Use broad-to-specific code strategies as appropriate.")

    df = st.session_state.plan["exposures_df"]

    with st.expander("Guidance", expanded=False):
        st.markdown("""
        - **Domain**: Medication, Procedure, Condition, Lab, etc.
        - **Code System**: RxNorm for meds, ICD-10-CM for diagnoses, CPT/HCPCS for procedures, LOINC for labs.
        - **Role**: Exposure / Exposure A / Exposure B / Index Condition.
        - **Window**: Relative to the index date (e.g., -90 to 0 days before index for new-user designs).
        """)

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "Domain": st.column_config.SelectboxColumn(options=["Medication","Procedure","Condition","Lab"], required=True),
            "Code System": st.column_config.SelectboxColumn(options=CODE_SYSTEMS, required=True),
            "Code": st.column_config.TextColumn(help="Accepts wildcards like G30% if your platform supports it."),
            "Description": st.column_config.TextColumn(),
            "Role": st.column_config.SelectboxColumn(options=["Exposure","Exposure A","Exposure B","Index Condition"], required=True),
            "Include?": st.column_config.CheckboxColumn(default=True),
            "Window Start (days)": st.column_config.NumberColumn(),
            "Window End (days)": st.column_config.NumberColumn(),
        },
        use_container_width=True,
        hide_index=True,
    )
    st.session_state.plan["exposures_df"] = edited



def step_outcomes():
    st.subheader("4) Outcome Definitions")
    st.caption("Define primary/secondary outcomes and time-at-risk windows.")
    df = st.session_state.plan["outcomes_df"]

    with st.expander("Guidance", expanded=False):
        st.markdown("""
        - **Outcome Type**: Primary vs Secondary.
        - **Time-at-risk**: Start after index to avoid immortal time bias; choose a clinically meaningful end.
        - Consider multiple definitions (diagnosis codes + meds/procedures) to test robustness.
        """)

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "Outcome Type": st.column_config.SelectboxColumn(options=["Primary","Secondary"], required=True),
            "Code System": st.column_config.SelectboxColumn(options=CODE_SYSTEMS, required=True),
            "Code": st.column_config.TextColumn(),
            "Description": st.column_config.TextColumn(),
            "Time-at-risk Start (days)": st.column_config.NumberColumn(),
            "Time-at-risk End (days)": st.column_config.NumberColumn(),
        },
        use_container_width=True,
        hide_index=True,
    )
    st.session_state.plan["outcomes_df"] = edited



def step_cohort_criteria():
    st.subheader("5) Cohort Inclusion/Exclusion Criteria")
    cc = st.session_state.plan["cohort_criteria"]

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        cc["age_min"] = st.number_input("Min age", min_value=0, max_value=120, value=int(cc["age_min"]))
    with c2:
        cc["age_max"] = st.number_input("Max age", min_value=0, max_value=120, value=int(cc["age_max"]))
    with c3:
        cc["sex"] = st.multiselect("Sex (optional)", ["Male","Female","Other/Unknown"], default=cc.get("sex", []))

    d1, d2, d3, d4 = st.columns([1,1,1,1])
    with d1:
        cc["index_date_start"] = st.text_input("Index date start (YYYY-MM-DD)", cc["index_date_start"])    
    with d2:
        cc["index_date_end"] = st.text_input("Index date end (YYYY-MM-DD)", cc["index_date_end"])    
    with d3:
        cc["washout_days"] = st.number_input("Washout (days w/o outcome/exposure before index)", min_value=0, value=int(cc["washout_days"]))
    with d4:
        cc["min_history_days"] = st.number_input("Minimum prior history (days)", min_value=0, value=int(cc["min_history_days"]))

    cc["site_inclusion"] = st.text_input("Site/network constraints (optional)", cc["site_inclusion"])    
    cc["notes"] = st.text_area("Additional inclusion/exclusion notes", cc.get("notes",""))



def step_covariates_balancing():
    st.subheader("6) Covariates & Balancing")
    st.caption("Select high-level covariate domains and specify any additional details.")

    st.session_state.plan["covariates"] = st.multiselect(
        "Covariate domains",
        COVARIATE_CATEGORIES,
        default=st.session_state.plan.get("covariates", []),
    )

    st.markdown("**Optional: list specific covariates** (e.g., 'Hypertension (ICD-10 I10)', 'LDL lab value')")
    cov_df = st.session_state.plan["covariates_df"]
    cov_edited = st.data_editor(
        cov_df,
        num_rows="dynamic",
        column_config={
            "Category": st.column_config.SelectboxColumn(options=COVARIATE_CATEGORIES + ["Other"], required=False),
            "Detail": st.column_config.TextColumn(),
        },
        use_container_width=True,
        hide_index=True,
    )
    st.session_state.plan["covariates_df"] = cov_edited

    st.markdown("---")
    st.markdown("**Balancing / Control for Confounding**")
    bal = st.session_state.plan["balancing"]
    b1, b2, b3, b4 = st.columns([1,1,1,1])
    with b1:
        bal["method"] = st.selectbox("Method", BALANCING_METHODS, index=BALANCING_METHODS.index(bal["method"]))
    with b2:
        bal["match_ratio"] = st.number_input("Match ratio (Exp:Ctrl)", min_value=1, max_value=5, value=int(bal["match_ratio"]))
    with b3:
        bal["caliper"] = st.text_input("Caliper (if applicable)", value=str(bal["caliper"]))
    with b4:
        bal["replacement"] = st.checkbox("Matching with replacement", value=bool(bal["replacement"]))

    bal_checks = st.multiselect("Post-match checks", ["Standardized differences < 0.1", "Variance ratios", "Love plot", "Visual overlap of PS"], default=bal.get("post_match_checks", []))
    bal["post_match_checks"] = bal_checks



def step_analysis():
    st.subheader("7) Analysis Plan")
    an = st.session_state.plan["analysis"]

    a1, a2 = st.columns([1,1])
    with a1:
        an["methods"] = st.multiselect("Effect estimation methods", ANALYSIS_METHODS, default=an.get("methods", []))
        an["followup"] = st.selectbox("Follow-up definition", FOLLOWUP_CHOICES, index=FOLLOWUP_CHOICES.index(an["followup"]))
        if an["followup"] == "Fixed window (e.g., 365 days)":
            an["followup_days"] = st.number_input("Fixed follow-up length (days)", min_value=1, max_value=3650, value=int(an.get("followup_days", 365)))
    with a2:
        an["censoring"] = st.multiselect("Censoring rules", ["Outcome","Death","Loss to follow-up","End of data window"], default=an.get("censoring", []))
        an["multiple_testing"] = st.text_input("Multiple testing approach (if many outcomes)", an.get("multiple_testing",""))



def step_subgroups_sensitivity():
    st.subheader("8) Subgroups & Sensitivity Analyses")
    st.session_state.plan["subgroups"] = st.tags_input("Subgroups (e.g., age bands, sex, baseline risk)", value=st.session_state.plan.get("subgroups", []))
    st.session_state.plan["sensitivity"] = st.tags_input("Sensitivity analyses (free text)", value=st.session_state.plan.get("sensitivity", []))



def step_feasibility_ethics():
    st.subheader("9) Feasibility & Ethics")
    fe = st.session_state.plan["feasibility"]
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        fe["min_n_per_arm"] = st.number_input("Minimum n per arm", min_value=0, value=int(fe["min_n_per_arm"]))
    with e2:
        fe["min_events"] = st.number_input("Minimum outcome events", min_value=0, value=int(fe["min_events"]))
    with e3:
        fe["network_scope"] = st.selectbox("Network scope", ["Global network","Regional subset","Single institution"], index=["Global network","Regional subset","Single institution"].index(fe["network_scope"]))
    with e4:
        fe["lookback_required_days"] = st.number_input("Required lookback (days)", min_value=0, value=int(fe["lookback_required_days"]))

    et = st.session_state.plan["ethics"]
    et["irb"] = st.text_input("IRB / Ethics notes", et.get("irb",""))
    et["data_governance"] = st.text_input("Data governance notes", et.get("data_governance",""))


# ----------------------
# Preview, Export & Viz
# ----------------------

def build_flowchart(plan: Dict[str, Any]) -> str:
    """Return a Graphviz DOT string for a simple cohort flow."""
    comp = plan.get("comparison_type", "Comparison")
    title = plan["meta"].get("title") or "Study Plan"

    exposures = plan["exposures_df"][plan["exposures_df"]["Include?"].fillna(False)] if not plan["exposures_df"].empty else pd.DataFrame()

    exp_nodes = []
    for i, row in exposures.iterrows():
        role = str(row.get("Role", "Exposure")).replace("\"", "'")
        desc = str(row.get("Description", row.get("Code",""))).replace("\"", "'")
        exp_nodes.append(f'"{role}: {desc}"')

    outcomes = plan["outcomes_df"] if not plan["outcomes_df"].empty else pd.DataFrame()
    out_nodes = []
    for i, row in outcomes.iterrows():
        odesc = str(row.get("Description", row.get("Code","Outcome"))).replace("\"", "'")
        otype = row.get("Outcome Type", "Outcome")
        out_nodes.append(f'"{otype}: {odesc}"')

    exp_cluster = " | ".join(exp_nodes) if exp_nodes else "Exposure/Index"
    out_cluster = " | ".join(out_nodes) if out_nodes else "Outcomes"

    dot = f"""
    digraph G {{
      rankdir=LR;
      node [shape=box, style=rounded];
      label=\"{title}\\n({comp})\"; labelloc=t; fontsize=18;

      subgraph cluster_0 {{
        label=\"Cohort Definitions\"; style=dashed;
        exp [label=\"{exp_cluster}\", shape=box];
        crit [label=\"Inclusion/Exclusion\nAge {plan['cohort_criteria']['age_min']}-{plan['cohort_criteria']['age_max']}\nWashout {plan['cohort_criteria']['washout_days']}d\", shape=box];
      }}

      subgraph cluster_1 {{
        label=\"Analysis\"; style=dashed;
        bal [label=\"Balancing: {plan['balancing']['method']}\nChecks: {'; '.join(plan['balancing'].get('post_match_checks', []))}\"]; 
        meth [label=\"Methods: {'; '.join(plan['analysis'].get('methods', []))}\nFollow-up: {plan['analysis'].get('followup', '')}\"]; 
      }}

      outcomes [label=\"{out_cluster}\", shape=box];

      exp -> crit -> bal -> meth -> outcomes;
    }}
    """
    return dot


def plan_to_markdown(plan: Dict[str, Any]) -> str:
    meta = plan["meta"]
    pico = plan["pico"]

    exposures_md = "\n".join(
        [f"- **{r.Role}** | {r['Code System']} {r.Code} — {r.Description} (window {r['Window Start (days)']},{r['Window End (days)']}; include={bool(r['Include?'])})"
         for _, r in plan["exposures_df"].fillna("").iterrows() if str(r.get("Code","")) or str(r.get("Description",""))]
    ) or "- _None specified_"

    outcomes_md = "\n".join(
        [f"- **{r['Outcome Type']}** | {r['Code System']} {r.Code} — {r.Description} (TAR {r['Time-at-risk Start (days)']},{r['Time-at-risk End (days)']})"
         for _, r in plan["outcomes_df"].fillna("").iterrows() if str(r.get("Code","")) or str(r.get("Description",""))]
    ) or "- _None specified_"

    covars_list = plan.get("covariates", [])
    covars_md = "\n".join([f"- {c}" for c in covars_list]) or "- _None selected_"

    covars_df_md = "\n".join(
        [f"  - {r.Category}: {r.Detail}" for _, r in plan["covariates_df"].fillna("").iterrows() if str(r.get("Detail",""))]
    )

    md = dedent(f"""
    # {meta.get('title') or 'Untitled Study'}
    **PI:** {meta.get('pi','')}  
    **Team:** {meta.get('team','')}  
    **Date:** {meta.get('date','')}  
    **Clinical Domain:** {meta.get('domain','')}

    ---
    ## Research Question
    {plan.get('research_question','')}

    ## PICO
    - **Population:** {pico.get('population','')}
    - **Intervention/Exposure:** {pico.get('intervention','')}
    - **Comparator:** {pico.get('comparator','')}
    - **Outcome:** {pico.get('outcome','')}

    ## Comparison Framework
    {plan.get('comparison_type','')}

    ## Exposure / Index Definitions
    {exposures_md}

    ## Outcome Definitions
    {outcomes_md}

    ## Cohort Inclusion/Exclusion
    - Age: {plan['cohort_criteria']['age_min']}–{plan['cohort_criteria']['age_max']}
    - Sex: {', '.join(plan['cohort_criteria'].get('sex', [])) or 'Any'}
    - Index date range: {plan['cohort_criteria'].get('index_date_start') or '—'} to {plan['cohort_criteria'].get('index_date_end') or '—'}
    - Washout: {plan['cohort_criteria']['washout_days']} days  
    - Minimum prior history: {plan['cohort_criteria']['min_history_days']} days  
    - Site/network: {plan['cohort_criteria'].get('site_inclusion','Any site')}  
    - Notes: {plan['cohort_criteria'].get('notes','')}

    ## Covariates & Balancing
    **Covariate domains**\n{covars_md}
    {covars_df_md if covars_df_md else ''}

    **Balancing approach**  
    - Method: {plan['balancing']['method']}  
    - Match ratio: {plan['balancing']['match_ratio']}  
    - Caliper: {plan['balancing']['caliper']}  
    - Replacement: {plan['balancing']['replacement']}  
    - Post-match checks: {', '.join(plan['balancing'].get('post_match_checks', []))}

    ## Analysis Plan
    - Methods: {', '.join(plan['analysis'].get('methods', []))}  
    - Follow-up: {plan['analysis'].get('followup','')}  
    - Fixed follow-up (days): {plan['analysis'].get('followup_days','—')}  
    - Censoring: {', '.join(plan['analysis'].get('censoring', []))}  
    - Multiple testing: {plan['analysis'].get('multiple_testing','')}

    ## Subgroups
    {', '.join(plan.get('subgroups', [])) or '—'}

    ## Sensitivity Analyses
    {', '.join(plan.get('sensitivity', [])) or '—'}

    ## Feasibility & Data Requirements
    - Minimum n per arm: {plan['feasibility']['min_n_per_arm']}  
    - Minimum events: {plan['feasibility']['min_events']}  
    - Network scope: {plan['feasibility']['network_scope']}  
    - Required lookback: {plan['feasibility']['lookback_required_days']} days

    ## Ethics & Governance
    - IRB: {plan['ethics'].get('irb','')}  
    - Data governance: {plan['ethics'].get('data_governance','')}
    """)
    return md


def step_preview_export():
    st.subheader("10) Preview & Export")

    # Quick health checks
    issues = []
    if not st.session_state.plan["comparison_type"]:
        issues.append("Select a comparison framework.")
    if st.session_state.plan["exposures_df"].dropna(how="all").empty:
        issues.append("Define at least one exposure/index row.")
    if st.session_state.plan["outcomes_df"].dropna(how="all").empty:
        issues.append("Define at least one outcome.")

    if issues:
        st.warning("\n".join([f"• {i}" for i in issues]))

    # Flowchart
    with st.expander("Study flow diagram", expanded=True):
        dot = build_flowchart(st.session_state.plan)
        st.graphviz_chart(dot, use_container_width=True)

    # Summary markdown
    md = plan_to_markdown(st.session_state.plan)
    st.markdown(md)

    # Downloads
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "Download Markdown",
            data=md.encode("utf-8"),
            file_name="study_plan.md",
            mime="text/markdown",
        )
    with colB:
        json_bytes = json.dumps(serialize_plan(st.session_state.plan), indent=2).encode("utf-8")
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="study_plan.json",
            mime="application/json",
        )

    st.caption("You can re-upload the JSON in the sidebar to continue later.")


# ------------------
# Serialization I/O
# ------------------

def serialize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    # Convert DataFrames to list-of-dicts for JSON
    p = {k: v for k, v in plan.items()}
    p["exposures_df"] = plan["exposures_df"].to_dict(orient="records")
    p["outcomes_df"] = plan["outcomes_df"].to_dict(orient="records")
    p["covariates_df"] = plan["covariates_df"].to_dict(orient="records")
    return p


def load_plan(uploaded_file):
    data = json.load(uploaded_file)
    # Defensive defaults
    st.session_state.plan.update({k: data.get(k, st.session_state.plan.get(k)) for k in st.session_state.plan.keys()})
    # Restore dataframes
    st.session_state.plan["exposures_df"] = pd.DataFrame(data.get("exposures_df", []))
    if st.session_state.plan["exposures_df"].empty:
        st.session_state.plan["exposures_df"] = EMPTY_EXPOSURES_DF.copy()
    st.session_state.plan["outcomes_df"] = pd.DataFrame(data.get("outcomes_df", []))
    if st.session_state.plan["outcomes_df"].empty:
        st.session_state.plan["outcomes_df"] = EMPTY_OUTCOMES_DF.copy()
    st.session_state.plan["covariates_df"] = pd.DataFrame(data.get("covariates_df", []))
    if st.session_state.plan["covariates_df"].empty:
        st.session_state.plan["covariates_df"] = EMPTY_COVARS_DF.copy()


# ---------
# Main App
# ---------

def main():
    header()
    init_state()
    sidebar_nav()

    step = int(st.session_state.step)
    if step == 1:
        step_basics()
    elif step == 2:
        step_comparison()
    elif step == 3:
        step_exposure()
    elif step == 4:
        step_outcomes()
    elif step == 5:
        step_cohort_criteria()
    elif step == 6:
        step_covariates_balancing()
    elif step == 7:
        step_analysis()
    elif step == 8:
        step_subgroups_sensitivity()
    elif step == 9:
        step_feasibility_ethics()
    else:
        step_preview_export()

    # Footer metrics
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Exposures/Index entries", value=len(st.session_state.plan["exposures_df"]))
    with m2:
        st.metric("Outcome definitions", value=len(st.session_state.plan["outcomes_df"]))
    with m3:
        st.metric("Covariate details", value=len(st.session_state.plan["covariates_df"]))
    with m4:
        st.metric("Subgroups", value=len(st.session_state.plan.get("subgroups", [])))


if __name__ == "__main__":
    main()
