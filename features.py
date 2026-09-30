"""Add-on features for MFI RiskRadar.

Nothing in your original app.py logic is changed. app.py only calls the functions below:
  features.render_evaluation_extras(...)   -> extra panels after each credit evaluation
  features.render_extra_sections()         -> dashboard, disbursement & repayments, lookup, backup
  features.sidebar_policy_settings()       -> lending policy limits + SMS options in the sidebar

New data lives in NEW csv files, so your loan_records.csv schema is never touched:
  loan_extras.csv   (probability, risk grade, policy result, interest rate per loan)
  loan_tracker.csv  (disbursement dates)
  repayments.csv    (every repayment received)
"""
import io
import os
import re
import smtplib
import urllib.parse
import zipfile
from datetime import date, datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

LOAN_RECORDS_CSV = "loan_records.csv"
EXTRAS_CSV = "loan_extras.csv"
TRACKER_CSV = "loan_tracker.csv"
REPAY_CSV = "repayments.csv"
NOTIFY_LOG_CSV = "notifications_log.csv"
DATA_FILES = [LOAN_RECORDS_CSV, EXTRAS_CSV, TRACKER_CSV, REPAY_CSV, NOTIFY_LOG_CSV]

EXTRA_COLS = ["loan_id", "interest_rate", "approval_probability", "risk_grade", "policy_result",
              "policy_breaches", "recommendation", "suggested_max_loan"]
TRACKER_COLS = ["loan_id", "disbursed_at"]
REPAY_COLS = ["loan_id", "paid_at", "amount", "note"]
NOTIFY_COLS = ["timestamp", "loan_id", "channel", "ok", "detail"]
DEFAULT_RATE = 15.0

DEFAULT_POLICY = {"max_pti": 40.0, "max_lti": 200.0, "max_lta": 100.0, "min_score": 550, "buffer": True}


# ============================================================ small helpers
def _read(path, cols=None) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=cols or [])


def _append(path, row: dict, cols: list):
    exists = os.path.exists(path)
    pd.DataFrame([row], columns=cols).to_csv(path, mode="a" if exists else "w", header=not exists, index=False)


def _inst() -> str:
    return st.session_state.get("inst_name", "GAWFA") or "GAWFA"


def money(x) -> str:
    return f"{float(x):,.0f}"


def monthly_payment(principal, annual_rate_pct, term) -> float:
    term = int(term)
    if term <= 0 or principal <= 0:
        return 0.0
    r = annual_rate_pct / 100 / 12
    if r <= 0:
        return principal / term
    f = (1 + r) ** term
    return principal * r * f / (f - 1)


def normalize_phone(phone):
    if not phone or str(phone).strip().upper() == "N/A":
        return None
    p = re.sub(r"[^\d+]", "", str(phone))
    if p.startswith("+"):
        return p if len(p) >= 10 else None
    if p.startswith("00"):
        return "+" + p[2:]
    if p.startswith("220") and len(p) >= 10:
        return "+" + p
    if len(p) == 7:
        return "+220" + p
    return None


def wa_link(phone, text: str) -> str:
    num = normalize_phone(phone)
    q = urllib.parse.quote(text)
    return f"https://wa.me/{num.lstrip('+')}?text={q}" if num else f"https://wa.me/?text={q}"


# ===================================================== notifications (email + SMS)
def _secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


def email_configured() -> bool:
    return all(_secret(k) for k in ("SMTP_SERVER", "SMTP_PORT", "SMTP_EMAIL", "SMTP_PASSWORD"))


def sms_provider() -> str:
    return str(_secret("SMS_PROVIDER", "budgetsms")).lower()


def sms_configured() -> bool:
    if sms_provider() == "africastalking":
        return all(_secret(k) for k in ("AT_USERNAME", "AT_API_KEY"))
    return all(_secret(k) for k in ("BUDGETSMS_USERNAME", "BUDGETSMS_USERID", "BUDGETSMS_HANDLE"))


def send_email_detailed(to, subject: str, text: str, html: str, pdf_bytes=None, pdf_name="Decision_Letter.pdf"):
    """Returns (ok, reason). Works with Gmail (port 587, App Password) and any SMTP server."""
    if not email_configured():
        return False, "email not set up (add the SMTP_* keys to secrets)"
    to = str(to or "").strip()
    if "@" not in to or any(c in to for c in "\r\n ") or to.upper() == "N/A":
        return False, "no valid email address for this applicant"
    server, port = str(_secret("SMTP_SERVER")), int(_secret("SMTP_PORT"))
    sender, pwd = str(_secret("SMTP_EMAIL")), str(_secret("SMTP_PASSWORD"))
    msg = MIMEMultipart("mixed")
    msg["Subject"], msg["From"], msg["To"] = subject, formataddr((_inst(), sender)), to
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt)
    if pdf_bytes:
        part = MIMEApplication(pdf_bytes, _subtype="pdf")
        part.add_header("Content-Disposition", "attachment", filename=pdf_name)
        msg.attach(part)
    try:
        if port == 465:
            smtp = smtplib.SMTP_SSL(server, port, timeout=20)
        else:
            smtp = smtplib.SMTP(server, port, timeout=20)
            smtp.ehlo(); smtp.starttls(); smtp.ehlo()
        with smtp:
            smtp.login(sender, pwd)
            smtp.sendmail(sender, [to], msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError:
        return False, "login failed - Gmail needs an App Password, not your normal password"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def send_sms_detailed(phone, message: str):
    """Returns (ok, note). Providers: BudgetSMS (default, covers all Gambian networks) or Africa's Talking."""
    if not sms_configured():
        return False, f"SMS not set up (add the {sms_provider()} keys to secrets)"
    to = normalize_phone(phone)
    if not to:
        return False, "no valid phone number (use +220 followed by 7 digits)"
    inst_sender = re.sub(r"[^A-Za-z0-9]", "", _inst())[:11] or "RiskRadar"
    try:
        if sms_provider() == "africastalking":
            sandbox = str(_secret("AT_SANDBOX", "false")).lower() == "true"
            url = ("https://api.sandbox.africastalking.com/version1/messaging" if sandbox
                   else "https://api.africastalking.com/version1/messaging")
            data = {"username": _secret("AT_USERNAME"), "to": to, "message": message[:459]}
            if _secret("AT_SENDER_ID"):
                data["from"] = _secret("AT_SENDER_ID")
            r = requests.post(url, data=data, headers={"apiKey": str(_secret("AT_API_KEY")), "Accept": "application/json"}, timeout=15)
            if r.status_code not in (200, 201):
                return False, f"gateway HTTP {r.status_code}: {r.text[:80]}"
            recips = r.json().get("SMSMessageData", {}).get("Recipients", [])
            if any(x.get("status") == "Success" for x in recips):
                return True, ""
            return False, "gateway rejected: " + (recips[0].get("status", "unknown") if recips else r.text[:80])

        test = str(_secret("SMS_TEST_MODE", "false")).lower() == "true"
        url = "https://api.budgetsms.net/testsms/" if test else "https://api.budgetsms.net/sendsms/"
        params = {"username": _secret("BUDGETSMS_USERNAME"), "userid": _secret("BUDGETSMS_USERID"),
                  "handle": _secret("BUDGETSMS_HANDLE"), "from": str(_secret("BUDGETSMS_SENDER", inst_sender))[:11],
                  "to": to.lstrip("+"), "msg": message}
        r = requests.get(url, params=params, timeout=15)
        body = r.text.strip()
        if r.status_code == 200 and body.upper().startswith("OK"):
            return True, "TEST MODE - accepted but NOT delivered (set SMS_TEST_MODE = \"false\")" if test else ""
        return False, f"gateway said: {body[:80] or 'HTTP ' + str(r.status_code)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def send_real_sms(phone, message: str) -> bool:
    return send_sms_detailed(phone, message)[0]


def _log_notification(loan_id, channel, ok, detail):
    _append(NOTIFY_LOG_CSV, {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "loan_id": loan_id,
                             "channel": channel, "ok": bool(ok), "detail": detail}, NOTIFY_COLS)


def build_messages(name, loan_id, status, amount, term, rate, inst):
    first = (str(name).split() or ["Customer"])[0]
    approved = status == "Approved"
    label = "APPROVED" if approved else "NOT APPROVED"
    subject = f"Loan application {label} [{loan_id}] - {inst}"
    if approved:
        sms = (f"{inst}: Dear {first}, your loan application {loan_id} for GMD {amount:,.0f} is APPROVED, "
               f"subject to verification. Please visit our office for the next steps.")
        body = (f"Good news - your application has been <b>approved</b>, subject to standard verification. "
                f"Please visit our office with your ID to complete the process. "
                f"Your monthly instalment will be about <b>GMD {monthly_payment(amount, rate, term):,.2f}</b> over {int(term)} months at {rate}% p.a.")
    else:
        sms = (f"{inst}: Dear {first}, your loan application {loan_id} was not approved at this time. "
               f"Please visit our office to discuss your options.")
        body = ("After careful review we are unable to approve your application at this time. "
                "Please visit our office - our team will be glad to discuss your options and how to strengthen a future application.")
    html = (f"<div style='font-family:Arial,sans-serif;max-width:560px'>"
            f"<h2 style='color:#064E3B'>{escape(inst)}</h2><p>Dear {escape(str(name))},</p>"
            f"<p>Your loan application <b>{escape(str(loan_id))}</b> for <b>GMD {amount:,.2f}</b> is <b>{label}</b>.</p>"
            f"<p>{body}</p><p>Kind regards,<br>{escape(inst)}</p></div>")
    text = re.sub(r"<[^>]+>", "", html.replace("</p>", "\n\n").replace("<br>", "\n"))
    return subject, text, html, sms


def notify_applicant(name, phone, email, loan_id, status, loan_amount, loan_term, interest_rate, pdf_bytes=None):
    """Send the decision by email (with PDF letter attached) and SMS, show the outcome, log it."""
    inst = _inst()
    subject, text, html, sms = build_messages(name, loan_id, status, float(loan_amount), loan_term, float(interest_rate), inst)

    if not st.session_state.get("notify_email", True):
        em = (False, "email sending is switched off in the sidebar")
    elif str(email).lower().endswith("@example.com"):
        em = (False, "demo address skipped")
    else:
        em = send_email_detailed(email, subject, text, html, pdf_bytes, f"Loan_Decision_{loan_id}.pdf")

    if not st.session_state.get("notify_sms", True):
        sm = (False, "SMS sending is switched off in the sidebar")
    elif normalize_phone(phone) == "+2201234567":
        sm = (False, "demo number skipped")
    else:
        sm = send_sms_detailed(phone, sms)

    _log_notification(loan_id, "email", em[0], em[1] or str(email))
    _log_notification(loan_id, "sms", sm[0], sm[1] or str(phone))

    st.markdown("### 📨 Applicant Notifications")
    c1, c2 = st.columns(2)
    if em[0]:
        c1.success(f"📧 Email sent to {email}" + (" (decision letter attached)" if pdf_bytes else ""))
    else:
        c1.warning(f"📧 Email not sent - {em[1]}")
    if sm[0]:
        c2.success(f"📲 SMS sent to {phone}" + (f" - {sm[1]}" if sm[1] else ""))
    else:
        c2.warning(f"📲 SMS not sent - {sm[1]}")
    return {"email": em, "sms": sm}


def to_model_units(X):
    """The saved model was trained with loan_term in YEARS (loan_approval_dataset.csv); the app form uses
    MONTHS. Convert column 5 (loan_term) before scaling/predicting."""
    X = np.array(X, dtype=float, copy=True)
    X[:, 5] = X[:, 5] / 12.0
    return X


def _p_approve(model, xs) -> np.ndarray:
    classes = list(model.classes_)
    idx = classes.index(1) if 1 in classes else len(classes) - 1
    return model.predict_proba(xs)[:, idx]


# ============================================================ lending policy
def get_policy() -> dict:
    return {k: st.session_state.get(f"pol_{k}", v) for k, v in DEFAULT_POLICY.items()}


def policy_checks(p, loan_amount, income, total_assets, bank_assets, score, rate, term) -> list:
    pay = monthly_payment(loan_amount, rate, term)
    m_inc = income / 12 if income > 0 else 0
    pti = pay / m_inc * 100 if m_inc > 0 else 999.0
    lti = loan_amount / income * 100 if income > 0 else 999.0
    lta = loan_amount / total_assets * 100 if total_assets > 0 else 999.0
    rules = [
        {"Rule": "Payment-to-income (monthly)", "Actual": f"{pti:.1f}%", "Limit": f"≤ {p['max_pti']:.0f}%", "ok": pti <= p["max_pti"]},
        {"Rule": "Loan-to-income (annual)", "Actual": f"{lti:.1f}%", "Limit": f"≤ {p['max_lti']:.0f}%", "ok": lti <= p["max_lti"]},
        {"Rule": "Loan-to-asset cover", "Actual": f"{lta:.1f}%", "Limit": f"≤ {p['max_lta']:.0f}%", "ok": lta <= p["max_lta"]},
        {"Rule": "Minimum credit score", "Actual": f"{int(score)}", "Limit": f"≥ {int(p['min_score'])}", "ok": score >= p["min_score"]},
    ]
    if p["buffer"]:
        rules.append({"Rule": "Liquid buffer ≥ 1 instalment", "Actual": f"{money(bank_assets)} GMD",
                      "Limit": f"≥ {money(pay)} GMD", "ok": bank_assets >= pay})
    return rules


def policy_result(rules: list) -> str:
    fails = [r for r in rules if not r["ok"]]
    score_fail = any(r["Rule"] == "Minimum credit score" and not r["ok"] for r in rules)
    if not fails:
        return "PASS"
    return "FAIL" if (score_fail or len(fails) >= 2) else "REVIEW"


def risk_grade(prob: float, pol: str) -> str:
    grades = ["A", "B", "C", "D", "E"]
    i = 0 if prob >= 0.90 else 1 if prob >= 0.75 else 2 if prob >= 0.55 else 3 if prob >= 0.35 else 4
    i += {"PASS": 0, "REVIEW": 1, "FAIL": 2}[pol]
    return grades[min(i, 4)]


def suggested_max_loan(p, loan_amount, income, total_assets, bank_assets, score, rate, term):
    """Largest loan (<= requested) that satisfies every policy rule. None if no amount can."""
    ok = lambda amt: all(r["ok"] for r in policy_checks(p, amt, income, total_assets, bank_assets, score, rate, term))
    if ok(loan_amount):
        return float(loan_amount)
    if not ok(1.0):
        return None
    lo, hi = 1.0, float(loan_amount)
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return float(np.floor(lo / 500) * 500)


def counterfactuals(model, scaler, row) -> list:
    """Smallest single changes (score up / loan down / term down) that make the MODEL approve."""
    base = np.array(row, dtype=float)
    tips = []

    def first_hit(rows):
        if not rows:
            return None
        pr = _p_approve(model, scaler.transform(to_model_units(np.array(rows))))
        hit = np.where(pr >= 0.5)[0]
        return rows[hit[0]] if len(hit) else None

    rows = []
    for s in range(int(base[6]) + 5, 901, 5):
        r = base.copy(); r[6] = s; rows.append(r)
    h = first_hit(rows)
    if h is not None:
        tips.append(f"Credit score of **{int(h[6])}** or higher (currently {int(base[6])}).")

    rows = []
    for f in np.linspace(0.95, 0.05, 19):
        r = base.copy(); r[4] = base[4] * f; rows.append(r)
    h = first_hit(rows)
    if h is not None:
        tips.append(f"Loan amount of **{money(h[4])} GMD** or less (requested {money(base[4])}).")

    rows = []
    for t in range(int(base[5]) - 2, 1, -2):
        r = base.copy(); r[5] = t; rows.append(r)
    h = first_hit(rows)
    if h is not None:
        tips.append(f"Repayment term of **{int(h[5])}** or shorter (requested {int(base[5])}).")
    return tips


# ================================================================ PDFs
def agreement_pdf(loan_id, name, phone, amount, term, rate, pay, officer, branch, inst) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    ss = getSampleStyleSheet()
    h = ParagraphStyle("H", parent=ss["Heading1"], fontSize=18, textColor=colors.HexColor("#064E3B"), spaceAfter=6)
    b = ParagraphStyle("B", parent=ss["Normal"], fontSize=10, leading=14)
    e = lambda x: escape(str(x if x not in (None, "") else "N/A"))
    total = pay * term
    story = [Paragraph(f"{e(inst).upper()} - LOAN AGREEMENT", h),
             Paragraph(f"Reference: {e(loan_id)} | Date: {date.today():%d %B %Y}", b),
             HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#064E3B"), spaceAfter=10)]
    terms = [("Lender", inst), ("Borrower", name), ("Borrower phone", phone),
             ("Principal", f"{amount:,.2f} GMD"), ("Interest rate", f"{rate}% per annum (reducing balance)"),
             ("Term", f"{int(term)} months"), ("Monthly instalment", f"{pay:,.2f} GMD"),
             ("Total repayable", f"{total:,.2f} GMD"), ("Loan officer / Branch", f"{e(officer)} / {e(branch)}")]
    t = Table([[Paragraph(f"<b>{k}</b>", b), Paragraph(e(v), b)] for k, v in terms], colWidths=[150, 380])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9FAFB")),
                           ("PADDING", (0, 0), (-1, -1), 6)]))
    story += [t, Spacer(1, 12)]
    clauses = [
        "1. The Lender agrees to lend the Principal to the Borrower, and the Borrower agrees to repay it with interest in equal monthly instalments, the first falling due one month after disbursement.",
        "2. Every payment must be made to the Lender's designated account or officer and a receipt obtained. Payments are applied first to interest due, then to principal.",
        "3. If an instalment is not paid when due, the Lender may charge the late-payment fee set out in its published fee schedule and may contact the Borrower and any guarantor to recover the arrears.",
        "4. The Borrower may repay the loan early in full or in part at any time.",
        "5. The Borrower confirms the information given in the application is true and consents to the Lender using it for credit assessment and loan administration.",
        "6. Any dispute will first be discussed between the parties and, failing agreement, handled under the laws of The Gambia.",
    ]
    story += [Paragraph("<b>Terms and conditions</b>", b), Spacer(1, 4)]
    story += [Paragraph(c, b) for c in clauses]
    story += [Spacer(1, 6), Paragraph("<i>Template for guidance. Have it reviewed by your lawyer before use.</i>", b), Spacer(1, 10),
              Paragraph("<b>Repayment schedule</b>", b), Spacer(1, 4)]

    r = rate / 100 / 12
    bal, rows = float(amount), [["Month", "Payment", "Principal", "Interest", "Balance"]]
    for m in range(1, int(term) + 1):
        interest = bal * r
        princ = pay - interest
        bal = max(0.0, bal - princ)
        rows.append([m, f"{pay:,.2f}", f"{princ:,.2f}", f"{interest:,.2f}", f"{bal:,.2f}"])
    st_ = Table(rows, repeatRows=1, colWidths=[50, 100, 100, 100, 110])
    st_.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#064E3B")),
                             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
                             ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story += [st_, Spacer(1, 30)]
    sig = Table([["______________________", "______________________", "______________________"],
                 ["Borrower", "Loan Officer", "Witness"]], colWidths=[177, 177, 177])
    story.append(sig)
    doc.build(story)
    buf.seek(0)
    return buf


def statement_pdf(row: dict, reps: pd.DataFrame, inst: str) -> io.BytesIO:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    ss = getSampleStyleSheet()
    h = ParagraphStyle("H", parent=ss["Heading1"], fontSize=16, textColor=colors.HexColor("#064E3B"))
    b = ParagraphStyle("B", parent=ss["Normal"], fontSize=10, leading=14)
    story = [Paragraph(f"{escape(inst)} - STATEMENT OF ACCOUNT", h),
             Paragraph(f"Loan {escape(str(row['loan_id']))} | {escape(str(row['applicant_name']))} | "
                       f"Generated {datetime.now():%Y-%m-%d %H:%M}", b), Spacer(1, 8)]
    summ = [["Principal", f"{row['loan_amount']:,.2f} GMD"], ["Instalment", f"{row['installment']:,.2f} GMD"],
            ["Total paid", f"{row['paid']:,.2f} GMD"], ["Outstanding principal", f"{row['outstanding']:,.2f} GMD"],
            ["Arrears", f"{row['arrears']:,.2f} GMD"], ["Days past due", str(int(row['dpd']))]]
    t = Table(summ, colWidths=[200, 200])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")), ("PADDING", (0, 0), (-1, -1), 5)]))
    story += [t, Spacer(1, 12), Paragraph("<b>Payments received</b>", b), Spacer(1, 4)]
    data = [["Date", "Amount (GMD)", "Note"]] + [[str(r.paid_at), f"{r.amount:,.2f}", escape(str(r.note if pd.notna(r.note) else ""))]
                                              for r in reps.itertuples()]
    if len(data) == 1:
        data.append(["-", "-", "No payments yet"])
    pt = Table(data, repeatRows=1, colWidths=[110, 120, 250])
    pt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#064E3B")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB"))]))
    story.append(pt)
    doc.build(story)
    buf.seek(0)
    return buf


# ================================================ 1) panels after an evaluation
def render_evaluation_extras(model, scaler, input_data, status, loan_id, applicant_name, applicant_phone,
                             loan_amount, loan_term, interest_rate, income_annum, cibil_score, total_assets,
                             monthly_pay, officer_name, branch_name):
    row = np.array(input_data, dtype=float)[0]
    bank_assets = row[10]
    prob = float(_p_approve(model, scaler.transform(to_model_units(row.reshape(1, -1))))[0])
    pol = get_policy()
    rules = policy_checks(pol, loan_amount, income_annum, total_assets, bank_assets, cibil_score, interest_rate, loan_term)
    pres = policy_result(rules)
    grade = risk_grade(prob, pres)
    breaches = "; ".join(r["Rule"] for r in rules if not r["ok"])
    max_loan = suggested_max_loan(pol, loan_amount, income_annum, total_assets, bank_assets, cibil_score, interest_rate, loan_term)

    if status == "Approved" and pres == "PASS":
        rec = "✅ Recommend APPROVAL - model approves and all lending policy limits are met."
        box = st.success
    elif status == "Approved":
        rec = f"🟠 Refer to credit committee - model approves but policy is breached: {breaches}."
        box = st.warning
    elif pres == "PASS":
        rec = "🟠 Model rejects but policy limits are met - manual review recommended."
        box = st.warning
    else:
        rec = f"🔴 Recommend REJECTION - model rejects and policy is breached: {breaches or 'n/a'}."
        box = st.error

    st.markdown("---")
    st.markdown("### 🧠 Extended Risk Intelligence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approval Probability", f"{prob * 100:.1f}%")
    c2.metric("Risk Grade", grade, help="A = lowest risk ... E = highest. Model probability, downgraded for policy breaches.")
    c3.metric("Policy Check", pres)
    c4.metric("Suggested Max Loan", f"{money(max_loan)} GMD" if max_loan is not None else "None")
    box(rec)

    with st.expander("🛡️ Lending policy check (limits are set in the sidebar)", expanded=pres != "PASS"):
        tbl = pd.DataFrame(rules)
        tbl["Result"] = tbl["ok"].map({True: "✅ Pass", False: "❌ Breach"})
        st.dataframe(tbl.drop(columns=["ok"]), hide_index=True, use_container_width=True)
        if max_loan is None:
            st.caption("No loan amount can pass the policy for this applicant (for example, credit score below the minimum).")
        elif max_loan < loan_amount:
            st.caption(f"Reducing the loan to about {money(max_loan)} GMD would satisfy every policy limit.")
        else:
            st.caption("Requested amount already satisfies every policy limit.")

    if status != "Approved":
        with st.expander("🔁 What could change the model's decision?", expanded=True):
            tips = counterfactuals(model, scaler, row)
            if tips:
                for t in tips:
                    st.markdown(f"- {t}")
                st.caption("Each change is tested alone with everything else unchanged.")
            else:
                st.write("No single change flips the decision. Several factors would need to improve together.")

    inst = _inst()
    msg = f"{inst}: Dear {applicant_name}, your loan application {loan_id} for GMD {loan_amount:,.0f} is {status.upper()}."
    s1, s2 = st.columns(2)
    s1.markdown(f"[💬 Send decision on WhatsApp]({wa_link(applicant_phone, msg)})")
    if status == "Approved":
        pdf = agreement_pdf(loan_id, applicant_name, applicant_phone, float(loan_amount), int(loan_term),
                            float(interest_rate), float(monthly_pay), officer_name, branch_name, inst)
        s2.download_button("📄 Download Loan Agreement + Schedule (PDF)", pdf.getvalue(),
                           file_name=f"Loan_Agreement_{loan_id}.pdf", mime="application/pdf",
                           key=f"agr_{loan_id}", use_container_width=True)

    _append(EXTRAS_CSV, {"loan_id": loan_id, "interest_rate": interest_rate, "approval_probability": round(prob, 4),
                         "risk_grade": grade, "policy_result": pres, "policy_breaches": breaches,
                         "recommendation": rec[:200], "suggested_max_loan": max_loan if max_loan is not None else ""},
            EXTRA_COLS)


# ==================================================== portfolio calculations
def _months_elapsed(start, today) -> int:
    m = (today.year - start.year) * 12 + (today.month - start.month)
    return max(0, m - (1 if today.day < start.day else 0))


def portfolio_table() -> pd.DataFrame:
    recs, trk, rep, ext = _read(LOAN_RECORDS_CSV), _read(TRACKER_CSV, TRACKER_COLS), _read(REPAY_CSV, REPAY_COLS), _read(EXTRAS_CSV, EXTRA_COLS)
    if recs.empty or trk.empty:
        return pd.DataFrame()
    d = trk.drop_duplicates("loan_id").merge(recs, on="loan_id", how="inner")
    rates = dict(zip(ext["loan_id"], ext["interest_rate"])) if not ext.empty else {}
    paid_map = rep.groupby("loan_id")["amount"].sum().to_dict() if not rep.empty else {}
    today = pd.Timestamp.today().normalize()
    rows = []
    for r in d.to_dict("records"):
        term, amt = int(r["loan_term_months"]), float(r["loan_amount"])
        rate = float(rates.get(r["loan_id"], DEFAULT_RATE))
        pay = monthly_payment(amt, rate, term)
        total = pay * term
        paid = float(paid_map.get(r["loan_id"], 0.0))
        start = pd.Timestamp(r["disbursed_at"])
        expected = pay * min(term, _months_elapsed(start, today))
        covered = min(term, int((paid + 2.0) / pay + 1e-9)) if pay > 0 else term
        next_due = start + pd.DateOffset(months=covered + 1) if covered < term else pd.NaT
        dpd = max(0, (today - next_due).days) if pd.notna(next_due) else 0
        rows.append({"loan_id": r["loan_id"], "applicant_name": r["applicant_name"], "applicant_phone": r["applicant_phone"],
                     "branch_name": r.get("branch_name"), "loan_amount": amt, "term": term, "rate": rate,
                     "disbursed_at": start.date(), "installment": round(pay, 2), "total_due": round(total, 2),
                     "paid": round(paid, 2), "arrears": round(max(0.0, expected - paid - 2.0), 2),
                     "outstanding": round(amt * max(0.0, 1 - paid / total) if total > 0 else 0.0, 2),
                     "next_due": next_due.date() if pd.notna(next_due) else None, "dpd": int(dpd)})
    return pd.DataFrame(rows)


def _par(pt: pd.DataFrame, days: int) -> float:
    tot = pt["outstanding"].sum()
    return float(pt[pt["dpd"] > days]["outstanding"].sum() / tot * 100) if tot > 0 else 0.0


# ============================================================== 2) sections
def render_extra_sections():
    st.markdown("---")
    st.markdown("## 🚀 Portfolio Tools")
    t1, t2, t3, t4 = st.tabs(["📈 Dashboard", "💵 Disbursements & Repayments", "🔎 Loan Lookup", "💾 Backup & Restore"])
    with t1:
        _tab_dashboard()
    with t2:
        _tab_tracker()
    with t3:
        _tab_lookup()
    with t4:
        _tab_backup()


def _tab_dashboard():
    recs = _read(LOAN_RECORDS_CSV)
    if recs.empty:
        st.info("No evaluations saved yet. Run a credit evaluation first.")
        return
    recs["timestamp"] = pd.to_datetime(recs["timestamp"], errors="coerce")
    recs["branch_name"] = recs["branch_name"].fillna("").replace("", "Unspecified")
    recs["officer_name"] = recs["officer_name"].fillna("").replace("", "Unspecified")
    appr = recs[recs["loan_status"] == "Approved"]
    k = st.columns(4)
    k[0].metric("Applications", len(recs))
    k[1].metric("Approval rate", f"{len(appr) / len(recs) * 100:.1f}%")
    k[2].metric("Approved volume (GMD)", money(appr["loan_amount"].sum()))
    k[3].metric("Expected loss on approved", money(appr["expected_loss"].sum()))

    ext = _read(EXTRAS_CSV, EXTRA_COLS)
    if not ext.empty:
        m = recs.merge(ext.drop_duplicates("loan_id"), on="loan_id", how="left")
        flagged = m[(m["loan_status"] == "Approved") & (m["policy_result"].isin(["REVIEW", "FAIL"]))]
        c1, c2 = st.columns(2)
        c1.metric("Approved by model but policy breached", len(flagged))
        gd = m["risk_grade"].value_counts().reindex(list("ABCDE")).fillna(0)
        c2.bar_chart(gd)
        c2.caption("Risk grade distribution")

    cmap = {"Approved": "#059669", "Rejected": "#DC2626"}
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(recs, x="cibil_score", color="loan_status", nbins=20, color_discrete_map=cmap,
                                title="Credit score distribution"), use_container_width=True)
    recs["month"] = recs["timestamp"].dt.to_period("M").astype(str)
    mm = recs.groupby(["month", "loan_status"]).size().reset_index(name="count")
    b.plotly_chart(px.bar(mm, x="month", y="count", color="loan_status", barmode="group",
                          color_discrete_map=cmap, title="Applications per month"), use_container_width=True)
    br = recs.groupby("branch_name").agg(Applications=("loan_id", "count"),
                                         Approval_rate=("loan_status", lambda s: round((s == "Approved").mean() * 100, 1)),
                                         Volume=("loan_amount", "sum")).reset_index()
    st.subheader("By branch")
    st.dataframe(br, hide_index=True, use_container_width=True)
    off = recs.groupby("officer_name").agg(Applications=("loan_id", "count"),
                                           Approval_rate=("loan_status", lambda s: round((s == "Approved").mean() * 100, 1))).reset_index()
    st.subheader("By loan officer")
    st.dataframe(off, hide_index=True, use_container_width=True)


def _tab_tracker():
    recs, trk, rep = _read(LOAN_RECORDS_CSV), _read(TRACKER_CSV, TRACKER_COLS), _read(REPAY_CSV, REPAY_COLS)
    if recs.empty:
        st.info("No loans yet.")
        return

    st.subheader("1. Disburse an approved loan")
    pending = recs[(recs["loan_status"] == "Approved") & (~recs["loan_id"].isin(trk["loan_id"]))]
    if pending.empty:
        st.caption("No approved loans are waiting for disbursement.")
    else:
        opts = {f"{r.loan_id} - {r.applicant_name} - {money(r.loan_amount)} GMD": r.loan_id for r in pending.itertuples()}
        pick = st.selectbox("Approved loan", list(opts), key="xt_disb_pick")
        ddate = st.date_input("Disbursement date", value=date.today(), key="xt_disb_date")
        if st.button("Confirm disbursement", key="xt_disb_btn"):
            _append(TRACKER_CSV, {"loan_id": opts[pick], "disbursed_at": ddate.isoformat()}, TRACKER_COLS)
            st.success("Disbursed.")
            st.rerun()

    pt = portfolio_table()
    if pt.empty:
        st.info("Disburse a loan to start tracking repayments.")
        return

    st.subheader("2. Record a repayment")
    active = pt[pt["outstanding"] > 0.5]
    if active.empty:
        st.caption("All disbursed loans are fully repaid.")
    else:
        opts = {f"{r.loan_id} - {r.applicant_name} (instalment {money(r.installment)})": r.loan_id for r in active.itertuples()}
        pick = st.selectbox("Loan", list(opts), key="xt_rep_pick")
        r1, r2, r3 = st.columns(3)
        amt = r1.number_input("Amount (GMD)", min_value=0.0, step=100.0, key="xt_rep_amt")
        pdate = r2.date_input("Date paid", value=date.today(), key="xt_rep_date")
        note = r3.text_input("Note", key="xt_rep_note")
        if st.button("Save repayment", key="xt_rep_btn"):
            if amt <= 0:
                st.error("Enter an amount greater than 0.")
            else:
                _append(REPAY_CSV, {"loan_id": opts[pick], "paid_at": pdate.isoformat(), "amount": float(amt), "note": note}, REPAY_COLS)
                st.success("Repayment saved.")
                st.rerun()

    st.subheader("3. Portfolio health")
    k = st.columns(5)
    k[0].metric("Disbursed (GMD)", money(pt["loan_amount"].sum()))
    k[1].metric("Outstanding principal", money(pt["outstanding"].sum()))
    k[2].metric("Total arrears", money(pt["arrears"].sum()))
    k[3].metric("PAR 30", f"{_par(pt, 30):.1f}%", help="% of outstanding principal more than 30 days late")
    k[4].metric("PAR 90", f"{_par(pt, 90):.1f}%")
    st.dataframe(pt.drop(columns=["applicant_phone"]), hide_index=True, use_container_width=True)

    od = pt[pt["dpd"] > 0].sort_values("dpd", ascending=False)
    st.subheader("4. Overdue borrowers")
    if od.empty:
        st.success("No overdue loans 🎉")
        return
    inst = _inst()
    for r in od.itertuples():
        text = (f"{inst}: Dear {r.applicant_name}, your loan {r.loan_id} instalment is overdue. "
                f"Arrears: GMD {r.arrears:,.0f}. Please pay as soon as possible. Thank you.")
        st.markdown(f"**{r.applicant_name}** - {r.dpd} days late - arrears {money(r.arrears)} GMD - "
                    f"[💬 WhatsApp reminder]({wa_link(r.applicant_phone, text)})")
    if sms_configured() and st.session_state.get("notify_sms", True):
        if st.checkbox(f"Yes, SMS all {len(od)} overdue borrowers", key="xt_sms_ok") and st.button("📲 Send SMS reminders", key="xt_sms_btn"):
            sent = sum(send_real_sms(r.applicant_phone, f"{inst}: Dear {r.applicant_name}, your loan {r.loan_id} instalment is overdue. "
                                     f"Arrears: GMD {r.arrears:,.0f}. Please pay soon. Thank you.") for r in od.itertuples())
            st.success(f"{sent} of {len(od)} SMS accepted by the gateway.")
    else:
        st.caption("Add your SMS keys to secrets (and keep 'SMS the applicant' ticked in the sidebar) to send reminders automatically.")


def _tab_lookup():
    recs = _read(LOAN_RECORDS_CSV)
    if recs.empty:
        st.info("No loans yet.")
        return
    q = st.text_input("Search by name, phone or loan ID", key="xt_q")
    v = recs
    if q:
        mask = (recs["applicant_name"].astype(str).str.contains(q, case=False, na=False)
                | recs["applicant_phone"].astype(str).str.contains(q, case=False, na=False)
                | recs["loan_id"].astype(str).str.contains(q, case=False, na=False))
        v = recs[mask]
    c1, c2 = st.columns(2)
    stat = c1.multiselect("Status", sorted(recs["loan_status"].dropna().unique()), key="xt_stat")
    if stat:
        v = v[v["loan_status"].isin(stat)]
    st.dataframe(v, hide_index=True, use_container_width=True)
    c2.download_button("📄 Download these rows (CSV)", v.to_csv(index=False).encode(), "loans_filtered.csv", "text/csv", key="xt_dl")

    st.subheader("Notifications")
    if len(v):
        rs = st.selectbox("Resend the decision (SMS + email) for", [""] + v["loan_id"].tolist(), key="xt_resend")
        if rs and st.button("📨 Resend decision notification", key="xt_resend_btn"):
            r = recs[recs["loan_id"] == rs].iloc[0]
            ext = _read(EXTRAS_CSV, EXTRA_COLS)
            rate = float(dict(zip(ext["loan_id"], ext["interest_rate"])).get(rs, DEFAULT_RATE)) if not ext.empty else DEFAULT_RATE
            notify_applicant(r["applicant_name"], r["applicant_phone"], r["applicant_email"], rs, r["loan_status"],
                             float(r["loan_amount"]), int(r["loan_term_months"]), rate)
    nlog = _read(NOTIFY_LOG_CSV, NOTIFY_COLS)
    if not nlog.empty:
        st.dataframe(nlog.iloc[::-1].head(50), hide_index=True, use_container_width=True)

    pt = portfolio_table()
    if pt.empty:
        st.caption("Statements are available for disbursed loans.")
        return
    pick = st.selectbox("Statement of account for", [""] + pt["loan_id"].tolist(), key="xt_stmt")
    if pick:
        row = pt[pt["loan_id"] == pick].iloc[0].to_dict()
        rep = _read(REPAY_CSV, REPAY_COLS)
        rep = rep[rep["loan_id"] == pick]
        st.dataframe(rep, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Statement of account (PDF)", statement_pdf(row, rep, _inst()).getvalue(),
                           f"Statement_{pick}.pdf", "application/pdf", key=f"xt_stmt_dl_{pick}")


def _tab_backup():
    st.warning("On Streamlit Community Cloud the app's files are wiped whenever the app restarts or redeploys. "
               "Download a backup regularly and restore it here if that happens.")
    present = [f for f in DATA_FILES if os.path.exists(f)]
    if present:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in present:
                z.write(f)
        st.download_button(f"💾 Download backup ({len(present)} files)", buf.getvalue(),
                           f"riskradar_backup_{datetime.now():%Y%m%d_%H%M}.zip", "application/zip", key="xt_bk")
    else:
        st.caption("Nothing to back up yet.")
    up = st.file_uploader("Restore from a backup .zip (overwrites current data)", type=["zip"], key="xt_restore")
    if up is not None and st.button("Restore now", key="xt_restore_btn"):
        try:
            with zipfile.ZipFile(io.BytesIO(up.getvalue())) as z:
                restored = []
                for name in z.namelist():
                    base = os.path.basename(name)
                    if base in DATA_FILES:             # only our own files, never arbitrary paths
                        with open(base, "wb") as fh:
                            fh.write(z.read(name))
                        restored.append(base)
            st.success(f"Restored: {', '.join(restored) or 'nothing recognised in the zip'}")
            st.rerun()
        except Exception as e:
            st.error(f"Could not restore: {e}")


# ================================================================ 3) sidebar
def sidebar_policy_settings():
    with st.expander("🛡️ Lending Policy Limits"):
        st.caption("Affordability rules checked on every evaluation, on top of the AI model.")
        st.number_input("Max monthly payment-to-income (%)", 5.0, 100.0, DEFAULT_POLICY["max_pti"], 1.0, key="pol_max_pti")
        st.number_input("Max loan-to-annual-income (%)", 20.0, 1000.0, DEFAULT_POLICY["max_lti"], 10.0, key="pol_max_lti")
        st.number_input("Max loan-to-asset (%)", 10.0, 500.0, DEFAULT_POLICY["max_lta"], 5.0, key="pol_max_lta")
        st.number_input("Minimum credit score", 300, 900, DEFAULT_POLICY["min_score"], 5, key="pol_min_score")
        st.checkbox("Require liquid balance ≥ 1 instalment", value=DEFAULT_POLICY["buffer"], key="pol_buffer")
    with st.expander("📨 Notifications (SMS + email)", expanded=True):
        st.text_input("Institution name (shown in messages)", "GAWFA", key="inst_name")
        st.checkbox("Email the applicant (with decision letter PDF)", value=True, key="notify_email")
        st.checkbox("SMS the applicant", value=True, key="notify_sms")
        st.caption(("✅ Email is set up" if email_configured() else "❌ Email not set up (SMTP_* secrets)"))
        st.caption((f"✅ SMS set up ({sms_provider()})" if sms_configured() else f"❌ SMS not set up ({sms_provider()} secrets)"))
        if str(_secret("SMS_TEST_MODE", "false")).lower() == "true":
            st.caption("⚠️ SMS TEST MODE is on - nothing is delivered")
        tp = st.text_input("Test phone number", placeholder="+220 7xxxxxx", key="test_phone")
        if st.button("Send test SMS", key="test_sms_btn"):
            ok, note = send_sms_detailed(tp, f"{_inst()}: test message from MFI RiskRadar.")
            (st.success if ok else st.error)("SMS sent. " + note if ok else f"Failed - {note}")
        te = st.text_input("Test email address", key="test_email")
        if st.button("Send test email", key="test_email_btn"):
            ok, note = send_email_detailed(te, f"Test from {_inst()}", "This is a test message.", "<p>This is a test message.</p>")
            (st.success if ok else st.error)("Email sent." if ok else f"Failed - {note}")