"""Generate the team briefing PDF for the SIH deck.

Written for teammates who are not ML people: plain language, every technical
term defined the first time it appears, and slide-by-slide guidance.

    python scripts/make_team_brief.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "RAAHAT_team_brief.pdf"

INK = colors.HexColor("#14181D")
SLATE = colors.HexColor("#3A4450")
MIST = colors.HexColor("#8A96A4")
RULE = colors.HexColor("#DDE3E9")
PAPER = colors.HexColor("#F2F4F6")
GREEN = colors.HexColor("#1B8A3F")
ORANGE = colors.HexColor("#F07C00")
RED = colors.HexColor("#D31F26")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=24, leading=28, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica",
                          fontSize=11, leading=15, textColor=SLATE, spaceAfter=10),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=15, leading=19, textColor=INK,
                         spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11.5, leading=15, textColor=INK,
                         spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=9.5, leading=13.5, textColor=INK,
                           alignment=TA_LEFT, spaceAfter=6),
    "small": ParagraphStyle("sm", parent=ss["Normal"], fontName="Helvetica",
                            fontSize=8.5, leading=12, textColor=SLATE, spaceAfter=5),
    "note": ParagraphStyle("n", parent=ss["Normal"], fontName="Helvetica-Oblique",
                           fontSize=9, leading=13, textColor=SLATE,
                           leftIndent=8, spaceAfter=6),
    "cell": ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=8.5, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cb", parent=ss["Normal"], fontName="Helvetica-Bold",
                            fontSize=8.5, leading=11, textColor=INK),
    "slide": ParagraphStyle("sl", parent=ss["Heading2"], fontName="Helvetica-Bold",
                            fontSize=12.5, leading=16, textColor=colors.white,
                            spaceBefore=0, spaceAfter=0),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def bullets(items, style="body"):
    return ListFlowable(
        [ListItem(Paragraph(i, S[style]), leftIndent=12) for i in items],
        bulletType="bullet", start="•", leftIndent=14, bulletFontSize=8,
        spaceBefore=1, spaceAfter=6,
    )


def rule(space_before=4, space_after=8):
    return HRFlowable(width="100%", thickness=0.7, color=RULE,
                      spaceBefore=space_before, spaceAfter=space_after)


def table(rows, widths, header=True, highlight=None, align_right=None):
    data = []
    for ri, row in enumerate(rows):
        out = []
        for ci, c in enumerate(row):
            sty = "cellb" if (header and ri == 0) else "cell"
            if highlight and ri in highlight:
                sty = "cellb"
            out.append(Paragraph(str(c), S[sty]))
        data.append(out)
    t = Table(data, colWidths=widths, hAlign="LEFT")
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), PAPER),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.8, MIST)]
    for r in (highlight or []):
        cmds.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#EAF4EE")))
    if align_right:
        for c in align_right:
            cmds.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(cmds))
    return t


def slide_header(n, title, colour):
    t = Table([[Paragraph(f"SLIDE {n} — {title}", S["slide"])]],
              colWidths=[170 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colour),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="RAAHAT - Team Briefing", author="RAAHAT team",
    )
    W = 170 * mm
    st = []

    # ---------------------------------------------------------- cover ---
    st += [
        P("RAAHAT", "title"),
        P("<b>R</b>egime-<b>A</b>ware <b>A</b>djustment of <b>H</b>eavy-rainfall "
          "<b>A</b>lerts &amp; <b>T</b>hresholds", "sub"),
        rule(0, 8),
        P("<b>Team briefing — everything you need to build the deck.</b> "
          "Written for non-technical readers. Every technical word is explained "
          "the first time it appears. Slide-by-slide guidance starts on page 4.", "body"),
        Spacer(1, 6),
        P("<b>What the project is, in one paragraph.</b> When heavy rain is coming, "
          "someone in a government office must decide whether to warn a district. "
          "They use computer weather predictions, and those predictions are wrong "
          "in ways that repeat. RAAHAT is a layer that sits on top of those "
          "predictions and corrects them before a warning goes out. It does not "
          "forecast weather. It cleans up somebody else's forecast.", "body"),
        Spacer(1, 4),
        P("<b>The one idea that makes it different.</b> Weather models don't make "
          "random mistakes — the mistake depends on what kind of weather is "
          "happening. The same model over-predicts drizzle in a dry spell and "
          "badly under-predicts a Bay of Bengal storm. Everyone today corrects "
          "them the same way regardless. We work out what kind of weather is "
          "coming first, then apply the correction that situation deserves.", "body"),
        rule(),
        P("HEADLINE NUMBERS (all measured on real data the model had never seen)", "h2"),
        table([
            ["Measure", "Raw European forecast", "RAAHAT", "Change"],
            ["Heavy-rain detection score", "0.054", "<b>0.180</b>", "<b>3.3x better</b>"],
            ["Catches heavy rain (of 168 events)", "1 in 15", "<b>1 in 3</b>", "POD 0.065  to  0.339"],
            ["Rainfall error (mm/day)", "13.76", "<b>12.64</b>", "8% lower"],
            ["Agreement with reality", "0.502", "<b>0.578</b>", "higher"],
            ["False alarms", "0.756", "0.723", "slightly fewer"],
        ], [58 * mm, 40 * mm, 34 * mm, 38 * mm], highlight=[1]),
        Spacer(1, 4),
        P("Tested on August–September 2025, 197 districts, 12,017 district-days. "
          "The test data was locked away and opened exactly once, after the model "
          "was frozen — so these numbers cannot have been tuned to look good.", "small"),
        rule(),
        P("THE RESULT WE LEAD WITH", "h2"),
        P("Our clever step — recognising the weather situation — helps <b>most where "
          "the theory says it should</b>, and not elsewhere. That pattern is much "
          "stronger evidence than a flat improvement everywhere.", "body"),
        table([
            ["Weather situation", "Raw forecast", "Without our step", "With our step"],
            ["Low-pressure system (storm)", "0.023", "0.171", "<b>0.232  (+36%)</b>"],
            ["Ordinary unsettled weather", "0.046", "0.139", "0.142  (+2%)"],
        ], [55 * mm, 33 * mm, 40 * mm, 42 * mm], highlight=[1]),
        Spacer(1, 3),
        P("In storm conditions the correction is <b>ten times</b> better than the raw "
          "forecast. In ordinary weather our extra step adds nothing — which is "
          "exactly right, because there is no distinctive situation to recognise.", "small"),
    ]

    # ------------------------------------------------------ page 2: words ---
    st += [PageBreak(), P("Words your audience may not know", "h1"),
           P("Use these definitions on slides or in the script. Judges will ask.", "small"),
           rule(0, 6)]

    gloss = [
        ["Post-processing", "Correcting someone else's forecast after they produced it. "
         "This is what we do. We are not a weather model."],
        ["NWP (Numerical Weather Prediction)", "The big physics simulations that produce "
         "forecasts — run by Europe, Germany, USA, India. We consume their output."],
        ["Lead time / Day-3 forecast", "How far ahead. 'Day 3' means predicted three days "
         "before it happens. Warnings are usually decided around day 3."],
        ["Regime", "The kind of weather situation. We use four: active monsoon spell, "
         "break (a lull), a low-pressure system (storm), and ordinary unsettled."],
        ["LPS / Low-pressure system", "A monsoon depression — the spinning storms that "
         "drift in from the Bay of Bengal and cause most severe flooding."],
        ["POD (Probability of Detection)", "Of all the heavy-rain events that happened, "
         "what fraction did we warn about? Higher is better."],
        ["FAR (False Alarm Ratio)", "Of all our warnings, what fraction were wrong? "
         "Lower is better."],
        ["CSI (Critical Success Index)", "A single score combining both — rewards "
         "catching events, penalises false alarms. Our headline number."],
        ["RMSE", "Average size of the error in millimetres. Lower is better."],
        ["Calibration", "Does '70% chance' actually happen 70% of the time? A warning "
         "system that says 70% and means 30% is dangerous."],
        ["Held-out test", "Data deliberately hidden from the model during building, "
         "used once at the end. The only honest measure of how good something is."],
        ["Ablation", "Switching one part off to see whether it mattered. We switched "
         "off our regime step to prove it earns its place."],
    ]
    st += [table([["Term", "What it means"]] + gloss, [48 * mm, 122 * mm])]

    st += [
        rule(10, 6),
        P("How the system works, in five steps", "h1"),
        bullets([
            "<b>1. Read someone else's forecast.</b> We take the European (ECMWF) and "
            "German (ICON) weather models' predictions for the next 1–5 days.",
            "<b>2. Work out what kind of weather is coming.</b> From the predicted "
            "pressure patterns and rainfall across a grid over India, we identify "
            "whether a storm, an active spell, a lull, or ordinary weather is due.",
            "<b>3. Apply the matching correction.</b> Each situation gets the "
            "adjustment it deserves, learned from two monsoons of past mistakes.",
            "<b>4. Produce a range, not a single number.</b> Instead of '180 mm' we "
            "give 'most likely 180 mm, with a 41% chance of exceeding 204.5 mm'. "
            "Those cut-offs are IMD's own official heavy / very heavy / extremely "
            "heavy boundaries.",
            "<b>5. Turn it into a colour.</b> Green / yellow / orange / red, using a "
            "dial the user controls for how cautious they want to be.",
        ]),
        rule(6, 6),
        P("Two safety rules worth mentioning on stage", "h2"),
        bullets([
            "<b>It can never quietly lower a warning.</b> If the raw forecast would "
            "have said orange, we cannot say yellow — unless we are genuinely "
            "confident it is dry. A tool that silently downgrades a real warning is "
            "worse than no tool.",
            "<b>Warnings don't flicker.</b> A district cannot flip orange–yellow–orange "
            "between updates. A real escalation passes through untouched.",
        ]),
    ]

    # ------------------------------------------------- page 3: honesty ---
    st += [
        PageBreak(),
        P("What we must say honestly (this wins marks, it doesn't lose them)", "h1"),
        P("Judges trust teams that state their own limits before being asked. "
          "Every item here is measured, not guessed.", "small"),
        rule(0, 6),
        P("1. We miss most heavy-rain events", "h2"),
        P("Of 168 heavy-rain events, <b>108 (64%) were never flagged</b> — the model "
          "correctly knows about 1% of days will be heavy, but often cannot say "
          "<i>which</i> 1%. We improved detection from 1-in-15 to 1-in-3; that is "
          "real progress, not a solved problem. Say the 1-in-3, not 'we detect "
          "heavy rain'.", "body"),
        P("2. The clever part only clearly helps in storms", "h2"),
        P("Averaged across all weather, our regime step adds very little and is "
          "within measurement noise. Split by situation, it adds 36% in "
          "low-pressure-system conditions. <b>Quote the split figure, and say it is "
          "a split figure.</b>", "body"),
        P("3. The probabilities are under-confident", "h2"),
        P("When the system said '29% chance of heavy rain', it actually happened "
          "63% of the time. We diagnosed why — the calibration step had too few "
          "examples and flattened the top of its range — and fixed it (error down "
          "4.6x). The fix could not be re-tested on the same held-out data, so the "
          "reported numbers are the conservative ones.", "body"),
        P("4. Two of six weather situations cannot be detected", "h2"),
        P("Western Disturbances and coastal-easterly regimes need upper-atmosphere "
          "data that is not freely available. They are shown as zero rather than "
          "quietly dropped.", "body"),
        P("5. Small sample", "h2"),
        P("Two monsoons, 197 districts. For the rarest rainfall there simply are not "
          "enough events, and we mark those cells 'insufficient data' rather than "
          "printing a confident-looking number built on four events.", "body"),
        rule(8, 6),
        P("The framing that protects us", "h2"),
        P("<b>We are not competing with IMD.</b> We consume the same kind of output "
          "they do and calibrate it. Their warnings mean 'heavy rain at isolated "
          "places within a district'; ours estimates a district-wide average. "
          "Different quantities. Say this on the slide before a judge says it to "
          "you — it turns a weakness into evidence that we understand the domain.", "body"),
        rule(8, 6),
        P("Where the data comes from (all free, no payment, no special access)", "h2"),
        table([
            ["Data", "Source", "Notes"],
            ["Actual rainfall (the truth)", "India Meteorological Department, Pune",
             "Official 0.25° gauge-based grid, 16 years (2010–2025)"],
            ["Forecasts", "ECMWF (Europe) + ICON (Germany) via Open-Meteo",
             "Free, no key, CC BY 4.0 licence"],
            ["Pressure patterns", "ERA5 reanalysis via Open-Meteo", "Used to spot storms"],
            ["District boundaries", "Datameet, Census 2011", "Open licence (ODbL)"],
        ], [42 * mm, 58 * mm, 70 * mm]),
        Spacer(1, 3),
        P("Required credit line for the deck and any screen: "
          "<i>“Weather data by Open-Meteo.com (CC BY 4.0). Observed rainfall: India "
          "Meteorological Department, Pune.”</i>", "small"),
    ]

    # -------------------------------------------- page 4: build status ---
    st += [
        PageBreak(),
        P("What is built, and what comes next", "h1"),
        P("Use this to decide what the deck may claim. Anything in the second or "
          "third table must be described as planned, not shown as working.", "small"),
        rule(0, 6),
        P("BUILT AND MEASURED — safe to claim outright", "h2"),
        table([
            ["Piece", "What it does", "Evidence"],
            ["Data pipeline", "IMD actual rainfall + European and German forecasts, "
             "197 districts, two monsoons", "0% missing; matched a known flood event"],
            ["Weather-situation recogniser", "Identifies storm / active spell / lull / "
             "ordinary, 1–5 days ahead", "Finds real storms, e.g. the Aug 2024 Gujarat "
             "depression"],
            ["The correction model", "Adjusts the forecast per situation; outputs a "
             "range plus probabilities", "3.3x detection vs raw forecast"],
            ["Comparison tests", "Five alternatives tested side by side",
             "Table on Slide 4"],
            ["Held-out evaluation", "Tested once on hidden data, model frozen first",
             "Full audit trail recorded"],
            ["Warning colours", "Green/yellow/orange/red, never silently downgrades",
             "16 automated tests"],
            ["Explanations", "Top three reasons for each forecast, plus similar past days",
             "12 automated tests"],
            ["Error Atlas", "Measures how wrong forecasts are per situation and region",
             "Built; needs more data to fill every cell"],
        ], [36 * mm, 76 * mm, 58 * mm]),
        Spacer(1, 6),
        P("NEXT — planned before the finale, describe as 'in progress'", "h2"),
        table([
            ["Piece", "Why it matters", "State"],
            ["The screen", "Map, district panel, sliders — what judges actually look at",
             "Project set up; no code written yet"],
            ["All-India expansion", "197 to 641 districts, ~2.8x more events; would fill "
             "the Atlas and the empty rows of our evidence table",
             "Started; data fetch stalled at 12%"],
            ["Offline demo mode", "So the presentation cannot fail on venue wifi",
             "Not started"],
            ["Web service check", "Confirm the service serves the screen correctly",
             "Written, not yet verified"],
            ["2023 as extra training data", "A third monsoon, single-lead only",
             "Confirmed available; not ingested"],
        ], [36 * mm, 76 * mm, 58 * mm]),
        Spacer(1, 6),
        P("LATER — mention as future work only", "h2"),
        table([
            ["Piece", "Why it is not done"],
            ["A more advanced correction model (CSGD mixture)",
             "The current one already works; this would refine the rainfall range. "
             "We found and documented a technical obstacle and two ways round it."],
            ["Benchmark against IMD's own issued warnings",
             "Needs a registration on IMD's public API that has not been completed. "
             "<b>This is the single most time-critical item</b> — the archive cannot be "
             "backfilled, so every day without it is lost permanently."],
            ["Hindi bulletins, mobile view, more regions",
             "Straightforward but not yet needed for the demo."],
        ], [62 * mm, 108 * mm]),
        Spacer(1, 5),
        P("<b>How to present the roadmap.</b> Put the built/next split on Slide 4 "
          "(Feasibility) as a small two-column box. A team that shows a credible "
          "next step scores better than one that implies everything is finished — "
          "and far better than one caught claiming something that does not run.", "note"),
    ]

    # ------------------------------------------------ slides 1 & 2 ---
    st += [PageBreak(), P("Slide-by-slide guidance", "h1"),
           P("One idea per slide. Under 25 words of body text per slide. The four "
             "IMD warning colours (green/yellow/orange/red) should be the only "
             "strong colours anywhere in the deck.", "small"), Spacer(1, 6)]

    st += [
        slide_header(1, "INTRODUCTION / PROBLEM", INK), Spacer(1, 5),
        P("<b>The one message:</b> Weather models make the same mistake every time a "
          "particular weather pattern occurs — so the mistake is predictable, and "
          "therefore fixable.", "body"),
        P("<b>Put on the slide</b>", "h2"),
        bullets([
            "A single hook number: <b>“At three days out, the operational forecast "
            "catches one heavy-rain event in fifteen.”</b>",
            "One sentence on what we build: a correction layer between the forecast "
            "and the district warning desk.",
            "One line stating we do NOT build a weather model and do NOT compete "
            "with IMD.",
            "Who uses it: IMD duty forecaster, District Collector / disaster "
            "management officer.",
        ]),
        P("<b>Best visual:</b> one scatter plot — predicted rainfall against what "
          "actually fell, with dots coloured by weather situation, so the different "
          "situations visibly sit on different lines. That single picture is the "
          "whole argument.", "note"),
        P("<b>Do NOT put:</b> team photos, college logos, a generic 'India faces "
          "floods' paragraph, or more than three lines of text.", "small"),
        Spacer(1, 10),
        slide_header(2, "TECHNICAL STACK", SLATE), Spacer(1, 5),
        P("<b>The one message:</b> Everything is free, open and runs on an ordinary "
          "laptop — no GPU, no paid data, no special government access.", "body"),
        P("<b>Put on the slide</b>", "h2"),
        table([
            ["Layer", "What we use", "Status"],
            ["Data", "IMD gridded rainfall; ECMWF + ICON forecasts via Open-Meteo",
             "<b>Working</b> — free, open licences"],
            ["Learning", "LightGBM (decision-tree model), scikit-learn",
             "<b>Working</b> — trains in minutes, CPU only"],
            ["Storage", "Parquet files + DuckDB", "<b>Working</b> — no database server"],
            ["Service", "Python FastAPI", "<b>Written</b>, end-to-end check pending"],
            ["Screen", "React + MapLibre map", "<b>Planned</b> — set up, not yet built"],
            ["Checks", "74 automated tests", "<b>Working</b> — all passing"],
        ], [26 * mm, 76 * mm, 68 * mm]),
        Spacer(1, 3),
        P("<b>Keep the status column on the slide.</b> Judges are far harder on a team "
          "that implies something works than on one that says plainly what is built "
          "and what is next. See the build-status page for the full picture.", "small"),
        Spacer(1, 4),
        P("<b>Say out loud:</b> “No GPU anywhere. Trains in minutes. Every input is "
          "free and openly licensed.” That is a genuine competitive claim — many "
          "teams cannot say it.", "note"),
        P("<b>Explain if asked why not deep learning:</b> we have two monsoons of "
          "data. A large neural network would simply memorise it. Decision trees "
          "are the right size of tool for the size of data we have — and we can "
          "explain every prediction, which a neural network cannot.", "small"),
    ]

    # ------------------------------------------------ slides 3 & 4 ---
    st += [
        PageBreak(),
        slide_header(3, "ARCHITECTURE / HOW IT WORKS", INK), Spacer(1, 5),
        P("<b>The one message:</b> We predict the weather situation from the forecast "
          "itself — so it works three days ahead, not just in hindsight.", "body"),
        P("<b>Draw four boxes left to right. Not ten.</b>", "h2"),
        table([
            ["1. DATA", "2. RECOGNISE", "3. CORRECT", "4. DECIDE"],
            ["Forecasts from ECMWF &amp; ICON.<br/>Actual rainfall from IMD.<br/>"
             "Pressure patterns.",
             "Work out the weather situation for each day, 1–5 days ahead, "
             "from the <b>forecast</b> — not from what already happened.",
             "Apply the correction that situation deserves. Output a range of "
             "possible rainfall, not one number.",
             "Convert to a probability of crossing IMD's official thresholds, "
             "then to a warning colour."],
        ], [40 * mm, 43 * mm, 43 * mm, 44 * mm]),
        Spacer(1, 6),
        P("<b>Three callouts to annotate the diagram with</b> — these are what a "
          "judge should read even if they read nothing else:", "h2"),
        bullets([
            "On box 2: <i>“The weather situation is FORECAST, not diagnosed after "
            "the fact. That is what makes it usable at day 3.”</i>",
            "On box 3: <i>“We blend situations rather than picking one. Real days "
            "are mixtures — a storm inside an active spell is one day, not two.”</i>",
            "On box 4: <i>“The caution level is a dial the operator controls, not a "
            "number we hard-coded.”</i>",
        ]),
        P("<b>Good second visual:</b> a bar chart of the weather-situation "
          "probabilities changing as a storm approaches, with the correction size "
          "tracking underneath. That is the most watchable fifteen seconds you have.", "note"),
        Spacer(1, 10),
        slide_header(4, "FEASIBILITY AND VIABILITY", ORANGE), Spacer(1, 5),
        P("<b>The one message:</b> It is already built and already measured on data "
          "it had never seen. This is not a proposal.", "body"),
        P("<b>Put the evidence table on the slide</b> — this is the most important "
          "table in the deck:", "h2"),
        table([
            ["What was tested", "Detection score", "Verdict"],
            ["Long-term averages (do nothing)", "0.000", "no skill"],
            ["Raw European forecast", "0.054", "the starting point"],
            ["Blending several models (what's done today)", "0.069", "barely helps"],
            ["Standard textbook correction", "0.138", "helps, but worsens accuracy"],
            ["RAAHAT without our clever step", "0.171", "post-processing works"],
            ["RAAHAT, full", "<b>0.180</b>", "<b>3.3x the raw forecast</b>"],
        ], [70 * mm, 34 * mm, 66 * mm], highlight=[6]),
        Spacer(1, 4),
        bullets([
            "<b>Feasible:</b> free data, no GPU, trains in minutes, 74 automated tests.",
            "<b>Honest:</b> the test data was locked and opened once, after the model "
            "was frozen. Point at the row where we did not improve and say so.",
            "<b>Scalable:</b> runs as a nightly batch job. Rainfall forecasts update "
            "four times a day — nothing here needs to be real-time.",
            "<b>Risks we name first:</b> two monsoons of data; we miss 64% of events; "
            "two weather situations we cannot yet detect.",
        ]),
        Spacer(1, 2),
        P("<b>Add a small two-column roadmap box to this slide</b> — it is the "
          "natural home for it and it answers 'what would you do next?' before a "
          "judge asks:", "h2"),
        table([
            ["Working today", "Next"],
            ["Data pipeline, situation recogniser, correction model, warning colours, "
             "explanations, held-out evaluation",
             "The forecaster's screen; all-India (197 to 641 districts); offline demo "
             "mode; benchmark against IMD's own issued warnings"],
        ], [85 * mm, 85 * mm]),
    ]

    # ---------------------------------------------------- slide 5 ---
    st += [
        PageBreak(),
        slide_header(5, "IMPACT AND BENEFITS", GREEN), Spacer(1, 5),
        P("<b>The one message:</b> Better warnings, earlier, with fewer false alarms — "
          "using forecasts India already produces.", "body"),
        P("<b>Put on the slide</b>", "h2"),
        table([
            ["Benefit", "The evidence"],
            ["<b>Catches 5x more heavy-rain events</b>",
             "1 in 15  to  1 in 3 at three days ahead"],
            ["<b>Buys about three days of lead time</b>",
             "Our day-5 forecast is as accurate as the raw day-2 forecast"],
            ["<b>Biggest gain exactly where it matters</b>",
             "+36% in low-pressure-system conditions — the storms that cause "
             "most flood deaths"],
            ["<b>Fewer unnecessary red alerts</b>",
             "False alarms fall from 0.756 to 0.514 in the conservative setting"],
            ["<b>Costs almost nothing to run</b>",
             "Free data, ordinary laptop, minutes to train"],
        ], [58 * mm, 112 * mm]),
        Spacer(1, 6),
        P("<b>Who benefits</b>", "h2"),
        bullets([
            "<b>Duty forecasters</b> — a district table with corrected rainfall, "
            "probabilities, a suggested colour, and the reason behind it.",
            "<b>District Collectors and disaster management officers</b> — a map and "
            "a plain-language bulletin with a confidence number.",
            "<b>People in affected districts</b> — fewer missed warnings, and fewer "
            "false red alerts that erode trust in the next real one.",
            "<b>Rain-fed farmers</b> — better three-day guidance for sowing, "
            "harvesting and irrigation decisions.",
        ]),
        P("<b>The closing line:</b> “Every input is free and open. It trains in "
          "minutes on a laptop with no GPU. It sits upstream of India's forecasting "
          "systems, not against them.”", "note"),
        rule(8, 6),
        P("Final reminders for whoever builds the deck", "h1"),
        bullets([
            "<b>One idea per slide.</b> Under 25 words of body text.",
            "<b>Every number carries its sample size.</b> Write “0.180 (168 events)”, "
            "not “0.180”.",
            "<b>Use only the four IMD warning colours</b> as strong colours. "
            "Everything else grey. When a district is red it must be the only red "
            "thing on the screen.",
            "<b>Show one place we did not improve.</b> It buys more credibility than "
            "anything else in the presentation.",
            "<b>Never imply live real-time capability.</b> Everything shown is replay "
            "of real past dates, and saying so is a strength.",
        ]),
        Spacer(1, 6),
        rule(2, 4),
        P("Generated from the project's own results files. All figures traceable to "
          "results/final/ and models/frozen/. Prepared for the RAAHAT team, SIH.", "small"),
    ]

    doc.build(st)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
