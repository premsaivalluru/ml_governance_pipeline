import os
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically calculate the total page count
    and print 'Page X of Y' in the footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#4B5563"))
        
        # Header
        self.drawString(54, 750, "AI Model Governance & Validation Report")
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        self.drawString(54, 40, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.line(54, 52, 558, 52)
        self.restoreState()

def make_progress_bar(val: float, bar_width: int = 80, bar_height: int = 10) -> Table:
    """Creates a native ReportLab vector progress bar for values ranging from 0.0 to 1.0."""
    val = max(0.0, min(1.0, val))
    filled_width = max(1, int(val * bar_width))
    empty_width = max(1, bar_width - filled_width)
    
    # Color coding based on status thresholds
    if val >= 0.85:
        color = colors.HexColor("#10B981") # Green
    elif val >= 0.70:
        color = colors.HexColor("#F59E0B") # Amber
    else:
        color = colors.HexColor("#EF4444") # Red
        
    bar = Table([["", ""]], colWidths=[filled_width, empty_width], rowHeights=[bar_height])
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), color),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    return bar

def make_diff_progress_bar(val: float, bar_width: int = 80, bar_height: int = 10) -> Table:
    """Creates a progress bar optimized for difference metrics (where lower is better, like bias parity difference)."""
    val = max(0.0, min(1.0, val))
    filled_width = max(1, int(val * bar_width))
    empty_width = max(1, bar_width - filled_width)
    
    # For difference: <= 0.10 is compliant (green), > 0.10 indicates bias (red)
    if val <= 0.10:
        color = colors.HexColor("#10B981")
    else:
        color = colors.HexColor("#EF4444")
        
    bar = Table([["", ""]], colWidths=[filled_width, empty_width], rowHeights=[bar_height])
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), color),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    return bar

def generate_pdf_report(
    output_path: str,
    model_details: Dict[str, Any],
    risk_score_info: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    fairness_results: Dict[str, Any] = None,
    drift_results: Dict[str, Any] = None,
    agent_reports: Dict[str, Any] = None
) -> str:
    """
    Generates a professional, print-ready PDF governance report with visual dashboard elements.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SubSectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#3B82F6"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=6
    )

    bold_body_style = ParagraphStyle(
        'BoldBodyText',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )

    story = []

    # 1. Document Title
    story.append(Spacer(1, 10))
    story.append(Paragraph("AI MODEL GOVERNANCE REPORT", title_style))
    story.append(Paragraph("Comprehensive Technical Validation & Regulatory Audit", body_style))
    story.append(Spacer(1, 15))

    # 2. Executive Dashboard (KPI Scorecards)
    story.append(Paragraph("1. Executive Summary Dashboard", h1_style))
    
    status = risk_score_info.get("status", "REJECTED")
    risk_level = risk_score_info.get("risk_level", "High")
    risk_score = risk_score_info.get("score", 0)
    
    status_color = "#10B981" if "APPROV" in status.upper() else ("#FBBF24" if "COND" in status.upper() or "REV" in status.upper() else "#EF4444")
    risk_color = "#10B981" if "LOW" in risk_level.upper() else ("#FBBF24" if "MED" in risk_level.upper() else "#EF4444")

    # Scorecard Styles
    kpi_title_style = ParagraphStyle(
        'KPITitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#4B5563"),
        alignment=1
    )
    kpi_val_style = ParagraphStyle(
        'KPIVal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1
    )
    kpi_status_style = ParagraphStyle(
        'KPIStatus',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor(status_color),
        alignment=1
    )
    kpi_risk_style = ParagraphStyle(
        'KPIRisk',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor(risk_color),
        alignment=1
    )

    kpi_data = [
        [
            Paragraph("GOVERNANCE RISK SCORE", kpi_title_style),
            Paragraph("OVERALL RISK LEVEL", kpi_title_style),
            Paragraph("DEPLOYMENT RECOMMENDATION", kpi_title_style)
        ],
        [
            Paragraph(f"{risk_score}/100", kpi_val_style),
            Paragraph(risk_level.upper(), kpi_risk_style),
            Paragraph(status.upper(), kpi_status_style)
        ]
    ]
    
    kpi_table = Table(kpi_data, colWidths=[160, 160, 180])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 15))

    # Model metadata block (Compact)
    story.append(Paragraph("Model Asset Details", h2_style))
    model_data = [
        [Paragraph("Model Name", bold_body_style), Paragraph(str(model_details.get("model_name", "N/A")), body_style)],
        [Paragraph("Version", bold_body_style), Paragraph(str(model_details.get("version", "N/A")), body_style)],
        [Paragraph("Algorithm", bold_body_style), Paragraph(str(model_details.get("algorithm", "N/A")), body_style)],
        [Paragraph("Evaluation Date", bold_body_style), Paragraph(datetime.now().strftime("%Y-%m-%d"), body_style)]
    ]
    model_table = Table(model_data, colWidths=[150, 350])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F9FAFB")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 15))

    # 3. Model Performance with Gauges
    story.append(Paragraph("2. Model Performance Evaluation", h1_style))
    story.append(Paragraph("The performance agent audited validation metrics against baseline thresholds. The results are detailed below:", body_style))
    story.append(Spacer(1, 5))
    
    perf_headers = [
        Paragraph("Metric", bold_body_style), 
        Paragraph("Value", bold_body_style), 
        Paragraph("Visual Gauge", bold_body_style), 
        Paragraph("Threshold", bold_body_style), 
        Paragraph("Status", bold_body_style)
    ]
    perf_rows = [perf_headers]
    
    thresholds = {"accuracy": 0.85, "roc_auc": 0.85, "precision": 0.80, "recall": 0.80, "f1_score": 0.80}
    
    # Normalize and extract metrics
    display_perf = {}
    for k, v in performance_metrics.items():
        if isinstance(v, (int, float)):
            display_perf[k] = float(v)
            
    class_report = performance_metrics.get("classification_report", {})
    if isinstance(class_report, dict):
        for avg_key in ["weighted avg", "macro avg", "1"]:
            if avg_key in class_report and isinstance(class_report[avg_key], dict):
                sub = class_report[avg_key]
                if "precision" in sub and "precision" not in display_perf:
                    display_perf["precision"] = float(sub["precision"])
                if "recall" in sub and "recall" not in display_perf:
                    display_perf["recall"] = float(sub["recall"])
                if "f1-score" in sub and "f1_score" not in display_perf:
                    display_perf["f1_score"] = float(sub["f1-score"])
                break
                
    for metric_name, val in display_perf.items():
        if metric_name in thresholds:
            thresh = thresholds[metric_name]
            passed = val >= thresh
            status_text = "<font color='#10B981'><b>PASS</b></font>" if passed else "<font color='#EF4444'><b>FAIL</b></font>"
            
            perf_rows.append([
                Paragraph(metric_name.replace("_", " ").title(), body_style),
                Paragraph(f"{val:.4f}", body_style),
                make_progress_bar(val), # Progress bar table element
                Paragraph(f">= {thresh:.2f}", body_style),
                Paragraph(status_text, body_style)
            ])
            
    perf_table = Table(perf_rows, colWidths=[130, 60, 110, 100, 100])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E5E7EB")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 15))

    # 4. Fairness and Bias Audit with Diff Gauges
    if fairness_results and "error" not in fairness_results:
        story.append(Paragraph("3. Algorithmic Fairness Analysis", h1_style))
        story.append(Paragraph(f"Protected attribute evaluated: <b>{fairness_results.get('protected_attribute', 'N/A')}</b>", body_style))
        story.append(Spacer(1, 5))
        
        fair_rows = [
            [
                Paragraph("Fairness Metric", bold_body_style), 
                Paragraph("Value", bold_body_style), 
                Paragraph("Visual Status", bold_body_style), 
                Paragraph("Regulatory Tolerance", bold_body_style), 
                Paragraph("Assessment", bold_body_style)
            ],
            [
                Paragraph("Disparate Impact Ratio", body_style),
                Paragraph(f"{fairness_results['disparate_impact_ratio']:.4f}", body_style),
                # Normalized mapping of impact ratio center (ideal=1.0)
                make_progress_bar(min(1.0, fairness_results['disparate_impact_ratio'] / 1.25)),
                Paragraph("0.80 - 1.25", body_style),
                Paragraph("<font color='#10B981'><b>COMPLIANT</b></font>" if 0.8 <= fairness_results['disparate_impact_ratio'] <= 1.25 else "<font color='#EF4444'><b>BIAS DETECTED</b></font>", body_style)
            ]
        ]
        
        dp_val = fairness_results['demographic_parity_difference']
        fair_rows.append([
            Paragraph("Demographic Parity Diff", body_style),
            Paragraph(f"{dp_val:.4f}", body_style),
            make_diff_progress_bar(dp_val), # Color shifts to red if > 0.10
            Paragraph("<= 0.10", body_style),
            Paragraph("<font color='#10B981'><b>COMPLIANT</b></font>" if dp_val <= 0.10 else "<font color='#EF4444'><b>BIAS DETECTED</b></font>", body_style)
        ])
        
        if fairness_results.get("equal_opportunity_difference") is not None:
            eo_val = fairness_results["equal_opportunity_difference"]
            fair_rows.append([
                Paragraph("Equal Opportunity Diff", body_style),
                Paragraph(f"{eo_val:.4f}", body_style),
                make_diff_progress_bar(eo_val),
                Paragraph("<= 0.10", body_style),
                Paragraph("<font color='#10B981'><b>COMPLIANT</b></font>" if eo_val <= 0.10 else "<font color='#EF4444'><b>BIAS DETECTED</b></font>", body_style)
            ])
            
        fair_table = Table(fair_rows, colWidths=[140, 70, 100, 100, 90])
        fair_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E5E7EB")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(fair_table)
        story.append(Spacer(1, 15))

    # 5. Data Drift Detection
    if drift_results:
        story.append(Paragraph("4. Data & Feature Drift Analysis", h1_style))
        drift_share = drift_results.get("drift_share", 0.0)
        drift_status = "DRIFT DETECTED" if drift_results.get("detected", False) else "STABLE"
        drift_color = "#EF4444" if drift_status == "DRIFT DETECTED" else "#10B981"
        
        story.append(Paragraph(f"Feature Drift Status: <font color='{drift_color}'><b>{drift_status}</b></font> ({drift_share*100:.1f}% features drifted)", body_style))
        if drift_results.get("drifted_features"):
            story.append(Paragraph(f"<b>Affected Features:</b> {', '.join(drift_results['drifted_features'])}", body_style))
        else:
            story.append(Paragraph("No significant drift detected across model features.", body_style))
        story.append(Spacer(1, 15))

    # Page Break for Agent Details
    story.append(PageBreak())

    # 6. CrewAI Governance Agents Detail Audit (Formatted as Scorecard Cards)
    story.append(Paragraph("5. CrewAI Agent Validation Breakdown", h1_style))
    story.append(Paragraph("The model package was audited by five specialized AI agents. Their full reports are compiled below:", body_style))
    story.append(Spacer(1, 10))

    if agent_reports:
        for agent_key, report_data in agent_reports.items():
            if not isinstance(report_data, dict):
                continue
            
            agent_name = report_data.get("agent_name", agent_key.replace("_", " ").title())
            score = report_data.get("score")
            status = report_data.get("status", "N/A")
            findings = report_data.get("findings", [])
            recommendations = report_data.get("recommendations", [])
            
            status_color = "#10B981" if status == "PASS" else ("#FBBF24" if status == "WARNING" else "#EF4444")
            
            # Format findings and recommendations as bullet text
            findings_text = "<br/>".join([f"• {f}" for f in findings[:3]])
            recs_text = "<br/>".join([f"• {r}" for r in recommendations[:3]])
            
            agent_card_data = [
                [Paragraph(f"<b>{agent_name}</b> (Score: {score}/100) — Status: <font color='{status_color}'><b>{status}</b></font>", h2_style)],
                [Paragraph("<b>Key Audit Findings:</b>", bold_body_style)],
                [Paragraph(findings_text, body_style)],
                [Paragraph("<b>Actionable Recommendations:</b>", bold_body_style)],
                [Paragraph(recs_text, body_style)]
            ]
            
            agent_card_table = Table(agent_card_data, colWidths=[500])
            agent_card_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
                ('PADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (0,0), 6),
                ('BOTTOMPADDING', (-1,-1), (-1,-1), 6),
                ('LINEBELOW', (0,0), (0,0), 1, colors.HexColor("#E2E8F0")),
            ]))
            
            story.append(agent_card_table)
            story.append(Spacer(1, 12))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path
