import os
import json
import time
from typing import Dict, Any, List

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class ReportGenerator:
    """
    Generates downloadable PDF execution reports containing query details,
    textual answers, confidence scores, visual overlays metadata, and auditable execution trace.
    """

    @staticmethod
    def generate_pdf(
        query_result: Dict[str, Any],
        output_pdf_path: str = "SatQuery_AI_Report.pdf"
    ) -> str:
        if not HAS_REPORTLAB:
            print("[Warning] `reportlab` not installed. Creating fallback text report.")
            txt_path = output_pdf_path.replace(".pdf", ".txt")
            with open(txt_path, "w") as f:
                f.write(json.dumps(query_result, indent=2))
            return txt_path

        doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=12
        )
        story.append(Paragraph("SatQuery AI — Execution & Evidence Report", title_style))
        story.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Query & Answer Box
        story.append(Paragraph("<b>User Natural Language Query:</b>", styles['Heading2']))
        story.append(Paragraph(f"<i>\"{query_result.get('query', query_result.get('execution_trace', {}).get('query', 'N/A'))}\"</i>", styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>SatQuery AI Generated Response:</b>", styles['Heading2']))
        # Use 'analysis' first (tool outputs put caption/analysis there), fall back to 'answer'
        answer_text = query_result.get('analysis') or query_result.get('answer') or query_result.get('result') or 'N/A'
        story.append(Paragraph(str(answer_text), styles['Normal']))
        story.append(Spacer(1, 10))

        # Confidence & Execution Info Table
        trace = query_result.get('execution_trace', {})
        table_data = [
            ["Metric / Field", "Value"],
            ["Selected Task", str(trace.get('selected_task', 'N/A'))],
            ["Confidence Score", f"{query_result.get('confidence', 0.0) * 100:.1f}%"],
            ["Tools Executed", ", ".join(trace.get('selected_tools', []))],
            ["Execution Latency", f"{trace.get('execution_time_ms', 0)} ms"],
            ["Status", str(trace.get('status', 'SUCCESS'))]
        ]
        t = Table(table_data, colWidths=[200, 300])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1'))
        ]))
        story.append(t)
        story.append(Spacer(1, 14))

        # Auditable Execution Trace Log JSON
        story.append(Paragraph("<b>Auditable Execution Trace (JSON):</b>", styles['Heading2']))
        json_str = json.dumps(trace, indent=2)
        trace_style = ParagraphStyle('TraceStyle', parent=styles['Code'], fontSize=8, leading=10)
        story.append(Paragraph(f"<pre>{json_str}</pre>", trace_style))

        doc.build(story)
        print(f"[Report Generator] PDF Report saved to {output_pdf_path}")
        return output_pdf_path