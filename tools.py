"""Tool definitions for the RAG chatbot."""
import os
import re
import markdown
from datetime import datetime
from langchain_core.tools import tool
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def _remove_chartjs_blocks(text):
    """Remove Chart.js JSON blocks and replace with text description.

    Args:
        text: Text potentially containing Chart.js blocks

    Returns:
        Cleaned text with Chart.js blocks converted to descriptions
    """
    import re
    # Pattern to match Chart.js blocks
    pattern = r"```chartjs\s*([\s\S]*?)```"

    def replace_chart(match):
        chart_json = match.group(1).strip()
        return (
            "[📊 CHART DATA INCLUDED]\n"
            "Note: This report includes chart visualization data. "
            "For interactive charts, please view this report in the web application.\n"
        )

    # Replace all Chart.js blocks with description
    cleaned = re.sub(pattern, replace_chart, text)
    return cleaned


@tool
def generate_pdf_tool(report_text: str):
    """Generate a PDF document from provided text.

    Useful when the user asks to generate, save, or download content.
    Converts markdown to reportlab-compatible HTML.
    Note: Chart.js JSON blocks are converted to text descriptions.

    Args:
        report_text: The content to include in the PDF (supports markdown)

    Returns:
        Success message with file path, or error message
    """
    output_dir = "outputs"
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"Summary_{timestamp}.pdf")

        # Remove Chart.js JSON blocks (can't render in PDF)
        # Replace with text placeholder
        cleaned_text = _remove_chartjs_blocks(report_text)

        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Convert markdown to HTML
        html_text = markdown.markdown(cleaned_text, extensions=['extra', 'nl2br'])

        # Clean HTML for reportlab compatibility
        clean_html = html_text.replace('<p>', '').replace('</p>', '<br/><br/>')
        clean_html = clean_html.replace('<strong>', '<b>').replace('</strong>', '</b>')
        clean_html = clean_html.replace('<em>', '<i>').replace('</em>', '</i>')
        clean_html = clean_html.replace('<ul>', '').replace('</ul>', '')
        clean_html = clean_html.replace('<li>', ' • ').replace('</li>', '<br/>')
        clean_html = re.sub(
            r'<(?!/?(b|i|u|br|strike|link|a|font|super|sub|bullet|span|strong|em))[^>]+>',
            '',
            clean_html
        )

        story = []
        story.append(Paragraph("<b>DocuBot: Professional Summary Report</b>", styles['Title']))
        story.append(Spacer(1, 20))

        body_style = styles['Normal']
        body_style.leading = 14

        for part in clean_html.split('<br/><br/>'):
            if part.strip():
                story.append(Paragraph(part.strip(), body_style))
                story.append(Spacer(1, 10))

        doc.build(story)
        return f"PDF created successfully at: {file_path}"

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot create output directory: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid report content: {e}")


# Tool registry
ALL_TOOLS = [generate_pdf_tool]
