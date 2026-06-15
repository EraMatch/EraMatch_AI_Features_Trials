"""PDF Renderer — WeasyPrint-based CV-to-PDF rendering with Jinja2 templates.

Extracted from cvs-eramatch.ipynb. Renders CVSchema objects to single-page A4 PDFs
using the 22 HTML+Jinja2 templates in the templates/ directory.

Usage:
    from pdf_renderer import render_cv_pdf, get_template_for_tier
    from schema import CVSchema
    from pathlib import Path

    ok = render_cv_pdf(cv, "t1_classic", Path("output.pdf"))
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from jinja2 import Template
import weasyprint

from schema import CVSchema

logger = logging.getLogger(__name__)

# Directory containing .html Jinja2 templates
TEMPLATES_DIR = Path(__file__).parent / "templates"

TIER_TEMPLATES: Dict[str, List[str]] = {
    "T1": [
        "T1_classic",
        "T1_modern",
        "T1_academic",
        "T1_functional",
        "T1_executive",
        "T1_fresher",
    ],
    "T2": [
        "T2_sidebar_left",
        "T2_sidebar_right",
        "T2_two_col",
        "T2_sidebar_icons",
        "T2_sidebar_photo",
        "T2_sidebar_timeline",
    ],
    "T3": [
        "T3_table",
        "T3_header_footer",
        "T3_nested_tables",
        "T3_card_layout",
        "T3_europass",
    ],
    "T4": ["T1_classic", "T2_sidebar_left", "T3_table", "T2_sidebar_icons"],
    "T5": ["T5_creative", "T5_minimal", "T5_infographic", "T5_magazine", "T5_dark"],
}


def get_template_for_tier(tier: str) -> str:
    """Return the first (primary) template name for a given tier.

    Args:
        tier: Tier string like "T1", "T2", etc.

    Returns:
        Template name string (e.g. "T1_classic").

    Raises:
        ValueError: If tier is not recognized.
    """
    tier_upper = tier.upper()
    if tier_upper not in TIER_TEMPLATES:
        raise ValueError(
            f"Unknown tier '{tier}'. Valid tiers: {list(TIER_TEMPLATES.keys())}"
        )
    return TIER_TEMPLATES[tier_upper][0]


TEMPLATE_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "T1_classic": {
        "page_width": 595,
        "page_height": 842,
        "sections": [
            {
                "type": "header",
                "label": "name",
                "y_range": (50, 80),
                "x_range": (50, 545),
            },
            {
                "type": "contact_block",
                "label": "contact_info",
                "y_range": (80, 100),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "summary",
                "y_range": (100, 120),
                "x_range": (50, 545),
            },
            {
                "type": "summary_block",
                "label": "summary_text",
                "y_range": (120, 150),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "experience",
                "y_range": (150, 170),
                "x_range": (50, 545),
            },
            {
                "type": "experience_entry",
                "label": "job_1",
                "y_range": (170, 220),
                "x_range": (50, 545),
            },
            {
                "type": "experience_entry",
                "label": "job_2",
                "y_range": (220, 270),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "education",
                "y_range": (270, 290),
                "x_range": (50, 545),
            },
            {
                "type": "education_entry",
                "label": "edu_1",
                "y_range": (290, 320),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "skills",
                "y_range": (320, 340),
                "x_range": (50, 545),
            },
            {
                "type": "skill_list",
                "label": "skills_text",
                "y_range": (340, 370),
                "x_range": (50, 545),
            },
        ],
    },
    "T2_sidebar_left": {
        "page_width": 595,
        "page_height": 842,
        "sections": [
            {
                "type": "sidebar",
                "label": "sidebar",
                "y_range": (0, 842),
                "x_range": (0, 208),
            },
            {
                "type": "header",
                "label": "name",
                "y_range": (20, 50),
                "x_range": (220, 545),
            },
            {
                "type": "contact_block",
                "label": "contact_info",
                "y_range": (50, 80),
                "x_range": (220, 545),
            },
            {
                "type": "section_header",
                "label": "experience",
                "y_range": (80, 100),
                "x_range": (220, 545),
            },
            {
                "type": "experience_entry",
                "label": "job_1",
                "y_range": (100, 150),
                "x_range": (220, 545),
            },
            {
                "type": "section_header",
                "label": "education",
                "y_range": (150, 170),
                "x_range": (220, 545),
            },
            {
                "type": "education_entry",
                "label": "edu_1",
                "y_range": (170, 200),
                "x_range": (220, 545),
            },
            {
                "type": "skill_list",
                "label": "skills_in_sidebar",
                "y_range": (200, 350),
                "x_range": (20, 188),
            },
        ],
    },
    "T2_sidebar_right": {
        "page_width": 595,
        "page_height": 842,
        "sections": [
            {
                "type": "sidebar",
                "label": "sidebar",
                "y_range": (0, 842),
                "x_range": (387, 595),
            },
            {
                "type": "header",
                "label": "name",
                "y_range": (20, 50),
                "x_range": (50, 380),
            },
            {
                "type": "contact_block",
                "label": "contact_info",
                "y_range": (50, 80),
                "x_range": (50, 380),
            },
            {
                "type": "section_header",
                "label": "experience",
                "y_range": (80, 100),
                "x_range": (50, 380),
            },
            {
                "type": "experience_entry",
                "label": "job_1",
                "y_range": (100, 150),
                "x_range": (50, 380),
            },
            {
                "type": "section_header",
                "label": "education",
                "y_range": (150, 170),
                "x_range": (50, 380),
            },
            {
                "type": "education_entry",
                "label": "edu_1",
                "y_range": (170, 200),
                "x_range": (50, 380),
            },
            {
                "type": "skill_list",
                "label": "skills_in_sidebar",
                "y_range": (200, 350),
                "x_range": (407, 575),
            },
        ],
    },
    "T2_two_col": {
        "page_width": 595,
        "page_height": 842,
        "sections": [
            {
                "type": "header",
                "label": "name",
                "y_range": (30, 60),
                "x_range": (50, 545),
            },
            {
                "type": "contact_block",
                "label": "contact_info",
                "y_range": (60, 80),
                "x_range": (50, 545),
            },
            {
                "type": "column_left",
                "label": "experience_col",
                "y_range": (80, 400),
                "x_range": (50, 290),
            },
            {
                "type": "column_right",
                "label": "education_skills_col",
                "y_range": (80, 400),
                "x_range": (305, 545),
            },
        ],
    },
    "T3_table": {
        "page_width": 595,
        "page_height": 842,
        "sections": [
            {
                "type": "header",
                "label": "name",
                "y_range": (30, 60),
                "x_range": (50, 545),
            },
            {
                "type": "contact_block",
                "label": "contact_info",
                "y_range": (60, 90),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "summary",
                "y_range": (90, 110),
                "x_range": (50, 545),
            },
            {
                "type": "table",
                "label": "experience_table",
                "y_range": (110, 300),
                "x_range": (50, 545),
            },
            {
                "type": "table",
                "label": "education_table",
                "y_range": (300, 400),
                "x_range": (50, 545),
            },
            {
                "type": "table",
                "label": "skills_table",
                "y_range": (400, 500),
                "x_range": (50, 545),
            },
        ],
    },
    "T5_creative": {
        "page_width": 595,
        "page_height": 842,
        "sections": [
            {
                "type": "hero_header",
                "label": "hero",
                "y_range": (0, 120),
                "x_range": (0, 595),
            },
            {
                "type": "header",
                "label": "name",
                "y_range": (10, 50),
                "x_range": (20, 575),
            },
            {
                "type": "contact_block",
                "label": "contact_info",
                "y_range": (50, 80),
                "x_range": (20, 575),
            },
            {
                "type": "section_header",
                "label": "experience",
                "y_range": (120, 150),
                "x_range": (20, 575),
            },
            {
                "type": "experience_entry",
                "label": "job_1",
                "y_range": (150, 200),
                "x_range": (20, 575),
            },
            {
                "type": "section_header",
                "label": "education",
                "y_range": (200, 230),
                "x_range": (20, 575),
            },
            {
                "type": "section_header",
                "label": "skills",
                "y_range": (230, 260),
                "x_range": (20, 575),
            },
        ],
    },
    "T5_minimal": {
        "page_width": 595,
        "page_height": 842,
        "sections": [
            {
                "type": "header_centered",
                "label": "name",
                "y_range": (50, 90),
                "x_range": (100, 495),
            },
            {
                "type": "contact_block",
                "label": "contact_info",
                "y_range": (90, 110),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "about",
                "y_range": (110, 130),
                "x_range": (50, 545),
            },
            {
                "type": "summary_block",
                "label": "summary_text",
                "y_range": (130, 160),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "experience",
                "y_range": (160, 180),
                "x_range": (50, 545),
            },
            {
                "type": "experience_entry",
                "label": "job_1",
                "y_range": (180, 230),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "education",
                "y_range": (230, 250),
                "x_range": (50, 545),
            },
            {
                "type": "section_header",
                "label": "skills",
                "y_range": (250, 270),
                "x_range": (50, 545),
            },
        ],
    },
}

# Fallbacks: templates without explicit layout annotations use T1_classic
_FALLBACK_TEMPLATES = [
    "T1_modern",
    "T1_academic",
    "T1_functional",
    "T1_executive",
    "T1_fresher",
    "T2_sidebar_icons",
    "T2_sidebar_photo",
    "T2_sidebar_timeline",
    "T3_header_footer",
    "T3_nested_tables",
    "T3_card_layout",
    "T3_europass",
    "T5_infographic",
    "T5_magazine",
    "T5_dark",
]
for _tmpl in _FALLBACK_TEMPLATES:
    if _tmpl not in TEMPLATE_LAYOUTS:
        TEMPLATE_LAYOUTS[_tmpl] = TEMPLATE_LAYOUTS["T1_classic"]


def get_template_layout(template_name: str) -> dict:
    """Get layout annotations for a template.

    Since we control the templates, the structure is known exactly.
    Falls back to T1_classic layout for unknown templates.

    Args:
        template_name: Template name like "T1_classic", "T2_sidebar_left", etc.

    Returns:
        Dict with 'page_width', 'page_height', and 'sections' list.
    """
    return TEMPLATE_LAYOUTS.get(template_name, TEMPLATE_LAYOUTS["T1_classic"])


def _template_filename(template_name: str) -> str:
    """Convert template name to .html filename.

    "T1_classic" → "t1_classic.html"
    "t1_classic" → "t1_classic.html"
    """
    return template_name.lower() + ".html"


def _template_key(template_name: str) -> str:
    """ "t1_classic" → "T1_classic"; "T1_classic" → "T1_classic"."""
    if (
        template_name[0].islower()
        and len(template_name) > 1
        and template_name[1].isdigit()
    ):
        return template_name[0].upper() + template_name[1:]
    return template_name


def render_cv_pdf(
    cv: CVSchema,
    template_name: str,
    output_path: Path,
) -> bool:
    """Render a CVSchema object to a single-page A4 PDF using a Jinja2 template.

    Args:
        cv: CVSchema object with all CV data.
        template_name: Template name (e.g. "t1_classic" or "T1_classic").
            Both lowercase and title-case forms are accepted.
        output_path: Path to write the PDF file.

    Returns:
        True on success, False on failure (errors are logged, not raised).
    """
    try:
        fname = _template_filename(template_name)
        tmpl_path = TEMPLATES_DIR / fname

        if not tmpl_path.exists():
            logger.error(f"Template file not found: {tmpl_path}")
            return False

        tmpl_html = tmpl_path.read_text(encoding="utf-8")
        template = Template(tmpl_html)
        rendered_html = template.render(cv=cv)

        weasyprint.HTML(string=rendered_html).write_pdf(str(output_path))

        if output_path.exists() and output_path.stat().st_size > 0:
            return True
        else:
            logger.error(f"PDF output is empty or missing: {output_path}")
            return False

    except Exception as e:
        logger.error(f"render_cv_pdf failed for template '{template_name}': {e}")
        return False


ALL_TEMPLATES: List[str] = sorted(
    name for names in TIER_TEMPLATES.values() for name in names
)


__all__ = [
    "render_cv_pdf",
    "get_template_for_tier",
    "get_template_layout",
    "TIER_TEMPLATES",
    "TEMPLATE_LAYOUTS",
    "ALL_TEMPLATES",
    "TEMPLATES_DIR",
]
