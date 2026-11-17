import plotly.graph_objects as go
import plotly.io as pio

# Bloomberg specific color constants
BBG_BG = "#000000"
BBG_TEXT = "#FFB900"  # Amber text
BBG_GRID = "#333333"
BBG_UP = "#00FF00"
BBG_DOWN = "#FF0000"


def setup_bloomberg_theme():
    """Registers and sets the default Bloomberg template in plotly.io"""
    bbg_template = go.layout.Template()
    bbg_template.layout.plot_bgcolor = BBG_BG
    bbg_template.layout.paper_bgcolor = BBG_BG
    bbg_template.layout.font = dict(family="Consolas, monospace", color=BBG_TEXT)
    bbg_template.layout.xaxis = dict(
        showgrid=True, gridcolor=BBG_GRID, gridwidth=0.5, linecolor="#555555"
    )
    bbg_template.layout.yaxis = dict(
        showgrid=True, gridcolor=BBG_GRID, gridwidth=0.5, linecolor="#555555"
    )

    pio.templates["bloomberg"] = bbg_template
    pio.templates.default = "bloomberg"
