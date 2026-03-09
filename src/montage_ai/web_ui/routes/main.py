"""
Main Routes - HTML Page Serving
"""

from flask import render_template
from . import main_bp


@main_bp.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@main_bp.route("/montage")
def montage_page():
    """Montage editor page."""
    return render_template("montage.html")


@main_bp.route("/creative")
def creative_page():
    """Creative director page."""
    return render_template("creative.html")


@main_bp.route("/shorts")
def shorts_page():
    """Shorts Studio page."""
    return render_template("shorts.html")


@main_bp.route("/transcript")
def transcript_page():
    """Transcript editor page."""
    return render_template("transcript.html")


@main_bp.route("/gallery")
def gallery_page():
    """Output gallery page."""
    return render_template("gallery.html")


@main_bp.route("/settings")
def settings_page():
    """Settings page."""
    return render_template("settings.html")


@main_bp.route("/features")
def features_page():
    """Feature flags page."""
    return render_template("features.html")
