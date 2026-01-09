"""
Montage AI - Web API Server

Provides user-friendly upload interface for videos, music, and assets.
Eliminates need for manual file copying to data directories.

Features:
- Multi-part file upload (videos, music, logos)
- Simple web interface for non-technical users
- Job status tracking
- Output file download

Architecture:
  User Browser → Upload Form → FastAPI → /data/{input,music,assets} → Editor Job
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import get_settings

# Get settings instance (fresh instantiation, avoids cached dataclass issue)
settings = get_settings()

# Directories (from centralized config)
INPUT_DIR = str(settings.paths.input_dir)
MUSIC_DIR = str(settings.paths.music_dir)
ASSETS_DIR = str(settings.paths.assets_dir)
OUTPUT_DIR = str(settings.paths.output_dir)

# Ensure directories exist
settings.paths.ensure_directories()

# Allowed file extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mxf", ".mts", ".m2ts", ".ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}

app = FastAPI(
    title="Montage AI API",
    description="Upload videos, music, and assets to create AI-powered montages",
    version="0.1.0"
)

# Enable CORS for web uploads
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# UPLOAD ENDPOINTS
# ============================================================================

@app.post("/api/upload/video")
async def upload_video(files: List[UploadFile] = File(...)):
    """
    Upload one or more video files.

    Accepts: .mp4, .mov, .avi, .mkv, .webm
    """
    uploaded = []

    for file in files:
        # Validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format: {ext}. Allowed: {', '.join(VIDEO_EXTENSIONS)}"
            )

        # Save to input directory
        file_path = os.path.join(INPUT_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)
        uploaded.append({
            "filename": file.filename,
            "size_mb": round(file_size_mb, 2),
            "path": file_path
        })

    return {
        "status": "success",
        "message": f"Uploaded {len(uploaded)} video(s)",
        "files": uploaded
    }


@app.post("/api/upload/music")
async def upload_music(files: List[UploadFile] = File(...)):
    """
    Upload one or more music files.

    Accepts: .mp3, .wav, .m4a, .aac, .flac
    """
    uploaded = []

    for file in files:
        # Validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio format: {ext}. Allowed: {', '.join(AUDIO_EXTENSIONS)}"
            )

        # Save to music directory
        file_path = os.path.join(MUSIC_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)
        uploaded.append({
            "filename": file.filename,
            "size_mb": round(file_size_mb, 2),
            "path": file_path
        })

    return {
        "status": "success",
        "message": f"Uploaded {len(uploaded)} music file(s)",
        "files": uploaded
    }


@app.post("/api/upload/logo")
async def upload_logo(files: List[UploadFile] = File(...)):
    """
    Upload logo/watermark images.

    Accepts: .png, .jpg, .jpeg, .svg
    """
    uploaded = []

    for file in files:
        # Validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {ext}. Allowed: {', '.join(IMAGE_EXTENSIONS)}"
            )

        # Save to assets directory
        file_path = os.path.join(ASSETS_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)
        uploaded.append({
            "filename": file.filename,
            "size_mb": round(file_size_mb, 2),
            "path": file_path
        })

    return {
        "status": "success",
        "message": f"Uploaded {len(uploaded)} logo(s)",
        "files": uploaded
    }


# ============================================================================
# FILE MANAGEMENT
# ============================================================================

@app.get("/api/files")
async def list_files():
    """List all uploaded files."""

    def scan_dir(directory, file_type):
        files = []
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                if os.path.isfile(path):
                    stat = os.stat(path)
                    files.append({
                        "filename": filename,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "type": file_type,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        return files

    return {
        "videos": scan_dir(INPUT_DIR, "video"),
        "music": scan_dir(MUSIC_DIR, "music"),
        "logos": scan_dir(ASSETS_DIR, "logo"),
        "outputs": scan_dir(OUTPUT_DIR, "output")
    }


@app.delete("/api/files/{file_type}/{filename}")
async def delete_file(file_type: str, filename: str):
    """Delete a file."""

    dir_map = {
        "video": INPUT_DIR,
        "music": MUSIC_DIR,
        "logo": ASSETS_DIR,
        "output": OUTPUT_DIR
    }

    if file_type not in dir_map:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")

    file_path = os.path.join(dir_map[file_type], filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    os.remove(file_path)

    return {"status": "success", "message": f"Deleted {filename}"}


@app.delete("/api/files/clear/{file_type}")
async def clear_directory(file_type: str):
    """Clear all files of a specific type."""

    dir_map = {
        "video": INPUT_DIR,
        "music": MUSIC_DIR,
        "logo": ASSETS_DIR,
        "output": OUTPUT_DIR
    }

    if file_type not in dir_map:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}")

    directory = dir_map[file_type]
    removed_count = 0

    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                removed_count += 1

    return {
        "status": "success",
        "message": f"Cleared {removed_count} file(s) from {file_type}"
    }


# ============================================================================
# OUTPUT DOWNLOAD
# ============================================================================

@app.get("/api/download/{filename}")
async def download_output(filename: str):
    """Download a generated montage or timeline export."""

    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
    )


# ============================================================================
# RENDER CONTROL (Future: Trigger montage creation via API)
# ============================================================================

@app.post("/api/render")
async def trigger_render(
    background_tasks: BackgroundTasks,
    style: str = Form("dynamic"),
    stabilize: bool = Form(False),
    upscale: bool = Form(False),
    export_timeline: bool = Form(False)
):
    """
    Trigger montage rendering (placeholder for future implementation).

    Currently, rendering is done via K8s Job or docker-compose.
    Future: This endpoint will trigger the editor.py directly.
    """

    # TODO: Implement background rendering
    # This would require moving editor.py into a worker queue system

    return {
        "status": "info",
        "message": "Render triggering not yet implemented. Use K8s Job or docker-compose to render.",
        "instructions": {
            "kubernetes": "kubectl apply -k deploy/k3s/overlays/dev/",
            "docker": "./montage-ai.sh run " + style
        }
    }


# ============================================================================
# WEB INTERFACE
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Simple web interface for file uploads."""

    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Montage AI - Upload</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        h1 { font-size: 2em; margin-bottom: 10px; }
        .subtitle { opacity: 0.9; }
        .content { padding: 30px; }
        .upload-section {
            background: #f7f9fc;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            border: 2px dashed #cbd5e0;
            transition: all 0.3s;
        }
        .upload-section:hover { border-color: #667eea; }
        .upload-section h2 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #2d3748;
        }
        input[type="file"] {
            display: block;
            width: 100%;
            padding: 12px;
            margin-bottom: 15px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: white;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button:active { transform: translateY(0); }
        .status {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            display: none;
        }
        .status.success { background: #c6f6d5; color: #22543d; display: block; }
        .status.error { background: #fed7d7; color: #742a2a; display: block; }
        .file-list {
            margin-top: 20px;
            padding: 20px;
            background: #f7f9fc;
            border-radius: 8px;
        }
        .file-item {
            padding: 10px;
            background: white;
            margin-bottom: 8px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .icon { font-size: 1.5em; margin-right: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Montage AI</h1>
            <p class="subtitle">Upload your videos, music, and logos</p>
        </div>

        <div class="content">
            <!-- Video Upload -->
            <div class="upload-section">
                <h2>🎥 Videos</h2>
                <input type="file" id="videoFiles" multiple accept=".mp4,.mov,.avi,.mkv,.webm">
                <button onclick="uploadFiles('video')">Upload Videos</button>
                <div id="videoStatus" class="status"></div>
            </div>

            <!-- Music Upload -->
            <div class="upload-section">
                <h2>🎵 Music</h2>
                <input type="file" id="musicFiles" multiple accept=".mp3,.wav,.m4a,.aac,.flac">
                <button onclick="uploadFiles('music')">Upload Music</button>
                <div id="musicStatus" class="status"></div>
            </div>

            <!-- Logo Upload -->
            <div class="upload-section">
                <h2>🏷️ Logo / Watermark</h2>
                <input type="file" id="logoFiles" multiple accept=".png,.jpg,.jpeg,.svg">
                <button onclick="uploadFiles('logo')">Upload Logo</button>
                <div id="logoStatus" class="status"></div>
            </div>

            <!-- File List -->
            <div class="file-list">
                <h2>📁 Uploaded Files</h2>
                <button onclick="loadFiles()">Refresh List</button>
                <div id="fileList" style="margin-top: 15px;"></div>
            </div>
        </div>
    </div>

    <script>
        async function uploadFiles(type) {
            const inputId = type + 'Files';
            const statusId = type + 'Status';
            const files = document.getElementById(inputId).files;

            if (files.length === 0) {
                showStatus(statusId, 'Please select files first', 'error');
                return;
            }

            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }

            try {
                showStatus(statusId, 'Uploading...', 'success');
                const response = await fetch(`/api/upload/${type}`, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    showStatus(statusId, `✅ ${result.message}`, 'success');
                    document.getElementById(inputId).value = '';
                    loadFiles();
                } else {
                    showStatus(statusId, `❌ ${result.detail}`, 'error');
                }
            } catch (error) {
                showStatus(statusId, `❌ Upload failed: ${error}`, 'error');
            }
        }

        function showStatus(elementId, message, type) {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.className = `status ${type}`;
        }

        async function loadFiles() {
            try {
                const response = await fetch('/api/files');
                const data = await response.json();

                const listHtml = `
                    <h3>Videos (${data.videos.length})</h3>
                    ${data.videos.map(f => `<div class="file-item">${f.filename} (${f.size_mb} MB)</div>`).join('') || '<p>No videos uploaded</p>'}

                    <h3>Music (${data.music.length})</h3>
                    ${data.music.map(f => `<div class="file-item">${f.filename} (${f.size_mb} MB)</div>`).join('') || '<p>No music uploaded</p>'}

                    <h3>Logos (${data.logos.length})</h3>
                    ${data.logos.map(f => `<div class="file-item">${f.filename} (${f.size_mb} MB)</div>`).join('') || '<p>No logos uploaded</p>'}

                    <h3>Outputs (${data.outputs.length})</h3>
                    ${data.outputs.map(f => `<div class="file-item">
                        <span>${f.filename} (${f.size_mb} MB)</span>
                        <a href="/api/download/${f.filename}" download>Download</a>
                    </div>`).join('') || '<p>No outputs yet</p>'}
                `;

                document.getElementById('fileList').innerHTML = listHtml;
            } catch (error) {
                console.error('Failed to load files:', error);
            }
        }

        // Load files on page load
        loadFiles();
    </script>
</body>
</html>
    """


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for K8s liveness/readiness probes."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "directories": {
            "input": os.path.exists(INPUT_DIR),
            "music": os.path.exists(MUSIC_DIR),
            "assets": os.path.exists(ASSETS_DIR),
            "output": os.path.exists(OUTPUT_DIR)
        }
    }


if __name__ == "__main__":
    # Development server
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
