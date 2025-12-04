// Montage AI - Frontend JavaScript (Vanilla - KISS principle)

// API base URL
const API_BASE = '/api';

// State
let pollInterval = null;

// =============================================================================
// File Upload
// =============================================================================

document.getElementById('videoUpload')?.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    for (const file of files) {
        await uploadFile(file, 'video');
    }
    refreshFiles();
});

document.getElementById('musicUpload')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        await uploadFile(file, 'music');
        refreshFiles();
    }
});

async function uploadFile(file, type) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);

    // Create progress UI element
    const progressId = `upload-progress-${Date.now()}`;
    const listId = type === 'video' ? 'videoList' : 'musicList';
    const listElement = document.getElementById(listId);

    const progressHTML = `
        <div id="${progressId}" class="upload-progress">
            <div class="upload-info">
                <span class="upload-filename">${file.name}</span>
                <span class="upload-size">${formatFileSize(file.size)}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
            <span class="upload-status">Uploading...</span>
        </div>
    `;
    listElement.insertAdjacentHTML('afterbegin', progressHTML);

    try {
        // Use XMLHttpRequest for progress tracking
        const result = await uploadFileWithProgress(formData, progressId);

        // Update UI on success
        const progressEl = document.getElementById(progressId);
        if (progressEl) {
            progressEl.querySelector('.upload-status').textContent = '✓ Uploaded';
            progressEl.classList.add('upload-complete');
            setTimeout(() => progressEl.remove(), 2000);
        }

        console.log(`Uploaded ${type}:`, result);
    } catch (error) {
        console.error('Upload error:', error);

        // Update UI on error
        const progressEl = document.getElementById(progressId);
        if (progressEl) {
            progressEl.querySelector('.upload-status').textContent = `✗ Failed: ${error.message}`;
            progressEl.classList.add('upload-error');
        }
    }
}

function uploadFileWithProgress(formData, progressId) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                const progressEl = document.getElementById(progressId);
                if (progressEl) {
                    progressEl.querySelector('.progress-fill').style.width = `${percentComplete}%`;
                    progressEl.querySelector('.upload-status').textContent = `Uploading... ${Math.round(percentComplete)}%`;
                }
            }
        });

        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const result = JSON.parse(xhr.responseText);
                    resolve(result);
                } catch (e) {
                    reject(new Error('Invalid server response'));
                }
            } else {
                try {
                    const error = JSON.parse(xhr.responseText);
                    reject(new Error(error.error || 'Upload failed'));
                } catch (e) {
                    reject(new Error(`Upload failed with status ${xhr.status}`));
                }
            }
        });

        xhr.addEventListener('error', () => {
            reject(new Error('Network error'));
        });

        xhr.addEventListener('abort', () => {
            reject(new Error('Upload cancelled'));
        });

        xhr.open('POST', `${API_BASE}/upload`);
        xhr.send(formData);
    });
}

async function refreshFiles() {
    try {
        const response = await fetch(`${API_BASE}/files`);
        const data = await response.json();

        // Update video list
        const videoList = document.getElementById('videoList');
        if (data.videos.length > 0) {
            videoList.innerHTML = `<ul>${data.videos.map(f => `<li>✓ ${f}</li>`).join('')}</ul>`;
        } else {
            videoList.innerHTML = '<p>No videos uploaded</p>';
        }

        // Update music list
        const musicList = document.getElementById('musicList');
        if (data.music.length > 0) {
            musicList.innerHTML = `<ul>${data.music.map(f => `<li>✓ ${f}</li>`).join('')}</ul>`;
        } else {
            musicList.innerHTML = '<p>No music uploaded</p>';
        }

    } catch (error) {
        console.error('Error fetching files:', error);
    }
}

// =============================================================================
// Job Creation
// =============================================================================

async function createJob() {
    const style = document.getElementById('style').value;
    const prompt = document.getElementById('prompt').value;
    const stabilize = document.getElementById('stabilize').checked;
    const upscale = document.getElementById('upscale').checked;
    const enhance = document.getElementById('enhance').checked;
    const cgpu = document.getElementById('cgpu').checked;
    const llmClipSelection = document.getElementById('llmClipSelection').checked;
    const exportTimeline = document.getElementById('exportTimeline').checked;
    const generateProxies = document.getElementById('generateProxies').checked;

    // NEW: Video length control
    const targetDuration = parseFloat(document.getElementById('targetDuration').value) || 60;

    // NEW: Music trimming controls
    const useFullTrack = document.getElementById('useFullTrack').checked;
    const musicStart = useFullTrack ? 0 : (parseFloat(document.getElementById('musicStart').value) || 0);
    const musicDuration = useFullTrack ? null : (parseFloat(document.getElementById('musicDuration').value) || null);

    // Calculate musicEnd from start + duration if trimming
    const musicEnd = (musicDuration && !useFullTrack) ? musicStart + musicDuration : null;

    const jobData = {
        style,
        prompt,
        stabilize,
        upscale,
        enhance,
        cgpu,
        llm_clip_selection: llmClipSelection,
        export_timeline: exportTimeline,
        generate_proxies: generateProxies,
        target_duration: targetDuration,
        music_start: musicStart,
        music_end: musicEnd
    };

    try {
        const response = await fetch(`${API_BASE}/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(jobData)
        });

        if (!response.ok) {
            const error = await response.json();
            alert(`Job creation failed: ${error.error}`);
            return;
        }

        const job = await response.json();
        console.log('Job created:', job);

        // Start polling for updates
        startPolling();

        // Refresh job list
        await refreshJobs();

    } catch (error) {
        console.error('Error creating job:', error);
        alert(`Job creation failed: ${error.message}`);
    }
}

// =============================================================================
// Job Listing
// =============================================================================

async function refreshJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();

        const jobsList = document.getElementById('jobsList');

        if (data.jobs.length === 0) {
            jobsList.innerHTML = '<p class="loading">No jobs yet. Create your first montage!</p>';
            return;
        }

        jobsList.innerHTML = data.jobs.map(job => renderJob(job)).join('');

    } catch (error) {
        console.error('Error fetching jobs:', error);
        document.getElementById('jobsList').innerHTML = '<p class="loading">Error loading jobs</p>';
    }
}

function renderJob(job) {
    const statusClass = `status-${job.status}`;
    const statusEmoji = {
        'queued': '⏳',
        'running': '▶️',
        'completed': '✅',
        'failed': '❌',
        'timeout': '⏱️'
    }[job.status] || '❓';

    // View Details button for all jobs
    const detailsButton = `
        <button onclick="showJobDetails('${job.id}')" class="btn-details">
            📋 View Details & Logs
        </button>
    `;

    // Progress bar for queued/running jobs
    let progressSection = '';
    if (job.status === 'queued') {
        const queuePos = job.queue_position || 0;
        progressSection = `
            <div class="job-progress-section">
                <div class="job-progress-info">
                    <span>⏳ Waiting in queue...</span>
                    ${queuePos > 0 ? `<span>Position: ${queuePos}</span>` : ''}
                </div>
            </div>
        `;
    } else if (job.status === 'running') {
        progressSection = `
            <div class="job-progress-section">
                <div class="job-progress-info">
                    <span>▶️ Processing video...</span>
                    <span>${getElapsedTime(job.started_at)}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill indeterminate"></div>
                </div>
            </div>
        `;
    }

    let downloads = '';
    if (job.status === 'completed' && job.output_file) {
        downloads = `
            <div class="download-links">
                <a href="${API_BASE}/download/${job.output_file}" class="download-btn">
                    📥 Download Video
                </a>
                ${job.timeline_files ? renderTimelineDownloads(job.timeline_files) : ''}
            </div>
        `;
    }

    let errorSection = '';
    if (job.status === 'failed' && job.error) {
        errorSection = `<pre class="error-message">${job.error}</pre>`;
    }

    let completedInfo = '';
    if (job.status === 'completed' && job.completed_at) {
        const duration = getJobDuration(job.started_at, job.completed_at);
        completedInfo = `<div class="job-duration">✓ Completed in ${duration}</div>`;
    }

    const options = job.options || {};
    const enabledOptions = Object.entries(options)
        .filter(([_, v]) => v === true)
        .map(([k, _]) => k)
        .join(', ') || 'none';

    return `
        <div class="job-card ${statusClass}">
            <div class="job-header">
                <span class="job-id">Job ${job.id}</span>
                <span class="job-status ${statusClass}">${statusEmoji} ${job.status}</span>
            </div>
            <div class="job-details">
                <div class="job-detail">
                    <strong>Style</strong>
                    <span>${job.style}</span>
                </div>
                <div class="job-detail">
                    <strong>Created</strong>
                    <span>${formatDate(job.created_at)}</span>
                </div>
                <div class="job-detail">
                    <strong>Options</strong>
                    <span>${enabledOptions}</span>
                </div>
            </div>
            ${options.prompt ? `<div style="margin-bottom: 1rem;"><strong>Prompt:</strong> "${options.prompt}"</div>` : ''}
            ${progressSection}
            ${completedInfo}
            ${downloads}
            ${errorSection}
            <div style="margin-top: 1rem;">
                ${detailsButton}
            </div>
        </div>
    `;
}

function renderTimelineDownloads(timelineFiles) {
    let html = '';

    if (timelineFiles.otio && timelineFiles.otio.length > 0) {
        html += `<a href="${API_BASE}/download/${timelineFiles.otio[0]}" class="download-btn">📽️ OTIO</a>`;
    }
    if (timelineFiles.edl && timelineFiles.edl.length > 0) {
        html += `<a href="${API_BASE}/download/${timelineFiles.edl[0]}" class="download-btn">📝 EDL</a>`;
    }
    if (timelineFiles.csv && timelineFiles.csv.length > 0) {
        html += `<a href="${API_BASE}/download/${timelineFiles.csv[0]}" class="download-btn">📊 CSV</a>`;
    }

    return html;
}

// =============================================================================
// Polling
// =============================================================================

function startPolling() {
    if (pollInterval) return; // Already polling

    pollInterval = setInterval(async () => {
        await refreshJobs();

        // Check if all jobs are completed/failed
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        const hasRunning = data.jobs.some(j => j.status === 'running' || j.status === 'queued');

        if (!hasRunning) {
            stopPolling();
        }
    }, 3000); // Poll every 3 seconds
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// =============================================================================
// Job Details & Logs
// =============================================================================

async function showJobDetails(jobId) {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2>Job ${jobId} - Details & Logs</h2>
                <button onclick="closeModal()" class="btn-close">✕</button>
            </div>
            <div class="modal-body">
                <div class="loading">Loading...</div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Fetch job details
    try {
        const [jobResponse, logsResponse, instructionsResponse] = await Promise.all([
            fetch(`${API_BASE}/jobs/${jobId}`),
            fetch(`${API_BASE}/jobs/${jobId}/logs?lines=500`),
            fetch(`${API_BASE}/jobs/${jobId}/creative-instructions`)
        ]);

        const jobData = await jobResponse.json();
        const logsData = logsResponse.ok ? await logsResponse.json() : null;
        const instructionsData = instructionsResponse.ok ? await instructionsResponse.json() : null;

        // Render details
        const modalBody = modal.querySelector('.modal-body');
        modalBody.innerHTML = renderJobDetails(jobData, logsData, instructionsData);

        // Auto-refresh logs for running jobs
        if (jobData.status === 'running') {
            const refreshInterval = setInterval(async () => {
                const updatedLogs = await fetch(`${API_BASE}/jobs/${jobId}/logs?lines=500`);
                if (updatedLogs.ok) {
                    const data = await updatedLogs.json();
                    const logsEl = document.getElementById('job-logs-content');
                    if (logsEl) {
                        logsEl.textContent = data.logs;
                        logsEl.scrollTop = logsEl.scrollHeight; // Scroll to bottom
                    }
                }

                // Stop refreshing if modal is closed or job completed
                if (!document.querySelector('.modal-overlay')) {
                    clearInterval(refreshInterval);
                }
            }, 3000);
        }

    } catch (error) {
        console.error('Error loading job details:', error);
        modal.querySelector('.modal-body').innerHTML = `
            <div class="error-message">Failed to load job details: ${error.message}</div>
        `;
    }
}

function renderJobDetails(job, logs, instructions) {
    let html = '<div class="job-details-tabs">';

    // Creative Instructions Tab
    if (instructions && instructions.creative_prompt) {
        html += `
            <div class="detail-section">
                <h3>🎬 Creative Director Instructions</h3>
                <div class="detail-card">
                    <div class="detail-row">
                        <strong>Prompt:</strong>
                        <span>"${instructions.creative_prompt}"</span>
                    </div>
                    <div class="detail-row">
                        <strong>Style:</strong>
                        <span>${instructions.style}</span>
                    </div>
                    <div class="detail-row">
                        <strong>Options:</strong>
                        <span>${Object.entries(instructions.options)
                            .filter(([_, v]) => v === true)
                            .map(([k, _]) => k)
                            .join(', ') || 'none'}</span>
                    </div>
                </div>
            </div>
        `;
    }

    // Logs Tab
    if (logs) {
        const logsAvailable = logs.logs && logs.logs.length > 0;
        html += `
            <div class="detail-section">
                <h3>📋 Processing Logs</h3>
                ${logsAvailable ? `
                    <div class="detail-info">
                        Showing last ${logs.returned_lines} of ${logs.total_lines} lines
                    </div>
                    <pre id="job-logs-content" class="logs-viewer">${logs.logs}</pre>
                    <button onclick="downloadLogs('${job.id}')" class="btn-secondary" style="margin-top: 0.5rem;">
                        💾 Download Full Logs
                    </button>
                ` : `
                    <div class="detail-info">No logs available yet. Job may not have started.</div>
                `}
            </div>
        `;
    }

    // Job Status
    html += `
        <div class="detail-section">
            <h3>ℹ️ Job Status</h3>
            <div class="detail-card">
                <div class="detail-row">
                    <strong>Status:</strong>
                    <span>${job.status}</span>
                </div>
                <div class="detail-row">
                    <strong>Created:</strong>
                    <span>${formatDate(job.created_at)}</span>
                </div>
                ${job.started_at ? `
                    <div class="detail-row">
                        <strong>Started:</strong>
                        <span>${formatDate(job.started_at)}</span>
                    </div>
                ` : ''}
                ${job.completed_at ? `
                    <div class="detail-row">
                        <strong>Completed:</strong>
                        <span>${formatDate(job.completed_at)}</span>
                    </div>
                ` : ''}
            </div>
        </div>
    `;

    html += '</div>';
    return html;
}

function closeModal() {
    const modal = document.querySelector('.modal-overlay');
    if (modal) {
        modal.remove();
    }
}

function downloadLogs(jobId) {
    window.open(`${API_BASE}/jobs/${jobId}/logs?lines=10000`, '_blank');
}

// =============================================================================
// Helpers
// =============================================================================

function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds}s`;
    } else if (seconds < 3600) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}m ${secs}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }
}

function getElapsedTime(startTime) {
    const elapsed = Math.floor((new Date() - new Date(startTime)) / 1000);
    return formatDuration(elapsed);
}

function getJobDuration(startTime, endTime) {
    const duration = Math.floor((new Date(endTime) - new Date(startTime)) / 1000);
    return formatDuration(duration);
}

// =============================================================================
// Duration Presets & Music Controls
// =============================================================================

function initDurationPresets() {
    const presetButtons = document.querySelectorAll('.preset-btn');
    const durationInput = document.getElementById('targetDuration');

    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active from all
            presetButtons.forEach(b => b.classList.remove('active'));
            // Add active to clicked
            btn.classList.add('active');

            const duration = btn.getAttribute('data-duration');
            if (duration === 'custom') {
                durationInput.focus();
            } else {
                durationInput.value = duration;
            }
        });
    });

    // When user types custom value, activate "Custom" button
    durationInput.addEventListener('input', () => {
        const value = durationInput.value;
        const matchingBtn = Array.from(presetButtons).find(btn => btn.getAttribute('data-duration') === value);

        presetButtons.forEach(b => b.classList.remove('active'));
        if (matchingBtn) {
            matchingBtn.classList.add('active');
        } else {
            const customBtn = Array.from(presetButtons).find(btn => btn.getAttribute('data-duration') === 'custom');
            if (customBtn) customBtn.classList.add('active');
        }
    });
}

function initMusicControls() {
    const useFullTrackCheckbox = document.getElementById('useFullTrack');
    const musicTrimSection = document.getElementById('musicTrimSection');

    useFullTrackCheckbox.addEventListener('change', () => {
        if (useFullTrackCheckbox.checked) {
            musicTrimSection.style.display = 'none';
        } else {
            musicTrimSection.style.display = 'block';
        }
    });
}

// =============================================================================
// Init
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    refreshFiles();
    refreshJobs();

    // Initialize new UI controls
    initDurationPresets();
    initMusicControls();

    // Check for running jobs and start polling if needed
    fetch(`${API_BASE}/jobs`)
        .then(r => r.json())
        .then(data => {
            const hasRunning = data.jobs.some(j => j.status === 'running' || j.status === 'queued');
            if (hasRunning) {
                startPolling();
            }
        });
});
