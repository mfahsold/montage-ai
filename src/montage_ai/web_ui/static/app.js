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

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            alert(`Upload failed: ${error.error}`);
            return;
        }

        const result = await response.json();
        console.log(`Uploaded ${type}:`, result);
    } catch (error) {
        console.error('Upload error:', error);
        alert(`Upload failed: ${error.message}`);
    }
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
    const exportTimeline = document.getElementById('exportTimeline').checked;
    const generateProxies = document.getElementById('generateProxies').checked;

    const jobData = {
        style,
        prompt,
        stabilize,
        upscale,
        enhance,
        cgpu,
        export_timeline: exportTimeline,
        generate_proxies: generateProxies
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
        errorSection = `<pre style="color: var(--danger); font-size: 0.875rem; background: #fee2e2; padding: 0.5rem; border-radius: 0.25rem; overflow-x: auto;">${job.error}</pre>`;
    }

    const options = job.options || {};
    const enabledOptions = Object.entries(options)
        .filter(([_, v]) => v === true)
        .map(([k, _]) => k)
        .join(', ') || 'none';

    return `
        <div class="job-card">
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
            ${downloads}
            ${errorSection}
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

// =============================================================================
// Init
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    refreshFiles();
    refreshJobs();

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
