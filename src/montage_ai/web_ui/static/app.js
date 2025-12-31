
// Montage AI - Poly-Chrome Archive / Cyber-NLE Frontend
// Matches the new "Cyber Deck" UI structure

const API_BASE = '/api';
let pollInterval = null;
let backendDefaults = {};
let systemStatus = {};

// =============================================================================
// Configuration
// =============================================================================

const TOGGLE_CONFIG = [
    { id: 'enhance', label: 'Enhance', desc: 'Color/sharpness tweak.', default: false, badges: [{ type: 'quality', text: 'Quality+' }] },
    { id: 'stabilize', label: 'Stabilize', desc: 'Reduce camera shake.', default: false, badges: [{ type: 'quality', text: 'Quality+' }, { type: 'cost', text: 'Slower' }] },
    { id: 'upscale', label: 'Upscale', desc: 'AI 4x upscaling.', default: false, badges: [{ type: 'quality', text: 'Quality++' }, { type: 'cost', text: 'GPU Heavy' }] },
    { id: 'cgpu', label: 'Cloud GPU', desc: 'Offload to cloud.', default: false, badges: [{ type: 'info', text: 'Remote' }] },
    { id: 'llm_clip_selection', label: 'LLM Clip Selection', desc: 'Semantic scene analysis.', default: false, badges: [{ type: 'quality', text: 'Smart Cuts' }] },
    { id: 'creative_loop', label: 'Creative Loop', desc: 'LLM refines cuts iteratively.', default: false, badges: [{ type: 'quality', text: 'Agentic' }, { type: 'cost', text: '2-3x Time' }] },
    { id: 'export_timeline', label: 'Export Timeline', desc: 'OTIO/EDL for NLEs.', default: false, badges: [{ type: 'info', text: 'Resolve/Premiere' }] },
    { id: 'generate_proxies', label: 'Generate Proxies', desc: 'Faster NLE editing.', default: false, badges: [{ type: 'cost', text: 'Extra Files' }] },
    { id: 'preserve_aspect', label: 'Preserve Aspect', desc: 'Letterbox vs crop.', default: false, badges: [{ type: 'info', text: 'Safe Area' }] }
];

// Pipeline phases for progress display
const PIPELINE_PHASES = [
    { id: 'setup', label: 'Setup', keywords: ['Setup', 'Initializing', 'Starting'] },
    { id: 'audio', label: 'Audio', keywords: ['Audio', 'Beat', 'Music', 'BPM'] },
    { id: 'scenes', label: 'Scenes', keywords: ['Scene', 'Detecting', 'Analysis'] },
    { id: 'selection', label: 'Selection', keywords: ['Select', 'Clip', 'Choosing'] },
    { id: 'render', label: 'Render', keywords: ['Render', 'Encoding', 'Writing', 'FFmpeg'] }
];

// File validation config
const FILE_VALIDATION = {
    video: {
        extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'mts', 'm2ts'],
        maxSizeMB: 2000,
        warnSizeMB: 500
    },
    music: {
        extensions: ['mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg'],
        maxSizeMB: 100,
        warnSizeMB: 50
    }
};

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    fetchStatus().then(() => renderToggles());
    
    // Attach listeners to static inputs
    ['style', 'targetDuration', 'prompt'].forEach(id => {
        document.getElementById(id)?.addEventListener('input', updateRunSummary);
    });
    
    updateRunSummary(); // Initial render

    refreshFiles();
    refreshJobs();
    startPolling();
    
    // Update duration display
    const durationSlider = document.getElementById('targetDuration');
    if (durationSlider) {
        durationSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            document.getElementById('durationValue').textContent = val === '0' ? 'Auto' : `${val}s`;
        });
    }
});

// =============================================================================
// UI Rendering
// =============================================================================

function renderToggles() {
    const container = document.getElementById('toggles-container');
    if (!container) return;

    container.innerHTML = TOGGLE_CONFIG.map(toggle => {
        const badgesHtml = toggle.badges?.map(b =>
            `<span class="toggle-badge ${b.type}">${b.text}</span>`
        ).join('') || '';

        return `
            <label class="checkbox-item-enhanced">
                <input type="checkbox" id="${toggle.id}" ${getDefaultValue(toggle.id) ? 'checked' : ''}>
                <div class="toggle-content">
                    <div class="toggle-label">${toggle.label}</div>
                    <div class="toggle-desc">${toggle.desc}</div>
                    ${badgesHtml ? `<div class="toggle-badges">${badgesHtml}</div>` : ''}
                </div>
            </label>
        `;
    }).join('');

    // Attach listeners for summary updates
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', updateRunSummary);
    });
}

function updateRunSummary() {
    const summary = document.getElementById('run-summary');
    if (!summary) return;

    const style = document.getElementById('style')?.value || 'dynamic';
    const duration = document.getElementById('targetDuration')?.value || '0';
    const activeToggles = TOGGLE_CONFIG
        .filter(t => document.getElementById(t.id)?.checked)
        .map(t => t.label);
    const gpuStatus = systemStatus.gpu || 'unknown';
    const cgpuStatus = systemStatus.cgpu || 'unknown';
    const encoder = systemStatus.encoder || 'auto';

    summary.innerHTML = `
        <div style="font-family: var(--font-body); color: var(--fg);">
            <strong>> PREFLIGHT_CHECK:</strong><br>
            <span style="opacity: 0.8;">• STYLE:</span> ${style.toUpperCase()}<br>
            <span style="opacity: 0.8;">• DURATION:</span> ${duration === '0' ? 'AUTO (Match Audio)' : duration + 's'}<br>
            <span style="opacity: 0.8;">• MODULES:</span> ${activeToggles.length ? activeToggles.join(', ') : 'Standard Pipeline'}<br>
            <span style="opacity: 0.8;">• GPU:</span> ${gpuStatus} / Encoder: ${encoder}<br>
            <span style="opacity: 0.8;">• CGPU:</span> ${cgpuStatus}
        </div>
    `;
}

// =============================================================================
// File Validation & Upload
// =============================================================================

function validateFile(file, type) {
    const config = FILE_VALIDATION[type];
    if (!config) return { valid: true };

    const ext = file.name.split('.').pop().toLowerCase();
    const sizeMB = file.size / (1024 * 1024);

    // Check extension
    if (!config.extensions.includes(ext)) {
        return {
            valid: false,
            error: `Format .${ext} not supported`,
            hint: `Allowed: ${config.extensions.join(', ')}`
        };
    }

    // Check max size
    if (sizeMB > config.maxSizeMB) {
        return {
            valid: false,
            error: `File too large (${sizeMB.toFixed(0)}MB)`,
            hint: `Maximum: ${config.maxSizeMB}MB`
        };
    }

    // Check warning size
    if (sizeMB > config.warnSizeMB) {
        return {
            valid: true,
            warning: `Large file (${sizeMB.toFixed(0)}MB) - upload may take time`
        };
    }

    return { valid: true };
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

document.getElementById('videoUpload')?.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    for (const file of files) {
        const validation = validateFile(file, 'video');
        if (!validation.valid) {
            showToast(`${validation.error}: ${file.name}`, 'error');
            continue;
        }
        if (validation.warning) {
            showToast(validation.warning, 'warning');
        }
        await uploadFile(file, 'video');
    }
    refreshFiles();
});

document.getElementById('musicUpload')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        const validation = validateFile(file, 'music');
        if (!validation.valid) {
            showToast(`${validation.error}: ${file.name}`, 'error');
            return;
        }
        if (validation.warning) {
            showToast(validation.warning, 'warning');
        }
        await uploadFile(file, 'music');
        refreshFiles();
    }
});

async function uploadFile(file, type) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);

    const listId = type === 'video' ? 'videoList' : 'musicList';
    const listElement = document.getElementById(listId);
    const fileSize = formatFileSize(file.size);

    // Loading indicator with file size
    const loadingId = `loading-${Date.now()}`;
    listElement.insertAdjacentHTML('afterbegin', `
        <div id="${loadingId}" class="file-item uploading">
            <span>> UPLOADING ${file.name}</span>
            <span class="file-size">${fileSize}</span>
        </div>
    `);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || 'Upload failed');
        }

        const result = await response.json();
        console.log(`Uploaded ${type}:`, result);
        showToast(`UPLOAD_COMPLETE: ${file.name}`, 'success');
    } catch (error) {
        console.error('Upload error:', error);
        showToast(`UPLOAD_FAILED: ${error.message}`, 'error');
    } finally {
        document.getElementById(loadingId)?.remove();
    }
}

async function refreshFiles() {
    try {
        const response = await fetch(`${API_BASE}/files`);
        const data = await response.json();

        const videoList = document.getElementById('videoList');
        const musicList = document.getElementById('musicList');

        if (videoList) {
            videoList.innerHTML = data.videos.length 
                ? data.videos.map(f => `<div class="file-item"><span>✓ ${f}</span></div>`).join('')
                : '<div class="file-item" style="opacity:0.5">[ NO_INPUT_STREAMS ]</div>';
        }

        if (musicList) {
            musicList.innerHTML = data.music.length 
                ? data.music.map(f => `<div class="file-item"><span>✓ ${f}</span></div>`).join('')
                : '<div class="file-item" style="opacity:0.5">[ NO_AUDIO_TRACK ]</div>';
        }
    } catch (error) {
        console.error('Error refreshing files:', error);
    }
}

// =============================================================================
// Job Creation
// =============================================================================

function buildJobPayload() {
    const getVal = (id) => document.getElementById(id)?.value;
    const getCheck = (id) => document.getElementById(id)?.checked || false;

    const jobData = {
        style: getVal('style') || 'dynamic',
        prompt: getVal('prompt') || '',
        target_duration: parseFloat(getVal('targetDuration')) || 0,
        music_start: parseFloat(getVal('musicStart')) || 0,
        music_end: getVal('musicEnd') ? parseFloat(getVal('musicEnd')) : null
    };

    TOGGLE_CONFIG.forEach(toggle => {
        jobData[toggle.id] = getCheck(toggle.id);
    });

    return jobData;
}

function validateBeforeJob() {
    const videoList = document.getElementById('videoList')?.textContent || '';
    const musicList = document.getElementById('musicList')?.textContent || '';

    if (videoList.includes('NO_INPUT_STREAMS')) {
        showToast('Please add at least one video', 'warning');
        return false;
    }
    if (musicList.includes('NO_AUDIO_TRACK')) {
        showToast('Please add a music track', 'warning');
        return false;
    }

    // Check cgpu status if enabled
    const cgpuEnabled = document.getElementById('cgpu')?.checked;
    if (cgpuEnabled && systemStatus.cgpu === 'unavailable') {
        showToast('Cloud GPU enabled but not available - will use local processing', 'warning');
    }

    // Check GPU encoder status for heavy operations
    const upscaleEnabled = document.getElementById('upscale')?.checked;
    const stabilizeEnabled = document.getElementById('stabilize')?.checked;
    if ((upscaleEnabled || stabilizeEnabled) && systemStatus.gpu === 'unknown') {
        showToast('GPU encoder not detected - heavy operations may be slow', 'info');
    }

    return true;
}

async function createJob(isPreview = false) {
    if (!validateBeforeJob()) return;
    const jobData = buildJobPayload();
    if (isPreview) {
        jobData.preset = 'fast';
    }

    try {
        // Visual feedback
        const btn = document.querySelector('.voxel-btn.primary');
        btn.innerText = '> TRANSMITTING...';
        btn.disabled = true;
        btn.classList.add('btn-busy');

        const response = await fetch(`${API_BASE}/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(jobData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Server Error');
        }

        const job = await response.json();
        console.log('Job created:', job);
        
        // Immediate refresh
        await refreshJobs();
        showToast('JOB_INITIATED: Sequence started', 'success');

    } catch (error) {
        console.error('Error creating job:', error);
        showToast(`EXECUTION_ERROR: ${error.message}`, 'error');
    } finally {
        const btn = document.querySelector('.voxel-btn.primary');
        if (btn) {
            btn.innerText = '▶ Create Montage';
            btn.disabled = false;
            btn.classList.remove('btn-busy');
        }
    }
}

// =============================================================================
// B-Roll Planning (Semantic Search)
// =============================================================================

/**
 * DRY helper: Display result in B-roll results div
 * @param {string} message - Message to display (prefix with "> ")
 * @param {string} type - 'success' | 'error' | 'info' | 'loading'
 */
function showBrollResult(message, type = 'info') {
    const resultsDiv = document.getElementById('brollResults');
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = message;
    const colors = {
        success: 'var(--primary)',
        error: 'var(--warning)',
        info: 'var(--text-dim)',
        loading: 'inherit'
    };
    resultsDiv.style.color = colors[type] || 'inherit';
}

async function analyzeFootage() {
    showBrollResult('> ANALYZING_FOOTAGE...', 'loading');

    try {
        const response = await fetch(`${API_BASE}/broll/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();

        if (response.ok) {
            showBrollResult(
                `> ANALYZED ${result.analyzed} FILES<br>` +
                `> MEMORY: ${result.memory_stats?.temporal_entries || 0} segments indexed`,
                'success'
            );
        } else {
            showBrollResult(`> ERROR: ${result.error}`, 'error');
        }
    } catch (error) {
        showBrollResult(`> ERROR: ${error.message}`, 'error');
    }
}

async function searchBroll() {
    const query = document.getElementById('brollQuery').value.trim();

    if (!query) {
        showToast('Enter a search query', 'warning');
        return;
    }

    showBrollResult(`> SEARCHING: "${query}"...`, 'loading');

    try {
        const response = await fetch(`${API_BASE}/broll/suggest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5 })
        });

        const result = await response.json();

        if (response.ok && result.suggestions?.length > 0) {
            const matches = result.suggestions.map((s, i) =>
                `  ${i + 1}. ${s.video_path?.split('/').pop() || 'clip'} ` +
                `[${s.start_time?.toFixed(1) || 0}s-${s.end_time?.toFixed(1) || 0}s] ` +
                `(${(s.similarity_score * 100).toFixed(0)}%)`
            ).join('<br>');
            showBrollResult(`> FOUND ${result.count} MATCHES:<br>${matches}`, 'success');
        } else if (response.ok) {
            showBrollResult('> NO MATCHES FOUND. Try analyzing footage first.', 'info');
        } else {
            showBrollResult(`> ERROR: ${result.error}`, 'error');
        }
    } catch (error) {
        showBrollResult(`> ERROR: ${error.message}`, 'error');
    }
}

// =============================================================================
// Job Listing & Monitoring
// =============================================================================

let eventSource = null;

function startPolling() {
    // Initial fetch to populate UI
    refreshJobs();

    // Setup SSE for real-time updates (Hardware Nah: Push instead of Pull)
    if (eventSource) eventSource.close();
    
    eventSource = new EventSource(`${API_BASE}/stream`);
    
    eventSource.onmessage = (e) => {
        // Generic message
        console.log('SSE Message:', e.data);
    };

    eventSource.addEventListener('job_update', (e) => {
        const data = JSON.parse(e.data);
        // Only update active job UI to save DOM operations
        updateActiveJobUI({
            id: data.job_id,
            status: data.status,
            phase: data.phase
        });
    });

    eventSource.addEventListener('job_complete', (e) => {
        const data = JSON.parse(e.data);
        showToast(`Job ${data.job_id.substring(0,8)} Completed!`, 'success');
        refreshJobs(); // Full refresh to update gallery
    });

    eventSource.onerror = () => {
        // Fallback to polling if SSE fails (e.g. connection lost)
        console.warn('SSE connection lost, falling back to polling');
        eventSource.close();
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(refreshJobs, 5000); // Slower polling as fallback
    };
}

async function refreshJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        
        const activeJob = data.jobs.find(j => j.status === 'running' || j.status === 'queued');
        const completedJobs = data.jobs.filter(j => j.status === 'completed' || j.status === 'failed');

        updateActiveJobUI(activeJob);
        updateGalleryUI(completedJobs);

    } catch (error) {
        console.error('Error fetching jobs:', error);
    }
}

async function updateActiveJobUI(job) {
    const section = document.getElementById('jobStatusSection');
    const statusCard = document.getElementById('currentJobStatus');
    const terminal = document.getElementById('terminalOutput');

    if (!job) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    
    // Update Status Card
    const phase = job.phase || {};
    const percent = phase.progress_percent || 0;
    
    statusCard.innerHTML = `
        <div class="job-header">
            <span class="job-id">#${job.id.substring(0, 8)}</span>
            <span class="job-phase">${(phase.name || job.status).toUpperCase()}</span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: ${percent}%"></div>
        </div>
        <div class="job-details">
            ${phase.label || 'Processing...'}
        </div>
    `;

    // Update Terminal
    try {
        const logResponse = await fetch(`${API_BASE}/jobs/${job.id}/logs`);
        const logData = await logResponse.json();
        if (logData.logs) {
            // Keep only last 20 lines to avoid DOM bloat
            const lines = logData.logs.split('\n').slice(-20);
            terminal.innerHTML = lines.map(line => 
                `<div class="terminal-line">${formatLogLine(line)}</div>`
            ).join('');
            terminal.scrollTop = terminal.scrollHeight;
        }
    } catch (e) {
        console.warn('Log fetch failed', e);
    }
}

function formatLogLine(line) {
    if (line.includes('[INFO]')) return `<span class="log-info">${line}</span>`;
    if (line.includes('[WARNING]')) return `<span class="log-warn">${line}</span>`;
    if (line.includes('[ERROR]')) return `<span class="log-error">${line}</span>`;
    return `<span class="log-dim">${line}</span>`;
}

function updateGalleryUI(jobs) {
    const gallery = document.getElementById('jobList'); // Reusing ID for gallery grid
    if (!gallery) return;

    if (jobs.length === 0) {
        gallery.innerHTML = '<div class="empty-state">No recent jobs found.</div>';
        return;
    }

    gallery.innerHTML = jobs.map(job => `
        <div class="gallery-item ${job.status}">
            <div class="gallery-header">
                <span class="gallery-id">#${job.id.substring(0, 6)}</span>
                <span class="gallery-date">${new Date(job.created_at).toLocaleTimeString()}</span>
            </div>
            <div class="gallery-info">
                <div class="gallery-style">${job.style.toUpperCase()}</div>
                ${job.output_file ? 
                    `<a href="${API_BASE}/download/${job.output_file}" class="voxel-btn primary small">Download MP4</a>` : 
                    `<span class="status-text">${job.status}</span>`
                }
            </div>
        </div>
    `).join('');
}


// =============================================================================
// Status & Defaults
// =============================================================================

async function fetchStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (!response.ok) return;
        const data = await response.json();
        systemStatus = {
            gpu: data.system?.gpu_encoder || 'unknown',
            encoder: data.system?.encoder || 'auto',
            cgpu: data.system?.cgpu_available ? 'available' : 'unavailable'
        };
        backendDefaults = data.defaults || {};
        // Update defaults in TOGGLE_CONFIG if provided
        TOGGLE_CONFIG.forEach(t => {
            if (backendDefaults[t.id] !== undefined) {
                t.default = backendDefaults[t.id];
            }
        });
    } catch (e) {
        console.warn('Status fetch failed', e);
    }
}

function getDefaultValue(id) {
    if (backendDefaults && backendDefaults[id] !== undefined) {
        return backendDefaults[id];
    }
    const cfg = TOGGLE_CONFIG.find(t => t.id === id);
    return cfg ? cfg.default : false;
}

function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span style="font-weight: bold;">[ ${type.toUpperCase()} ]</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease-in';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
