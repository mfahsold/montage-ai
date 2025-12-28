
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

async function analyzeFoootage() {
    const resultsDiv = document.getElementById('brollResults');
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = '> ANALYZING_FOOTAGE...';

    try {
        const response = await fetch(`${API_BASE}/broll/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();

        if (response.ok) {
            resultsDiv.innerHTML = `> ANALYZED ${result.analyzed} FILES<br>` +
                `> MEMORY: ${result.memory_stats?.temporal_entries || 0} segments indexed`;
            resultsDiv.style.color = 'var(--primary)';
        } else {
            resultsDiv.innerHTML = `> ERROR: ${result.error}`;
            resultsDiv.style.color = 'var(--warning)';
        }
    } catch (error) {
        resultsDiv.innerHTML = `> ERROR: ${error.message}`;
        resultsDiv.style.color = 'var(--warning)';
    }
}

async function searchBroll() {
    const query = document.getElementById('brollQuery').value.trim();
    const resultsDiv = document.getElementById('brollResults');

    if (!query) {
        showToast('Enter a search query', 'warning');
        return;
    }

    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = `> SEARCHING: "${query}"...`;

    try {
        const response = await fetch(`${API_BASE}/broll/suggest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5 })
        });

        const result = await response.json();

        if (response.ok && result.suggestions?.length > 0) {
            resultsDiv.innerHTML = `> FOUND ${result.count} MATCHES:<br>` +
                result.suggestions.map((s, i) =>
                    `  ${i + 1}. ${s.video_path?.split('/').pop() || 'clip'} ` +
                    `[${s.start_time?.toFixed(1) || 0}s-${s.end_time?.toFixed(1) || 0}s] ` +
                    `(${(s.similarity_score * 100).toFixed(0)}%)`
                ).join('<br>');
            resultsDiv.style.color = 'var(--primary)';
        } else if (response.ok) {
            resultsDiv.innerHTML = '> NO MATCHES FOUND. Try analyzing footage first.';
            resultsDiv.style.color = 'var(--text-dim)';
        } else {
            resultsDiv.innerHTML = `> ERROR: ${result.error}`;
            resultsDiv.style.color = 'var(--warning)';
        }
    } catch (error) {
        resultsDiv.innerHTML = `> ERROR: ${error.message}`;
        resultsDiv.style.color = 'var(--warning)';
    }
}

// =============================================================================
// Job Listing & Monitoring
// =============================================================================

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(refreshJobs, 3000);
}

async function refreshJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        const jobList = document.getElementById('jobList'); // Matches new HTML ID

        if (!jobList) return;

        if (data.jobs.length === 0) {
            jobList.innerHTML = `
                <div class="job-item">
                    <span>[WAITING]</span>
                    <span>System initialized. Ready for input.</span>
                </div>`;
            return;
        }

        jobList.innerHTML = data.jobs.map(job => renderJob(job)).join('');

    } catch (error) {
        console.error('Error fetching jobs:', error);
    }
}

function getPhaseIndex(phaseText) {
    if (!phaseText) return -1;
    const text = phaseText.toLowerCase();
    for (let i = 0; i < PIPELINE_PHASES.length; i++) {
        for (const keyword of PIPELINE_PHASES[i].keywords) {
            if (text.includes(keyword.toLowerCase())) return i;
        }
    }
    return -1;
}

function renderPhaseChips(phaseText, status) {
    if (status === 'completed') {
        return PIPELINE_PHASES.map(p =>
            `<span class="phase-chip completed">${p.label}</span>`
        ).join('<span class="phase-connector active"></span>');
    }
    if (status === 'failed' || status === 'queued') {
        return PIPELINE_PHASES.map(p =>
            `<span class="phase-chip pending">${p.label}</span>`
        ).join('<span class="phase-connector"></span>');
    }

    const activeIndex = getPhaseIndex(phaseText);
    return PIPELINE_PHASES.map((p, i) => {
        let chipClass = 'pending';
        if (i < activeIndex) chipClass = 'completed';
        else if (i === activeIndex) chipClass = 'active';
        return `<span class="phase-chip ${chipClass}">${p.label}</span>`;
    }).join('<span class="phase-connector' + (activeIndex > 0 ? ' active' : '') + '"></span>');
}

function renderCompletionCard(job) {
    const warnings = [];
    if (systemStatus.gpu === 'unknown' || systemStatus.gpu === 'none') {
        warnings.push('CPU encoding used (no GPU encoder detected)');
    }

    return `
        <div class="completion-card">
            <div class="header">
                <span class="title">MONTAGE_COMPLETE</span>
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-label">Style</span>
                    <span class="stat-value">${(job.style || 'dynamic').toUpperCase()}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Duration</span>
                    <span class="stat-value">${job.duration ? job.duration.toFixed(1) + 's' : '--'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Cuts</span>
                    <span class="stat-value">${job.cut_count || '--'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Render</span>
                    <span class="stat-value">${job.render_time ? job.render_time.toFixed(0) + 's' : '--'}</span>
                </div>
            </div>
            <div class="actions">
                <a href="${API_BASE}/download/${job.output_file}" class="voxel-btn primary" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;">
                    Download
                </a>
                <a href="${API_BASE}/jobs/${job.id}/logs" target="_blank" class="voxel-btn" style="padding: 0.4rem 0.8rem; font-size: 0.85rem;">
                    View Logs
                </a>
            </div>
            ${warnings.length ? `<div class="warning-banner">${warnings.join(' | ')}</div>` : ''}
        </div>
    `;
}

function renderJob(job) {
    let statusColor = '#fff';
    let statusLabel = job.status.toUpperCase();

    if (job.status === 'running') statusColor = '#0ff';
    if (job.status === 'completed') statusColor = '#0f0';
    if (job.status === 'failed') statusColor = '#f00';
    if (job.status === 'queued') statusColor = '#ff0';

    const phaseChips = renderPhaseChips(job.phase, job.status);
    const logLink = `<a href="${API_BASE}/jobs/${job.id}/logs" target="_blank" class="status-badge" style="color: #888; border-color: #888; text-decoration: none;">[ LOGS ]</a>`;

    // Completed jobs get enhanced card
    if (job.status === 'completed') {
        return `
            <div class="job-item" style="flex-direction: column; align-items: stretch;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="status-badge" style="color: ${statusColor}; border-color: ${statusColor}">
                        [ ${statusLabel} ]
                    </span>
                    <span style="font-size: 0.8em; opacity: 0.7;">ID: ${job.id}</span>
                </div>
                <div class="phase-progress" style="margin: 0.5rem 0;">
                    ${phaseChips}
                </div>
                ${renderCompletionCard(job)}
            </div>
        `;
    }

    // Running/queued/failed jobs
    const statusMessage = job.error || (job.status === 'running' ? job.phase || 'Processing...' : 'Queued');
    return `
        <div class="job-item" style="flex-direction: column; align-items: stretch;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="status-badge" style="color: ${statusColor}; border-color: ${statusColor}">
                    [ ${statusLabel} ]
                </span>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <span style="font-size: 0.8em; opacity: 0.7;">ID: ${job.id}</span>
                    ${logLink}
                </div>
            </div>
            <div class="phase-progress" style="margin: 0.5rem 0;">
                ${phaseChips}
            </div>
            <div style="font-size: 0.8em; color: ${job.status === 'failed' ? '#f00' : '#888'};">
                > ${statusMessage}
            </div>
        </div>
    `;
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
