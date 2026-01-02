/* =============================================================================
 * Montage AI - Streamlined App Logic (Consolidated Toggles + Cloud Acceleration)
 * Strategy: Outcome-based UI, Quality Profiles, Single Cloud Toggle
 * ============================================================================= */

const API_BASE = '/api';
let pollInterval = null;
let systemStatus = {};

// =============================================================================
// CONSOLIDATED CONFIGURATION - Quality-Based Approach
// =============================================================================

// Quality profiles bundle enhance/stabilize/upscale
const QUALITY_PROFILES = {
    preview: {
        name: 'Preview',
        desc: 'Fast render (360p) - See the cut quickly',
        icon: '⚡',
        settings: { enhance: false, stabilize: false, upscale: false, resolution: '360p' },
        time_multiplier: 0.3
    },
    standard: {
        name: 'Standard',
        desc: 'Balanced quality & speed (1080p)',
        icon: '⚖️',
        settings: { enhance: true, stabilize: false, upscale: false, resolution: '1080p' },
        time_multiplier: 1.0
    },
    high: {
        name: 'High Quality',
        desc: 'Polish everything (1080p + stabilize)',
        icon: '✨',
        settings: { enhance: true, stabilize: true, upscale: false, resolution: '1080p' },
        time_multiplier: 2.0
    },
    master: {
        name: 'Master',
        desc: 'Archive grade (4K upscale + HEVC)',
        icon: '🎬',
        settings: { enhance: true, stabilize: true, upscale: true, resolution: '4k' },
        time_multiplier: 4.0
    }
};

// Output modes (purpose-driven)
const OUTPUT_MODES = {
    social: {
        name: 'Social Ready',
        desc: 'Short-form vertical (9:16)',
        icon: '📱',
        settings: { shorts_mode: true, captions: true, preserve_aspect: false }
    },
    landscape: {
        name: 'Standard',
        desc: 'Horizontal (16:9)',
        icon: '🖥️',
        settings: { shorts_mode: false, captions: false, preserve_aspect: true }
    },
    pro_handoff: {
        name: 'Pro Handoff',
        desc: 'Export timeline for NLE',
        icon: '🎛️',
        settings: { shorts_mode: false, export_timeline: true, generate_proxies: true }
    }
};

// AI features (opt-in advanced)
const AI_FEATURES = {
    smart_selection: {
        name: 'Smart Clip Selection',
        desc: 'LLM picks best moments',
        icon: '🎯'
    },
    creative_loop: {
        name: 'Creative Loop',
        desc: 'Iterate cuts automatically',
        icon: '🔄'
    },
    story_engine: {
        name: 'Story Engine',
        desc: 'Narrative tension curve',
        icon: '📈'
    }
};

// Story arc presets
const STORY_ARCS = [
    { id: '', name: 'Auto', desc: 'Match style preset' },
    { id: 'hero_journey', name: 'Hero Journey', desc: 'Classic 3-act' },
    { id: 'mtv_energy', name: 'MTV Energy', desc: 'High throughout' },
    { id: 'slow_burn', name: 'Slow Burn', desc: 'Late climax' },
    { id: 'documentary', name: 'Documentary', desc: 'Observational' }
];

// Style presets
const STYLES = [
    { id: 'dynamic', name: 'Dynamic', desc: 'Position-aware pacing' },
    { id: 'vlog', name: 'Vlog', desc: 'Face-focused, talking head' },
    { id: 'sport', name: 'Sport', desc: 'Action-focused, high energy' },
    { id: 'hitchcock', name: 'Hitchcock', desc: 'Suspense build' },
    { id: 'mtv', name: 'MTV', desc: 'Rapid-fire cuts' },
    { id: 'action', name: 'Action', desc: 'High energy' },
    { id: 'documentary', name: 'Documentary', desc: 'Observational' },
    { id: 'minimalist', name: 'Minimalist', desc: 'Long takes' },
    { id: 'wes_anderson', name: 'Wes Anderson', desc: 'Symmetric/stylized' }
];

// Current state
let currentState = {
    quality: 'standard',
    output: 'landscape',
    style: 'dynamic',
    cloudAcceleration: false,
    aiFeatures: {},
    storyArc: ''
};

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await fetchStatus();
    renderUI();
    refreshFiles();
    refreshJobs();
    startPolling();
    loadSavedState();
});

// =============================================================================
// UI RENDERING
// =============================================================================

function renderUI() {
    renderQualityCards();
    renderOutputModes();
    renderStyleSelector();
    renderCloudToggle();
    renderAIFeatures();
    renderStoryArcSelector();
    updateSummary();
}

function renderQualityCards() {
    const container = document.getElementById('quality-cards');
    if (!container) return;

    container.innerHTML = Object.entries(QUALITY_PROFILES).map(([id, profile]) => `
        <div class="quality-card ${currentState.quality === id ? 'selected' : ''}" 
             onclick="selectQuality('${id}')" data-quality="${id}">
            <div class="quality-icon">${profile.icon}</div>
            <div class="quality-name">${profile.name}</div>
            <div class="quality-desc">${profile.desc}</div>
            <div class="quality-time">~${profile.time_multiplier}x render time</div>
        </div>
    `).join('');
}

function renderOutputModes() {
    const container = document.getElementById('output-modes');
    if (!container) return;

    container.innerHTML = Object.entries(OUTPUT_MODES).map(([id, mode]) => `
        <button class="output-btn ${currentState.output === id ? 'selected' : ''}"
                onclick="selectOutput('${id}')" data-output="${id}">
            <span class="output-icon">${mode.icon}</span>
            <span class="output-name">${mode.name}</span>
        </button>
    `).join('');
}

function renderStyleSelector() {
    const select = document.getElementById('style');
    if (!select) return;

    select.innerHTML = STYLES.map(style => `
        <option value="${style.id}" ${currentState.style === style.id ? 'selected' : ''}>
            ${style.name} — ${style.desc}
        </option>
    `).join('');

    select.addEventListener('change', (e) => {
        currentState.style = e.target.value;
        updateSummary();
    });
}

function renderCloudToggle() {
    const container = document.getElementById('cloud-toggle');
    if (!container) return;

    const available = systemStatus.cgpu_available || false;
    const status = available ? 'Available' : 'Not configured';

    container.innerHTML = `
        <label class="cloud-toggle-label ${!available ? 'disabled' : ''}">
            <input type="checkbox" id="cloudAcceleration" 
                   ${currentState.cloudAcceleration ? 'checked' : ''}
                   ${!available ? 'disabled' : ''}
                   onchange="toggleCloud(this.checked)">
            <div class="cloud-toggle-content">
                <div class="cloud-toggle-icon">☁️</div>
                <div class="cloud-toggle-text">
                    <div class="cloud-toggle-name">Cloud Acceleration</div>
                    <div class="cloud-toggle-status">${status}</div>
                </div>
                <div class="cloud-toggle-switch"></div>
            </div>
        </label>
        <div class="cloud-features">
            <span class="cloud-feature">🎤 Transcription</span>
            <span class="cloud-feature">🔊 Voice Isolation</span>
            <span class="cloud-feature">📐 Upscaling</span>
            <span class="cloud-feature">🎬 Render</span>
        </div>
    `;
}

function renderAIFeatures() {
    const container = document.getElementById('ai-features');
    if (!container) return;

    container.innerHTML = Object.entries(AI_FEATURES).map(([id, feature]) => `
        <label class="ai-feature-toggle">
            <input type="checkbox" 
                   ${currentState.aiFeatures[id] ? 'checked' : ''}
                   onchange="toggleAIFeature('${id}', this.checked)">
            <span class="ai-feature-icon">${feature.icon}</span>
            <span class="ai-feature-name">${feature.name}</span>
        </label>
    `).join('');
}

function renderStoryArcSelector() {
    const select = document.getElementById('storyArc');
    if (!select) return;

    select.innerHTML = STORY_ARCS.map(arc => `
        <option value="${arc.id}" ${currentState.storyArc === arc.id ? 'selected' : ''}>
            ${arc.name} ${arc.desc ? '— ' + arc.desc : ''}
        </option>
    `).join('');

    select.addEventListener('change', (e) => {
        currentState.storyArc = e.target.value;
        updateSummary();
    });
}

// =============================================================================
// STATE MANAGEMENT
// =============================================================================

function selectQuality(quality) {
    currentState.quality = quality;
    document.querySelectorAll('.quality-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.quality === quality);
    });
    updateSummary();
    saveState();
}

function selectOutput(output) {
    currentState.output = output;
    document.querySelectorAll('.output-btn').forEach(btn => {
        btn.classList.toggle('selected', btn.dataset.output === output);
    });
    updateSummary();
    saveState();
}

function toggleCloud(enabled) {
    currentState.cloudAcceleration = enabled;
    updateSummary();
    saveState();
}

function toggleAIFeature(feature, enabled) {
    currentState.aiFeatures[feature] = enabled;
    updateSummary();
    saveState();
}

function updateSummary() {
    const summary = document.getElementById('run-summary');
    if (!summary) return;

    const qualityProfile = QUALITY_PROFILES[currentState.quality];
    const outputMode = OUTPUT_MODES[currentState.output];
    const activeAI = Object.entries(currentState.aiFeatures)
        .filter(([_, v]) => v)
        .map(([k]) => AI_FEATURES[k].name);

    const duration = document.getElementById('targetDuration')?.value || '0';

    summary.innerHTML = `
        <div class="summary-row">
            <span class="summary-label">Quality:</span>
            <span class="summary-value">${qualityProfile.icon} ${qualityProfile.name}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Output:</span>
            <span class="summary-value">${outputMode.icon} ${outputMode.name}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Style:</span>
            <span class="summary-value">${currentState.style}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Duration:</span>
            <span class="summary-value">${duration === '0' ? 'Match audio' : duration + 's'}</span>
        </div>
        <div class="summary-row">
            <span class="summary-label">Cloud:</span>
            <span class="summary-value">${currentState.cloudAcceleration ? '☁️ Enabled' : 'Local'}</span>
        </div>
        ${activeAI.length ? `
        <div class="summary-row">
            <span class="summary-label">AI Features:</span>
            <span class="summary-value">${activeAI.join(', ')}</span>
        </div>
        ` : ''}
        <div class="summary-row estimate">
            <span class="summary-label">Est. Time:</span>
            <span class="summary-value">~${Math.round(qualityProfile.time_multiplier * 5)}min</span>
        </div>
    `;
}

// =============================================================================
// JOB CREATION
// =============================================================================

async function createJob(isPreview = false) {
    const quality = isPreview ? 'preview' : currentState.quality;
    const qualityProfile = QUALITY_PROFILES[quality];
    const outputMode = OUTPUT_MODES[currentState.output];

    const options = {
        // Quality settings (bundled)
        ...qualityProfile.settings,
        
        // Output mode settings
        ...outputMode.settings,
        
        // AI features
        llm_clip_selection: currentState.aiFeatures.smart_selection || false,
        creative_loop: currentState.aiFeatures.creative_loop || false,
        story_engine: currentState.aiFeatures.story_engine || false,
        
        // Cloud acceleration (single toggle)
        cgpu: currentState.cloudAcceleration,
        
        // Other settings
        story_arc: currentState.storyArc,
        quality_profile: quality,
        target_duration: parseFloat(document.getElementById('targetDuration')?.value || 0),
        music_start: parseFloat(document.getElementById('musicStart')?.value || 0),
        music_end: parseFloat(document.getElementById('musicEnd')?.value || 0) || null,
        prompt: document.getElementById('prompt')?.value || ''
    };

    const payload = {
        style: currentState.style,
        options: options,
        preset: isPreview ? 'fast' : 'standard'
    };

    try {
        showToast('Creating montage...', 'info');
        const response = await fetch(`${API_BASE}/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (data.id) {
            showToast(`Job #${data.id} started!`, 'success');
            showJobStatus(data.id);
            refreshJobs();
        } else {
            throw new Error(data.error || 'Failed to create job');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

// =============================================================================
// FILE MANAGEMENT
// =============================================================================

async function refreshFiles() {
    try {
        const response = await fetch(`${API_BASE}/files`);
        const data = await response.json();
        
        renderFileList('videoList', data.videos || [], 'video');
        renderFileList('musicList', data.music || [], 'audio');
    } catch (error) {
        console.error('Failed to refresh files:', error);
    }
}

function renderFileList(containerId, files, type) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (files.length === 0) {
        container.innerHTML = `<div class="no-files">No ${type} files uploaded</div>`;
        return;
    }

    container.innerHTML = files.slice(0, 10).map(file => `
        <div class="file-item">
            <span class="file-icon">${type === 'video' ? '🎬' : '🎵'}</span>
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatSize(file.size)}</span>
        </div>
    `).join('');

    if (files.length > 10) {
        container.innerHTML += `<div class="file-more">+${files.length - 10} more files</div>`;
    }
}

// =============================================================================
// JOB MONITORING
// =============================================================================

async function refreshJobs() {
    try {
        const response = await fetch(`${API_BASE}/jobs`);
        const data = await response.json();
        renderJobList(data.jobs || []);
    } catch (error) {
        console.error('Failed to refresh jobs:', error);
    }
}

function renderJobList(jobs) {
    const container = document.getElementById('jobList');
    if (!container) return;

    if (jobs.length === 0) {
        container.innerHTML = `<div class="no-jobs">No montages created yet</div>`;
        return;
    }

    container.innerHTML = jobs.slice(0, 8).map(job => `
        <div class="job-card ${job.status}" onclick="showJobDetails('${job.id}')">
            <div class="job-header">
                <span class="job-id">#${job.id}</span>
                <span class="job-status-badge ${job.status}">${job.status}</span>
            </div>
            <div class="job-style">${job.style || 'dynamic'}</div>
            ${job.output_file ? `
                <a href="${API_BASE}/download/${job.output_file}" class="job-download" onclick="event.stopPropagation()">
                    ⬇️ Download
                </a>
            ` : ''}
        </div>
    `).join('');
}

function showJobStatus(jobId) {
    const section = document.getElementById('jobStatusSection');
    if (section) {
        section.style.display = 'block';
        section.scrollIntoView({ behavior: 'smooth' });
    }
    startJobPolling(jobId);
}

let jobPollInterval = null;

function startJobPolling(jobId) {
    if (jobPollInterval) clearInterval(jobPollInterval);
    
    const poll = async () => {
        try {
            const response = await fetch(`${API_BASE}/jobs/${jobId}`);
            const job = await response.json();
            updateJobDisplay(job);
            
            if (job.status === 'completed' || job.status === 'failed') {
                clearInterval(jobPollInterval);
                refreshJobs();
            }
        } catch (error) {
            console.error('Job poll failed:', error);
        }
    };
    
    poll();
    jobPollInterval = setInterval(poll, 2000);
}

function updateJobDisplay(job) {
    const statusEl = document.getElementById('currentJobStatus');
    if (!statusEl) return;

    const phase = job.phase || {};
    const progress = phase.progress || 0;

    statusEl.innerHTML = `
        <div class="job-header">
            <span class="job-id">#${job.id}</span>
            <span class="job-phase">${phase.name || job.status}</span>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: ${progress}%"></div>
        </div>
        <div class="job-details">
            ${phase.message || `Status: ${job.status}`}
        </div>
    `;
}

// =============================================================================
// FILE UPLOAD
// =============================================================================

document.getElementById('videoUpload')?.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    await uploadFiles(files, 'video');
});

document.getElementById('musicUpload')?.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    await uploadFiles(files, 'music');
});

async function uploadFiles(files, type) {
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);

        try {
            showToast(`Uploading ${file.name}...`, 'info');
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                showToast(`Uploaded ${file.name}`, 'success');
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            showToast(`Upload failed: ${error.message}`, 'error');
        }
    }
    refreshFiles();
}

// =============================================================================
// SYSTEM STATUS
// =============================================================================

async function fetchStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        systemStatus = await response.json();
        renderCloudToggle();
    } catch (error) {
        console.error('Status fetch failed:', error);
    }
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(() => {
        refreshJobs();
    }, 10000);
}

// =============================================================================
// PERSISTENCE
// =============================================================================

function saveState() {
    localStorage.setItem('montage_state', JSON.stringify(currentState));
}

function loadSavedState() {
    try {
        const saved = localStorage.getItem('montage_state');
        if (saved) {
            const parsed = JSON.parse(saved);
            currentState = { ...currentState, ...parsed };
            renderUI();
        }
    } catch (error) {
        console.error('Failed to load state:', error);
    }
}

// =============================================================================
// UTILITIES
// =============================================================================

function formatSize(bytes) {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }, 10);
}

// =============================================================================
// B-ROLL & ADVANCED
// =============================================================================

async function analyzeFootage() {
    try {
        showToast('Analyzing footage...', 'info');
        const response = await fetch(`${API_BASE}/broll/analyze`, { method: 'POST' });
        const data = await response.json();
        showToast(data.message || 'Analysis complete', 'success');
    } catch (error) {
        showToast('Analysis failed', 'error');
    }
}

async function searchBroll() {
    const query = document.getElementById('brollQuery')?.value;
    if (!query) {
        showToast('Enter a search query', 'warning');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/broll/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        const results = document.getElementById('brollResults');
        if (results) {
            results.style.display = 'block';
            results.innerHTML = data.results?.map(r => `
                <div class="broll-result">
                    <span>${r.file}</span>
                    <span>${r.score?.toFixed(2)}</span>
                </div>
            `).join('') || 'No results found';
        }
    } catch (error) {
        showToast('Search failed', 'error');
    }
}

// Prompt chips
document.querySelectorAll('.chip')?.forEach(chip => {
    chip.addEventListener('click', () => {
        const prompt = document.getElementById('prompt');
        if (prompt) {
            prompt.value = chip.textContent;
            updateSummary();
        }
    });
});

// Duration slider
document.getElementById('targetDuration')?.addEventListener('input', (e) => {
    const display = document.getElementById('durationValue');
    if (display) {
        display.textContent = e.target.value === '0' ? 'Auto' : `${e.target.value}s`;
    }
    updateSummary();
});
