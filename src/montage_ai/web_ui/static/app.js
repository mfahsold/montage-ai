
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
    // Core toggles (always visible)
    { id: 'shorts_mode', label: 'Shorts Mode', desc: '9:16 Vertical + Smart Crop.', default: false, category: 'core', badges: [{ type: 'info', text: 'TikTok/Reels' }, { type: 'quality', text: 'AI Reframing' }] },
    { id: 'captions', label: 'Burn-in Captions', desc: 'Auto-transcribed subtitles.', default: false, category: 'core', badges: [{ type: 'info', text: 'Social Ready' }] },
    { id: 'export_timeline', label: 'Export Timeline', desc: 'OTIO/EDL for NLEs.', default: false, category: 'core', badges: [{ type: 'info', text: 'Resolve/Premiere' }] },
    { id: 'cloud_acceleration', label: 'Cloud Acceleration', desc: 'Offload AI tasks to cloud GPU (upscaling, transcription, LLM).', default: false, category: 'core', badges: [{ type: 'info', text: 'Auto-Fallback' }] },
    
    // Advanced AI features (collapsible)
    { id: 'llm_clip_selection', label: 'LLM Clip Selection', desc: 'Semantic scene analysis.', default: false, category: 'advanced', badges: [{ type: 'quality', text: 'Smart Cuts' }, { type: 'cost', text: 'LLM Cost' }] },
    { id: 'creative_loop', label: 'Creative Loop', desc: 'LLM refines cuts iteratively.', default: false, category: 'advanced', badges: [{ type: 'quality', text: 'Agentic' }, { type: 'cost', text: '2-3x Time' }] },
    { id: 'story_engine', label: 'Story Engine', desc: 'Narrative tension-based editing.', default: false, category: 'advanced', badges: [{ type: 'quality', text: 'Cinematic' }] },
    
    // Pro export options (collapsible)
    { id: 'generate_proxies', label: 'Generate Proxies', desc: 'Faster NLE editing.', default: false, category: 'pro', badges: [{ type: 'cost', text: 'Extra Files' }] },
    { id: 'preserve_aspect', label: 'Preserve Aspect', desc: 'Letterbox vs crop.', default: false, category: 'pro', badges: [{ type: 'info', text: 'Safe Area' }] }
];

// Pipeline phases for progress display
const PIPELINE_PHASES = [
    { id: 'setup', label: 'Setup', keywords: ['Setup', 'Initializing', 'Starting'] },
    { id: 'audio', label: 'Audio', keywords: ['Audio', 'Beat', 'Music', 'BPM'] },
    { id: 'scenes', label: 'Scenes', keywords: ['Scene', 'Detecting', 'Analysis'] },
    { id: 'selection', label: 'Selection', keywords: ['Select', 'Clip', 'Choosing'] },
    { id: 'render', label: 'Render', keywords: ['Render', 'Encoding', 'Writing', 'FFmpeg'] }
];

// Story arc presets (matches backend story_arc.py)
const STORY_ARC_PRESETS = [
    { id: 'hero_journey', name: 'Hero Journey', desc: 'Classic 3-act with climax at 70%' },
    { id: 'mtv_energy', name: 'MTV Energy', desc: 'High energy throughout' },
    { id: 'slow_burn', name: 'Slow Burn', desc: 'Gradual build to late climax' },
    { id: 'documentary', name: 'Documentary', desc: 'Natural, observational flow' },
    { id: 'three_act', name: 'Three Act', desc: 'Classic dramatic structure' },
    { id: 'fichtean_curve', name: 'Fichtean Curve', desc: 'Rising action with mini-crises' },
    { id: 'linear_build', name: 'Linear Build', desc: 'Steady tension increase' },
    { id: 'constant', name: 'Constant', desc: 'Flat energy (music videos)' }
];

// Quality profile presets (strategy-aligned with clear value props)
const QUALITY_PROFILES = [
    { 
        id: 'preview', 
        name: '🚀 Preview', 
        desc: '360p Fast Iteration',
        details: 'No enhancements. Quick rough cut review.',
        settings: { enhance: false, stabilize: false, upscale: false, resolution: '360p' }
    },
    { 
        id: 'standard', 
        name: '📺 Standard', 
        desc: '1080p Social Media',
        details: 'Color grading. General use.',
        settings: { enhance: true, stabilize: false, upscale: false, resolution: '1080p' }
    },
    { 
        id: 'high', 
        name: '✨ High', 
        desc: '1080p Professional',
        details: 'Grading + stabilization. Pro delivery.',
        settings: { enhance: true, stabilize: true, upscale: false, resolution: '1080p' }
    },
    { 
        id: 'master', 
        name: '🎬 Master', 
        desc: '4K Broadcast',
        details: 'All enhancements + AI upscaling. Cinema, archival.',
        settings: { enhance: true, stabilize: true, upscale: true, resolution: '4k' }
    }
];

// File validation config
const FILE_VALIDATION = {
    video: {
        extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', 'mxf', 'mts', 'm2ts', 'ts'],
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
// DRY API Helper
// =============================================================================

/**
 * DRY: Centralized API fetch with error handling
 * @param {string} endpoint - API endpoint (without /api prefix)
 * @param {object} options - fetch options (method, body, etc)
 * @returns {Promise<object>} - parsed JSON response
 */
async function apiCall(endpoint, options = {}) {
    const url = `${API_BASE}/${endpoint}`;
    const defaultOptions = {
        headers: { 'Content-Type': 'application/json' }
    };

    try {
        const response = await fetch(url, { ...defaultOptions, ...options });
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(data.error || `API Error: ${response.status}`);
        }

        return data;
    } catch (error) {
        console.error(`API call failed: ${endpoint}`, error);
        throw error;
    }
}

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    fetchStatus().then(() => {
        renderToggles();
        renderStoryArcSelector();
        renderQualitySelector();
    });
    fetchTransparency();

    // Attach listeners to static inputs
    ['style', 'targetDuration', 'prompt', 'storyArc', 'qualityProfile'].forEach(id => {
        document.getElementById(id)?.addEventListener('input', updateRunSummary);
    });

    // Make prompt chips clickable
    initPromptChips();

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

    // Load saved preset if exists
    loadPreset();
});

// =============================================================================
// UI Rendering
// =============================================================================

function renderStoryArcSelector() {
    const container = document.getElementById('storyArc-container');
    if (!container) return;

    container.innerHTML = `
        <label for="storyArc">Story Arc</label>
        <select id="storyArc" class="voxel-select">
            <option value="">Auto (match style)</option>
            ${STORY_ARC_PRESETS.map(arc =>
                `<option value="${arc.id}">${arc.name} - ${arc.desc}</option>`
            ).join('')}
        </select>
        <div class="helper">Controls tension curve over time when Story Engine is enabled.</div>
    `;
}

function renderQualitySelector() {
    const container = document.getElementById('qualityProfile-container');
    if (!container) return;

    // Render as visual cards instead of dropdown (strategy: outcome-based UI)
    container.innerHTML = `
        <label style="display: block; margin-bottom: 0.75rem; font-weight: 600;">Quality Profile</label>
        <div class="quality-profile-cards" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1rem;">
            ${QUALITY_PROFILES.map(qp => `
                <div class="quality-card ${qp.id === 'standard' ? 'selected' : ''}" 
                     data-profile="${qp.id}"
                     onclick="selectQualityProfile('${qp.id}')"
                     style="
                         background: var(--card-bg, #1a1a24);
                         border: 2px solid var(--border, #2a2a3a);
                         border-radius: 10px;
                         padding: 1rem;
                         cursor: pointer;
                         transition: all 0.2s;
                         text-align: center;
                     ">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">${qp.name.split(' ')[0]}</div>
                    <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem;">${qp.name.substring(qp.name.indexOf(' ') + 1)}</div>
                    <div style="font-size: 0.75rem; opacity: 0.7; margin-bottom: 0.5rem;">${qp.desc}</div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">${qp.details}</div>
                </div>
            `).join('')}
        </div>
        <input type="hidden" id="qualityProfile" value="standard">
        <div class="helper">Choose based on your output goal. Higher quality = longer processing time.</div>
    `;

    // Add styles for selected state
    const style = document.createElement('style');
    style.textContent = `
        .quality-card:hover {
            border-color: #7C3AED !important;
            transform: translateY(-2px);
        }
        .quality-card.selected {
            border-color: #7C3AED !important;
            background: rgba(124, 58, 237, 0.15) !important;
        }
    `;
    if (!document.querySelector('#quality-card-styles')) {
        style.id = 'quality-card-styles';
        document.head.appendChild(style);
    }
}

function selectQualityProfile(profileId) {
    // Update hidden input
    const input = document.getElementById('qualityProfile');
    if (input) input.value = profileId;

    // Update UI selection
    document.querySelectorAll('.quality-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`[data-profile="${profileId}"]`)?.classList.add('selected');

    // Update run summary
    updateRunSummary();
    
    showToast(`Quality profile: ${QUALITY_PROFILES.find(q => q.id === profileId)?.name}`, 'info');
}

function initPromptChips() {
    const chips = document.querySelectorAll('.chips .chip');
    const promptInput = document.getElementById('prompt');

    chips.forEach(chip => {
        chip.style.cursor = 'pointer';
        chip.addEventListener('click', () => {
            if (promptInput) {
                promptInput.value = chip.textContent;
                promptInput.dispatchEvent(new Event('input'));
                showToast(`Prompt set: "${chip.textContent}"`, 'info');
            }
        });
    });
}

function renderToggles() {
    const container = document.getElementById('toggles-container');
    if (!container) return;

    // Separate toggles by category
    const coreToggles = TOGGLE_CONFIG.filter(t => t.category === 'core');
    const advancedToggles = TOGGLE_CONFIG.filter(t => t.category === 'advanced');
    const proToggles = TOGGLE_CONFIG.filter(t => t.category === 'pro');

    const renderToggleGroup = (toggles) => {
        return toggles.map(toggle => {
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
    };

    container.innerHTML = `
        <div style="margin-bottom: 1.5rem;">
            ${renderToggleGroup(coreToggles)}
        </div>
        
        <details style="margin-bottom: 1rem;">
            <summary style="cursor: pointer; font-weight: 600; padding: 0.5rem; background: rgba(124,58,237,0.1); border-radius: 6px; margin-bottom: 0.75rem;">
                🤖 Advanced AI Features (Optional)
            </summary>
            <div style="padding-left: 1rem;">
                ${renderToggleGroup(advancedToggles)}
                <div class="helper" style="margin-top: 0.5rem; font-size: 0.8rem;">
                    ⚠️ These features use LLM and increase processing time. Enable for maximum quality.
                </div>
            </div>
        </details>
        
        ${proToggles.length > 0 ? `
        <details style="margin-bottom: 1rem;">
            <summary style="cursor: pointer; font-weight: 600; padding: 0.5rem; background: rgba(16,185,129,0.1); border-radius: 6px; margin-bottom: 0.75rem;">
                🎬 Pro Export Options
            </summary>
            <div style="padding-left: 1rem;">
                ${renderToggleGroup(proToggles)}
            </div>
        </details>
        ` : ''}
    `;

    // Attach listeners for summary updates
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', updateRunSummary);
    });
    
    // Handle cloud acceleration toggle specially (replaces old cgpu toggle)
    const cloudAccelToggle = document.getElementById('cloud_acceleration');
    if (cloudAccelToggle) {
        cloudAccelToggle.addEventListener('change', (e) => {
            // When cloud acceleration is enabled, inform backend
            if (e.target.checked) {
                showToast('Cloud Acceleration: Enables AI upscaling, fast transcription, LLM direction with auto-fallback', 'info');
            }
        });
    }
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

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[ch]));
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
    await refreshFiles();

    // Auto-Preview Logic
    const autoPreview = document.getElementById('autoPreview')?.checked;
    const musicList = document.getElementById('musicList')?.textContent || '';
    if (autoPreview && !musicList.includes('NO_AUDIO_TRACK')) {
        showToast('Auto-starting preview render...', 'info');
        createJob(true);
    }
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
        await refreshFiles();

        // Auto-Preview Logic
        const autoPreview = document.getElementById('autoPreview')?.checked;
        const videoList = document.getElementById('videoList')?.textContent || '';
        if (autoPreview && !videoList.includes('NO_INPUT_STREAMS')) {
            showToast('Auto-starting preview render...', 'info');
            createJob(true);
        }
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
        const response = await fetch(`${API_BASE}/files?details=1`);
        const data = await response.json();

        const videoList = document.getElementById('videoList');
        const musicList = document.getElementById('musicList');

        if (videoList) {
            const videoItems = (data.video_items && data.video_items.length)
                ? data.video_items
                : (data.videos || []).map(name => ({ name }));

            videoList.innerHTML = videoItems.length 
                ? videoItems.map(item => {
                    const name = escapeHtml(item.name || '');
                    const desc = item.description ? escapeHtml(item.description) : '';
                    const size = item.size_bytes ? formatFileSize(item.size_bytes) : '';
                    const supported = item.supported !== false;
                    const warnClass = supported ? '' : 'warning';
                    const statusPrefix = supported ? '✓' : '⚠';
                    const title = desc ? ` title="${desc}"` : '';
                    const sizeHtml = size ? `<span class="file-size">${size}</span>` : '';
                    return `<div class="file-item ${warnClass}"><span${title}>${statusPrefix} ${name}</span>${sizeHtml}</div>`;
                }).join('')
                : '<div class="file-item" style="opacity:0.5">[ NO_INPUT_STREAMS ]</div>';
        }

        if (musicList) {
            const musicItems = (data.music_items && data.music_items.length)
                ? data.music_items
                : (data.music || []).map(name => ({ name }));

            musicList.innerHTML = musicItems.length 
                ? musicItems.map(item => {
                    const name = escapeHtml(item.name || '');
                    const size = item.size_bytes ? formatFileSize(item.size_bytes) : '';
                    const sizeHtml = size ? `<span class="file-size">${size}</span>` : '';
                    return `<div class="file-item"><span>✓ ${name}</span>${sizeHtml}</div>`;
                }).join('')
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

    // Get quality profile and extract settings
    const qualityProfile = getVal('qualityProfile') || 'standard';
    const profileSettings = QUALITY_PROFILES.find(p => p.id === qualityProfile)?.settings || {};

    const jobData = {
        style: getVal('style') || 'dynamic',
        prompt: getVal('prompt') || '',
        target_duration: parseFloat(getVal('targetDuration')) || 0,
        music_start: parseFloat(getVal('musicStart')) || 0,
        music_end: getVal('musicEnd') ? parseFloat(getVal('musicEnd')) : null,
        story_arc: getVal('storyArc') || '',
        quality_profile: qualityProfile
    };

    // Apply quality profile settings
    jobData.enhance = profileSettings.enhance || false;
    jobData.stabilize = profileSettings.stabilize || false;
    jobData.upscale = profileSettings.upscale || false;

    // Handle cloud acceleration toggle (replaces old cgpu/cgpu_gpu toggles)
    const cloudAcceleration = getCheck('cloud_acceleration');
    jobData.cgpu = cloudAcceleration;  // Enable LLM features
    jobData.cgpu_gpu = cloudAcceleration && jobData.upscale;  // Only use GPU cloud if upscaling

    // Add other toggles (except old enhance/stabilize/upscale/cgpu which are now in quality profile)
    TOGGLE_CONFIG.forEach(toggle => {
        if (toggle.id !== 'cloud_acceleration') {  // Skip our consolidated toggle
            jobData[toggle.id] = getCheck(toggle.id);
        }
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

    // Check cloud acceleration status if enabled
    const cloudAccelEnabled = document.getElementById('cloud_acceleration')?.checked;
    if (cloudAccelEnabled && systemStatus.cgpu === 'unavailable') {
        showToast('Cloud Acceleration enabled but unavailable - will auto-fallback to local processing', 'warning');
    }

    // Check GPU encoder status for heavy operations
    const qualityProfile = document.getElementById('qualityProfile')?.value;
    const isHeavyProfile = qualityProfile === 'high' || qualityProfile === 'master';
    if (isHeavyProfile && systemStatus.gpu === 'unknown') {
        showToast('GPU encoder not detected - high quality profiles may be slow', 'info');
    }

    return true;
}

async function createJob(isPreview = false) {
    if (!validateBeforeJob()) return;
    const jobData = buildJobPayload();
    if (isPreview) {
        jobData.preset = 'fast';
        jobData.quality_profile = 'preview';
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
        updateGlobalProgress(data);
    });

    eventSource.addEventListener('job_complete', (e) => {
        const data = JSON.parse(e.data);
        showToast(`Job ${data.job_id.substring(0,8)} Completed!`, 'success');
        refreshJobs(); // Full refresh to update gallery
        hideGlobalProgress();
    });

    eventSource.addEventListener('job_failed', (e) => {
        const data = JSON.parse(e.data);
        showToast(`Job Failed: ${data.error}`, 'error');
        refreshJobs();
        hideGlobalProgress();
    });

    eventSource.onerror = () => {
        // Fallback to polling if SSE fails (e.g. connection lost)
        console.warn('SSE connection lost, falling back to polling');
        eventSource.close();
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(refreshJobs, 5000); // Slower polling as fallback
    };
}

function updateGlobalProgress(data) {
    const bar = document.getElementById('global-progress-bar');
    const fill = document.getElementById('progress-fill');
    const label = document.getElementById('progress-label');
    const eta = document.getElementById('progress-eta');
    
    if (!bar || !data.phase) return;
    
    bar.style.display = 'block';
    fill.style.width = `${data.phase.progress_percent}%`;
    label.textContent = `> JOB #${data.job_id.substring(0,6)}: ${data.phase.label.toUpperCase()}`;
    
    // Simple ETA calculation (mock for now, could be real later)
    if (data.phase.progress_percent > 0) {
        eta.textContent = `${data.phase.progress_percent}%`;
    }
}

function hideGlobalProgress() {
    const bar = document.getElementById('global-progress-bar');
    if (bar) {
        setTimeout(() => {
            bar.style.display = 'none';
        }, 2000);
    }
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

async function finalizeJob(jobId) {
    if (!confirm('Render this montage in High Quality (1080p)? This may take a few minutes.')) return;
    
    try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}/finalize`, { method: 'POST' });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Failed to start finalize job');
        }
        const result = await response.json();
        showToast('Started High Quality Render', 'success');
        // Refresh jobs list immediately
        fetchJobs();
    } catch (e) {
        showToast(e.message, 'error');
    }
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
                <div class="gallery-actions" style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                    ${job.output_file ? 
                        `<a href="${API_BASE}/download/${job.output_file}" class="voxel-btn primary small">Download</a>` : 
                        `<span class="status-text">${job.status}</span>`
                    }
                    ${(job.status === 'completed' && job.options?.quality_profile === 'preview') ? 
                        `<button onclick="finalizeJob('${job.id}')" class="voxel-btn small secondary" title="Render in High Quality">Finalize (1080p)</button>` : ''
                    }
                    ${job.status === 'completed' || job.status === 'failed' ? 
                        `<button onclick="showDecisions('${job.id}')" class="voxel-btn small secondary" title="View AI Decisions">Log</button>` : ''
                    }
                </div>
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

function formatBackendStatus(value) {
    if (value === true) return 'enabled';
    if (value === false) return 'disabled';
    return 'unknown';
}

function renderTransparencyCard(data) {
    const container = document.getElementById('transparencyContent');
    if (!container) return;

    const policy = data.policy || {};
    const explainability = data.explainability || {};
    const llm = data.llm_backends || {};
    const ossStack = data.oss_stack || [];
    const scope = data.scope || [];
    const outOfScope = data.out_of_scope || [];

    const ossList = ossStack.length
        ? ossStack.map(item => `<li>${item.name} — ${item.purpose}</li>`).join('')
        : '<li>OSS stack details unavailable.</li>';

    const scopeList = scope.length
        ? scope.map(item => `<li>${item}</li>`).join('')
        : '<li>Scope details unavailable.</li>';

    const outOfScopeList = outOfScope.length
        ? outOfScope.map(item => `<li>${item}</li>`).join('')
        : '<li>Out-of-scope details unavailable.</li>';

    container.innerHTML = `
        <div class="transparency-block">
            <h3>DATA & CONTROL</h3>
            <ul class="transparency-list">
                <li><strong>Data:</strong> ${policy.data_handling || 'Local by default.'}</li>
                <li><strong>Training:</strong> ${policy.training || 'No training on user footage.'}</li>
                <li><strong>Control:</strong> ${policy.control || 'User-configurable pipeline.'}</li>
            </ul>
        </div>
        <div class="transparency-block">
            <h3>EXPLAINABILITY</h3>
            <ul class="transparency-list">
                <li>${explainability.decision_logs || 'Decision logs available when enabled.'}</li>
            </ul>
        </div>
        <div class="transparency-block">
            <h3>MODEL BACKENDS</h3>
            <ul class="transparency-list">
                <li>OpenAI-compatible: ${formatBackendStatus(llm.openai_compatible)}</li>
                <li>Google AI: ${formatBackendStatus(llm.google_ai)}</li>
                <li>cgpu proxy: ${formatBackendStatus(llm.cgpu)}</li>
                <li>Ollama (local): ${formatBackendStatus(llm.ollama)}</li>
            </ul>
        </div>
        <div class="transparency-block">
            <h3>OSS STACK</h3>
            <ul class="transparency-list">
                ${ossList}
            </ul>
        </div>
        <div class="transparency-block">
            <h3>SCOPE</h3>
            <ul class="transparency-list">
                ${scopeList}
            </ul>
        </div>
        <div class="transparency-block">
            <h3>OUT OF SCOPE</h3>
            <ul class="transparency-list">
                ${outOfScopeList}
            </ul>
        </div>
    `;
}

async function fetchTransparency() {
    try {
        const data = await apiCall('transparency');
        renderTransparencyCard(data);
    } catch (error) {
        const container = document.getElementById('transparencyContent');
        if (container) {
            container.innerHTML = '<div class="helper">Transparency data unavailable.</div>';
        }
    }
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

// =============================================================================
// Preset Save/Load (DRY for End Users)
// =============================================================================

const PRESET_STORAGE_KEY = 'montage_ai_preset';

function savePreset() {
    const preset = buildJobPayload();
    try {
        localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(preset));
        showToast('Preset saved to browser', 'success');
    } catch (e) {
        showToast('Failed to save preset', 'error');
    }
}

function loadPreset() {
    try {
        const saved = localStorage.getItem(PRESET_STORAGE_KEY);
        if (!saved) return;

        const preset = JSON.parse(saved);

        // Apply saved values
        const setVal = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.value = val;
        };
        const setCheck = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.checked = !!val;
        };

        setVal('style', preset.style);
        setVal('prompt', preset.prompt);
        setVal('targetDuration', preset.target_duration);
        setVal('musicStart', preset.music_start);
        if (preset.music_end) setVal('musicEnd', preset.music_end);
        setVal('storyArc', preset.story_arc || '');
        setVal('qualityProfile', preset.quality_profile || 'standard');

        // Apply toggles
        TOGGLE_CONFIG.forEach(toggle => {
            if (preset[toggle.id] !== undefined) {
                setCheck(toggle.id, preset[toggle.id]);
            }
        });

        // Update duration display
        const durationValue = document.getElementById('durationValue');
        if (durationValue) {
            const dur = preset.target_duration || 0;
            durationValue.textContent = dur === 0 ? 'Auto' : `${dur}s`;
        }

        updateRunSummary();
        console.log('Preset loaded from browser storage');
    } catch (e) {
        console.warn('Failed to load preset', e);
    }
}

function clearPreset() {
    try {
        localStorage.removeItem(PRESET_STORAGE_KEY);
        showToast('Preset cleared', 'info');
    } catch (e) {
        // Ignore
    }
}

// =============================================================================
// Transparency / Decisions Modal
// =============================================================================

async function showDecisions(jobId) {
    const modal = document.getElementById('decisionsModal');
    const content = document.getElementById('decisionsContent');
    
    if (!modal || !content) return;
    
    content.innerHTML = '<div class="helper">Loading decisions...</div>';
    modal.style.display = 'flex';
    
    try {
        const data = await apiCall(`jobs/${jobId}/decisions`);
        
        if (data.available === false) {
            content.innerHTML = `
                <div class="transparency-block">
                    <h3>⚠️ No Decisions Log</h3>
                    <p>${data.message || 'Director\'s log is not available for this job.'}</p>
                </div>
            `;
            return;
        }

        let html = '';
        
        // Director's Commentary
        if (data.director_commentary) {
            html += `
                <div class="transparency-block">
                    <h3>🗣️ Director's Note</h3>
                    <p>${data.director_commentary}</p>
                </div>
            `;
        }
        
        // Style Details
        if (data.style) {
            html += `
                <div class="transparency-block">
                    <h3>🎨 Style: ${data.style.name.toUpperCase()}</h3>
                    <ul class="transparency-list">
                        <li><strong>Mood:</strong> ${data.style.mood}</li>
                        ${data.style.description ? `<li><strong>Description:</strong> ${data.style.description}</li>` : ''}
                    </ul>
                </div>
            `;
        }
        
        // Pacing
        if (data.pacing) {
            html += `
                <div class="transparency-block">
                    <h3>⏱️ Pacing</h3>
                    <ul class="transparency-list">
                        <li><strong>Speed:</strong> ${data.pacing.speed}</li>
                        <li><strong>Variation:</strong> ${data.pacing.variation}</li>
                        <li><strong>Intro:</strong> ${data.pacing.intro_duration_beats} beats</li>
                    </ul>
                </div>
            `;
        }
        
        // Raw JSON toggle
        html += `
            <div style="margin-top: 1.5rem; border-top: 1px solid var(--border); padding-top: 1rem;">
                <button onclick="document.getElementById('rawJson').style.display = 'block'" class="voxel-btn small secondary">Show Raw JSON</button>
                <pre id="rawJson" style="display: none;">${JSON.stringify(data, null, 2)}</pre>
            </div>
        `;
        
        content.innerHTML = html;
        
    } catch (e) {
        content.innerHTML = `
            <div class="system-banner warning">
                <span class="icon">⚠️</span>
                <span>No decision logs found for this job. (Was Creative Director enabled?)</span>
            </div>
        `;
    }
}

function closeDecisions() {
    const modal = document.getElementById('decisionsModal');
    if (modal) modal.style.display = 'none';
}
