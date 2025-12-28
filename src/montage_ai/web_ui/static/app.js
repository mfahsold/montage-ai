
// Montage AI - Poly-Chrome Archive / Cyber-NLE Frontend
// Matches the new "Cyber Deck" UI structure

const API_BASE = '/api';
let pollInterval = null;

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
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

    const listId = type === 'video' ? 'videoList' : 'musicList';
    const listElement = document.getElementById(listId);
    
    // Simple loading indicator
    const loadingId = `loading-${Date.now()}`;
    listElement.insertAdjacentHTML('afterbegin', `
        <div id="${loadingId}" class="file-item" style="color: var(--primary)">
            <span>> UPLOADING ${file.name}...</span>
        </div>
    `);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const result = await response.json();
        console.log(`Uploaded ${type}:`, result);
    } catch (error) {
        console.error('Upload error:', error);
        alert(`UPLOAD_FAILED: ${file.name}`);
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

async function createJob() {
    // Safe element retrieval helper
    const getVal = (id) => document.getElementById(id)?.value;
    const getCheck = (id) => document.getElementById(id)?.checked || false;

    const jobData = {
        style: getVal('style') || 'dynamic',
        prompt: getVal('prompt') || '',
        stabilize: getCheck('stabilize'),
        upscale: getCheck('upscale'),
        cgpu: getCheck('cgpu'),
        llm_clip_selection: getCheck('llm_clip_selection'), // Matches HTML ID
        target_duration: parseFloat(getVal('targetDuration')) || 0,
        music_start: parseFloat(getVal('musicStart')) || 0,
        music_end: getVal('musicEnd') ? parseFloat(getVal('musicEnd')) : null,
        
        // Defaults for missing UI elements
        enhance: true,
        export_timeline: false,
        generate_proxies: false,
        preserve_aspect: false
    };

    try {
        // Visual feedback
        const btn = document.querySelector('.voxel-btn.primary');
        const originalText = btn.innerText;
        btn.innerText = '> TRANSMITTING...';
        btn.disabled = true;

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

    } catch (error) {
        console.error('Error creating job:', error);
        alert(`EXECUTION_ERROR: ${error.message}`);
    } finally {
        const btn = document.querySelector('.voxel-btn.primary');
        if (btn) {
            btn.innerText = '▶ EXECUTE_MONTAGE_SEQUENCE';
            btn.disabled = false;
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
        alert('Enter a search query');
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

function renderJob(job) {
    let statusColor = '#fff';
    let statusLabel = job.status.toUpperCase();
    
    if (job.status === 'running') statusColor = '#0ff'; // Cyan
    if (job.status === 'completed') statusColor = '#0f0'; // Green
    if (job.status === 'failed') statusColor = '#f00'; // Red
    if (job.status === 'queued') statusColor = '#ff0'; // Yellow

    const downloadLink = job.status === 'completed' 
        ? `<a href="${API_BASE}/download/${job.output_file}" class="status-badge" style="color: #0f0; border-color: #0f0; text-decoration: none;">[ DOWNLOAD ]</a>`
        : '';

    const logLink = `<a href="${API_BASE}/jobs/${job.id}/logs" target="_blank" class="status-badge" style="color: #888; border-color: #888; text-decoration: none;">[ LOGS ]</a>`;

    return `
        <div class="job-item">
            <span class="status-badge" style="color: ${statusColor}; border-color: ${statusColor}">
                [ ${statusLabel} ]
            </span>
            <div style="display: flex; flex-direction: column; gap: 0.25rem;">
                <span>ID: ${job.id} // STYLE: ${job.style.toUpperCase()}</span>
                <span style="font-size: 0.8em; opacity: 0.7;">${job.error || 'Processing stream...'}</span>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                ${downloadLink}
                ${logLink}
            </div>
        </div>
    `;
}
