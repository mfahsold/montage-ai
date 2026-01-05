/**
 * Enhanced Job Progress Display
 * 
 * Displays real-time progress with:
 * - Phase indicator
 * - Progress bar with percentage
 * - Time remaining estimate
 * - Current step information
 * - Cancel button
 */

class JobProgressDisplay {
    constructor(containerId, jobId) {
        this.container = document.getElementById(containerId);
        this.jobId = jobId;
        this.eventSource = null;
        this.isComplete = false;
        
        this.render();
        this.startListening();
    }
    
    render() {
        this.container.innerHTML = `
            <div class="job-progress-container voxel-card">
                <div class="progress-header">
                    <h3 class="progress-title">Processing Job</h3>
                    <button class="voxel-btn voxel-btn-secondary text-xs" id="cancel-btn-${this.jobId}">
                        Cancel
                    </button>
                </div>
                
                <div class="progress-phase">
                    <span class="phase-label">PHASE:</span>
                    <span class="phase-name" id="phase-${this.jobId}">Queued</span>
                </div>
                
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar-${this.jobId}">
                        <div class="progress-fill" id="progress-fill-${this.jobId}" style="width: 0%"></div>
                    </div>
                    <span class="progress-percent" id="progress-percent-${this.jobId}">0%</span>
                </div>
                
                <div class="progress-info">
                    <div class="info-row">
                        <span class="info-label">Status:</span>
                        <span class="info-value" id="status-${this.jobId}">Initializing...</span>
                    </div>
                    <div class="info-row" id="step-row-${this.jobId}" style="display: none;">
                        <span class="info-label">Step:</span>
                        <span class="info-value" id="step-${this.jobId}">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Elapsed:</span>
                        <span class="info-value" id="elapsed-${this.jobId}">0s</span>
                    </div>
                    <div class="info-row" id="remaining-row-${this.jobId}" style="display: none;">
                        <span class="info-label">Remaining:</span>
                        <span class="info-value" id="remaining-${this.jobId}">Calculating...</span>
                    </div>
                </div>
            </div>
        `;
        
        // Attach cancel handler
        document.getElementById(`cancel-btn-${this.jobId}`).addEventListener('click', () => {
            this.cancel();
        });
    }
    
    startListening() {
        this.eventSource = new EventSource(`/api/progress/${this.jobId}`);
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.update(data);
            } catch (error) {
                console.error('Failed to parse progress data:', error);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('SSE connection error:', error);
            if (this.eventSource.readyState === EventSource.CLOSED) {
                this.stopListening();
            }
        };
    }
    
    update(data) {
        // Update phase
        document.getElementById(`phase-${this.jobId}`).textContent = data.phase_name;
        
        // Update progress bar
        const percent = Math.round(data.progress * 100);
        document.getElementById(`progress-fill-${this.jobId}`).style.width = `${percent}%`;
        document.getElementById(`progress-percent-${this.jobId}`).textContent = `${percent}%`;
        
        // Update status message
        document.getElementById(`status-${this.jobId}`).textContent = data.message;
        
        // Update step info (if available)
        if (data.current_step && data.total_steps) {
            const stepRow = document.getElementById(`step-row-${this.jobId}`);
            stepRow.style.display = 'flex';
            document.getElementById(`step-${this.jobId}`).textContent = 
                `${data.current_step_num}/${data.total_steps} - ${data.current_step}`;
        }
        
        // Update elapsed time
        document.getElementById(`elapsed-${this.jobId}`).textContent = 
            this.formatTime(data.time_elapsed);
        
        // Update remaining time (if available)
        if (data.time_remaining !== null && data.time_remaining !== undefined) {
            const remainingRow = document.getElementById(`remaining-row-${this.jobId}`);
            remainingRow.style.display = 'flex';
            document.getElementById(`remaining-${this.jobId}`).textContent = 
                this.formatTime(data.time_remaining);
        }
        
        // Check if complete
        if (data.phase === 'complete') {
            this.onComplete();
        } else if (data.phase === 'error') {
            this.onError(data.message);
        } else if (data.phase === 'cancelled') {
            this.onCancelled();
        }
    }
    
    formatTime(seconds) {
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return `${mins}m ${secs}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${mins}m`;
        }
    }
    
    cancel() {
        if (confirm('Are you sure you want to cancel this job?')) {
            fetch(`/api/jobs/${this.jobId}/cancel`, {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'cancelled') {
                    this.onCancelled();
                }
            })
            .catch(error => {
                console.error('Failed to cancel job:', error);
                alert('Failed to cancel job. Please try again.');
            });
        }
    }
    
    onComplete() {
        this.isComplete = true;
        this.stopListening();
        
        // Update UI for complete state
        document.getElementById(`cancel-btn-${this.jobId}`).style.display = 'none';
        document.getElementById(`phase-${this.jobId}`).textContent = 'Complete';
        document.getElementById(`status-${this.jobId}`).textContent = 'Job completed successfully!';
        
        // Add success styling
        this.container.querySelector('.job-progress-container').classList.add('progress-complete');
        
        // Show download button
        this.showDownloadButton();
    }
    
    onError(message) {
        this.isComplete = true;
        this.stopListening();
        
        // Update UI for error state
        document.getElementById(`cancel-btn-${this.jobId}`).style.display = 'none';
        document.getElementById(`phase-${this.jobId}`).textContent = 'Error';
        document.getElementById(`status-${this.jobId}`).textContent = message || 'Job failed';
        
        // Add error styling
        this.container.querySelector('.job-progress-container').classList.add('progress-error');
    }
    
    onCancelled() {
        this.isComplete = true;
        this.stopListening();
        
        // Update UI for cancelled state
        document.getElementById(`cancel-btn-${this.jobId}`).style.display = 'none';
        document.getElementById(`phase-${this.jobId}`).textContent = 'Cancelled';
        document.getElementById(`status-${this.jobId}`).textContent = 'Job was cancelled';
        
        // Add cancelled styling
        this.container.querySelector('.job-progress-container').classList.add('progress-cancelled');
    }
    
    showDownloadButton() {
        const infoDiv = this.container.querySelector('.progress-info');
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'voxel-btn voxel-btn-primary w-full mt-4';
        downloadBtn.textContent = 'Download Video';
        downloadBtn.onclick = () => {
            window.location.href = `/api/jobs/${this.jobId}/download`;
        };
        infoDiv.appendChild(downloadBtn);
    }
    
    stopListening() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
    
    destroy() {
        this.stopListening();
        this.container.innerHTML = '';
    }
}

// CSS Styles for Progress Display (add to voxel-dark.css)
const progressStyles = `
.job-progress-container {
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.progress-title {
    font-size: 1.25rem;
    margin: 0;
}

.progress-phase {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
    font-size: 0.875rem;
}

.phase-label {
    color: var(--muted-fg);
    font-weight: 700;
    letter-spacing: 0.05em;
}

.phase-name {
    color: var(--primary);
    font-weight: 700;
}

.progress-bar-container {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.progress-bar {
    flex: 1;
    height: 24px;
    background: var(--input-bg);
    border: 2px solid var(--border);
    position: relative;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    transition: width 0.3s ease;
    position: relative;
}

.progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, 
        transparent, 
        rgba(255, 255, 255, 0.3), 
        transparent
    );
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.progress-percent {
    font-size: 0.875rem;
    font-weight: 700;
    color: var(--primary);
    min-width: 3rem;
    text-align: right;
}

.progress-info {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.info-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
}

.info-label {
    color: var(--muted-fg);
    font-weight: 700;
}

.info-value {
    color: var(--fg);
}

/* State-specific styles */
.progress-complete {
    border-color: var(--success);
}

.progress-complete .progress-fill {
    background: var(--success);
}

.progress-error {
    border-color: var(--error);
}

.progress-error .progress-fill {
    background: var(--error);
}

.progress-cancelled {
    border-color: var(--muted);
    opacity: 0.7;
}
`;

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = JobProgressDisplay;
}
