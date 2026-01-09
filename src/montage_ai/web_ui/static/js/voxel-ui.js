/**
 * Montage AI - Voxel UI JavaScript Helpers
 *
 * Loading states, toasts, and form utilities for the Voxel Design System.
 * Works with templates/components/voxel.html macros.
 */

const VoxelUI = {
    // ==========================================================================
    // LOADING STATES
    // ==========================================================================

    /**
     * Show a loading overlay with optional progress
     * @param {string} id - Overlay element ID (default: 'loading-overlay')
     * @param {string} message - Loading message to display
     */
    showLoading(id = 'loading-overlay', message = 'Processing...') {
        const overlay = document.getElementById(id);
        if (overlay) {
            overlay.style.display = 'flex';
            const msgEl = overlay.querySelector('.loading-message');
            if (msgEl) msgEl.textContent = message;
            this.updateProgress(id, 0);
        }
    },

    /**
     * Hide a loading overlay
     * @param {string} id - Overlay element ID
     */
    hideLoading(id = 'loading-overlay') {
        const overlay = document.getElementById(id);
        if (overlay) {
            overlay.style.display = 'none';
        }
    },

    /**
     * Update loading overlay progress
     * @param {string} id - Overlay element ID
     * @param {number} percent - Progress percentage (0-100)
     */
    updateProgress(id, percent) {
        const progressFill = document.getElementById(`${id}-progress`);
        const progressText = document.getElementById(`${id}-text`);

        if (progressFill) progressFill.style.width = `${percent}%`;
        if (progressText) progressText.textContent = `${Math.round(percent)}%`;
    },

    /**
     * Set a button to loading state
     * @param {string|HTMLElement} btn - Button element or ID
     * @param {boolean} loading - Whether loading
     */
    setButtonLoading(btn, loading = true) {
        const button = typeof btn === 'string' ? document.getElementById(btn) : btn;
        if (!button) return;

        if (loading) {
            button.classList.add('is-loading');
            button.disabled = true;
        } else {
            button.classList.remove('is-loading');
            button.disabled = false;
        }
    },

    // ==========================================================================
    // TOAST NOTIFICATIONS
    // ==========================================================================

    /**
     * Show a toast notification
     * @param {string} message - Toast message
     * @param {string} type - 'success' | 'error' | 'warning' | 'info'
     * @param {number} duration - Auto-dismiss time in ms (0 = no auto-dismiss)
     */
    toast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('voxel-toast-container');
        if (!container) {
            console.warn('VoxelUI: No toast container found. Add {{ toast_container() }} to your template.');
            return;
        }

        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };

        const toast = document.createElement('div');
        toast.className = `voxel-alert voxel-alert-${type} voxel-toast`;
        toast.innerHTML = `
            <span class="alert-icon">${icons[type] || icons.info}</span>
            <span class="alert-message">${message}</span>
            <button type="button" class="alert-close" onclick="VoxelUI.dismissToast(this.parentElement)">✕</button>
        `;

        container.appendChild(toast);

        if (duration > 0) {
            setTimeout(() => this.dismissToast(toast), duration);
        }

        return toast;
    },

    /**
     * Dismiss a toast notification
     * @param {HTMLElement} toast - Toast element
     */
    dismissToast(toast) {
        if (!toast || toast.classList.contains('toast-exit')) return;

        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 200);
    },

    // Convenience methods
    success(message, duration) { return this.toast(message, 'success', duration); },
    error(message, duration) { return this.toast(message, 'error', duration); },
    warning(message, duration) { return this.toast(message, 'warning', duration); },
    info(message, duration) { return this.toast(message, 'info', duration); },

    // ==========================================================================
    // API HELPERS
    // ==========================================================================

    /**
     * Fetch with automatic loading state and error handling
     * @param {string} url - API endpoint
     * @param {Object} options - Fetch options + extra config
     * @returns {Promise<Object>} - Response JSON
     */
    async apiFetch(url, options = {}) {
        const {
            method = 'GET',
            body = null,
            loadingId = null,
            loadingMessage = 'Processing...',
            buttonId = null,
            showToast = true,
            ...fetchOptions
        } = options;

        // Show loading state
        if (loadingId) this.showLoading(loadingId, loadingMessage);
        if (buttonId) this.setButtonLoading(buttonId, true);

        try {
            const response = await fetch(url, {
                method,
                headers: {
                    'Content-Type': 'application/json',
                    ...(fetchOptions.headers || {})
                },
                body: body ? JSON.stringify(body) : null,
                ...fetchOptions
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }

            if (showToast && data.success !== false) {
                this.success(data.message || 'Operation completed');
            }

            return data;

        } catch (err) {
            if (showToast) {
                this.error(err.message || 'An error occurred');
            }
            throw err;

        } finally {
            if (loadingId) this.hideLoading(loadingId);
            if (buttonId) this.setButtonLoading(buttonId, false);
        }
    },

    /**
     * POST request with loading state
     */
    async post(url, body, options = {}) {
        return this.apiFetch(url, { method: 'POST', body, ...options });
    },

    // ==========================================================================
    // SSE (Server-Sent Events) Helper
    // ==========================================================================

    /**
     * Subscribe to job updates via SSE
     * @param {string} jobId - Job ID to monitor
     * @param {Object} callbacks - Event callbacks
     * @returns {EventSource} - EventSource instance for cleanup
     */
    subscribeToJob(jobId, callbacks = {}) {
        const {
            onProgress = () => {},
            onComplete = () => {},
            onError = () => {},
            onPhase = () => {}
        } = callbacks;

        const eventSource = new EventSource('/api/stream');

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // Filter for our job
                if (data.job_id !== jobId) return;

                switch (data.status) {
                    case 'processing':
                        onProgress(data);
                        break;
                    case 'completed':
                        onComplete(data);
                        eventSource.close();
                        break;
                    case 'failed':
                        onError(data);
                        eventSource.close();
                        break;
                    case 'phase':
                        onPhase(data);
                        break;
                }
            } catch (e) {
                console.error('SSE parse error:', e);
            }
        };

        eventSource.onerror = (err) => {
            console.error('SSE connection error:', err);
            onError({ error: 'Connection lost' });
        };

        return eventSource;
    },

    // ==========================================================================
    // FORM UTILITIES
    // ==========================================================================

    /**
     * Get form data as object
     * @param {string|HTMLFormElement} form - Form element or ID
     * @returns {Object} - Form data as key-value pairs
     */
    getFormData(form) {
        const formEl = typeof form === 'string' ? document.getElementById(form) : form;
        if (!formEl) return {};

        const data = {};
        const formData = new FormData(formEl);

        for (const [key, value] of formData.entries()) {
            // Handle checkboxes
            const input = formEl.elements[key];
            if (input && input.type === 'checkbox') {
                data[key] = input.checked;
            } else if (input && input.type === 'number') {
                data[key] = parseFloat(value) || 0;
            } else {
                data[key] = value;
            }
        }

        return data;
    },

    /**
     * Populate form from object
     * @param {string|HTMLFormElement} form - Form element or ID
     * @param {Object} data - Data to populate
     */
    setFormData(form, data) {
        const formEl = typeof form === 'string' ? document.getElementById(form) : form;
        if (!formEl) return;

        for (const [key, value] of Object.entries(data)) {
            const input = formEl.elements[key];
            if (!input) continue;

            if (input.type === 'checkbox') {
                input.checked = Boolean(value);
            } else if (input.type === 'radio') {
                const radio = formEl.querySelector(`input[name="${key}"][value="${value}"]`);
                if (radio) radio.checked = true;
            } else {
                input.value = value;
            }
        }
    },

    // ==========================================================================
    // INITIALIZATION
    // ==========================================================================

    /**
     * Initialize Voxel UI behaviors
     * Call this on DOMContentLoaded
     */
    init() {
        // Auto-update range slider values
        document.querySelectorAll('.voxel-range').forEach(range => {
            const valueEl = document.getElementById(`${range.id}-value`);
            if (valueEl) {
                range.addEventListener('input', () => {
                    valueEl.textContent = range.value;
                });
            }
        });

        // Add loading state to forms with data-loading-submit
        document.querySelectorAll('form[data-loading-submit]').forEach(form => {
            form.addEventListener('submit', () => {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) this.setButtonLoading(submitBtn, true);
            });
        });

        console.log('[VoxelUI] Initialized');
    }
};

// Auto-init on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => VoxelUI.init());
} else {
    VoxelUI.init();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoxelUI;
}
