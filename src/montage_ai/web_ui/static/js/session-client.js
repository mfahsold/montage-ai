/**
 * Montage AI - Session Client
 * Shared logic for managing editing sessions, assets, and state.
 */

class SessionClient {
    constructor(type) {
        this.type = type; // 'shorts' | 'transcript'
        this.sessionId = new URLSearchParams(window.location.search).get('session_id');
        this.state = {};
        this.assets = {};
        
        // Event listeners
        this.onStateChange = null;
        this.onAssetLoaded = null;
    }

    async init() {
        if (this.sessionId) {
            await this.loadSession(this.sessionId);
        } else {
            await this.createSession();
        }
        return this.sessionId;
    }

    async createSession() {
        try {
            const res = await fetch('/api/session/create', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ type: this.type })
            });
            const data = await res.json();
            this.sessionId = data.id;
            
            // Update URL without reload
            const newUrl = `${window.location.pathname}?session_id=${this.sessionId}`;
            window.history.pushState({path: newUrl}, '', newUrl);
            
            this.showToast('New Session Created', 'success');
        } catch (e) {
            this.showToast('Failed to create session: ' + e.message, 'error');
            throw e;
        }
    }

    async loadSession(id) {
        try {
            const res = await fetch(`/api/session/${id}`);
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            this.state = data.state || {};
            this.assets = data.assets || {};
            
            if (this.onStateChange) this.onStateChange(this.state);
            
            // Check for main video
            const videoAsset = Object.values(this.assets).find(a => a.type === 'video');
            if (videoAsset && this.onAssetLoaded) {
                this.onAssetLoaded(videoAsset);
            }
            
            this.showToast('Session Loaded', 'success');
        } catch (e) {
            this.showToast('Failed to load session: ' + e.message, 'error');
            throw e;
        }
    }

    async uploadAsset(file, type = 'video') {
        if (!this.sessionId) await this.createSession();
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        
        this.showToast('Uploading...', 'info');

        try {
            const res = await fetch(`/api/session/${this.sessionId}/asset`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (data.error) throw new Error(data.error);
            
            this.assets = data.session.assets;
            this.showToast('Upload complete', 'success');
            
            return data.asset;
        } catch (error) {
            console.error('Upload failed:', error);
            this.showToast('Upload failed: ' + error.message, 'error');
            throw error;
        }
    }

    async analyze(type, params = {}) {
        if (!this.sessionId) return;
        
        this.showToast(`Running ${type} analysis...`, 'info');
        try {
            const payload = { type, ...params };
            const res = await fetch(`/api/session/${this.sessionId}/analyze`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            this.showToast('Analysis complete', 'success');
            return data;
        } catch (e) {
            this.showToast('Analysis failed: ' + e.message, 'error');
            throw e;
        }
    }

    async removeFillers() {
        if (!this.sessionId) return;
        
        this.showToast('Removing filler words...', 'info');
        try {
            const res = await fetch(`/api/session/${this.sessionId}/remove_fillers`, {
                method: 'POST'
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            this.showToast(`Removed ${data.count} filler words`, 'success');
            
            // Update local state if edits returned
            if (data.edits) {
                this.state.edits = data.edits;
                if (this.onStateChange) this.onStateChange(this.state);
            }
            
            return data;
        } catch (e) {
            this.showToast('Failed to remove fillers: ' + e.message, 'error');
            throw e;
        }
    }

    async saveState(updates) {
        if (!this.sessionId) return;
        
        // Merge updates locally
        this.state = { ...this.state, ...updates };
        
        try {
            await fetch(`/api/session/${this.sessionId}/state`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(updates)
            });
        } catch (e) {
            console.error('Failed to save state:', e);
            this.showToast('Failed to save state', 'error');
        }
    }

    async renderPreview(params) {
        if (!this.sessionId) return;
        
        try {
            const res = await fetch(`/api/session/${this.sessionId}/render_preview`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            return data;
        } catch (e) {
            this.showToast('Preview failed: ' + e.message, 'error');
            throw e;
        }
    }

    async export(format) {
        if (!this.sessionId) return;
        
        this.showToast(`Exporting to ${format.toUpperCase()}...`, 'info');
        try {
            const res = await fetch(`/api/session/${this.sessionId}/export`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ format })
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            
            this.showToast('Export complete', 'success');
            
            // Trigger download
            if (data.url) {
                const link = document.createElement('a');
                link.href = data.url;
                link.download = '';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            
            return data;
        } catch (e) {
            this.showToast('Export failed: ' + e.message, 'error');
            throw e;
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            bottom: 2rem;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            background: ${type === 'success' ? '#10B981' : type === 'error' ? '#EF4444' : type === 'warning' ? '#F59E0B' : '#7C3AED'};
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: opacity 0.3s;
        `;
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}
