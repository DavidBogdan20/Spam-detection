/**
 * Email Inbox - JavaScript for real email display
 * Handles fetching and displaying emails from connected email accounts
 */

// State
let emails = [];
let currentEmail = null;
let currentFilter = 'all';
let isConnected = false;

// API Base URL
const API_BASE = '/api';

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', () => {
    checkEmailStatus();
    checkModelStatus();
});

// ==================== API Functions ====================

async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || `API error: ${response.status}`);
        }

        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

async function checkModelStatus() {
    try {
        const status = await apiCall('/health');
        const statusDot = document.querySelector('.status-dot');

        if (status.model_loaded) {
            statusDot.style.background = 'var(--success)';
        } else {
            statusDot.style.background = 'var(--warning)';
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// ==================== Email Status ====================

async function checkEmailStatus() {
    try {
        const status = await apiCall('/email/status');
        updateConnectionUI(status.connected, status.email);

        if (status.connected) {
            await fetchEmails();
        }
    } catch (error) {
        console.error('Error checking email status:', error);
        updateConnectionUI(false, null);
    }
}

function updateConnectionUI(connected, email) {
    isConnected = connected;
    const connectionStatus = document.getElementById('connection-status');
    const connectPrompt = document.getElementById('connect-prompt');
    const emailBtnText = document.getElementById('email-btn-text');
    const refreshBtn = document.getElementById('refresh-btn');
    const connectForm = document.getElementById('email-connect-form');
    const connectedInfo = document.getElementById('email-connected-info');
    const connectBtn = document.getElementById('connect-btn');
    const disconnectBtn = document.getElementById('disconnect-btn');

    if (connected) {
        connectionStatus.textContent = `Connected to ${email}`;
        connectionStatus.style.color = 'var(--success)';
        connectPrompt.style.display = 'none';
        emailBtnText.textContent = 'Connected';
        refreshBtn.disabled = false;

        // Update modal
        if (connectForm) connectForm.style.display = 'none';
        if (connectedInfo) connectedInfo.style.display = 'block';
        if (connectBtn) connectBtn.style.display = 'none';
        if (disconnectBtn) disconnectBtn.style.display = 'inline-flex';
        const connectedEmail = document.getElementById('connected-email');
        if (connectedEmail) connectedEmail.textContent = email;
    } else {
        connectionStatus.textContent = 'Not connected';
        connectionStatus.style.color = 'var(--text-secondary)';
        connectPrompt.style.display = 'flex';
        emailBtnText.textContent = 'Connect Email';
        refreshBtn.disabled = true;

        // Update modal
        if (connectForm) connectForm.style.display = 'block';
        if (connectedInfo) connectedInfo.style.display = 'none';
        if (connectBtn) connectBtn.style.display = 'inline-flex';
        if (disconnectBtn) disconnectBtn.style.display = 'none';
    }
}

// ==================== Email Fetching ====================

async function fetchEmails() {
    const listContainer = document.getElementById('messages-list');
    listContainer.innerHTML = `
        <div class="loading-state">
            <div class="loading-spinner"></div>
            <p>Fetching your emails...</p>
        </div>
    `;

    try {
        const result = await apiCall('/email/fetch?limit=30');

        if (result.success) {
            emails = result.emails || [];
            updateCounts();
            renderEmails();
            showToast(`✓ Fetched ${emails.length} emails`, 'success');
        } else {
            throw new Error(result.message || 'Failed to fetch emails');
        }
    } catch (error) {
        listContainer.innerHTML = `
            <div class="loading-state">
                <div class="empty-icon">❌</div>
                <h3>Error fetching emails</h3>
                <p>${error.message}</p>
                <button class="btn btn-primary" onclick="fetchEmails()" style="margin-top: 16px;">
                    <span>🔄</span> Try Again
                </button>
            </div>
        `;
        showToast('Error fetching emails', 'error');
    }
}

async function refreshEmails() {
    if (!isConnected) {
        showToast('Please connect your email first', 'error');
        return;
    }
    await fetchEmails();
}

// ==================== Rendering ====================

function renderEmails() {
    const listContainer = document.getElementById('messages-list');

    if (emails.length === 0) {
        listContainer.innerHTML = `
            <div class="loading-state">
                <div class="empty-icon">📭</div>
                <p>No emails found</p>
            </div>
        `;
        return;
    }

    // Filter emails
    let filteredEmails = emails;
    if (currentFilter === 'spam') {
        filteredEmails = emails.filter(e => e.prediction === 'spam');
    } else if (currentFilter === 'ham') {
        filteredEmails = emails.filter(e => e.prediction === 'ham');
    }

    listContainer.innerHTML = filteredEmails.map((email, index) => `
        <div class="message-item ${email.prediction}" onclick="selectEmail(${index})" data-index="${index}">
            <div class="message-indicator ${email.prediction}">
                ${email.prediction === 'spam' ? '⚠️' : '✉️'}
            </div>
            <div class="message-content">
                <div class="message-subject">${escapeHtml(email.subject || '(No Subject)')}</div>
                <div class="message-from">${escapeHtml(email.from || 'Unknown')}</div>
                <div class="message-meta">
                    <span class="message-label ${email.prediction}">
                        ${email.prediction.toUpperCase()}
                    </span>
                    <span class="message-confidence">
                        ${(email.confidence * 100).toFixed(1)}% confidence
                    </span>
                    <span class="message-date">${email.date || ''}</span>
                </div>
            </div>
        </div>
    `).join('');
}

function updateCounts() {
    const emailCountEl = document.getElementById('email-count');
    const spamCount = emails.filter(e => e.prediction === 'spam').length;
    const hamCount = emails.filter(e => e.prediction === 'ham').length;

    if (emailCountEl) emailCountEl.textContent = emails.length;
}

// ==================== Email Selection ====================

function selectEmail(index) {
    // Update selection UI
    document.querySelectorAll('.message-item').forEach(el => {
        el.classList.remove('selected');
    });

    const selectedEl = document.querySelector(`[data-index="${index}"]`);
    if (selectedEl) {
        selectedEl.classList.add('selected');
    }

    currentEmail = emails[index];
    showEmailDetail(currentEmail);
}

function showEmailDetail(email) {
    const emptyState = document.querySelector('.detail-empty');
    const detailContent = document.getElementById('detail-content');

    emptyState.style.display = 'none';
    detailContent.style.display = 'block';

    // Update subject
    document.getElementById('detail-subject').textContent = email.subject || '(No Subject)';

    // Update from and date
    document.getElementById('detail-from').textContent = email.from || 'Unknown';
    document.getElementById('detail-date').textContent = email.date || '-';

    // Update classification badge
    const classification = document.getElementById('detail-classification');
    classification.innerHTML = `
        <span class="classification-badge ${email.prediction}">
            ${email.prediction === 'spam' ? '⚠️ SPAM' : '✅ SAFE'}
        </span>
    `;

    // Update confidence meter
    const confidence = email.confidence * 100;
    document.getElementById('confidence-fill').style.width = `${confidence}%`;
    document.getElementById('confidence-value').textContent = `${confidence.toFixed(1)}%`;

    // Update full content
    document.getElementById('message-full-content').textContent = email.content || '';

    // Update metadata
    document.getElementById('spam-prob').textContent = `${(email.spam_probability * 100).toFixed(1)}%`;
    document.getElementById('ham-prob').textContent = `${((1 - email.spam_probability) * 100).toFixed(1)}%`;
}

// ==================== Filtering & Sorting ====================

function filterEmails(filter) {
    currentFilter = filter;

    // Update tab UI
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.classList.remove('active');
        const tabText = tab.textContent.trim().toLowerCase();
        if ((filter === 'all' && tabText === 'all') ||
            (filter === 'spam' && tabText === 'spam') ||
            (filter === 'ham' && tabText === 'safe')) {
            tab.classList.add('active');
        }
    });

    renderEmails();
}

function sortEmails() {
    const sortBy = document.getElementById('sort-by').value;

    if (sortBy === 'date') {
        // Keep original order (newest first from server)
        emails.sort((a, b) => a.id.localeCompare(b.id));
    } else if (sortBy === 'confidence') {
        emails.sort((a, b) => b.confidence - a.confidence);
    } else if (sortBy === 'prediction') {
        emails.sort((a, b) => {
            if (a.prediction === 'spam' && b.prediction !== 'spam') return -1;
            if (a.prediction !== 'spam' && b.prediction === 'spam') return 1;
            return 0;
        });
    }

    renderEmails();
}

// ==================== Email Connection Modal ====================

function openEmailModal() {
    document.getElementById('email-modal').classList.add('active');
    checkEmailStatus();
}

function closeEmailModal() {
    document.getElementById('email-modal').classList.remove('active');
    document.getElementById('email-password').value = '';
    document.getElementById('email-error').style.display = 'none';
}

async function connectEmail() {
    const email = document.getElementById('email-address').value.trim();
    const password = document.getElementById('email-password').value;
    const provider = document.getElementById('email-provider').value;
    const errorDiv = document.getElementById('email-error');
    const connectBtn = document.getElementById('connect-btn');

    if (!email || !password) {
        errorDiv.textContent = 'Please enter both email and app password.';
        errorDiv.style.display = 'block';
        return;
    }

    connectBtn.disabled = true;
    connectBtn.innerHTML = '<span>⏳</span> Connecting...';
    errorDiv.style.display = 'none';

    try {
        const result = await apiCall('/email/connect', {
            method: 'POST',
            body: JSON.stringify({
                email: email,
                password: password,
                imap_server: provider || null
            })
        });

        if (result.success) {
            showToast('✓ Connected to ' + email, 'success');
            updateConnectionUI(true, email);
            await fetchEmails();
            closeEmailModal();
        } else {
            errorDiv.textContent = result.message || 'Connection failed.';
            errorDiv.style.display = 'block';
        }
    } catch (error) {
        errorDiv.textContent = error.message || 'Connection failed. Make sure you\'re using an App Password.';
        errorDiv.style.display = 'block';
    } finally {
        connectBtn.disabled = false;
        connectBtn.innerHTML = '<span>🔗</span> Connect & Fetch Emails';
    }
}

async function disconnectEmail() {
    try {
        await apiCall('/email/disconnect', { method: 'POST' });
        showToast('Disconnected from email', 'success');
        emails = [];
        updateConnectionUI(false, null);
        renderEmails();

        // Show connect prompt
        document.getElementById('connect-prompt').style.display = 'flex';
    } catch (error) {
        showToast('Error disconnecting', 'error');
    }
}

// ==================== Feedback ====================

async function submitFeedback(correctLabel) {
    if (!currentEmail) {
        showToast('No email selected', 'error');
        return;
    }

    try {
        const originalPrediction = currentEmail.prediction === 'spam' ? 1 : 0;

        await apiCall('/feedback', {
            method: 'POST',
            body: JSON.stringify({
                message_id: currentEmail.id,
                original_prediction: originalPrediction,
                correct_label: correctLabel,
                message: currentEmail.content
            })
        });

        const feedbackText = correctLabel === 1 ? 'Marked as spam' : 'Marked as safe';
        showToast(`✓ ${feedbackText}. Thank you!`, 'success');
    } catch (error) {
        showToast('Failed to submit feedback', 'error');
    }
}

// ==================== Toast Notifications ====================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastSlideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==================== Utilities ====================

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeEmailModal();
    }
});
