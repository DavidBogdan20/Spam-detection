/**
 * Spam Shield - Frontend Application
 * Handles message classification, feedback, and UI interactions
 */

// State
let messages = [];
let currentMessage = null;
let currentFilter = 'all';
let spamCount = 0;
let hamCount = 0;

// API Base URL
const API_BASE = '/api';

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', () => {
    loadMessages();
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
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showToast('Error communicating with server', 'error');
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

// ==================== Messages ====================

async function loadMessages() {
    const listContainer = document.getElementById('messages-list');
    
    try {
        const data = await apiCall(`/messages?limit=50&filter=${currentFilter}`);
        messages = data.messages || [];
        
        // Update counts
        spamCount = messages.filter(m => m.prediction === 'spam').length;
        hamCount = messages.filter(m => m.prediction === 'ham').length;
        
        updateCounts();
        renderMessages();
        
    } catch (error) {
        listContainer.innerHTML = `
            <div class="loading-state">
                <p>Failed to load messages. Please try again.</p>
            </div>
        `;
    }
}

function renderMessages() {
    const listContainer = document.getElementById('messages-list');
    
    if (messages.length === 0) {
        listContainer.innerHTML = `
            <div class="loading-state">
                <div class="empty-icon">📭</div>
                <p>No messages found</p>
            </div>
        `;
        return;
    }
    
    // Filter messages based on current filter
    let filteredMessages = messages;
    if (currentFilter === 'spam') {
        filteredMessages = messages.filter(m => m.prediction === 'spam');
    } else if (currentFilter === 'ham') {
        filteredMessages = messages.filter(m => m.prediction === 'ham');
    }
    
    listContainer.innerHTML = filteredMessages.map((msg, index) => `
        <div class="message-item ${msg.prediction}" onclick="selectMessage(${index})" data-index="${index}">
            <div class="message-indicator ${msg.prediction}">
                ${msg.prediction === 'spam' ? '⚠️' : '✉️'}
            </div>
            <div class="message-content">
                <div class="message-preview">${escapeHtml(msg.content)}</div>
                <div class="message-meta">
                    <span class="message-label ${msg.prediction}">
                        ${msg.prediction.toUpperCase()}
                    </span>
                    <span class="message-confidence">
                        ${(msg.confidence * 100).toFixed(1)}% confidence
                    </span>
                </div>
            </div>
        </div>
    `).join('');
}

function updateCounts() {
    document.getElementById('inbox-count').textContent = messages.length;
    document.getElementById('spam-count').textContent = spamCount;
    document.getElementById('ham-count').textContent = hamCount;
    document.getElementById('total-messages').textContent = `${messages.length} messages`;
}

function selectMessage(index) {
    // Update selection UI
    document.querySelectorAll('.message-item').forEach(el => {
        el.classList.remove('selected');
    });
    
    const selectedEl = document.querySelector(`[data-index="${index}"]`);
    if (selectedEl) {
        selectedEl.classList.add('selected');
    }
    
    // Update current message
    currentMessage = messages[index];
    
    // Show detail panel
    showMessageDetail(currentMessage);
}

function showMessageDetail(msg) {
    const emptyState = document.querySelector('.detail-empty');
    const detailContent = document.getElementById('detail-content');
    
    emptyState.style.display = 'none';
    detailContent.style.display = 'block';
    
    // Update classification badge
    const classification = document.getElementById('detail-classification');
    classification.innerHTML = `
        <span class="classification-badge ${msg.prediction}">
            ${msg.prediction === 'spam' ? '⚠️ SPAM' : '✅ SAFE'}
        </span>
    `;
    
    // Update confidence meter
    const confidence = msg.confidence * 100;
    document.getElementById('confidence-fill').style.width = `${confidence}%`;
    document.getElementById('confidence-value').textContent = `${confidence.toFixed(1)}%`;
    
    // Update full content
    document.getElementById('message-full-content').textContent = msg.content;
    
    // Update metadata
    document.getElementById('spam-prob').textContent = `${(msg.spam_probability * 100).toFixed(1)}%`;
    document.getElementById('ham-prob').textContent = `${((1 - msg.spam_probability) * 100).toFixed(1)}%`;
    document.getElementById('true-label').textContent = msg.true_label ? msg.true_label.toUpperCase() : '-';
    document.getElementById('msg-length').textContent = `${msg.content.length} chars`;
}

// ==================== Filtering & Sorting ====================

function filterMessages(filter) {
    currentFilter = filter;
    
    // Update tab UI
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    event.target.classList.add('active');
    
    // Re-render with filter
    renderMessages();
}

function sortMessages() {
    const sortBy = document.getElementById('sort-by').value;
    
    if (sortBy === 'confidence') {
        messages.sort((a, b) => b.confidence - a.confidence);
    } else if (sortBy === 'prediction') {
        messages.sort((a, b) => {
            if (a.prediction === 'spam' && b.prediction !== 'spam') return -1;
            if (a.prediction !== 'spam' && b.prediction === 'spam') return 1;
            return 0;
        });
    }
    
    renderMessages();
}

function refreshMessages() {
    loadMessages();
    showToast('Messages refreshed', 'success');
}

// ==================== Feedback ====================

async function submitFeedback(correctLabel) {
    if (!currentMessage) {
        showToast('No message selected', 'error');
        return;
    }
    
    try {
        const originalPrediction = currentMessage.prediction === 'spam' ? 1 : 0;
        
        await apiCall('/feedback', {
            method: 'POST',
            body: JSON.stringify({
                message_id: currentMessage.id,
                original_prediction: originalPrediction,
                correct_label: correctLabel,
                message: currentMessage.content
            })
        });
        
        const feedbackText = correctLabel === 1 ? 'Marked as spam' : 'Marked as safe';
        showToast(`✓ ${feedbackText}. Thank you for your feedback!`, 'success');
        
    } catch (error) {
        showToast('Failed to submit feedback', 'error');
    }
}

// ==================== Test Message Modal ====================

function openComposeModal() {
    document.getElementById('compose-modal').classList.add('active');
    document.getElementById('test-message').focus();
    document.getElementById('test-result').style.display = 'none';
}

function closeComposeModal() {
    document.getElementById('compose-modal').classList.remove('active');
    document.getElementById('test-message').value = '';
    document.getElementById('test-result').style.display = 'none';
}

async function testMessage() {
    const message = document.getElementById('test-message').value.trim();
    
    if (!message) {
        showToast('Please enter a message', 'error');
        return;
    }
    
    try {
        const result = await apiCall('/classify', {
            method: 'POST',
            body: JSON.stringify({ message })
        });
        
        // Show result
        const resultDiv = document.getElementById('test-result');
        resultDiv.style.display = 'block';
        
        const badge = document.getElementById('result-badge');
        badge.textContent = result.prediction.toUpperCase();
        badge.className = `result-badge ${result.prediction}`;
        
        document.getElementById('result-confidence').textContent = 
            `${(result.confidence * 100).toFixed(1)}% confident`;
        
        document.getElementById('result-bar-spam').style.width = 
            `${result.spam_probability * 100}%`;
        
    } catch (error) {
        showToast('Classification failed', 'error');
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
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'toastSlideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==================== Utilities ====================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Escape to close modal
    if (e.key === 'Escape') {
        closeComposeModal();
    }
    
    // Ctrl+Enter to submit test message
    if (e.ctrlKey && e.key === 'Enter') {
        const modal = document.getElementById('compose-modal');
        if (modal.classList.contains('active')) {
            testMessage();
        }
    }
});
