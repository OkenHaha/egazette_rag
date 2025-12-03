
let conversationHistory = [];
let currentFilters = {
    start_date: null,
    end_date: null
};

async function applyDateFilter() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    if (!startDate || !endDate) {
        alert('Please select both start and end dates');
        return;
    }
    
    try {
        const response = await fetch('/apply-date-filter', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                start_date: startDate,
                end_date: endDate
            })
        });
        
        const data = await response.json();
        currentFilters.start_date = startDate;
        currentFilters.end_date = endDate;
        
        const filterInfo = document.getElementById('filterInfo');
        filterInfo.innerHTML = `
            <div class="filter-info">
                ✅ Filter Applied: ${data.total_parent_ids} documents, ${data.total_chunks} chunks
            </div>
        `;
        
        addMessage(`📅 Applied date filter: ${startDate} to ${endDate} (${data.total_parent_ids} documents)`, 'assistant');
    } catch (error) {
        console.error('Error:', error);
        alert('Error applying date filter');
    }
}

function clearDateFilter() {
    document.getElementById('startDate').value = '';
    document.getElementById('endDate').value = '';
    document.getElementById('filterInfo').innerHTML = '';
    currentFilters.start_date = null;
    currentFilters.end_date = null;
    addMessage('🔄 Date filter cleared', 'assistant');
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    addMessage(message, 'user');
    input.value = '';
    
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    input.disabled = true;
    
    const typingId = addTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: message,
                conversation_history: conversationHistory,
                start_date: currentFilters.start_date,
                end_date: currentFilters.end_date
            })
        
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        removeTypingIndicator(typingId);
        addMessage(data.response, 'assistant', data.main_sources, data.other_sources, data.filter_info);
        
        conversationHistory.push(
            { role: 'user', content: message },
            { role: 'assistant', content: data.response }
        );
        
        if (conversationHistory.length > 10) {
            conversationHistory = conversationHistory.slice(-10);
        }
        
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage('Sorry, there was an error processing your request. Please try again.', 'assistant');
        console.error('Error:', error);
    } finally {
        sendButton.disabled = false;
        input.disabled = false;
        input.focus();
    }
}

function toggleSourceContent(sourceId) {
    const contentDiv = document.getElementById(`content-${sourceId}`);
    const toggleLink = document.getElementById(`toggle-${sourceId}`);
    
    if (contentDiv.style.display === 'none') {
        contentDiv.style.display = 'block';
        toggleLink.textContent = '▼ Hide full content';
    } else {
        contentDiv.style.display = 'none';
        toggleLink.textContent = '▶ Show full content';
    }
}

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function addMessage(text, sender, mainSources = null, otherSources = null, filterInfo = null) {
    const container = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    const messageId = Date.now();
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    //contentDiv.textContent = text;
    const rawHtml = marked.parse(text);
    // 2. Sanitize HTML (Security best practice to prevent XSS attacks)
    const cleanHtml = DOMPurify.sanitize(rawHtml);
    // 3. Set HTML
    contentDiv.innerHTML = cleanHtml;
    
    messageDiv.appendChild(contentDiv);
    
    // Add filter info if present
    if (filterInfo && filterInfo.applied) {
        const filterDiv = document.createElement('div');
        filterDiv.className = 'sources-section';
        filterDiv.innerHTML = `
            <div class="sources" style="background: #dbeafe; border-left-color: #0284c7;">
                <div class="sources-title" style="color: #0284c7;">
                    📅 Applied Date Filter
                    <span class="source-count" style="background: #0284c7">${filterInfo.filtered_documents} docs</span>
                </div>
                <div style="padding: 10px; color: #0369a1; font-size: 12px;">
                    <strong>Range:</strong> ${filterInfo.start_date} to ${filterInfo.end_date}<br>
                    <strong>Documents in range:</strong> ${filterInfo.filtered_documents}
                </div>
            </div>
        `;
        contentDiv.appendChild(filterDiv);
    }
    
    // Add main sources
    if (mainSources && mainSources.length > 0) {
        const sourcesSection = document.createElement('div');
        sourcesSection.className = 'sources-section';
        
        const mainSourcesDiv = document.createElement('div');
        mainSourcesDiv.className = 'sources';
        mainSourcesDiv.innerHTML = `
            <div class="sources-title">
                📚 Main Sources (Used in Answer)
                <span class="source-count">${mainSources.length}</span>
            </div>
        `;
        
        mainSources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            const sourceId = `${messageId}-${index}`;
            
            const pdfFileName = source.source.replace('.md', '.pdf');
            const pdfPath = `http://10.10.1.117:8008/${pdfFileName}`;
            
         
            
            let metadataHtml = '';
            if (source.metadata) {
                metadataHtml = `
                    <div>
                        <span class="source-date">📅 Published: ${formatDate(source.metadata.publicationdate)}</span>
                        
                    </div>
                `;
            }
            
            sourceItem.innerHTML = `
                <div class="source-meta">Serial: ${source.parent_id} | Chunk ${source.chunk_index}/${source.total_chunks}</div>
                <div class="source-link">
                    <a href="${pdfPath}" target="_blank">📄 View PDF: ${pdfFileName}</a>
                </div>
                <div class="toggle-content" id="toggle-${sourceId}" onclick="toggleSourceContent('${sourceId}')">
                    ▶ Show full content
                </div>
                <div class="source-content" id="content-${sourceId}" style="display: none;">
                    ${source.content}
                </div>
            `;
            mainSourcesDiv.appendChild(sourceItem);
        });
        
        sourcesSection.appendChild(mainSourcesDiv);
        contentDiv.appendChild(sourcesSection);
    }
    
    // Add other sources (collapsible)
    if (otherSources && otherSources.length > 0) {
        const otherSourcesDiv = document.createElement('div');
        otherSourcesDiv.className = 'sources';
        otherSourcesDiv.style.background = '#fef3c7';
        otherSourcesDiv.style.borderLeftColor = '#f59e0b';
        
        const headerHtml = `
            <div class="sources-title" style="color: #f59e0b;">
                🔖 Other Relevant Sources
                <span class="source-count" style="background: #f59e0b">${otherSources.length}</span>
            </div>
        `;
        otherSourcesDiv.innerHTML = headerHtml;
        
        otherSources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const pdfFileName = source.source.replace('.md', '.pdf');
            const pdfPath = `http://10.10.1.117:8008/${pdfFileName}`;
        
            
            let metadataHtml = '';
            if (source.metadata) {
                metadataHtml = `
                    <div style="margin-top: 5px;">
                        <span class="source-date">📅 ${formatDate(source.metadata.publicationdate)}</span>
                    </div>
                `;
            }
            
            sourceItem.innerHTML = `
                <div class="source-name">${index + 1}. ${source.source}</div>
                <div class="source-meta">Serial: ${source.parent_id} | Chunk ${source.chunk_index}/${source.total_chunks}</div>
                <div class="source-link">
                    <a href="${pdfPath}" target="_blank">📄 View PDF</a>
                </div>
                <div style="margin-top: 8px; color: #888; font-size: 12px; font-style: italic;">
                    ${source.content}
                </div>
            `;
            otherSourcesDiv.appendChild(sourceItem);
        });
        
        const sourcesSection = document.createElement('div');
        sourcesSection.className = 'sources-section';
        sourcesSection.appendChild(otherSourcesDiv);
        contentDiv.appendChild(sourcesSection);
    }
    
    messageDiv.appendChild(contentDiv);
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
}

function addTypingIndicator() {
    const container = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'typing-indicator-' + Date.now();
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    
    messageDiv.appendChild(typingDiv);
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    return messageDiv.id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

// Initialize date inputs with today's date and 30 days ago
window.addEventListener('load', function() {
    const today = new Date();
    const thirtyDaysAgo = new Date(today.getTime() - (30 * 24 * 60 * 60 * 1000));
    
    document.getElementById('startDate').valueAsDate = thirtyDaysAgo;
    document.getElementById('endDate').valueAsDate = today;
});