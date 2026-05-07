document.addEventListener("DOMContentLoaded", () => {
    const meetingsList = document.getElementById('meetings-list');
    const chatLog = document.getElementById('chat-log');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const currentMeetingTitle = document.getElementById('current-meeting-title');
    const meetingBadge = document.getElementById('meeting-badge');
    
    let activeMeetingId = null; // null means global mode
    let activeIngestions = {};
    let pollingInterval = null;

    // 1. Fetch meetings on load
    async function fetchMeetings() {
        try {
            const res = await fetch('/api/meetings');
            const data = await res.json();
            
            meetingsList.innerHTML = '';
            
            const allMeetings = [...data.meetings];
            
            // Add processing meetings that might not be in DB yet
            for (const [m_id, status] of Object.entries(activeIngestions)) {
                if (status === 'processing' && !allMeetings.find(m => m.meeting_id === m_id)) {
                    allMeetings.unshift({ meeting_id: m_id, processing: true });
                }
            }

            if (allMeetings.length === 0) {
                meetingsList.innerHTML = '<div class="sidebar-label">No meetings ingested yet.</div>';
                return;
            }

            allMeetings.forEach(mtg => {
                const btn = document.createElement('button');
                btn.className = 'meeting-item' + (activeMeetingId === mtg.meeting_id ? ' active' : '');
                
                let content = mtg.meeting_id;
                if (mtg.processing || activeIngestions[mtg.meeting_id] === 'processing') {
                    content += ' <span class="badge processing" style="float:right">Processing</span>';
                    btn.disabled = true;
                    btn.style.opacity = '0.7';
                }
                
                btn.innerHTML = content;
                btn.onclick = () => selectMeeting(mtg.meeting_id, btn);
                meetingsList.appendChild(btn);
            });
        } catch (err) {
            meetingsList.innerHTML = '<div style="color:red; font-size:12px;">Failed to load.</div>';
        }
    }

    // Background Poller
    async function checkStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            
            let changed = false;
            if (JSON.stringify(activeIngestions) !== JSON.stringify(data.active)) {
                activeIngestions = data.active;
                changed = true;
            }
            
            if (changed) fetchMeetings();
            
            // Stop polling if nothing is processing
            if (!Object.values(activeIngestions).includes('processing')) {
                clearInterval(pollingInterval);
                pollingInterval = null;
            }
        } catch (e) {}
    }
    
    // Start polling immediately
    checkStatus();
    if (!pollingInterval) pollingInterval = setInterval(checkStatus, 3000);

    // 2. Select a meeting mode
    function selectMeeting(meetingId, btnElement) {
        activeMeetingId = meetingId;
        
        // Update UI
        document.querySelectorAll('.meeting-item').forEach(b => b.classList.remove('active'));
        if (btnElement) btnElement.classList.add('active');
        
        currentMeetingTitle.textContent = meetingId ? `Analysis: ${meetingId}` : 'Global Knowledge Graph';
        meetingBadge.textContent = meetingId ? 'Deep Dive Mode' : 'Global Mode';
        meetingBadge.className = meetingId ? 'badge' : 'badge subtle';
        
        // Clear chat
        chatLog.innerHTML = `
            <div class="message ai greeting">
                <div class="avatar">AI</div>
                <div class="content">
                    Ready to analyze ${meetingId ? `<b>${meetingId}</b>` : 'all meetings'}. What would you like to know?
                </div>
            </div>
        `;
    }

    document.getElementById('new-chat-btn').onclick = () => selectMeeting(null, null);

    // 3. Handle sending requests
    async function sendMessage() {
        const query = promptInput.value.trim();
        if (!query) return;

        // Add User Message
        appendMessage('user', query);
        promptInput.value = '';
        sendBtn.disabled = true;

        // Create AI Message Container with cursor
        const aiMessageDiv = appendMessage('ai', '<span class="cursor"></span>');
        const contentDiv = aiMessageDiv.querySelector('.content');

        try {
            // Unify Global and Deep Dive modes to single-pass fetch to bypass Windows SSE issues
            contentDiv.innerHTML = activeMeetingId ? '<i>Analyzing transcript...</i>' : '<i>Searching memories...</i>';
            
            const reqBody = { query, stream: false };
            if (activeMeetingId) reqBody.meeting_id = activeMeetingId;

            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });
            
            const data = await res.json();
            
            if (data.error) {
                let errorMsg = data.error;
                // Mask API exhaustion/quota errors as "system overloaded" per user request
                if (errorMsg.toLowerCase().includes('exhausted') || errorMsg.toLowerCase().includes('quota') || errorMsg.toLowerCase().includes('429')) {
                    errorMsg = "⚠️ The system is currently overloaded. Please try again later.";
                }
                contentDiv.innerHTML = `<span style="color:red">${errorMsg}</span>`;
            } else {
                let text = marked.parse(data.answer);
                if (data.sources && data.sources.length) {
                    // Extract unique meeting IDs from chunk IDs (e.g. "25-5146_c10" -> "25-5146")
                    const meetingIds = [...new Set(data.sources.map(s => s.replace(/_c\d+$/, '')))];
                    text += `<br><br><small style="color:var(--text-muted)">📎 Based on ${data.sources.length} transcript excerpts from: ${meetingIds.join(', ')}</small>`;
                }
                contentDiv.innerHTML = text;
            }

        } catch (err) {
            contentDiv.innerHTML = `<span style="color:red">Connection error: ${err.message}</span>`;
        }

        sendBtn.disabled = false;
        promptInput.focus();
    }

    function appendMessage(role, htmlContent) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        
        let avatarSvg = role === 'user' 
            ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
            : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20"></path><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>';
            
        div.innerHTML = `
            <div class="avatar">${avatarSvg}</div>
            <div class="content">${htmlContent}</div>
        `;
        chatLog.appendChild(div);
        chatLog.scrollTop = chatLog.scrollHeight;
        return div;
    }

    // Event Listeners
    sendBtn.onclick = sendMessage;
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Audio Upload Integration
    const uploadBtn = document.getElementById('upload-btn');
    const audioUpload = document.getElementById('audio-upload');
    
    if (uploadBtn && audioUpload) {
        uploadBtn.onclick = () => audioUpload.click();
        
        audioUpload.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append("file", file);
            
            const originalText = uploadBtn.innerHTML;
            uploadBtn.innerHTML = "Uploading...";
            uploadBtn.disabled = true;
            
            try {
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                
                if (data.meeting_id) {
                    activeIngestions[data.meeting_id] = 'processing';
                    fetchMeetings();
                    if (!pollingInterval) pollingInterval = setInterval(checkStatus, 3000);
                }
            } catch (err) {
                alert("Upload failed: " + err.message);
            }
            
            uploadBtn.innerHTML = originalText;
            uploadBtn.disabled = false;
            audioUpload.value = '';
        };
    }

    // Auto-resize textarea
    promptInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value === '') this.style.height = 'auto';
    });

    // Boot
    fetchMeetings();
});
