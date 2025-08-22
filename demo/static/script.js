// Global state
let currentFolder = null;
let currentFile = null;
let currentStep = null;
let currentEpisodeData = null;
let allEpisodes = [];
let playbackState = {
    playing: false,
    currentStep: 0,
    speed: 1,
    totalReward: 0,
    intervalId: null,
    isDragging: false,
    pendingTimeouts: []  // Track all pending timeouts
};

// Loading overlay functions
function showLoadingOverlay(text = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');
    loadingText.textContent = text;
    overlay.classList.add('active');
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.remove('active');
}

// Create inline loading HTML with spinner
function createLoadingHTML(text = 'Loading...') {
    return `
        <div class="loading">
            <div class="spinner"></div>
            <div>${text}</div>
        </div>
    `;
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    loadFileList();
});

// Format training name from folder name with optional metadata
function formatTrainingName(folderName, metadata = null) {
    // Expected format: training_tag_YYYYMMDD_HHMMSS or training_YYYYMMDD_HHMMSS
    const parts = folderName.split('_');
    
    if (parts[0] !== 'training') {
        return folderName; // Return as-is if not in expected format
    }
    
    let tag = '';
    let dateStr = '';
    let timeStr = '';
    
    if (parts.length === 4) {
        // Format: training_tag_YYYYMMDD_HHMMSS
        tag = parts[1];
        dateStr = parts[2];
        timeStr = parts[3];
    } else if (parts.length === 3) {
        // Format: training_YYYYMMDD_HHMMSS
        dateStr = parts[1];
        timeStr = parts[2];
    } else {
        return folderName; // Return as-is if format doesn't match
    }
    
    // Parse date and time
    if (dateStr.length === 8 && timeStr.length === 6) {
        const year = dateStr.substring(0, 4);
        const month = dateStr.substring(4, 6);
        const day = dateStr.substring(6, 8);
        const hour = timeStr.substring(0, 2);
        const minute = timeStr.substring(2, 4);
        const second = timeStr.substring(4, 6);
        
        // Create a date object
        const date = new Date(
            parseInt(year),
            parseInt(month) - 1, // Month is 0-indexed
            parseInt(day),
            parseInt(hour),
            parseInt(minute),
            parseInt(second)
        );
        
        // Format the date nicely
        const options = {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        const formattedDate = date.toLocaleString('en-US', options);
        
        // Get observation mode if metadata is provided
        let modeText = '';
        if (metadata && metadata.config && metadata.config.mode) {
            const mode = metadata.config.mode;
            if (mode === 'partial') {
                modeText = ' with Partial Observations';
            } else if (mode === 'full') {
                modeText = ' with Full Observations';
            } else {
                modeText = ` in ${mode} mode`;
            }
        }
        
        if (tag) {
            return `Fine-tuning "${tag}"${modeText} at ${formattedDate}`;
        } else {
            return `Fine-tuning${modeText} at ${formattedDate}`;
        }
    }
    
    return folderName; // Return as-is if date/time parsing fails
}

// Load and display available trainings
async function loadFileList() {
    const fileListEl = document.getElementById('fileList');
    fileListEl.innerHTML = createLoadingHTML('Loading training files...');
    
    try {
        const response = await fetch('/api/trainings');
        const files = await response.json();
        
        if (files.length === 0) {
            fileListEl.innerHTML = '<div class="loading">No training files found</div>';
            return;
        }
        
        fileListEl.innerHTML = files.map(file => {
            // Create a simple metadata object with the mode
            const metadata = { config: { mode: file.mode } };
            const displayName = formatTrainingName(file.folder, metadata);
            const episodeInfo = `${file.episode_count} episodes | ${file.sample_steps} steps`;
            return `
                <div class="file-item" onclick="selectFolder('${file.folder}')">
                    <div class="file-name">${displayName}</div>
                    <div class="file-meta">
                        Size: ${file.size} | 
                        ${episodeInfo} | 
                        Last updated: ${new Date(file.modified * 1000).toLocaleString()}
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading files:', error);
        fileListEl.innerHTML = `<div class="loading">Error loading files: ${error.message}</div>`;
    }
}

// Select a folder and show sample steps
async function selectFolder(folderName) {
    currentFolder = folderName;
    showLoadingOverlay('Loading sample steps...');
    
    try {
        const response = await fetch(`/api/sample_steps/${folderName}`);
        const data = await response.json();
        
        if (!data.success) {
            hideLoadingOverlay();
            alert('Error loading sample steps: ' + data.error);
            return;
        }
        
        // Update UI with metadata
        const formattedName = formatTrainingName(folderName, data.training_metadata);
        document.getElementById('selectedFolderName').textContent = formattedName;
        
        // Display sample steps
        const stepListEl = document.getElementById('sampleStepList');
        
        if (data.sample_steps.length === 0) {
            stepListEl.innerHTML = '<div class="loading">No sample steps found</div>';
            hideLoadingOverlay();
            return;
        }
        
        // Sort sample steps by success rate (highest first)
        const sortedSteps = [...data.sample_steps].sort((a, b) => {
            const successRateA = a.total_episodes > 0 
                ? (a.terminal_successes / a.total_episodes) 
                : 0;
            const successRateB = b.total_episodes > 0 
                ? (b.terminal_successes / b.total_episodes) 
                : 0;
            return successRateB - successRateA; // Descending order
        });
        
        stepListEl.innerHTML = sortedSteps.map(step => {
            const successRate = step.total_episodes > 0 
                ? ((step.terminal_successes / step.total_episodes) * 100).toFixed(1)
                : '0.0';
            
            return `
                <div class="sample-step-item" onclick="selectSampleStep(${step.step})">
                    <div class="sample-step-number">Step ${step.step.toLocaleString()}</div>
                    <div class="sample-step-stats">
                        <div class="step-stat">
                            <span class="step-stat-label">Terminal Successes:</span>
                            <span class="step-stat-value success">${step.terminal_successes}</span>
                        </div>
                        <div class="step-stat">
                            <span class="step-stat-label">Total Episodes:</span>
                            <span class="step-stat-value">${step.total_episodes}</span>
                        </div>
                        <div class="step-stat">
                            <span class="step-stat-label">Success Rate:</span>
                            <span class="step-stat-value">${successRate}%</span>
                        </div>
                        <div class="step-stat">
                            <span class="step-stat-label">Total Samples:</span>
                            <span class="step-stat-value">${step.total_samples.toLocaleString()}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        hideLoadingOverlay();
        
        // Switch screens
        switchScreen('sampleStepScreen');
    } catch (error) {
        console.error('Error loading sample steps:', error);
        hideLoadingOverlay();
        alert('Error loading sample steps: ' + error.message);
    }
}

// Select a sample step and load episodes
async function selectSampleStep(stepNum) {
    currentStep = stepNum;
    showLoadingOverlay(`Loading episodes for step ${stepNum}...`);
    
    try {
        const response = await fetch(`/api/episodes/${currentFolder}/${stepNum}`);
        const data = await response.json();
        
        if (!data.success) {
            hideLoadingOverlay();
            alert('Error loading episodes: ' + data.error);
            return;
        }
        
        allEpisodes = data.episodes;
        
        // Update UI - We need to get metadata for the mode info
        // We'll make a simple request to get it
        const metadataResponse = await fetch(`/api/sample_steps/${currentFolder}`);
        const metadataData = await metadataResponse.json();
        const formattedName = formatTrainingName(currentFolder, metadataData.training_metadata);
        document.getElementById('selectedFileName').textContent = formattedName;
        document.getElementById('selectedStepNum').textContent = stepNum.toLocaleString();
        
        // Display statistics
        const statsEl = document.getElementById('statistics');
        statsEl.innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${data.statistics.total_episodes}</div>
                <div class="stat-label">Total Episodes</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${data.statistics.successful_episodes}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${data.statistics.terminal_accuracy}</div>
                <div class="stat-label">Terminal Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${data.statistics.total_samples}</div>
                <div class="stat-label">Total Samples</div>
            </div>
        `;
        
        // Display episodes
        displayEpisodes('all');
        
        hideLoadingOverlay();
        
        // Switch screens
        switchScreen('episodeSelectionScreen');
    } catch (error) {
        console.error('Error loading episodes:', error);
        hideLoadingOverlay();
        alert('Error loading episodes: ' + error.message);
    }
}

// Display episodes in grid
function displayEpisodes(filter, buttonElement) {
    const gridEl = document.getElementById('episodeGrid');
    gridEl.innerHTML = createLoadingHTML('Filtering episodes...');
    
    // Small delay to show loading state
    setTimeout(() => {
        let filteredEpisodes = allEpisodes;
        if (filter === 'success') {
            filteredEpisodes = allEpisodes.filter(e => e.success);
        } else if (filter === 'failed') {
            filteredEpisodes = allEpisodes.filter(e => !e.success);
        }
        
        if (filteredEpisodes.length === 0) {
            gridEl.innerHTML = '<div class="loading">No episodes match the filter</div>';
            return;
        }
    
        gridEl.innerHTML = filteredEpisodes.map(episode => `
            <div class="episode-card ${episode.success ? 'success' : 'failed'}" 
                 onclick="playEpisode(${episode.process_id}, ${episode.episode_id})">
                <div class="episode-number">P${episode.process_id} E${episode.episode_id}</div>
                <div class="episode-meta">
                    ${episode.total_samples} samples<br>
                    Reward: ${episode.total_reward.toFixed(2)}<br>
                    ${episode.success ? '✓ Success' : '✗ Failed'}
                </div>
            </div>
        `).join('');
        
        // Update filter buttons if button element provided
        if (buttonElement) {
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            buttonElement.classList.add('active');
        }
    }, 100); // Small delay to show loading state
}

// Filter episodes
function filterEpisodes(filter) {
    displayEpisodes(filter, event.target);
}

// Play a specific episode
async function playEpisode(processId, episodeId) {
    showLoadingOverlay(`Loading episode P${processId} E${episodeId}...`);
    
    try {
        const response = await fetch(`/api/episode/${currentFolder}/${processId}/${episodeId}`);
        const data = await response.json();
        
        if (!data.success) {
            hideLoadingOverlay();
            alert('Error loading episode: ' + data.error);
            return;
        }
        
        currentEpisodeData = data.data;
        
        // Update metadata display
        if (currentEpisodeData && currentEpisodeData.length > 0) {
            const firstStep = currentEpisodeData[0];
            const userGoal = firstStep.user_goal || 'Goal not specified';
            const callTreeId = firstStep.call_tree_id || 'Call tree not specified';
            
            document.getElementById('userGoal').textContent = userGoal;
            // Format the call tree ID nicely
            document.getElementById('callTreeId').textContent = callTreeId.startsWith('tree_') 
                ? callTreeId.replace('tree_', 'Call Tree #').replace(/_/g, ' - ')
                : callTreeId;
        } else {
            document.getElementById('userGoal').textContent = 'Goal information not available';
            document.getElementById('callTreeId').textContent = 'Call tree information not available';
        }
        
        // Reset playback state
        playbackState = {
            playing: true,
            currentStep: 0,
            speed: 1,
            totalReward: 0,
            intervalId: null,
            isDragging: false,
            pendingTimeouts: []
        };
        
        hideLoadingOverlay();
        
        // Switch to playback screen
        switchScreen('playbackScreen');
        
        // Clear display track
        document.getElementById('displayTrack').innerHTML = '';
        
        // Start playback
        startPlayback();
    } catch (error) {
        console.error('Error loading episode:', error);
        hideLoadingOverlay();
        alert('Error loading episode: ' + error.message);
    }
}

// Start playback
function startPlayback() {
    if (!currentEpisodeData || currentEpisodeData.length === 0) return;
    
    playbackState.playing = true;
    updatePlayPauseButton();
    
    // Initialize progress bar dragging
    initProgressBarDragging();
    
    // Start the playback loop
    playNextStep();
}

// Track if progress bar dragging is initialized
let progressBarInitialized = false;

// Initialize progress bar dragging
function initProgressBarDragging() {
    // Only initialize once
    if (progressBarInitialized) return;
    progressBarInitialized = true;
    
    const progressBar = document.getElementById('progressBar');
    const progressHandle = document.getElementById('progressHandle');
    
    // Click on progress bar to seek
    progressBar.addEventListener('click', (e) => {
        if (!playbackState.isDragging) {
            const rect = progressBar.getBoundingClientRect();
            const percent = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
            // Clear any pending timeouts before seeking
            clearPendingTimeouts();
            seekToPosition(percent, true);  // true = resume playback after seeking
        }
    });
    
    // Drag handle
    progressHandle.addEventListener('mousedown', (e) => {
        playbackState.isDragging = true;
        playbackState.playing = false;
        clearPendingTimeouts();  // Clear timeouts when starting drag
        updatePlayPauseButton();
        progressHandle.classList.add('dragging');
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (playbackState.isDragging) {
            const progressBar = document.getElementById('progressBar');
            const rect = progressBar.getBoundingClientRect();
            const percent = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
            progressHandle.style.left = `${percent}%`;
            document.getElementById('progressFill').style.width = `${percent}%`;
            
            // Update step number display (1-indexed)
            const targetStep = Math.floor((percent / 100) * (currentEpisodeData.length - 1));
            document.getElementById('progressText').textContent = 
                `${Math.min(targetStep + 1, currentEpisodeData.length)} / ${currentEpisodeData.length}`;
        }
    });
    
    document.addEventListener('mouseup', (e) => {
        if (playbackState.isDragging) {
            playbackState.isDragging = false;
            progressHandle.classList.remove('dragging');
            
            const progressBar = document.getElementById('progressBar');
            const rect = progressBar.getBoundingClientRect();
            const percent = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
            seekToPosition(percent, false);  // false = don't auto-resume after drag
        }
    });
}

// Seek to a specific position in the episode
function seekToPosition(percent, resumePlayback = false) {
    const targetStep = Math.floor((percent / 100) * (currentEpisodeData.length - 1));
    playbackState.currentStep = targetStep;
    
    // Rebuild display up to this point
    document.getElementById('displayTrack').innerHTML = '';
    document.getElementById('rewardAnimations').innerHTML = '';
    
    // Recalculate total reward up to this point
    playbackState.totalReward = 0;
    for (let i = 0; i < targetStep; i++) {
        if (currentEpisodeData[i].reward) {
            playbackState.totalReward += currentEpisodeData[i].reward;
        }
    }
    
    // Show last few observations/actions
    const startIdx = Math.max(0, targetStep - 3);
    for (let i = startIdx; i < targetStep; i++) {
        const step = currentEpisodeData[i];
        if (step.observation_text) {
            addContentBox('observation', 'Observation', step.observation_text, step);
        }
        // Remove > prefix from action text if present
        const actionText = step.action.startsWith('>') ? step.action.substring(1).trim() : step.action;
        // Determine action box class based on type and reward
        let actionClass = 'action-wait';
        if (step.action_type === 'press' || step.action_type !== 'wait') {
            actionClass = step.reward >= 0 ? 'action-press-positive' : 'action-press-negative';
        }
        addContentBox(
            actionClass,
            'Action',
            actionText,
            step
        );
    }
    
    // Update UI
    updateProgress();
    document.querySelector('.reward-value').textContent = playbackState.totalReward.toFixed(2);
    
    // Handle resuming playback based on parameter
    if (resumePlayback && targetStep < currentEpisodeData.length) {
        playbackState.playing = true;
        updatePlayPauseButton();
        playNextStep();
    }
}

// Play next step in the episode
function playNextStep() {
    // Clear any pending timeouts first
    clearPendingTimeouts();
    
    if (!playbackState.playing || playbackState.currentStep >= currentEpisodeData.length) {
        if (playbackState.currentStep >= currentEpisodeData.length) {
            // Episode complete
            playbackState.playing = false;
            updatePlayPauseButton();
        }
        return;
    }
    
    const step = currentEpisodeData[playbackState.currentStep];
    
    // Add content box for observation
    if (step.observation_text) {
        addContentBox('observation', 'Observation', step.observation_text, step);
    }
    
    // Add content box for action
    const actionTimeout = setTimeout(() => {
        if (!playbackState.playing) return;  // Check if still playing
        
        // Remove > prefix from action text if present
        const actionText = step.action.startsWith('>') ? step.action.substring(1).trim() : step.action;
        // Determine action box class based on type and reward
        let actionClass = 'action-wait';
        if (step.action_type === 'press' || step.action_type !== 'wait') {
            actionClass = step.reward >= 0 ? 'action-press-positive' : 'action-press-negative';
        }
        addContentBox(
            actionClass,
            'Action',
            actionText,
            step
        );
        
        // Update reward
        if (step.reward !== 0) {
            updateReward(step.reward);
        }
        
        // Update progress
        updateProgress();
        
        // Move to next step
        playbackState.currentStep++;
        
        // Schedule next step
        if (playbackState.playing) {
            const delay = 2000 / playbackState.speed;  // Slower base speed
            const nextTimeout = setTimeout(() => playNextStep(), delay);
            playbackState.pendingTimeouts.push(nextTimeout);
        }
    }, 750 / playbackState.speed);  // Slower base speed
    
    playbackState.pendingTimeouts.push(actionTimeout);
}

// Add a content box to the display
function addContentBox(type, label, text, stepData = null) {
    const track = document.getElementById('displayTrack');
    
    const box = document.createElement('div');
    box.className = `content-box ${type}`;
    
    // Create tooltip content if step data is available
    let tooltipContent = '';
    if (stepData) {
        tooltipContent = `
            <div class="tooltip-content">
                ${stepData.advantage !== undefined && stepData.advantage !== null ? `<div class="tooltip-row"><strong>Advantage:</strong> ${stepData.advantage.toFixed(4)}</div>` : ''}
                ${stepData.value !== undefined && stepData.value !== null ? `<div class="tooltip-row"><strong>Value:</strong> ${stepData.value.toFixed(4)}</div>` : ''}
                ${stepData.returnn !== undefined && stepData.returnn !== null ? `<div class="tooltip-row"><strong>Return:</strong> ${stepData.returnn.toFixed(4)}</div>` : ''}
                ${stepData.reward !== undefined && stepData.reward !== null ? `<div class="tooltip-row"><strong>Reward:</strong> ${stepData.reward.toFixed(4)}</div>` : ''}
                ${stepData.action_type ? `<div class="tooltip-row"><strong>Action Type:</strong> ${stepData.action_type}</div>` : ''}
            </div>
        `;
    }
    
    box.innerHTML = `
        <div class="content-type">${label}</div>
        <div class="content-text">${text}</div>
        ${tooltipContent}
    `;
    
    track.appendChild(box);
    
    // Check if mobile view (viewport width <= 768px)
    const isMobile = window.innerWidth <= 768;
    const boxes = track.querySelectorAll('.content-box');
    
    if (isMobile) {
        // Mobile behavior: Keep only the last 2 boxes visible, remove old ones
        const maxVisibleBoxes = 2;
        
        // Remove boxes that are too old
        if (boxes.length > maxVisibleBoxes) {
            for (let i = 0; i < boxes.length - maxVisibleBoxes; i++) {
                boxes[i].remove();
            }
        }
        
        // No translation on mobile - keep content centered
        track.style.transform = 'translateX(0)';
    } else {
        // Desktop behavior: Original sliding effect
        if (boxes.length > 3) {
            // Start fading older boxes
            for (let i = 0; i < boxes.length - 3; i++) {
                boxes[i].classList.add('fading');
            }
        }
        if (boxes.length > 5) {
            // Hide very old boxes
            for (let i = 0; i < boxes.length - 5; i++) {
                boxes[i].classList.add('hidden');
            }
        }
        
        // Slide the track to center the new content
        const offset = -(boxes.length - 1) * 320;
        track.style.transform = `translateX(${offset}px)`;
    }
}

// Update reward display
function updateReward(reward) {
    playbackState.totalReward += reward;
    
    const rewardEl = document.querySelector('.reward-value');
    rewardEl.textContent = playbackState.totalReward.toFixed(2);
    
    // Add color and animation
    rewardEl.classList.remove('positive', 'negative', 'reward-flash');
    if (playbackState.totalReward > 0) {
        rewardEl.classList.add('positive');
    } else if (playbackState.totalReward < 0) {
        rewardEl.classList.add('negative');
    }
    
    if (reward !== 0) {
        rewardEl.classList.add('reward-flash');
        setTimeout(() => rewardEl.classList.remove('reward-flash'), 500);
        
        // Create reward bubble animation
        createRewardBubble(reward);
    }
}

// Create animated reward bubble
function createRewardBubble(reward) {
    const container = document.getElementById('rewardAnimations');
    const bubble = document.createElement('div');
    bubble.className = `reward-bubble ${reward > 0 ? 'positive' : 'negative'}`;
    bubble.textContent = `${reward > 0 ? '+' : ''}${reward.toFixed(2)}`;
    
    // Position the bubble near the current content box
    const boxes = document.querySelectorAll('.content-box');
    if (boxes.length > 0) {
        const lastBox = boxes[boxes.length - 1];
        const rect = lastBox.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        bubble.style.left = `${rect.left - containerRect.left + rect.width / 2}px`;
    } else {
        bubble.style.left = '50%';
    }
    
    container.appendChild(bubble);
    
    // Remove after animation
    setTimeout(() => bubble.remove(), 2000);
}

// Update progress bar
function updateProgress() {
    const progress = Math.min((playbackState.currentStep / (currentEpisodeData.length - 1)) * 100, 100);
    document.getElementById('progressFill').style.width = `${progress}%`;
    document.getElementById('progressHandle').style.left = `${progress}%`;
    // Display 1-indexed current step and total steps (not 0-indexed)
    document.getElementById('progressText').textContent = 
        `${Math.min(playbackState.currentStep + 1, currentEpisodeData.length)} / ${currentEpisodeData.length}`;
}

// Toggle play/pause
function togglePlayPause() {
    playbackState.playing = !playbackState.playing;
    updatePlayPauseButton();
    
    if (playbackState.playing) {
        playNextStep();
    }
}

// Update play/pause button
function updatePlayPauseButton() {
    const btn = document.getElementById('playPauseBtn');
    if (playbackState.playing) {
        btn.innerHTML = '<span class="icon">⏸</span> Pause';
    } else {
        btn.innerHTML = '<span class="icon">▶</span> Play';
    }
}

// Clear all pending timeouts
function clearPendingTimeouts() {
    playbackState.pendingTimeouts.forEach(timeout => clearTimeout(timeout));
    playbackState.pendingTimeouts = [];
}

// Rewind playback to beginning
function rewindPlayback() {
    // Clear all pending timeouts first
    clearPendingTimeouts();
    
    // Stop playback
    playbackState.playing = false;
    playbackState.currentStep = 0;
    playbackState.totalReward = 0;
    
    // Clear display
    document.getElementById('displayTrack').innerHTML = '';
    document.getElementById('rewardAnimations').innerHTML = '';
    
    // Reset UI
    updateProgress();
    document.querySelector('.reward-value').textContent = '0.00';
    document.querySelector('.reward-value').classList.remove('positive', 'negative');
    updatePlayPauseButton();
    
    // Small delay before restarting to ensure clean state
    setTimeout(() => {
        // Start playing from beginning
        playbackState.playing = true;
        updatePlayPauseButton();
        playNextStep();
    }, 100);
}

// Stop playback and return to episode selection
function stopPlayback() {
    playbackState.playing = false;
    if (playbackState.intervalId) {
        clearInterval(playbackState.intervalId);
    }
    switchScreen('episodeSelectionScreen');
}

// Show file selection screen
function showFileSelection() {
    switchScreen('fileSelectionScreen');
}

// Show sample steps screen
function showSampleSteps() {
    if (currentFolder) {
        selectFolder(currentFolder);
    } else {
        switchScreen('sampleStepScreen');
    }
}

// Switch between screens
function switchScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    document.getElementById(screenId).classList.add('active');
}
