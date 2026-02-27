const API_BASE = "http://localhost:8000";

// DOM Elements
const navBtns = document.querySelectorAll('.nav-btn');
const views = document.querySelectorAll('.view');
const pageTitle = document.getElementById('page-title');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultsContainer = document.getElementById('results-container');
const loadingOverlay = document.getElementById('loading-overlay');

// View Switching
navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Update active nav
        navBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update view
        const targetId = btn.getAttribute('data-target');
        views.forEach(v => v.classList.remove('active'));
        document.getElementById(targetId).classList.add('active');
        
        // Update title
        pageTitle.innerText = btn.innerText;

        if (targetId === 'dashboard-view') {
            fetchStats();
        }
    });
});

// Drag & Drop Handling
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
    let dt = e.dataTransfer;
    let files = dt.files;
    handleFiles(files);
});

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

// File Processing
async function handleFiles(files) {
    if (files.length === 0) return;
    
    // Check sizes
    for(let f of files) {
        if(f.size > 5 * 1024 * 1024) {
            alert(`File ${f.name} exceeds 5MB limit.`);
            return;
        }
    }

    loadingOverlay.classList.remove('hidden');
    resultsContainer.innerHTML = ''; // clear old
    resultsContainer.classList.remove('hidden');

    try {
        if (files.length === 1) {
            await handleSingleUpload(files[0]);
        } else {
            await handleBatchUpload(files);
        }
    } catch (e) {
        alert("Inference Error: " + e.message);
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

async function handleSingleUpload(file) {
    const fd = new FormData();
    fd.append("file", file);

    const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: fd
    });
    
    if (!res.ok) {
        throw new Error(await res.text());
    }
    const data = await res.json();
    renderResultCard(data);
}

async function handleBatchUpload(files) {
    // Only handling max 10 to ensure CPU stability per backend limit
    if (files.length > 10) {
        alert("Batch limit is strictly 10 files. Sending first 10.");
    }
    
    const fd = new FormData();
    const len = Math.min(files.length, 10);
    for(let i=0; i<len; i++) {
        fd.append("files", files[i]);
    }

    const res = await fetch(`${API_BASE}/predict-batch`, {
        method: 'POST',
        body: fd
    });
    
    if (!res.ok) {
        throw new Error(await res.text());
    }
    const data = await res.json();
    
    data.batch_results.forEach(item => {
        if(item.status === 'success') {
            renderResultCard(item.data);
        } else {
            console.error(item.error);
        }
    });
}

function renderResultCard(data) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    let tagsHTML = '';
    if(data.defects.length === 0) {
        tagsHTML = `<span class="defect-tag" style="background: rgba(16,185,129,0.1); color: var(--accent-green); border-color: rgba(16,185,129,0.2);">Pass (Clean)</span>`;
    } else {
        data.defects.forEach(d => {
            tagsHTML += `<span class="defect-tag">${d.type} (${(d.confidence*100).toFixed(1)}%)</span>`;
        });
    }

    card.innerHTML = `
        <div class="res-img-wrap">
            <img src="data:image/jpeg;base64,${data.image_base64}" alt="Result Image">
        </div>
        <div class="res-meta">
            <h4>${data.filename}</h4>
            <p style="font-size: 0.85rem; color: var(--text-secondary);">Inference: ${data.inference_time_ms}ms</p>
            <div class="res-defects">
                ${tagsHTML}
            </div>
        </div>
    `;
    resultsContainer.appendChild(card);
}

// Stats Dashboard
async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if(!res.ok) return;
        const data = await res.json();

        // Update summaries
        document.getElementById('total-inspections').innerText = data.total_predictions || 0;
        document.getElementById('avg-confidence').innerText = data.average_confidence 
            ? (data.average_confidence * 100).toFixed(1) + '%' 
            : '--%';
        document.getElementById('common-defect').innerText = data.most_common_defect || '--';

        // Update table
        const tbody = document.querySelector('#history-table tbody');
        tbody.innerHTML = '';
        if(data.recent_history) {
            data.recent_history.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${row.filename}</td>
                    <td><span class="defect-tag" style="display:inline-block">${row.defect_type}</span></td>
                    <td>${(row.confidence * 100).toFixed(1)}%</td>
                    <td>${row.timestamp}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch(e) {
        console.error("Stats fetching failed.", e);
    }
}

// Initial fetch on load
fetchStats();
