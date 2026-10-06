/**
 * Disease Detection Interface
 * Handles file upload and disease prediction visualization
 */

let selectedFile = null;
let probabilityChart = null;
let featureChart = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    const fileInput = document.getElementById('file-input');
    const uploadZone = document.getElementById('upload-zone');
    const detectBtn = document.getElementById('detect-btn');
    const demoBtn = document.getElementById('demo-btn');

    // File input
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '#6366f1';
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';

        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect({ target: fileInput });
        }
    });

    // Buttons
    detectBtn.addEventListener('click', runDetection);
    demoBtn.addEventListener('click', runDemoDetection);
}

function handleFileSelect(event) {
    const file = event.target.files[0];

    if (file) {
        selectedFile = file;

        // Update file info
        document.getElementById('file-info').style.display = 'block';
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-size').textContent = formatFileSize(file.size);

        // Enable detect button
        document.getElementById('detect-btn').disabled = false;
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

async function runDetection() {
    if (!selectedFile) return;

    showLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            displayResults(result.prediction, result.feature_importance);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Detection error:', error);
        alert('Failed to perform detection. Please try again.');
    } finally {
        showLoading(false);
    }
}

async function runDemoDetection() {
    showLoading(true);
    updateProgress('Generating synthetic data...');

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ use_demo: true })
        });

        const result = await response.json();

        if (result.success) {
            displayResults(result.prediction, result.feature_importance);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Demo detection error:', error);
        alert('Failed to run demo. Please try again.');
    } finally {
        showLoading(false);
    }
}

function displayResults(prediction, featureImportance) {
    // Hide placeholder, show results
    document.getElementById('results-placeholder').style.display = 'none';
    document.getElementById('results-content').style.display = 'block';

    // Display predicted disease
    const technicalName = prediction.predicted_disease;
    const diseaseMap = {
        'Normal': 'Healthy Control',
        'Alzheimer_Mild': 'Alzheimer\'s (Mild)',
        'Alzheimer_Moderate': 'Alzheimer\'s (Moderate)',
        'Alzheimer_VeryMild': 'Alzheimer\'s (Very Mild)',
        'Glioma': 'Brain Tumor (Glioma)',
        'Meningioma': 'Brain Tumor (Meningioma)',
        'Pituitary': 'Brain Tumor (Pituitary)',
        'MS': 'Multiple Sclerosis',
        'Parkinson': 'Parkinson\'s Disease',
        'Tumor': 'Brain Tumor'
    };

    const displayTitle = diseaseMap[technicalName] || technicalName;
    const confidence = prediction.confidence;

    document.getElementById('disease-name').textContent = displayTitle;
    document.getElementById('confidence-value').textContent = (confidence * 100).toFixed(1) + '%';

    const tumorClasses = ['Glioma', 'Meningioma', 'Pituitary', 'Tumor'];
    const isTumor = tumorClasses.includes(technicalName) && prediction.threshold_met;

    // Color code based on severity / threshold met
    const badge = document.getElementById('disease-badge');
    const warningDiv = document.getElementById('threshold-warning');
    
    if (technicalName === 'Normal' || technicalName === 'Healthy') {
        badge.style.background = 'linear-gradient(135deg, #10b981, #059669)';
        if (warningDiv) warningDiv.style.display = 'none';
    } else if (!prediction.threshold_met) {
        badge.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
        document.getElementById('disease-name').textContent = `Inconclusive (${displayTitle})`;
        if (warningDiv) {
            warningDiv.style.display = 'flex';
            const threshold = (prediction.confidence_threshold !== undefined && prediction.confidence_threshold !== null) ? prediction.confidence_threshold : 0.75;
            document.getElementById('threshold-warning-text').textContent = 
                `Low confidence (${(confidence * 100).toFixed(1)}%) is below the clinical threshold (${(threshold * 100).toFixed(0)}%). Results are considered inconclusive.`;
        }
    } else {
        badge.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        if (warningDiv) warningDiv.style.display = 'none';
        
        // Automatically trigger simulation for Tumor-related classes
        if (isTumor) {
            triggerSimulation('tumor');
        }
    }

    // Display probabilities chart
    displayProbabilityChart(prediction.probabilities);

    // Display feature importance
    displayFeatureChart(featureImportance);

    // Display recommendations
    displayRecommendations(prediction.recommendations);
}

async function triggerSimulation(diseaseType) {
    const statusDiv = document.getElementById('simulation-status');
    if (statusDiv) statusDiv.style.display = 'block';

    updateProgress('Tumor detected. Forwarding to Neural Simulation...');

    try {
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_regions: 90,
                duration: 5.0,
                disease_type: 'tumor' // use lowercase as expected by backend simulation
            })
        });

        const result = await response.json();

        if (result.success) {
            if (statusDiv) {
                statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> Simulation generated for tumor progression.';
                statusDiv.style.color = '#10b981';
            }
            // Wait a moment so the user reads the message
            setTimeout(() => {
                alert('Simulation completed successfully. Check the simulation tab for detailed network dynamics.');
            }, 1000);
        } else {
            if (statusDiv) statusDiv.style.display = 'none';
            alert('Simulation error: ' + result.error);
        }
    } catch (error) {
        console.error('Simulation error:', error);
        if (statusDiv) statusDiv.style.display = 'none';
        alert('Failed to run simulation. Please try again.');
    }
}

function displayProbabilityChart(probabilities) {
    const ctx = document.getElementById('probability-chart').getContext('2d');

    // Destroy existing chart
    if (probabilityChart) {
        probabilityChart.destroy();
    }

    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities).map(v => v * 100);

    const colors = labels.map(label => {
        if (label === 'Healthy') return '#10b981';
        return '#6366f1';
    });

    probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: values,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { color: '#a0aec0' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                x: {
                    ticks: { color: '#a0aec0' },
                    grid: { display: false }
                }
            }
        }
    });
}

function displayFeatureChart(features) {
    const ctx = document.getElementById('feature-chart').getContext('2d');

    // Destroy existing chart
    if (featureChart) {
        featureChart.destroy();
    }

    const labels = Object.keys(features);
    const values = Object.values(features);

    featureChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels.map(l => l.replace('_', ' ').toUpperCase()),
            datasets: [{
                label: 'Importance',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.2)',
                borderColor: '#6366f1',
                borderWidth: 2,
                pointBackgroundColor: '#6366f1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    ticks: { color: '#a0aec0' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#a0aec0' }
                }
            }
        }
    });
}

function displayRecommendations(recommendations) {
    const list = document.getElementById('recommendations-list');
    list.innerHTML = '';

    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        li.style.marginBottom = '10px';
        li.style.color = '#e2e8f0';
        list.appendChild(li);
    });
}

function showLoading(show) {
    const modal = document.getElementById('loading-modal');
    modal.style.display = show ? 'flex' : 'none';

    if (show) {
        updateProgress('Preprocessing...');
        setTimeout(() => updateProgress('Running neural network...'), 1000);
        setTimeout(() => updateProgress('Analyzing results...'), 2000);
    }
}

function updateProgress(text) {
    const progressText = document.getElementById('progress-text');
    if (progressText) {
        progressText.textContent = text;
    }
}
