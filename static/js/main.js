// Main JavaScript file for AI Classification System

// Global variables
let isProcessing = false;
let notificationTimeout;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
    initializeFileUpload();
    initializeNotifications();
});

// Initialize Bootstrap tooltips
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize file upload with drag and drop
function initializeFileUpload() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        const container = input.closest('.card-body') || input.parentElement;
        
        // Add drag and drop functionality
        container.addEventListener('dragover', function(e) {
            e.preventDefault();
            container.classList.add('dragover');
        });
        
        container.addEventListener('dragleave', function(e) {
            e.preventDefault();
            container.classList.remove('dragover');
        });
        
        container.addEventListener('drop', function(e) {
            e.preventDefault();
            container.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                input.files = files;
                
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                input.dispatchEvent(event);
            }
        });
    });
}

// Initialize notification system
function initializeNotifications() {
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            if (alert.parentElement) {
                alert.classList.add('fade');
                setTimeout(() => {
                    alert.remove();
                }, 150);
            }
        }, 5000);
    });
}

// Show notification
function showNotification(message, type = 'info', duration = 5000) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of main content
    const main = document.querySelector('main');
    main.insertBefore(alertDiv, main.firstChild);
    
    // Auto-hide after duration
    if (duration > 0) {
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.classList.remove('show');
                setTimeout(() => {
                    alertDiv.remove();
                }, 150);
            }
        }, duration);
    }
}

// Show loading state
function showLoading(element, text = 'Loading...') {
    if (element) {
        element.disabled = true;
        element.dataset.originalText = element.innerHTML;
        element.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${text}`;
    }
}

// Hide loading state
function hideLoading(element) {
    if (element && element.dataset.originalText) {
        element.disabled = false;
        element.innerHTML = element.dataset.originalText;
        delete element.dataset.originalText;
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Validate image file
function validateImageFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        return { valid: false, message: 'Invalid file type. Please select an image file.' };
    }
    
    if (file.size > maxSize) {
        return { valid: false, message: 'File too large. Maximum size is 16MB.' };
    }
    
    return { valid: true };
}

// Preview image
function previewImage(file, previewElement) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        previewElement.src = e.target.result;
        previewElement.style.display = 'block';
    };
    
    reader.readAsDataURL(file);
}

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showNotification('Copied to clipboard!', 'success', 2000);
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
        showNotification('Failed to copy to clipboard', 'danger', 3000);
    });
}

// Download data as JSON
function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
}

// Format timestamp
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// API helper functions
const API = {
    // Generic API call
    call: async function(endpoint, options = {}) {
        try {
            const response = await fetch(endpoint, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    },
    
    // Get system statistics
    getStats: function() {
        return this.call('/api/stats');
    },
    
    // Discover new classes
    discoverClasses: function(method = 'kmeans') {
        return this.call('/api/discover_classes', {
            method: 'POST',
            body: JSON.stringify({ method })
        });
    },
    
    // Create class from cluster
    createClass: function(clusterId, className, description = '') {
        return this.call('/api/create_class', {
            method: 'POST',
            body: JSON.stringify({
                cluster_id: clusterId,
                class_name: className,
                description: description
            })
        });
    }
};

// Camera utilities
const Camera = {
    stream: null,
    
    // Start camera
    start: async function(videoElement) {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            
            videoElement.srcObject = this.stream;
            return true;
        } catch (error) {
            console.error('Error starting camera:', error);
            throw error;
        }
    },
    
    // Stop camera
    stop: function() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    },
    
    // Capture frame
    captureFrame: function(videoElement, canvasElement) {
        const ctx = canvasElement.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        return canvasElement.toDataURL('image/jpeg', 0.8);
    }
};

// Chart utilities
const Charts = {
    // Create bar chart
    createBarChart: function(canvasId, labels, data, options = {}) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: options.label || 'Data',
                    data: data,
                    backgroundColor: options.backgroundColor || 'rgba(54, 162, 235, 0.6)',
                    borderColor: options.borderColor || 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                ...options.chartOptions
            }
        });
    },
    
    // Create pie chart
    createPieChart: function(canvasId, labels, data, options = {}) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        
        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: options.backgroundColor || [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                    ]
                }]
            },
            options: {
                responsive: true,
                ...options.chartOptions
            }
        });
    }
};

// Form validation utilities
const Validation = {
    // Validate required fields
    validateRequired: function(form) {
        const requiredFields = form.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                this.showFieldError(field, 'This field is required');
                isValid = false;
            } else {
                this.clearFieldError(field);
            }
        });
        
        return isValid;
    },
    
    // Show field error
    showFieldError: function(field, message) {
        field.classList.add('is-invalid');
        
        let feedback = field.parentElement.querySelector('.invalid-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            field.parentElement.appendChild(feedback);
        }
        
        feedback.textContent = message;
    },
    
    // Clear field error
    clearFieldError: function(field) {
        field.classList.remove('is-invalid');
        
        const feedback = field.parentElement.querySelector('.invalid-feedback');
        if (feedback) {
            feedback.remove();
        }
    }
};

// Export utilities for use in other scripts
window.AIClassificationSystem = {
    API,
    Camera,
    Charts,
    Validation,
    showNotification,
    showLoading,
    hideLoading,
    formatFileSize,
    validateImageFile,
    previewImage,
    copyToClipboard,
    downloadJSON,
    formatTimestamp,
    debounce,
    throttle
};