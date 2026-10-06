/**
 * Dashboard functionality
 * Updates stats and provides interactivity
 */

document.addEventListener('DOMContentLoaded', () => {
    animateStats();
    fetchSystemStatus();
});

function animateStats() {
    // Animate stat values with counting effect
    const statValues = document.querySelectorAll('.stat-value');
    
    statValues.forEach(stat => {
        const target = parseInt(stat.textContent) || 95;
        const duration = 2000;
        const steps = 60;
        const increment = target / steps;
        let current = 0;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                stat.textContent = target + (stat.textContent.includes('%') ? '%' : '');
                clearInterval(timer);
            } else {
                const value = Math.floor(current);
                stat.textContent = value + (stat.textContent.includes('%') ? '%' : '');
            }
        }, duration / steps);
    });
}

async function fetchSystemStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('System status: Healthy');
        }
    } catch (error) {
        console.error('Failed to fetch system status:', error);
    }
}

// Add hover effects to action cards
const actionCards = document.querySelectorAll('.action-card');
actionCards.forEach(card => {
    card.addEventListener('mouseenter', () => {
        card.style.transform = 'translateY(-8px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', () => {
        card.style.transform = 'translateY(0) scale(1)';
    });
});
