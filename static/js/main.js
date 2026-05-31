// TextDifficulty.ai — Main JS Utilities

// Smooth page transitions
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.fade-in').forEach((el, i) => {
        el.style.animationDelay = `${i * 0.05}s`;
    });
});

// Toast notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    const colors = { info: 'var(--accent)', success: 'var(--green)', error: 'var(--red)', warning: 'var(--yellow)' };
    toast.style.cssText = `
        position: fixed; bottom: 24px; right: 24px; z-index: 9999;
        padding: 12px 20px; border-radius: 8px; font-size: 14px;
        background: var(--bg-card); color: var(--text-primary);
        border: 1px solid ${colors[type] || colors.info};
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        animation: fadeIn 0.3s ease;
        font-family: var(--font-main);
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
}
