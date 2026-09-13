const sunIcon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="4" stroke="currentColor" stroke-width="1.8"/><path d="M12 2V4M12 20V22M4 12H2M22 12H20M5 5L6.5 6.5M18.5 18.5L20 20M5 19L6.5 17.5M18.5 5.5L20 4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
const moonIcon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 14.5C18.9 15 17.7 15.3 16.5 15.3C11.8 15.3 8 11.5 8 6.8C8 5.6 8.3 4.4 8.8 3.3C5.4 4.6 3 7.9 3 11.8C3 16.9 7.1 21 12.2 21C16 21 19.3 18.7 20.7 15.3C20.5 14.7 20.2 14.6 20 14.5Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>';

function updateToggleIcon() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    document.querySelectorAll('.theme-toggle').forEach(btn => {
        btn.innerHTML = isDark ? sunIcon : moonIcon;
    });
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateToggleIcon();
}

(function() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
})();

document.addEventListener('DOMContentLoaded', updateToggleIcon);

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    document.getElementById('btn-' + tabName).classList.add('active');
}

function handleUploadSubmit(form) {
    const btn = form.querySelector('button[type="submit"]');
    const fileInput = form.querySelector('input[type="file"]');
    if (!fileInput.files.length) return true;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Processing video, this may take a few minutes...';
    return true;
}

let pendingForm = null;

function confirmAction(message, formElement) {
    pendingForm = formElement;
    document.getElementById('modal-message').textContent = message;
    document.getElementById('modal-overlay').classList.add('active');
    return false;
}

function modalConfirm() {
    document.getElementById('modal-overlay').classList.remove('active');
    if (pendingForm) pendingForm.submit();
}

function modalCancel() {
    document.getElementById('modal-overlay').classList.remove('active');
    pendingForm = null;
}

function checkVideoDuration(input) {
    return new Promise((resolve) => {
        if (!input.files.length) { resolve(true); return; }
        const file = input.files[0];
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = function() {
            window.URL.revokeObjectURL(video.src);
            if (video.duration > 30) {
                alert('Please record a video no longer than 30 seconds. Your video is ' + Math.round(video.duration) + ' seconds.');
                resolve(false);
            } else {
                resolve(true);
            }
        };
        video.src = URL.createObjectURL(file);
    });
}

async function handleUploadSubmitWithCheck(form) {
    const input = form.querySelector('input[type="file"]');
    const ok = await checkVideoDuration(input);
    if (!ok) return false;
    return handleUploadSubmit(form);
}

function triggerCameraInput() {
    document.getElementById('camera-input').click();
}

function triggerFileInput() {
    document.getElementById('file-input').click();
}

function onVideoSelected(input) {
    const nameDisplay = document.getElementById('selected-file-name');
    if (input.files.length) {
        nameDisplay.textContent = 'Selected: ' + input.files[0].name;
        // sync this file into the real hidden input the form actually submits
        const realInput = document.getElementById('video-input');
        const dt = new DataTransfer();
        dt.items.add(input.files[0]);
        realInput.files = dt.files;
    }
}

function toggleDateGroup(id) {
    const header = document.getElementById('header-' + id);
    const table = document.getElementById('table-' + id);
    header.classList.toggle('collapsed');
    table.classList.toggle('collapsed');
}
