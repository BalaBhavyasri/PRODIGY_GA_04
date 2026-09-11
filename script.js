document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const canvas = document.getElementById('paintCanvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const brushSizeInput = document.getElementById('brushSize');
    const brushSizeVal = document.getElementById('brushSizeVal');
    const clearBtn = document.getElementById('clearBtn');
    const fillBtn = document.getElementById('fillBtn');
    const colorTools = document.querySelectorAll('.color-tool');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    const translateBtn = document.getElementById('translateBtn');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadPreviewContainer = document.getElementById('uploadPreviewContainer');
    const uploadPreview = document.getElementById('uploadPreview');
    const removeUploadBtn = document.getElementById('removeUploadBtn');
    
    const inputWrapper = document.getElementById('inputWrapper');
    const inputImg = document.getElementById('inputImg');
    const outputWrapper = document.getElementById('outputWrapper');
    const outputImg = document.getElementById('outputImg');
    const targetBox = document.getElementById('targetBox');
    const targetImg = document.getElementById('targetImg');
    const loadingSpinner = outputWrapper.querySelector('.loading-spinner');
    
    const presetsGallery = document.getElementById('presetsGallery');
    // Drawing Canvas Variables
    let isDrawing = false;
    let currentColor = '#ff0000'; // Default is Red (Wall)
    let brushSize = 15;
    let activeTab = 'draw'; // 'draw' or 'upload'
    let uploadedImageBase64 = null;
    // Initialize Canvas with background color
    function initCanvas() {
        ctx.fillStyle = '#ff0000'; // Standard Facades Wall label (Red)
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    initCanvas();
    // Brush settings listeners
    brushSizeInput.addEventListener('input', (e) => {
        brushSize = e.target.value;
        brushSizeVal.textContent = `${brushSize}px`;
    });
    // Toolbar color selector
    colorTools.forEach(tool => {
        tool.addEventListener('click', () => {
            colorTools.forEach(t => t.classList.remove('active'));
            tool.classList.add('active');
            currentColor = tool.getAttribute('data-color');
        });
    });
    // Drawing event listeners
    function getMousePos(canvasDom, e) {
        const rect = canvasDom.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: ((clientX - rect.left) / rect.width) * canvasDom.width,
            y: ((clientY - rect.top) / rect.height) * canvasDom.height
        };
    }
    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }
    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath();
    }
    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        
        const pos = getMousePos(canvas, e);
        
        ctx.lineWidth = brushSize;
        ctx.lineCap = 'round';
        ctx.strokeStyle = currentColor;
        
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
    
    // Mobile Touch support
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);
    // Canvas Clear and Fill
    clearBtn.addEventListener('click', initCanvas);
    fillBtn.addEventListener('click', () => {
        ctx.fillStyle = currentColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    });
    // Tab buttons switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            activeTab = btn.getAttribute('data-tab');
            document.getElementById(`${activeTab}-tab`).classList.add('active');
        });
    });
    // File Upload handling
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        if (e.dataTransfer.files.length) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleImageFile(e.target.files[0]);
        }
    });
    function handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please drop/upload a valid image file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImageBase64 = e.target.result;
            uploadPreview.src = uploadedImageBase64;
            dropZone.style.display = 'none';
            uploadPreviewContainer.style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }
    removeUploadBtn.addEventListener('click', () => {
        uploadedImageBase64 = null;
        uploadPreview.src = '';
        uploadPreviewContainer.style.display = 'none';
        dropZone.style.display = 'block';
        fileInput.value = '';
    });
    // Load Presets Gallery
    async function loadPresets() {
        try {
            const response = await fetch('/presets');
            const files = await response.json();
            
            presetsGallery.innerHTML = '';
            
            if (files.length === 0) {
                presetsGallery.innerHTML = '<div class="gallery-loading">No validation samples found in facades/val/</div>';
                return;
            }
            files.forEach(filename => {
                const card = document.createElement('div');
                card.className = 'preset-card';
                card.innerHTML = `
                    <img src="" alt="${filename}" data-filename="${filename}">
                    <span>Sample ${filename.split('.')[0]}</span>
                `;
                
                // Add click listener
                card.addEventListener('click', () => loadPresetImage(filename));
                presetsGallery.appendChild(card);
                
                // Lazy-load a preview of validation combined image
                // To display a tiny thumbnail, we can query details or use the server endpoint
                loadThumbnail(filename, card.querySelector('img'));
            });
        } catch (err) {
            console.error('Failed to load presets:', err);
            presetsGallery.innerHTML = '<div class="gallery-loading">Error loading validation gallery presets.</div>';
        }
    }
    async function loadThumbnail(filename, imgElement) {
        try {
            const res = await fetch(`/presets/${filename}`);
            const data = await res.json();
            // Display label part of preset as thumbnail
            imgElement.src = data.label;
        } catch (e) {
            console.error(e);
        }
    }
    async function loadPresetImage(filename) {
        try {
            // Show loaders
            inputWrapper.classList.remove('empty');
            inputImg.style.display = 'none';
            
            const response = await fetch(`/presets/${filename}`);
            const data = await response.json();
            
            // 1. Load label into Input Results
            inputImg.src = data.label;
            inputImg.style.display = 'block';
            
            // 2. Load ground truth comparison
            targetImg.src = data.photo;
            targetBox.style.display = 'block';
            
            // 3. Load label color map back into canvas for editing
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Switch to draw tab automatically
                tabBtns[0].click();
            };
            img.src = data.label;
            
        } catch (err) {
            console.error('Failed to load preset images:', err);
            alert('Failed to load preset image files.');
        }
    }
    // Call Model Translate API
    translateBtn.addEventListener('click', async () => {
        let payloadImage = null;
        
        if (activeTab === 'draw') {
            payloadImage = canvas.toDataURL('image/png');
        } else {
            payloadImage = uploadedImageBase64;
        }
        
        if (!payloadImage) {
            alert('Please draw a layout or upload an image first.');
            return;
        }
        
        // Show input layout in result preview
        inputWrapper.classList.remove('empty');
        inputImg.src = payloadImage;
        inputImg.style.display = 'block';
        
        // Reset output image and show loading spinner
        outputWrapper.classList.remove('empty');
        outputImg.style.display = 'none';
        loadingSpinner.style.display = 'flex';
        
        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: payloadImage })
            });
            
            const data = await response.json();
            
            if (response.ok && data.image) {
                outputImg.src = data.image;
                outputImg.style.display = 'block';
            } else {
                alert(`Error: ${data.error || 'Failed to translate image.'}`);
            }
        } catch (err) {
            console.error(err);
            alert('A network error occurred while connecting to the server.');
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });
    // Initial setups
    loadPresets();
});