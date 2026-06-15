function app() {
    return {
        dragover: false,
        loading: false,
        results: null,
        metrics: null,
        modalOpen: false,
        modalImage: '',

        // Camera State
        cameraOpen: false,
        mediaStream: null,

        showDetections: false,
        showDepth: false,
        imgNaturalWidth: 0,
        imgNaturalHeight: 0,
        imgDisplayWidth: 0,
        chartInstances: {},

        // Toast State
        toasts: [],

        showToast(message, type = 'error') {
            const id = Date.now();
            this.toasts.push({ id, message, type });
            setTimeout(() => {
                this.toasts = this.toasts.filter(t => t.id !== id);
            }, 5000);
        },

        detectionEnabled: window.detectionEnabled !== false,

        // Pipeline Configuration
        pipelineConfig: {
            stretch: true,
            wb: true,
            gamma: true,
            sharp: true
        },
        currentFilename: null,

        // ... methods ...
        toggleModule(module) {
            this.pipelineConfig[module] = !this.pipelineConfig[module];
            // Auto reprocess if we have an active image
            if (this.results && this.currentFilename) {
                this.reprocessImage();
            }
        },

        reprocessImage() {
            if (!this.currentFilename || this.loading) return;

            this.loading = true;
            fetch('/reprocess', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: this.currentFilename,
                    config: this.pipelineConfig
                })
            })
                .then(async response => {
                    const contentType = response.headers.get("content-type");
                    if (contentType && contentType.indexOf("application/json") !== -1) {
                        return response.json().then(data => {
                            if (!response.ok) {
                                throw new Error(data.error || `Server Error: ${response.status}`);
                            }
                            return data;
                        });
                    } else {
                        const text = await response.text();
                        console.error("Reprocess Non-JSON Response:", text);
                        throw new Error(`Reprocess Failed: Server returned ${response.status}. Check logs.`);
                    }
                })
                .then(data => this.handleResponse(data))
                .catch(error => {
                    console.error("Reprocess error:", error);
                    this.showToast(error.message, 'error');
                })
                .finally(() => {
                    this.loading = false;
                });
        },

        handleResponse(data) {
            console.log("Handle Response Logic Triggered:", data);

            if (data.success) {
                console.log("Success! Setting results.");
                this.results = data.images;
                this.metrics = data.metrics;
                if (data.filename) this.currentFilename = data.filename;

                // Safety check for charts
                setTimeout(() => {
                    try {
                        if (data.images.histograms) this.renderHistograms(data.images.histograms);
                        if (data.images.composition) this.renderCompositionChart(data.images.composition);
                    } catch (e) {
                        console.error("Chart Rendering Error (Non-blocking):", e);
                    }
                }, 100);
            } else {
                console.error("Backend Error Recieved:", data.error);
                this.showToast(data.error, 'error');
                this.results = null; // Reset on error
            }
        },

        async startCamera() {
            try {
                this.cameraOpen = true;
                this.mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment' }
                });
                // Alpine x-ref usage: this.$refs.videoFeed
                // We need to wait for DOM update or allow Alpine to bind.
                // Best to use $nextTick or ensure element exists.
                this.$nextTick(() => {
                    const video = this.$refs.videoFeed;
                    if (video) {
                        video.srcObject = this.mediaStream;
                    }
                });
            } catch (err) {
                console.error("Error accessing camera:", err);
                this.showToast("Could not access camera. Ensure permissions are granted.", 'error');
                this.cameraOpen = false;
            }
        },

        stopCamera() {
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
                this.mediaStream = null;
            }
            this.cameraOpen = false;
        },

        capturePhoto() {
            const video = this.$refs.videoFeed;
            const canvas = this.$refs.cameraCanvas;

            if (video && canvas) {
                // Set canvas dimensions to match video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                const ctx = canvas.getContext('2d');
                // Draw normal (we css-mirrored the video for preview, but capture should probably be normal? 
                // Usually selfies are mirrored, world-facing are not. Let's keep it WYSIWYG relative to element if possible
                // but usually simpler to just draw image)

                // If we mirrored the video with CSS transform -scale-x-100, we should mirror the draw too if we want result to match preview
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);

                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Convert to blob
                canvas.toBlob((blob) => {
                    if (blob) {
                        // Create a file from blob
                        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
                        this.stopCamera();
                        this.uploadFile(file);
                    }
                }, 'image/jpeg', 0.95);
            }
        },

        get detections() {
            if (!this.results || !this.results.detections) return [];
            return this.results.detections;
        },

        get intermediateSteps() {
            if (!this.results) return {};
            return {
                'Adaptive Color': this.results.wb,
                'Adaptive Gamma': this.results.gamma,
                'Edge Preserving': this.results.sharp,
                'CLAHE': this.results.clahe,
                'Hist. Linearized': this.results.hist
            };
        },

        validateFile(file) {
            const maxSize = 16 * 1024 * 1024; // 16MB (Matching backend)
            const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];

            if (!allowedTypes.includes(file.type)) {
                this.showToast("Invalid file type. JPG, PNG, BMP, or WebP only.", 'error');
                return false;
            }
            if (file.size > maxSize) {
                this.showToast("File matches size limit (16MB). Please use a smaller image.", 'error');
                return false;
            }
            return true;
        },

        handleDrop(event) {
            this.dragover = false;
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                this.uploadFile(files[0]);
            }
        },

        handleFile(event) {
            const files = event.target.files;
            if (files.length > 0) {
                this.uploadFile(files[0]);
            }
            // Reset input so same file selection triggers change again if needed
            event.target.value = '';
        },

        uploadFile(file) {
            if (this.loading) return; // Prevent double submission
            if (!this.validateFile(file)) return;

            this.loading = true;
            this.results = null; // Clear previous results

            const formData = new FormData();
            formData.append('file', file);
            formData.append('config', JSON.stringify(this.pipelineConfig));

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
                .then(async response => {
                    const contentType = response.headers.get("content-type");
                    if (contentType && contentType.indexOf("application/json") !== -1) {
                        return response.json().then(data => {
                            if (!response.ok) {
                                throw new Error(data.error || `Server Error: ${response.status}`);
                            }
                            return data;
                        });
                    } else {
                        // If we get HTML or text, it's likely a crash or unhandled error
                        const text = await response.text();
                        console.error("Non-JSON Response:", text);
                        throw new Error(`Upload Failed: Server returned ${response.status} (${response.statusText}). Check logs.`);
                    }
                })
                .then(data => this.handleResponse(data))
                .catch(error => {
                    console.error('Upload Error:', error);
                    this.showToast(error.message, 'error');
                })
                .finally(() => {
                    this.loading = false;
                });
        },

        reset() {
            this.results = null;
            this.metrics = null;
            if (document.getElementById('fileInput')) document.getElementById('fileInput').value = '';
            this.stopCamera();
        },

        navigateToSection(sectionId) {
            if (sectionId === 'histogram') {
                if (!this.results) {
                    this.showToast("Please upload and process an image first to view the histogram analysis.", "info");
                    return;
                }
                this.$nextTick(() => {
                    const el = document.getElementById('histogram-analysis');
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                });
                return;
            }
            if (this.results) {
                this.reset();
            }
            this.$nextTick(() => {
                const elementId = sectionId === 'pipeline' ? 'pipeline-config-dashboard' : 'demonstration-carousel';
                const el = document.getElementById(elementId);
                if (el) {
                    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            });
        },

        formatLabel(key) {
            return key;
        },

        openModal(src) {
            this.modalImage = src;
            this.modalOpen = true;
        },

        updateBoxes() {
            const img = document.getElementById('detectionImage');
            if (img) {
                this.imgNaturalWidth = img.naturalWidth;
                this.imgNaturalHeight = img.naturalHeight;
                this.imgDisplayWidth = img.clientWidth;
            }
        },

        getBoxStyle(box) {
            // box is [x1, y1, x2, y2]
            const img = document.getElementById('detectionImage');
            if (!img || img.clientWidth === 0 || img.naturalWidth === 0) return {};

            const scale = img.clientWidth / img.naturalWidth;

            const x = box[0] * scale;
            const y = box[1] * scale;
            const w = (box[2] - box[0]) * scale;
            const h = (box[3] - box[1]) * scale;

            return {
                left: `${x}px`,
                top: `${y}px`,
                width: `${w}px`,
                height: `${h}px`
            };
        },

        setView(view) {
            this.showDetections = (view === 'detections');
            this.showDepth = (view === 'depth');

            if (view === 'depth') {
                this.loadDepthRaw();
            }
        },

        loadDepthRaw() {
            if (!this.results || !this.results.depth_raw) return;

            const rawImg = new Image();
            rawImg.src = '/static/images/results/' + this.results.depth_raw;
            rawImg.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = rawImg.width;
                canvas.height = rawImg.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(rawImg, 0, 0);
                this.depthCtx = ctx;
                this.depthWidth = rawImg.width;
                this.depthHeight = rawImg.height;
            };
        },

        handleDepthHover(event) {
            const img = event.target;
            if (!this.depthCtx || !img) return;

            const rect = img.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            // Scale to natural dimensions
            const scaleX = this.depthWidth / rect.width;
            const scaleY = this.depthHeight / rect.height;

            const realX = Math.floor(x * scaleX);
            const realY = Math.floor(y * scaleY);

            if (realX >= 0 && realX < this.depthWidth && realY >= 0 && realY < this.depthHeight) {
                const pixel = this.depthCtx.getImageData(realX, realY, 1, 1).data;
                // pixel[0] is Red channel. In our raw map (grayscale), R=G=B = distance value.
                // 0 = Near, 255 = Far.
                const distanceVal = pixel[0];

                // Calculate relative depth % (0% = closest, 100% = furthest)
                const depthPct = Math.round((distanceVal / 255.0) * 100);

                // Determine category
                let category = "Mid";
                // Our heuristic:
                // Near: High Red in original -> Low Val in Raw Map.
                // Far: Low Red in original -> High Val in Raw Map.
                if (distanceVal < 85) category = "Near"; // 0-85 (0-33%)
                else if (distanceVal > 170) category = "Far"; // 170-255 (66-100%)

                // Update tooltip
                const tooltip = document.getElementById('depthTooltip');
                if (tooltip) {
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 15) + 'px';
                    tooltip.style.top = (event.clientY + 15) + 'px';
                    tooltip.innerHTML = `
                        <div class="font-bold text-cyan-300">${category}</div>
                        <div class="text-xs text-white">Relative Depth: ${depthPct}%</div>
                    `;
                }
            }
        },

        hideDepthTooltip() {
            const tooltip = document.getElementById('depthTooltip');
            if (tooltip) tooltip.style.display = 'none';
        },

        renderCompositionChart(compositionData) {
            const ctx = document.getElementById('compositionChart');
            if (!ctx || !compositionData) return;

            // Destroy existing if any
            if (this.chartInstances['compositionChart']) {
                this.chartInstances['compositionChart'].destroy();
            }

            const labels = Object.keys(compositionData);
            const data = Object.values(compositionData);

            // Colors for specific classes
            const colorMap = {
                'scuba diver': '#facc15', // yellow
                'fish': '#22d3ee', // cyan
                'underwater rock': '#94a3b8', // slate
                'coral reef': '#f472b6', // pink
                'jellyfish': '#c084fc', // purple
                'sea turtle': '#4ade80', // green
                'shark': '#ef4444', // red
                'starfish': '#fb923c' // orange
            };

            const backgroundColors = labels.map(l => colorMap[l] || '#cbd5e1');

            this.chartInstances['compositionChart'] = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
                    datasets: [{
                        data: data,
                        backgroundColor: backgroundColors,
                        borderWidth: 0,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#e2e8f0',
                                usePointStyle: true,
                                padding: 20
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    return context.label + ': ' + context.raw + '%';
                                }
                            }
                        }
                    },
                    cutout: '70%'
                }
            });
        },

        downloadImage(filename, format = null) {
            console.log("Attempting download:", filename, format);
            if (!filename) {
                console.error("No filename provided for download");
                this.showToast("Error: No image available to download.", 'error');
                return;
            }

            const link = document.createElement('a');
            let url = `/download/${filename}`;

            if (format) {
                // If format specified, append query param
                url += `?format=${format}`;
            }

            link.href = url;

            // Set download attribute as hint
            if (format) {
                // Replace extension
                const base = filename.replace(/\.[^/.]+$/, "");
                // Map common formats
                const ext = format === 'jpeg' ? 'jpg' : format;
                link.download = `${base}.${ext}`;
            } else {
                link.download = filename;
            }

            link.target = "_blank"; // Open in new tab if logic fails, better fallback

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        },

        renderHistograms(histData) {
            if (!histData) return;

            const createChart = (canvasId, data) => {
                const ctx = document.getElementById(canvasId).getContext('2d');

                // Destroy existing
                if (this.chartInstances[canvasId]) {
                    this.chartInstances[canvasId].destroy();
                }

                // Create labels (0-255) - reduce density if needed, but Chartjs handles it
                const labels = Array.from({ length: 256 }, (_, i) => i);

                this.chartInstances[canvasId] = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Red',
                                data: data.r,
                                borderColor: 'rgba(239, 68, 68, 1)', // red-500
                                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: true,
                                tension: 0.4
                            },
                            {
                                label: 'Green',
                                data: data.g,
                                borderColor: 'rgba(34, 197, 94, 1)', // green-500
                                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: true,
                                tension: 0.4
                            },
                            {
                                label: 'Blue',
                                data: data.b,
                                borderColor: 'rgba(59, 130, 246, 1)', // blue-500
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: true,
                                tension: 0.4
                            },
                            {
                                label: 'Luminance',
                                data: data.y,
                                borderColor: 'rgba(255, 255, 255, 0.8)',
                                borderDash: [5, 5],
                                borderWidth: 1,
                                pointRadius: 0,
                                fill: false,
                                tension: 0.4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            mode: 'index',
                            intersect: false,
                        },
                        scales: {
                            x: {
                                display: true,
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                ticks: { color: '#94a3b8', maxTicksLimit: 10 }
                            },
                            y: {
                                display: true,
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                ticks: { color: '#94a3b8', display: false } // Hide counts, just shape matters
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#e2e8f0', usePointStyle: true }
                            }
                        }
                    }
                });
            };

            createChart('beforeChart', histData.before);
            createChart('afterChart', histData.after);
        }
    }
}
