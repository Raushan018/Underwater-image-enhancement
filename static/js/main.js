function app() {
    return {
        dragover: false,
        loading: false,
        results: null,
        metrics: null,
        modalOpen: false,
        modalImage: '',
        showDetections: false,
        imgNaturalWidth: 0,
        imgNaturalHeight: 0,
        imgDisplayWidth: 0,
        chartInstances: {},

        get detections() {
            if (!this.results || !this.results.detections) return [];
            return this.results.detections;
        },

        get intermediateSteps() {
            if (!this.results) return {};
            return {
                'White Balanced': this.results.wb,
                'Gamma Corrected': this.results.gamma,
                'Sharpened': this.results.sharp,
                'CLAHE': this.results.clahe,
                'Hist. Linearized': this.results.hist
            };
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
        },

        uploadFile(file) {
            this.loading = true;
            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    this.loading = false;
                    if (data.success) {
                        this.results = data.images;
                        this.metrics = data.metrics;

                        // Render histograms after a small delay to ensure DOM is ready/visible
                        setTimeout(() => {
                            this.renderHistograms(data.images.histograms);
                        }, 100);

                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    this.loading = false;
                    console.error('Error:', error);
                    alert('An error occurred during upload.');
                });
        },

        reset() {
            this.results = null;
            this.metrics = null;
            document.getElementById('fileInput').value = '';
        },

        formatLabel(key) {
            return key;
        },

        openModal(src) {
            this.modalImage = src;
            this.modalOpen = true;
        },

        downloadImage(filename, format = null) {
            if (!filename) return;

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

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
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
            // We need to scale from natural size to displayed size
            const img = document.getElementById('detectionImage');
            if (!img || img.clientWidth === 0) return {};

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
