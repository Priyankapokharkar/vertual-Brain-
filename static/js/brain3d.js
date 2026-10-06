/**
 * Virtual Brain Inference - 3D Visualization
 * Three.js-based brain and vessel rendering
 */

class BrainVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.brainMesh = null;
        this.vessels = [];
        this.blockages = [];
        this.animationId = null;
        this.animationId = null;
        this.isRotating = false;
        this.isSimulating = false;
        this.simulationData = null;
        this.currentSimStep = 0;
        this.regionMap = {};
        this.diseaseMap = {
            'Normal': 'Healthy Control',
            'Alzheimer_Mild': 'Alzheimer\'s (Mild)',
            'Alzheimer_Moderate': 'Alzheimer\'s (Moderate)',
            'Alzheimer_VeryMild': 'Alzheimer\'s (Very Mild)',
            'Glioma': 'Brain Tumor (Glioma)',
            'Meningioma': 'Brain Tumor (Meningioma)',
            'Pituitary': 'Brain Tumor (Pituitary)',
            'MS': 'Multiple Sclerosis',
            'Parkinson': 'Parkinson\'s Disease'
        };

        this.init();
    }

    init() {
        // Setup scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e27);

        // Setup camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 3);

        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        // Add lights
        this.setupLights();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Start animation
        this.animate();
    }

    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        // Directional lights
        const light1 = new THREE.DirectionalLight(0x6366f1, 0.8);
        light1.position.set(2, 2, 2);
        this.scene.add(light1);

        const light2 = new THREE.DirectionalLight(0xec4899, 0.4);
        light2.position.set(-2, -1, -2);
        this.scene.add(light2);

        // Point light for glow effect
        const pointLight = new THREE.PointLight(0x14b8a6, 1, 100);
        pointLight.position.set(0, 2, 0);
        this.scene.add(pointLight);
    }

    createBrainMesh(meshData) {
        // Create geometry from mesh data
        const geometry = new THREE.BufferGeometry();

        const vertices = new Float32Array(meshData.vertices);
        const indices = new Uint32Array(meshData.faces);

        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();

        // Handle regions and colors
        if (meshData.regions) {
            const count = vertices.length / 3;
            const colors = new Float32Array(count * 3);
            this.regionMap = {}; // Map region index to vertex indices

            // Initialize with default or region colors
            for (let i = 0; i < count; i++) {
                colors[i * 3] = 0.4;     // R
                colors[i * 3 + 1] = 0.4; // G
                colors[i * 3 + 2] = 0.9; // B
            }

            // Apply region colors
            Object.entries(meshData.regions).forEach(([name, region]) => {
                const regionId = parseInt(name.split('_')[1]);
                this.regionMap[regionId] = region.vertices;

                const color = region.color || [0.4, 0.4, 0.9];

                region.vertices.forEach(vIdx => {
                    colors[vIdx * 3] = color[0];
                    colors[vIdx * 3 + 1] = color[1];
                    colors[vIdx * 3 + 2] = color[2];
                });
            });

            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }

        // Create material with vertex colors
        const material = new THREE.MeshPhongMaterial({
            vertexColors: meshData.regions ? true : false,
            color: meshData.regions ? 0xffffff : 0x6366f1,
            emissive: 0x222222,
            shininess: 30,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide
        });

        // Create mesh
        this.brainMesh = new THREE.Mesh(geometry, material);
        this.scene.add(this.brainMesh);

        return this.brainMesh;
    }

    createVesselNetwork(vesselData) {
        this.vessels = [];

        vesselData.vessels.forEach(vessel => {
            const points = [
                new THREE.Vector3(...vessel.start),
                new THREE.Vector3(...vessel.end)
            ];

            const geometry = new THREE.TubeGeometry(
                new THREE.CatmullRomCurve3(points),
                10,
                vessel.radius,
                8,
                false
            );

            // Color based on vessel type
            let color;
            if (vessel.type === 'artery') {
                color = vessel.blockage > 0 ? 0x8b0000 : 0xff3333; // Dark red if blocked
            } else {
                color = vessel.blockage > 0 ? 0x00008b : 0x3333ff; // Dark blue if blocked
            }

            const material = new THREE.MeshPhongMaterial({
                color: color,
                emissive: vessel.blockage > 0 ? 0xff0000 : 0x000000,
                emissiveIntensity: vessel.blockage * 0.5,
                transparent: true,
                opacity: 0.8
            });

            const vesselMesh = new THREE.Mesh(geometry, material);
            vesselMesh.userData = vessel;

            this.vessels.push(vesselMesh);
            this.scene.add(vesselMesh);
        });
    }

    highlightBlockages(blockageData) {
        blockageData.forEach(blockage => {
            const geometry = new THREE.SphereGeometry(0.05, 16, 16);
            const material = new THREE.MeshBasicMaterial({
                color: 0xff0000,
                transparent: true,
                opacity: 0.8
            });

            const marker = new THREE.Mesh(geometry, material);
            marker.position.set(...blockage.location);

            // Pulsing animation
            marker.userData = {
                isBlockage: true,
                severity: blockage.severity,
                pulsePhase: Math.random() * Math.PI * 2
            };

            this.blockages.push(marker);
            this.scene.add(marker);
        });
    }

    loadPatientData(patientId, focalCoords = null) {
        document.getElementById('status-text').textContent = 'Loading...';
        this.clearScene();

        let url = `/api/visualization/${patientId}`;
        if (focalCoords) {
            url += `?focal_x=${focalCoords.x}&focal_y=${focalCoords.y}&focal_z=${focalCoords.z}`;
            if (focalCoords.radius) url += `&radius=${focalCoords.radius}`;
        }

        fetch(url)
            .then(response => response.json())
            .then(data => {
                if (data.success) {

                    // Create brain mesh
                    if (data.brain_mesh && document.getElementById('show-brain').checked) {
                        this.createBrainMesh(data.brain_mesh);
                    }

                    // Create vessel network
                    if (data.vessel_network && document.getElementById('show-vessels').checked) {
                        this.createVesselNetwork(data.vessel_network);
                    }

                    // Highlight blockages
                    if (data.blockage_analysis && document.getElementById('show-blockages').checked) {
                        this.highlightBlockages(data.blockage_analysis.blockages || []);
                    }

                    // Update info panel
                    this.updateInfoPanel(data);

                    document.getElementById('status-text').textContent = 'Loaded';
                } else {
                    console.error('Failed to load visualization data:', data.error);
                    document.getElementById('status-text').textContent = 'Error';
                }
            })
            .catch(error => {
                console.error('Error loading patient data:', error);
                document.getElementById('status-text').textContent = 'Error';
            });
    }

    updateInfoPanel(data) {
        if (data.vessel_network) {
            document.getElementById('vessel-count').textContent = data.vessel_network.num_vessels;
        }

        if (data.blockage_analysis) {
            document.getElementById('blockage-count').textContent = data.blockage_analysis.blockages_detected;

            const avgSeverity = data.blockage_analysis.average_severity;
            if (avgSeverity > 0) {
                document.getElementById('avg-severity').textContent =
                    `${(avgSeverity * 100).toFixed(1)}%`;
            } else {
                document.getElementById('avg-severity').textContent = 'None';
            }
        }
    }

    clearScene() {
        // Remove brain mesh
        if (this.brainMesh) {
            this.scene.remove(this.brainMesh);
            this.brainMesh.geometry.dispose();
            this.brainMesh.material.dispose();
            this.brainMesh = null;
        }

        // Remove vessels
        this.vessels.forEach(vessel => {
            this.scene.remove(vessel);
            vessel.geometry.dispose();
            vessel.material.dispose();
        });
        this.vessels = [];

        // Remove blockage markers
        this.blockages.forEach(blockage => {
            this.scene.remove(blockage);
            blockage.geometry.dispose();
            blockage.material.dispose();
        });
        this.blockages = [];
    }

    toggleRotation() {
        this.isRotating = !this.isRotating;
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        // Auto-rotation
        if (this.isRotating && this.brainMesh) {
            this.brainMesh.rotation.y += 0.005;
        }

        // Simulation Update
        if (this.isSimulating) {
            this.updateBrainColors();
        }

        // Pulse blockages
        const time = Date.now() * 0.001;
        this.blockages.forEach(blockage => {
            const scale = 1 + Math.sin(time * 2 + blockage.userData.pulsePhase) * 0.3;
            blockage.scale.set(scale, scale, scale);
        });

        // Update controls
        this.controls.update();

        // Render
        this.renderer.render(this.scene, this.camera);
    }

    runSimulation(diseaseType) {
        document.getElementById('status-text').textContent = 'Simulating...';

        fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_regions: 90,
                duration: 5.0,
                disease_type: diseaseType
            })
        })
            .then(res => res.json())
            .then(data => {
                if (data.success && data.activity) {
                    this.startSimulationPlayback(data.activity);
                    document.getElementById('status-text').textContent = 'Playing Simulation';
                } else {
                    console.error('Simulation failed');
                    document.getElementById('status-text').textContent = 'Sim Error';
                }
            })
            .catch(err => {
                console.error(err);
                document.getElementById('status-text').textContent = 'Error';
            });
    }

    startSimulationPlayback(activityData) {
        this.simulationData = activityData;
        this.currentSimStep = 0;
        this.isSimulating = true;
    }

    updateBrainColors() {
        if (!this.brainMesh || !this.simulationData || !this.isSimulating) return;

        const activity = this.simulationData[this.currentSimStep];
        if (!activity) {
            // Loop or stop
            this.currentSimStep = 0;
            return;
        }

        const colors = this.brainMesh.geometry.attributes.color.array;

        // Update colors based on activity
        // Activity is array of size [num_regions]
        activity.forEach((val, regionId) => {
            // Map value to color (heatmap: blue -> red)
            // Normalize val (assuming approx -1 to 1 range or similar)
            const intensity = Math.max(0, Math.min(1, (val + 0.5))); // Simple normalization

            const r = intensity;
            const g = 0.2;
            const b = 1.0 - intensity;

            if (this.regionMap && this.regionMap[regionId]) {
                const vertexIndices = this.regionMap[regionId];
                vertexIndices.forEach(vIdx => {
                    colors[vIdx * 3] = r;
                    colors[vIdx * 3 + 1] = g;
                    colors[vIdx * 3 + 2] = b;
                });
            }
        });

        this.brainMesh.geometry.attributes.color.needsUpdate = true;

        // Advance step (slow down if needed, currently 1 step per frame)
        if (this.currentSimStep < this.simulationData.length - 1) {
            this.currentSimStep++;
        } else {
            this.currentSimStep = 0; // Loop
        }
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.clearScene();
        this.renderer.dispose();
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    const viz = new BrainVisualization('visualization-canvas');

    // Expose instance globally
    window.brainViz = viz;

    // Load initial data if patient-id exists
    const patientInput = document.getElementById('patient-id');
    if (patientInput) {
        const pid = patientInput.value;
        viz.loadPatientData(pid);

        // Auto-run simulation if arrived from a detection result (e.g., pid is 'tumor')
        if (pid === 'tumor') {
            setTimeout(() => {
                viz.runSimulation('tumor');
            }, 1000); // Wait for mesh to load
        }
    } else {
        // Default load for pages without explicit patient control
        viz.loadPatientData('patient_001');
    }

    // Event listeners - Safe binding
    const loadBtn = document.getElementById('load-data-btn');
    if (loadBtn) {
        loadBtn.addEventListener('click', () => {
            const pid = document.getElementById('patient-id')?.value || 'patient_001';
            viz.loadPatientData(pid);
        });
    }

    const rotBtn = document.getElementById('toggle-rotation-btn');
    if (rotBtn) {
        rotBtn.addEventListener('click', () => {
            viz.toggleRotation();
        });
    }

    // Visibility toggles
    ['show-brain', 'show-vessels', 'show-blockages', 'show-arteries', 'show-veins'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', () => {
                const pid = document.getElementById('patient-id')?.value || 'patient_001';
                viz.loadPatientData(pid);
            });
        }
    });

    // Simulation Button
    const runSimBtn = document.getElementById('run-sim-btn');
    if (runSimBtn) {
        runSimBtn.addEventListener('click', () => {
            const disease = document.getElementById('sim-disease-type')?.value || 'healthy';
            viz.runSimulation(disease);
        });
    }

    // Nerve Damage Analysis - Image Upload
    const analyzeBtn = document.getElementById('analyze-scan-btn');
    const scanUpload = document.getElementById('tumor-scan-upload');

    if (analyzeBtn && scanUpload) {
        analyzeBtn.addEventListener('click', async () => {
            const file = scanUpload.files[0];
            if (!file) {
                alert('Please select a tumor image first.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                analyzeBtn.innerText = 'Analyzing...';
                analyzeBtn.disabled = true;

                const response = await fetch('/api/detect_tumor_spatial', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (result.success && result.is_tumor) {
                    const loc = result.localization;
                    const coords = {
                        x: loc.centroid[0],
                        y: loc.centroid[1],
                        z: loc.centroid[2],
                        radius: loc.radius
                    };

                    // Reload visualization with detected focal point
                    viz.loadPatientData('tumor', coords);

                    // Auto-run simulation for the detected tumor
                    setTimeout(() => {
                        viz.runSimulation('tumor');
                    }, 1500);

                    alert(`Tumor detected and localized at [${coords.x.toFixed(2)}, ${coords.y.toFixed(2)}]. Visualization updated.`);
                } else if (result.success) {
                    alert('No tumor detected in this scan.');
                } else {
                    alert('Error analyzing scan: ' + result.error);
                }
            } catch (err) {
                console.error(err);
                alert('Failed to analyze scan.');
            } finally {
                analyzeBtn.innerText = 'Analyze Spatial Damage';
                analyzeBtn.disabled = false;
            }
        });
    }
});
