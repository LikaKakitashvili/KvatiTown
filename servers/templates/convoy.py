from .base import render_template

_EXTRA_CSS = '''
.convoy-state {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.state-cruising  { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid #3fb950; }
.state-slow      { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid #d29922; }
.state-stopping  { background: rgba(248,81,73,0.20);  color: #f85149; border: 1px solid #f85149; }
.state-stopped   { background: rgba(248,81,73,0.10);  color: #8b949e; border: 1px solid #444; }
.state-unknown   { background: rgba(139,148,158,0.1); color: #8b949e; border: 1px solid #444; }

.sign-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
}
.sign-stop   { background: rgba(248,81,73,0.2);  color: #f85149; }
.sign-slow   { background: rgba(210,153,34,0.2); color: #d29922; }
.sign-none   { background: rgba(139,148,158,0.1); color: #6e7681; }

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid var(--border-color);
    font-size: 13px;
}
.info-row:last-child { border-bottom: none; }
.info-key  { color: var(--text-secondary); }
.info-val  { color: var(--text-primary); font-family: monospace; font-weight: 500; }

.control-group { margin-bottom: 16px; }
.control-group:last-child { margin-bottom: 0; }
.control-group label { display: block; margin-bottom: 6px; font-size: 13px; font-weight: 600; }
.control-row { display: flex; align-items: center; gap: 10px; }
.value-display { min-width: 50px; text-align: right; font-family: monospace; font-size: 12px; color: var(--text-secondary); }

.hsv-title { font-size: 12px; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.5px; margin: 14px 0 8px; }
.hsv-title.yellow { color: #f1c40f; }
.hsv-title.white  { color: #ecf0f1; }

#camDot { display:inline-block; width:8px; height:8px; border-radius:50%;
          background:#e74c3c; margin-right:6px; }
'''

_CONTENT = '''
<div class="container">
    <div class="video-section">
        <img src="/video" class="stream" alt="Camera Feed">
    </div>

    <div class="controls-section">

        <!-- Convoy Status -->
        <div class="card">
            <div class="card-header">
                Convoy Status
                <span id="camDot"></span>
            </div>

            <div style="margin-bottom:12px;text-align:center">
                <span id="stateLabel" class="convoy-state state-unknown">—</span>
            </div>

            <div class="info-row">
                <span class="info-key">Role</span>
                <span class="info-val" id="iRole">—</span>
            </div>
            <div class="info-row">
                <span class="info-key">Sign</span>
                <span id="iSign" class="sign-badge sign-none">NONE</span>
            </div>
            <div class="info-row">
                <span class="info-key">Speed</span>
                <span class="info-val" id="iSpeed">—</span>
            </div>
            <div class="info-row">
                <span class="info-key">Tags seen</span>
                <span class="info-val" id="iTags">—</span>
            </div>
            <div class="info-row">
                <span class="info-key">Frames</span>
                <span class="info-val" id="iFrames">—</span>
            </div>
        </div>

        <!-- Drive Control -->
        <div class="card">
            <div class="card-header">Drive Control</div>
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                <span id="runDot" style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#e74c3c;flex-shrink:0"></span>
                <span id="runLabel" style="font-size:13px;font-weight:600;color:var(--text-secondary)">STOPPED</span>
            </div>
            <div style="display:flex;gap:8px">
                <button onclick="driveStart()" class="button success" style="flex:1">▶  Start</button>
                <button onclick="driveStop()"  class="button danger"  style="flex:1">■  Stop</button>
            </div>
        </div>

        <!-- Lane Parameters -->
        <div class="card">
            <div class="card-header">Lane Control</div>

            <div class="control-group">
                <label>Lateral Gain (k_d)</label>
                <div class="control-row">
                    <input type="range" id="k_d" class="slider" min="0" max="1" step="0.01" value="0.07">
                    <span class="value-display" id="k_d_val">0.07</span>
                </div>
            </div>
            <div class="control-group">
                <label>Heading Gain (k_phi)</label>
                <div class="control-row">
                    <input type="range" id="k_phi" class="slider" min="0" max="2" step="0.01" value="0.18">
                    <span class="value-display" id="k_phi_val">0.18</span>
                </div>
            </div>
            <div class="control-group">
                <label>Base Speed <span style="color:var(--text-muted);font-size:11px">(0–1)</span></label>
                <div class="control-row">
                    <input type="range" id="const" class="slider" min="0" max="1" step="0.01" value="0.48">
                    <span class="value-display" id="const_val">0.48</span>
                </div>
            </div>
            <button onclick="applyConfig()" class="button success">Apply</button>
            <div id="cfg-status" class="status"></div>
        </div>

        <!-- HSV Calibration -->
        <div class="card">
            <div class="card-header">HSV Calibration</div>

            <div class="hsv-title yellow">Yellow line (left / dashed)</div>
            <div class="slider-group">
                <div class="slider-label"><span>Hue Low</span><span style="color:var(--text-muted)">0–179</span></div>
                <div class="slider-controls">
                    <input type="range" id="yLowH"  min="0" max="179" value="15" class="slider">
                    <input type="number" id="yLowH-input"  min="0" max="179" value="15" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Hue High</span><span style="color:var(--text-muted)">0–179</span></div>
                <div class="slider-controls">
                    <input type="range" id="yHighH" min="0" max="179" value="80" class="slider">
                    <input type="number" id="yHighH-input" min="0" max="179" value="80" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Saturation Low</span><span style="color:var(--text-muted)">0–255</span></div>
                <div class="slider-controls">
                    <input type="range" id="yLowS"  min="0" max="255" value="80" class="slider">
                    <input type="number" id="yLowS-input"  min="0" max="255" value="80" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Saturation High</span><span style="color:var(--text-muted)">0–255</span></div>
                <div class="slider-controls">
                    <input type="range" id="yHighS" min="0" max="255" value="255" class="slider">
                    <input type="number" id="yHighS-input" min="0" max="255" value="255" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Value Low</span><span style="color:var(--text-muted)">0–255</span></div>
                <div class="slider-controls">
                    <input type="range" id="yLowV"  min="0" max="255" value="80" class="slider">
                    <input type="number" id="yLowV-input"  min="0" max="255" value="80" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Value High</span><span style="color:var(--text-muted)">0–255</span></div>
                <div class="slider-controls">
                    <input type="range" id="yHighV" min="0" max="255" value="255" class="slider">
                    <input type="number" id="yHighV-input" min="0" max="255" value="255" class="input-box">
                </div>
            </div>

            <div class="hsv-title white" style="margin-top:18px">White line (right / solid)</div>
            <div class="slider-group">
                <div class="slider-label"><span>Hue Low</span><span style="color:var(--text-muted)">0–179</span></div>
                <div class="slider-controls">
                    <input type="range" id="wLowH"  min="0" max="179" value="0" class="slider">
                    <input type="number" id="wLowH-input"  min="0" max="179" value="0" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Hue High</span><span style="color:var(--text-muted)">0–179</span></div>
                <div class="slider-controls">
                    <input type="range" id="wHighH" min="0" max="179" value="179" class="slider">
                    <input type="number" id="wHighH-input" min="0" max="179" value="179" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Saturation Low</span><span style="color:var(--text-muted)">0–255</span></div>
                <div class="slider-controls">
                    <input type="range" id="wLowS"  min="0" max="255" value="0" class="slider">
                    <input type="number" id="wLowS-input"  min="0" max="255" value="0" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Saturation High</span><span style="color:var(--text-muted)">0–255</span></div>
                <div class="slider-controls">
                    <input type="range" id="wHighS" min="0" max="255" value="40" class="slider">
                    <input type="number" id="wHighS-input" min="0" max="255" value="40" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Value Low</span><span style="color:var(--text-muted)">0–255</span></div>
                <div class="slider-controls">
                    <input type="range" id="wLowV"  min="0" max="255" value="180" class="slider">
                    <input type="number" id="wLowV-input"  min="0" max="255" value="180" class="input-box">
                </div>
            </div>
            <div class="slider-group">
                <div class="slider-label"><span>Value High</span><span style="color:var(--text-muted)">0–255</span></div>
                <div class="slider-controls">
                    <input type="range" id="wHighV" min="0" max="255" value="255" class="slider">
                    <input type="number" id="wHighV-input" min="0" max="255" value="255" class="input-box">
                </div>
            </div>
            <div id="hsv-status" class="status"></div>
        </div>

    </div>
</div>
'''

_JS = '''
// ── Status poll ──────────────────────────────────────────────────
const STATE_CSS = {
    CRUISING: 'state-cruising',
    SLOW:     'state-slow',
    STOPPING: 'state-stopping',
    STOPPED:  'state-stopped',
};

function refreshStatus() {
    fetch('/status')
        .then(r => r.json())
        .then(d => {
            // Camera dot
            const cam = document.getElementById('camDot');
            cam.style.background = d.camera_ready ? '#3fb950' : '#e74c3c';
            cam.title = d.camera_ready ? 'Camera ready' : 'Camera not ready';

            // State badge
            const sl  = document.getElementById('stateLabel');
            const st  = (d.convoy_state || 'UNKNOWN').toUpperCase();
            sl.textContent = st;
            sl.className = 'convoy-state ' + (STATE_CSS[st] || 'state-unknown');

            // Role
            document.getElementById('iRole').textContent = (d.role || '—').toUpperCase();

            // Sign event
            const ev   = (d.event || '').toUpperCase();
            const sb   = document.getElementById('iSign');
            if (ev.includes('STOP')) {
                sb.textContent = 'STOP';
                sb.className = 'sign-badge sign-stop';
            } else if (ev.includes('SLOW')) {
                sb.textContent = 'SLOW';
                sb.className = 'sign-badge sign-slow';
            } else {
                sb.textContent = 'NONE';
                sb.className = 'sign-badge sign-none';
            }

            // Speed / tags / frames
            document.getElementById('iSpeed').textContent =
                typeof d.convoy_speed === 'number' ? d.convoy_speed.toFixed(3) : '—';
            const tags = Array.isArray(d.tag_ids) && d.tag_ids.length > 0
                ? d.tag_ids.join(', ') : '—';
            document.getElementById('iTags').textContent   = tags;
            document.getElementById('iFrames').textContent = d.frame_count || '—';

            // Running indicator
            const running = d.running === true;
            document.getElementById('runDot').style.background   = running ? '#3fb950' : '#e74c3c';
            document.getElementById('runLabel').textContent       = running ? 'RUNNING' : 'STOPPED';
            document.getElementById('runLabel').style.color       = running ? '#3fb950' : 'var(--text-secondary)';

            // Prefill lane control sliders from config
            if (d.config) {
                setSliderPair('k_d',   'k_d_val',   d.config.p_gain);
                setSliderPair('k_phi', 'k_phi_val', d.config.d_gain);
                setSliderPair('const', 'const_val', d.config.base_speed);
            }
        })
        .catch(() => {
            document.getElementById('stateLabel').className = 'convoy-state state-unknown';
            document.getElementById('stateLabel').textContent = 'offline';
        });
}

// ── Drive control ────────────────────────────────────────────────
function driveStart() {
    postJSON('/start', {})
        .then(() => { document.getElementById('runDot').style.background = '#3fb950';
                      document.getElementById('runLabel').textContent = 'RUNNING'; })
        .catch(() => showStatus('cfg-status', 'Start failed', 'error'));
}
function driveStop() {
    postJSON('/stop', {})
        .then(() => { document.getElementById('runDot').style.background = '#e74c3c';
                      document.getElementById('runLabel').textContent = 'STOPPED'; })
        .catch(() => showStatus('cfg-status', 'Stop failed', 'error'));
}

// ── Lane config ──────────────────────────────────────────────────
function setSliderPair(sliderId, valId, value) {
    if (value == null) return;
    const el = document.getElementById(sliderId);
    if (el && el.value == el.defaultValue) {   // only update if user hasn't touched it
        el.value = value;
        document.getElementById(valId).textContent = parseFloat(value).toFixed(2);
    }
}

document.getElementById('k_d').oninput = function() {
    document.getElementById('k_d_val').textContent = this.value; };
document.getElementById('k_phi').oninput = function() {
    document.getElementById('k_phi_val').textContent = this.value; };
document.getElementById('const').oninput = function() {
    document.getElementById('const_val').textContent = this.value; };

function applyConfig() {
    postJSON('/update_config', {
        k_d:   parseFloat(document.getElementById('k_d').value),
        k_phi: parseFloat(document.getElementById('k_phi').value),
        const: parseFloat(document.getElementById('const').value),
    })
    .then(() => showStatus('cfg-status', 'Applied!', 'success'))
    .catch(() => showStatus('cfg-status', 'Update failed', 'error'));
}

// ── HSV calibration ──────────────────────────────────────────────
fetch('/get_hsv')
    .then(r => r.json())
    .then(d => {
        setSliderValue('yLowH',  d.yellow_lower_h);
        setSliderValue('yHighH', d.yellow_upper_h);
        setSliderValue('yLowS',  d.yellow_lower_s);
        setSliderValue('yHighS', d.yellow_upper_s);
        setSliderValue('yLowV',  d.yellow_lower_v);
        setSliderValue('yHighV', d.yellow_upper_v);
        setSliderValue('wLowH',  d.white_lower_h);
        setSliderValue('wHighH', d.white_upper_h);
        setSliderValue('wLowS',  d.white_lower_s);
        setSliderValue('wHighS', d.white_upper_s);
        setSliderValue('wLowV',  d.white_lower_v);
        setSliderValue('wHighV', d.white_upper_v);
    });

const hsvMap = {
    yLowH:'yellow_lower_h', yHighH:'yellow_upper_h',
    yLowS:'yellow_lower_s', yHighS:'yellow_upper_s',
    yLowV:'yellow_lower_v', yHighV:'yellow_upper_v',
    wLowH:'white_lower_h',  wHighH:'white_upper_h',
    wLowS:'white_lower_s',  wHighS:'white_upper_s',
    wLowV:'white_lower_v',  wHighV:'white_upper_v',
};

Object.entries(hsvMap).forEach(([sid, key]) => {
    syncSliderInput(sid, () => {
        const payload = {};
        payload[key] = parseInt(document.getElementById(sid).value);
        postJSON('/update_hsv', payload)
            .then(() => showStatus('hsv-status', 'HSV updated', 'success'));
    });
});

// ── Poll loop ────────────────────────────────────────────────────
refreshStatus();
setInterval(refreshStatus, 800);
'''

CONVOY_TEMPLATE = render_template(
    'Convoy — {{ hostname }}',
    '{{ hostname }} — Convoy Controller',
    _CONTENT,
    extra_css=_EXTRA_CSS,
    extra_js=_JS,
)
