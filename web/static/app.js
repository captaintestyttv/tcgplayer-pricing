/* TCGPlayer Pricing - client-side JS */

// ---------------------------------------------------------------------------
// Table sorting
// ---------------------------------------------------------------------------
function sortTable(th, colIndex) {
    const table = th.closest("table");
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const type = th.dataset.sort || "string";

    // Toggle direction
    const isAsc = th.classList.contains("sort-asc");
    table.querySelectorAll("th").forEach(h => h.classList.remove("sort-asc", "sort-desc"));
    th.classList.add(isAsc ? "sort-desc" : "sort-asc");
    const dir = isAsc ? -1 : 1;

    rows.sort((a, b) => {
        let va = a.cells[colIndex].textContent.trim();
        let vb = b.cells[colIndex].textContent.trim();
        if (type === "number") {
            va = parseFloat(va.replace(/[$,]/g, "")) || 0;
            vb = parseFloat(vb.replace(/[$,]/g, "")) || 0;
            return (va - vb) * dir;
        }
        return va.localeCompare(vb) * dir;
    });

    rows.forEach(r => tbody.appendChild(r));
}

// ---------------------------------------------------------------------------
// Predictions filtering
// ---------------------------------------------------------------------------
function filterTable() {
    const signal = document.getElementById("filter-signal")?.value || "";
    const action = document.getElementById("filter-action")?.value || "";
    const name = (document.getElementById("filter-name")?.value || "").toLowerCase();

    document.querySelectorAll("#predictions-table tbody tr").forEach(row => {
        const matchSignal = !signal || row.dataset.signal === signal;
        const matchAction = !action || row.dataset.action === action;
        const matchName = !name || row.dataset.name.includes(name);
        row.style.display = (matchSignal && matchAction && matchName) ? "" : "none";
    });
}

// ---------------------------------------------------------------------------
// Background jobs
// ---------------------------------------------------------------------------
function triggerJob(type, body) {
    const url = "/api/" + type;
    fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: body ? JSON.stringify(body) : "{}",
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }
        showJobStatus(data);
        pollJob(data.id);
    })
    .catch(err => alert("Request failed: " + err));
}

function showJobStatus(job) {
    const el = document.getElementById("job-status");
    const text = document.getElementById("job-status-text");
    const progress = document.getElementById("job-progress");
    if (!el) return;

    el.hidden = false;
    text.textContent = `Job ${job.id} (${job.type}): ${job.status}`;
    if (job.status === "running") {
        progress.removeAttribute("value");
    } else {
        progress.value = 100;
        progress.max = 100;
    }
}

function pollJob(jobId) {
    const interval = setInterval(() => {
        fetch("/api/jobs/" + jobId)
        .then(r => r.json())
        .then(job => {
            showJobStatus(job);
            if (job.status !== "running") {
                clearInterval(interval);
                if (job.status === "failed") {
                    alert("Job failed: " + (job.error || "Unknown error"));
                }
            }
        });
    }, 2000);
}

// Import form handling
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("import-form");
    if (form) {
        form.addEventListener("submit", (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            fetch("/api/import", { method: "POST", body: formData })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    alert("Error: " + data.error);
                    return;
                }
                showJobStatus(data);
                pollJob(data.id);
            })
            .catch(err => alert("Upload failed: " + err));
        });
    }

    // Price chart on card page
    const chartCanvas = document.getElementById("price-chart");
    if (chartCanvas) {
        const tcgId = chartCanvas.dataset.tcgId;
        fetch("/api/card/" + tcgId + "/prices")
        .then(r => r.json())
        .then(data => renderPriceChart(chartCanvas, data));
    }

    // Calibration chart on backtest page
    if (window._calibrationData) {
        renderCalibrationChart();
    }

    // Feature importance chart on backtest page
    if (window._featureImportance) {
        renderFeatureImportanceChart();
    }
});

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------
function renderPriceChart(canvas, data) {
    const datasets = [];

    if (data.normal && Object.keys(data.normal).length > 0) {
        const sorted = Object.entries(data.normal).sort((a, b) => a[0].localeCompare(b[0]));
        datasets.push({
            label: "Normal",
            data: sorted.map(([d, p]) => ({x: d, y: p})),
            borderColor: "#4bc0c0",
            fill: false,
            tension: 0.1,
            pointRadius: 1,
        });
    }

    if (data.foil && Object.keys(data.foil).length > 0) {
        const sorted = Object.entries(data.foil).sort((a, b) => a[0].localeCompare(b[0]));
        datasets.push({
            label: "Foil",
            data: sorted.map(([d, p]) => ({x: d, y: p})),
            borderColor: "#ff9f40",
            fill: false,
            tension: 0.1,
            pointRadius: 1,
        });
    }

    if (data.buylist && Object.keys(data.buylist).length > 0) {
        const sorted = Object.entries(data.buylist).sort((a, b) => a[0].localeCompare(b[0]));
        datasets.push({
            label: "Buylist",
            data: sorted.map(([d, p]) => ({x: d, y: p})),
            borderColor: "#9966ff",
            fill: false,
            tension: 0.1,
            pointRadius: 1,
        });
    }

    if (datasets.length === 0) {
        canvas.parentElement.innerHTML = "<p>No price history available for this card.</p>";
        return;
    }

    new Chart(canvas, {
        type: "line",
        data: { datasets },
        options: {
            responsive: true,
            scales: {
                x: { type: "category", title: { display: true, text: "Date" } },
                y: { title: { display: true, text: "Price ($)" }, beginAtZero: false },
            },
            plugins: {
                legend: { position: "top" },
            },
        },
    });
}

function renderCalibrationChart() {
    const data = window._calibrationData;
    const canvas = document.getElementById("calibration-chart");
    if (!canvas || !data.length) return;

    new Chart(canvas, {
        type: "bar",
        data: {
            labels: data.map(b => b.range),
            datasets: [
                {
                    label: "Actual Spike Rate",
                    data: data.map(b => b.actual_spike_rate),
                    backgroundColor: "#4bc0c0",
                },
                {
                    label: "Avg Predicted Prob",
                    data: data.map(b => b.avg_predicted_prob),
                    backgroundColor: "#ff6384",
                },
            ],
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 1, title: { display: true, text: "Rate" } },
                x: { title: { display: true, text: "Probability Bin" } },
            },
        },
    });
}

function renderFeatureImportanceChart() {
    const data = window._featureImportance;
    const canvas = document.getElementById("feature-importance-chart");
    if (!canvas || !data.length) return;

    // data is sorted descending by importance already
    const top = data.slice(0, 15);

    new Chart(canvas, {
        type: "bar",
        data: {
            labels: top.map(f => f[0]),
            datasets: [{
                label: "Importance",
                data: top.map(f => f[1]),
                backgroundColor: "#36a2eb",
            }],
        },
        options: {
            indexAxis: "y",
            responsive: true,
            scales: {
                x: { beginAtZero: true, title: { display: true, text: "Importance" } },
            },
            plugins: { legend: { display: false } },
        },
    });
}

// ---------------------------------------------------------------------------
// Jobs page
// ---------------------------------------------------------------------------
function showJobLog(jobId) {
    fetch("/api/jobs/" + jobId)
    .then(r => r.json())
    .then(job => {
        document.getElementById("job-log-content").textContent = job.log || "(empty)";
        document.getElementById("job-log-error").textContent = job.error || "";
        document.getElementById("job-log-dialog").showModal();
    });
}

function refreshJobs() {
    location.reload();
}
