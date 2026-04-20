# ---------------------------------------------------------------------------
# CSS Styles
# ---------------------------------------------------------------------------

ERROR_ANALYSIS_CSS = """
/* Main content */
.main-content {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 32px 40px;
}

/* Summary cards */
.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin: 24px 0;
}
.summary-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  text-align: center;
}
.card-value {
  font-size: 24px;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 4px;
}
.card-label {
  font-size: 11px;
  color: var(--text-2);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* Error summary table */
.error-summary-container {
  margin: 32px 0;
}
.error-summary-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 12px;
}
.error-summary-scroll-wrapper {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}
.error-summary-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  background: var(--surface);
}
.error-summary-table th {
  padding: 10px 12px;
  text-align: left;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-2);
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  border-right: 1px solid var(--border-sub);
}
.error-summary-table th:last-child {
  border-right: none;
}
.error-summary-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border-sub);
  border-right: 1px solid var(--border-sub);
  vertical-align: top;
}
.error-summary-table td:last-child {
  border-right: none;
}
.error-summary-table tr:last-child td {
  border-bottom: none;
}
.error-summary-table .sticky-col {
  position: sticky;
  left: 0;
  background: var(--surface);
  font-weight: 600;
  color: var(--text-1);
  z-index: 10;
  box-shadow: 2px 0 4px rgba(0, 0, 0, 0.05);
}
.error-summary-table th.sticky-col {
  background: var(--bg);
  z-index: 11;
}
.error-summary-table .trace-cell {
  color: var(--brand);
  font-weight: 500;
}
.trace-expand-btn {
  margin-left: 6px;
  padding: 2px 8px;
  font-size: 11px;
  color: var(--brand);
  background: transparent;
  border: 1px solid var(--brand);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
}
.trace-expand-btn:hover {
  background: var(--brand);
  color: white;
}
.trace-list-hidden {
  display: none;
}
.trace-list-visible, .trace-list-hidden {
  display: inline;
}
.error-summary-warning {
  margin-top: 12px;
  padding: 12px;
  background: #fff3cd;
  border: 1px solid #ffc107;
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: #856404;
}

/* Selection controls */
.selection-section {
  margin: 32px 0;
}
.selection-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 16px;
}
.selection-controls {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* Search input */
.search-input {
  width: 100%;
  padding: 10px 12px;
  font-family: var(--font);
  font-size: 13px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-1);
  transition: border-color 0.2s;
}
.search-input:focus {
  outline: none;
  border-color: var(--brand);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}
.search-input::placeholder {
  color: var(--text-3);
}

/* Pair list selector */
.pair-list-container {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  margin-bottom: 20px;
  max-height: 400px;
  overflow-y: auto;
}
.pair-list {
  display: flex;
  flex-direction: column;
}
.pair-list-item {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-sub);
  cursor: pointer;
  font-size: 13px;
  color: var(--text-1);
  transition: background 0.15s ease;
}
.pair-list-item:hover {
  background: var(--bg);
}
.pair-list-item.selected {
  background: #e3f2fd;
  color: var(--brand);
  font-weight: 500;
}
.pair-list-item.sample-separator {
  border-bottom: 2px solid var(--border);
}
.pair-list-item:last-child {
  border-bottom: none;
}
.pair-list-item.hidden {
  display: none;
}
.pair-list-item.indented {
  padding-left: 32px;
}
.pair-list-sample-header {
  padding: 8px 16px;
  background: var(--bg);
  font-weight: 600;
  color: var(--text-1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border);
  cursor: default;
}
.pair-list-sample-header.hidden {
  display: none;
}
.pair-list-sample-header:hover {
  background: var(--bg);
}
.pair-list-sample-header-btn {
  padding: 4px 8px;
  font-size: 11px;
  font-weight: 500;
  background: var(--brand);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.2s;
}
.pair-list-sample-header-btn:hover {
  background: #4338ca;
}

/* Selected pairs chips */
.selected-pairs-section {
  margin-top: 16px;
}
.selected-pairs-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-2);
  margin-bottom: 8px;
}
.selected-pairs-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  min-height: 32px;
  max-height: 72px;
  padding: 8px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow-y: auto;
  overflow-x: hidden;
}
.selected-pairs-chips:empty::after {
  content: 'No pairs selected';
  color: var(--text-3);
  font-size: 12px;
}
.pair-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px 4px 12px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  font-size: 12px;
  color: var(--text-1);
}
.pair-chip-remove {
  cursor: pointer;
  color: var(--text-3);
  font-size: 14px;
  line-height: 1;
  padding: 2px;
  transition: color 0.2s;
}
.pair-chip-remove:hover {
  color: var(--score-lo);
}
.control-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.control-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-2);
}
.multi-select {
  width: 100%;
  min-height: 120px;
  padding: 8px;
  font-family: var(--font);
  font-size: 13px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-1);
}
.multi-select option {
  padding: 6px;
}
.btn-primary {
  padding: 10px 20px;
  background: var(--brand);
  color: white;
  border: none;
  border-radius: var(--radius-sm);
  font-family: var(--font);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}
.btn-primary:hover {
  background: #4338ca;
}
.btn-secondary {
  padding: 10px 20px;
  background: var(--surface);
  color: var(--text-1);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-family: var(--font);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.btn-secondary:hover {
  background: var(--bg);
  border-color: var(--text-3);
}

/* Plot container */
.plot-container {
  margin: 32px 0;
  padding: 24px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.plot-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 16px;
}
#dot-plot {
  width: 100%;
  height: 500px;
}

/* Heatmap comparison */
.heatmap-section {
  margin: 32px 0;
}
.heatmap-comparison {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 24px;
  margin-top: 16px;
}
.heatmap-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
}
.heatmap-panel-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 16px;
}
.heatmap-grid {
  display: grid;
  grid-template-columns: auto repeat(var(--judge-count, 5), 1fr);
  gap: 2px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.heatmap-row {
  display: contents;
}
.heatmap-label {
  background: var(--bg);
  padding: 8px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-2);
  display: flex;
  align-items: center;
}
.heatmap-header-cell {
  background: var(--bg);
  padding: 8px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-2);
  text-align: center;
}
.heatmap-cell {
  background: var(--surface);
  padding: 8px;
  font-size: 11px;
  font-weight: 600;
  text-align: center;
  cursor: help;
}
.cell-complete {
  background: #1976d2;
  color: white;
}
.cell-incomplete {
  background: #e0e0e0;
  color: var(--text-2);
}

/* Judge details */
.judge-details-container {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
}
.judge-details-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 12px;
}
.subgoal-selector {
  margin-bottom: 16px;
}
.subgoal-selector select {
  width: 100%;
  padding: 8px;
  font-family: var(--font);
  font-size: 13px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
}
.judge-item {
  margin-bottom: 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.judge-header {
  padding: 10px 14px;
  background: var(--bg);
  font-size: 12px;
  font-weight: 600;
  color: var(--text-1);
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.judge-header:hover {
  background: var(--border-sub);
}
.judge-content {
  padding: 14px;
  display: none;
}
.judge-content.open {
  display: block;
}
.judge-field {
  margin-bottom: 12px;
}
.judge-field-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-2);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
}
.judge-field-value {
  font-size: 12px;
  color: var(--text-1);
  line-height: 1.5;
}
.judge-field-code {
  font-family: var(--font-mono);
  font-size: 11px;
  background: var(--bg);
  padding: 12px;
  border-radius: var(--radius-sm);
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
}

/* Legend */
.legend {
  display: flex;
  gap: 16px;
  font-size: 11px;
  color: var(--text-2);
  margin-top: 12px;
  padding: 8px 12px;
  background: var(--bg);
  border-radius: var(--radius-sm);
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}
.legend-swatch {
  width: 16px;
  height: 16px;
  border-radius: 3px;
}
"""